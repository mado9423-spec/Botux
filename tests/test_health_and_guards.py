"""Last-moment checks before an order, health checks before signals, human-only resets."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from botux.app import make_gate_check
from botux.cli import main
from botux.clock import FakeClock
from botux.config import MarketSettings, Settings, StrategySettings, parse_settings
from botux.errors import ConfigError, LiveTradingNotAllowed, StorageError
from botux.exchange.client import BinanceSpotClient
from botux.exchange.rules import SymbolRules
from botux.execution.order_manager import OrderIntent, OrderManager
from botux.execution.paper import PaperBroker
from botux.killswitch import KillSwitch
from botux.models import OrderStatus, OrderType, Quote, Side
from botux.runner import TradingLoop
from botux.safety import ARM_PHRASE, arm_live_interactive
from botux.secrets_guard import Credentials, Secret
from botux.storage import Store
from tests.conftest import FakeExchange
from tests.test_live_gate import live
from tests.test_runner import Harness, harness

D = Decimal


@pytest.fixture
def cfg(settings: Settings) -> Settings:
    return dataclasses.replace(
        settings,
        strategy=StrategySettings("sma_cross", {"fast": 2, "slow": 3}),
        market=MarketSettings(poll_seconds=1.0),
    )


def buys(h: Harness) -> list[object]:
    return [o for o in h.orders() if o.side is Side.BUY]


class TestEntryGuardBeforeEveryNewOrder:
    def om(self, rules: SymbolRules, store: Store, clock: FakeClock):  # type: ignore[no-untyped-def]
        from botux.alerts import MemoryNotifier

        paper = PaperBroker(
            {"BTCUSDT": rules}, {"USDT": D(1000)}, fee_rate=D("0.001"), slippage_bps=D(5)
        )
        paper.set_quote(Quote("BTCUSDT", D("99.99"), D("100.01"), clock.now_ms()))
        return OrderManager(paper, store, {"BTCUSDT": rules}, MemoryNotifier(), clock=clock), paper

    def intent(
        self, key: str, side: Side = Side.BUY, *, is_exit: bool = False, qty: str = "0.1"
    ) -> OrderIntent:
        return OrderIntent(key, "BTCUSDT", side, OrderType.MARKET, D(qty), D(100), is_exit=is_exit)

    def test_a_veto_stops_the_order_before_anything_is_written_or_sent(
        self, rules: SymbolRules, store: Store, clock: FakeClock
    ) -> None:
        om, paper = self.om(rules, store, clock)
        om.set_entry_guard(lambda _i: "gate closed a moment ago")
        res = om.submit(self.intent("k1"))
        assert res.outcome == "blocked" and "gate closed" in res.reason
        assert store.list_orders() == [] and paper.placed_count == 0

    def test_the_guard_is_asked_for_every_new_order_but_never_blocks_an_exit(
        self, rules: SymbolRules, store: Store, clock: FakeClock
    ) -> None:
        om, _paper = self.om(rules, store, clock)
        asked: list[str] = []

        def guard(i: OrderIntent) -> str | None:
            asked.append(i.intent_key)
            return "no"

        om.set_entry_guard(guard)
        om.submit(self.intent("a"))
        om.submit(self.intent("b"))
        assert asked == ["a", "b"]
        om.set_entry_guard(lambda _i: None)
        assert om.submit(self.intent("c")).outcome == "accepted"
        om.set_entry_guard(guard)
        sell = om.submit(self.intent("d", Side.SELL, is_exit=True, qty="0.09"))
        assert sell.outcome == "accepted" and asked == ["a", "b"]  # exit: guard not consulted

    def test_a_gate_that_closes_between_the_cycle_start_and_the_order_blocks_the_entry(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        calls = {"n": 0}

        def closes_on_second_look() -> None:
            calls["n"] += 1
            if calls["n"] >= 2:
                raise LiveTradingNotAllowed("arming file removed")

        h.loop._gate_check = closes_on_second_look
        h.cycle(
            hours=1
        )  # the BUY signal: call 1 = cycle start (open), call 2 = just before the order
        assert buys(h) == [] and h.paper.placed_count == 0
        assert "arming file removed" in (h.loop.risk.hard_halt_reason or "")
        assert "Live gate closed" in h.notifier.titles()

    def test_an_open_gate_lets_the_entry_through(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.loop._gate_check = lambda: None
        h.cycle(hours=1)
        assert len(buys(h)) == 1


class TestSettingsWatchedWhileRunning:
    def make(self, settings: Settings, clock: FakeClock, reload):  # type: ignore[no-untyped-def]
        s = live(settings)
        arm_live_interactive(s, clock.now_ms(), lambda _p: ARM_PHRASE, lambda _t: None)
        client = BinanceSpotClient(
            credentials=Credentials(Secret("A" * 64), Secret("B" * 64)),
            http=FakeExchange().http(),
            clock=clock,
        )
        return s, make_gate_check(s, KillSwitch(s.paths.kill_file, clock), clock, client, reload)

    def test_unchanged_settings_pass(self, settings: Settings, clock: FakeClock) -> None:
        s, check = self.make(settings, clock, None)
        _, check2 = self.make(settings, clock, lambda: s)
        check()
        check2()

    def test_settings_edited_on_disk_close_the_gate(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        s = live(settings)
        edited = dataclasses.replace(
            s, risk=dataclasses.replace(s.risk, max_order_notional=D(5000))
        )
        _, check = self.make(settings, clock, lambda: edited)
        with pytest.raises(LiveTradingNotAllowed, match="changed on disk"):
            check()

    def test_an_unreadable_or_invalid_settings_file_closes_the_gate(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        def broken() -> Settings:
            raise ConfigError("unknown key(s) in [risk]: oops")

        _, check = self.make(settings, clock, broken)
        with pytest.raises(LiveTradingNotAllowed, match="unreadable or invalid"):
            check()


class TestSpreadAndStorageHealth:
    def test_a_wide_spread_means_no_entry_and_a_single_warning(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.clock.advance(3600)
        h.feed.mark_to_last_close()
        h.feed.bid, h.feed.ask = D("95.00"), D("105.00")  # ~10% wide
        h.loop.run_cycle()
        assert buys(h) == [] and h.paper.placed_count == 0
        assert h.notifier.titles().count("Entry skipped: wide spread") == 1

    def test_a_normal_spread_still_trades(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.cycle(hours=1)
        assert len(buys(h)) == 1

    def test_the_spread_limit_is_validated(self) -> None:
        for bad in ("0", "-0.1", "0.5"):
            with pytest.raises(ConfigError, match="max_spread_pct"):
                parse_settings({"risk": {"max_spread_pct": bad}})
        assert parse_settings({"risk": {"max_spread_pct": "0.01"}}).risk.max_spread_pct == D("0.01")

    def test_a_corrupt_database_is_refused_at_start(
        self, cfg: Settings, rules: SymbolRules, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = harness(cfg, rules, store)
        monkeypatch.setattr(store, "integrity_ok", lambda: False)
        with pytest.raises(StorageError, match="integrity"):
            TradingLoop(
                settings=cfg,
                feed=h.feed,
                broker=h.paper,
                order_manager=h.loop.order_manager,
                risk=h.loop.risk,
                positions=h.loop.positions,
                store=store,
                kill=h.loop._kill,
                notifier=h.notifier,
                clock=h.clock,
                strategy=h.loop._strategy,
                rules={"BTCUSDT": rules},
            )

    def test_integrity_check_reports_a_sound_database(self, store: Store) -> None:
        assert store.integrity_ok()

    def test_an_unwritable_database_stops_all_order_flow_and_is_counted_as_an_error(
        self, cfg: Settings, rules: SymbolRules, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        real = store.kv_set

        def failing(key: str, value: str) -> None:
            if key == "health:ts":
                raise sqlite3.OperationalError("disk I/O error")
            real(key, value)

        monkeypatch.setattr(store, "kv_set", failing)
        h.cycle(hours=1)  # a BUY signal that must not be acted on
        assert buys(h) == [] and h.paper.placed_count == 0
        assert h.loop.risk.consecutive_errors == 1
        (event,) = store.events("cycle_error")
        assert json.loads(event["detail"])["type"] == "StorageError"


class TestManualDailyLossReset:
    def trip(self, h: Harness) -> None:
        h.cycle()
        h.cycle(hours=1)  # long
        h.feed.bid, h.feed.ask = D("40.00"), D("40.01")
        h.loop.run_cycle()
        h.loop.run_cycle()
        assert h.loop.risk.daily_loss_halted

    def test_the_halt_survives_midnight_and_restarts(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        self.trip(h)
        h.clock.advance(26 * 3600)  # past UTC midnight
        h.feed.bid, h.feed.ask = D("40.00"), D("40.01")
        h.loop.run_cycle()
        assert h.loop.risk.daily_loss_halted
        assert json.loads(store.kv_get("risk_state") or "{}")["daily_loss_halted"] is True

    def test_a_human_request_lifts_it_once_with_a_fresh_reference(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        self.trip(h)
        store.kv_set("ack:daily_loss", "1")  # what `botux reset-daily-loss` writes
        h.loop.run_cycle()
        assert not h.loop.risk.daily_loss_halted
        assert not store.kv_get("ack:daily_loss")
        (event,) = store.events("daily_loss_reset")
        assert json.loads(event["detail"])["lifted"] is True
        assert "Daily loss limit reset by a human" in h.notifier.titles()
        # nothing re-trips it by itself: the new reference is the equity at reset time
        h.loop.run_cycle()
        assert not h.loop.risk.daily_loss_halted

    def test_a_request_while_not_halted_changes_nothing_but_the_reference(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        store.kv_set("ack:daily_loss", "1")
        h.loop.run_cycle()
        (event,) = store.events("daily_loss_reset")
        assert json.loads(event["detail"])["lifted"] is False


class TestResetCommand:
    @pytest.fixture
    def cfg_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "botux.toml"
        path.write_text(
            "[paths]\n"
            f'kill_file = "{tmp_path / "KILL"}"\n'
            f'database = "{tmp_path / "db.sqlite3"}"\n'
            f'alerts_file = "{tmp_path / "alerts.jsonl"}"\n'
            "[live]\n"
            f'arm_file = "{tmp_path / "armed.json"}"\n'
        )
        return str(path)

    def test_it_needs_an_explicit_confirmation_and_touches_no_network(
        self,
        cfg_path: str,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def no_http(*_a: object, **_k: object) -> None:
            raise AssertionError("network used")

        monkeypatch.setattr(httpx.Client, "request", no_http)
        Store(tmp_path / "db-paper.sqlite3").close()
        assert main(["-c", cfg_path, "reset-daily-loss", "--mode", "paper"]) == 2
        assert "--confirm-checked" in capsys.readouterr().err
        s = Store(tmp_path / "db-paper.sqlite3")
        assert not s.kv_get("ack:daily_loss")
        s.close()
        args = ["-c", cfg_path, "reset-daily-loss", "--mode", "paper", "--confirm-checked"]
        assert main(args) == 0
        s = Store(tmp_path / "db-paper.sqlite3")
        assert s.kv_get("ack:daily_loss")
        s.close()
        assert main(["-c", cfg_path, "reset-daily-loss", "--confirm-checked"]) == 2  # no live db


def test_order_status_enum_is_unchanged() -> None:
    assert OrderStatus.FILLED.is_terminal  # guards the import used above
