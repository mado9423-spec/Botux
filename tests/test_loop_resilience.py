"""The loop keeps protecting positions when parts of its environment fail, and does not trade on
stale signals after a restart."""

from __future__ import annotations

import dataclasses
import itertools
import json
import sqlite3
from decimal import Decimal

import pytest

from botux.alerts import MemoryNotifier
from botux.app import build_paper_app
from botux.config import MarketSettings, Settings, StrategySettings
from botux.errors import DataError
from botux.exchange.rules import SymbolRules
from botux.models import OrderStatus, Quote, Side
from botux.risk.manager import RiskManager
from botux.runner import TradingLoop
from botux.storage import Store
from tests.conftest import FakeExchange
from tests.test_runner import FakeFeed, Harness, harness

D = Decimal


@pytest.fixture
def cfg(settings: Settings) -> Settings:
    return dataclasses.replace(
        settings,
        strategy=StrategySettings("sma_cross", {"fast": 2, "slow": 3}),
        market=MarketSettings(poll_seconds=30.0),
    )


def enter_long(h: Harness) -> None:
    h.cycle()
    h.cycle(hours=1)
    assert h.paper.balances()["BTC"].free > 0


def sells(h: Harness) -> list:  # type: ignore[type-arg]
    return [o for o in h.orders() if o.side is Side.SELL]


class FlakyFeed(FakeFeed):
    """Quotes can fail per symbol (the same prices are served for every symbol)."""

    def __init__(self, *a: object, **k: object) -> None:
        super().__init__(*a, **k)  # type: ignore[arg-type]
        self.failing: set[str] = set()

    def quote(self, symbol: str) -> Quote:
        if symbol in self.failing:
            raise DataError(f"invalid book ticker for {symbol}")
        return super().quote(symbol)


def two_symbol_harness(cfg: Settings, rules: SymbolRules, store: Store) -> Harness:
    eth = dataclasses.replace(rules, symbol="ETHUSDT", base_asset="ETH")
    s = dataclasses.replace(
        cfg, market=dataclasses.replace(cfg.market, symbols=("BTCUSDT", "ETHUSDT"))
    )
    h = Harness.__new__(Harness)
    from botux.clock import FakeClock
    from tests.conftest import HOUR_MS, T0, candles_from_closes
    from tests.test_runner import SCENARIO

    h.clock = FakeClock(T0 + 4 * HOUR_MS + 5_000)
    h.candles = candles_from_closes(SCENARIO)
    h.feed = FlakyFeed(h.clock, h.candles)
    h.feed.mark_to_last_close()
    h.http = FakeExchange()
    h.app = build_paper_app(
        s, http=h.http.http(), clock=h.clock, store=store, feed=h.feed,
        rules={"BTCUSDT": rules, "ETHUSDT": eth},
    )  # fmt: skip
    h.loop = h.app.loop
    h.notifier = MemoryNotifier()
    h.loop._notifier = h.notifier
    h.loop._om._notifier = h.notifier
    h.store = store
    return h


class TestProtectionIsIsolated:
    def test_one_symbols_failing_quote_does_not_disable_anothers_stop_loss(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = two_symbol_harness(cfg, rules, store)
        enter_long(h)
        assert h.paper.balances()["BTC"].free > 0
        assert isinstance(h.feed, FlakyFeed)
        h.feed.failing = {"ETHUSDT"}
        h.feed.bid, h.feed.ask = D("90.00"), D("90.01")  # BTC falls through its stop
        h.loop.run_cycle()
        stops = [
            o for o in h.orders() if o.symbol == "BTCUSDT" and o.intent_key.startswith("stop:")
        ]
        assert stops and stops[0].status is OrderStatus.FILLED
        assert h.loop.risk.consecutive_errors == 1  # the failure is still counted and logged
        assert any("cycle_error" in e["kind"] for e in store.events())

    def test_after_a_restart_a_held_symbol_with_a_dead_quote_cannot_disable_anothers_stop(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        """No last price is known for the dead symbol (fresh process): it is valued at its
        cost, and the other symbol's stop-loss still runs."""
        h = two_symbol_harness(cfg, rules, store)
        enter_long(h)
        assert h.paper.balances()["BTC"].free > 0 and h.paper.balances()["ETH"].free > 0
        assert isinstance(h.feed, FlakyFeed)
        h.loop._last_mid.clear()  # what a restart does
        h.feed.failing = {"ETHUSDT"}
        h.feed.bid, h.feed.ask = D("90.00"), D("90.01")
        h.loop.run_cycle()
        stops = [
            o for o in h.orders() if o.symbol == "BTCUSDT" and o.intent_key.startswith("stop:")
        ]
        assert stops and stops[0].status is OrderStatus.FILLED

    def test_a_failing_database_write_cannot_skip_the_stop_loss(
        self, cfg: Settings, rules: SymbolRules, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = harness(cfg, rules, store)
        enter_long(h)

        def boom(*_a: object, **_k: object) -> None:
            raise sqlite3.OperationalError("database or disk is full")

        monkeypatch.setattr(store, "add_equity", boom)
        h.feed.bid, h.feed.ask = D("90.00"), D("90.01")
        h.loop.run_cycle()
        assert any(o.intent_key.startswith("stop:") for o in h.orders())
        assert h.loop.risk.consecutive_errors == 1

    def test_a_dead_database_cannot_turn_an_error_into_a_dead_loop(
        self, cfg: Settings, rules: SymbolRules, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()

        def boom(*_a: object, **_k: object) -> None:
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(store, "log_event", boom)
        monkeypatch.setattr(store, "kv_set", boom)
        for _ in range(h.loop._s.risk.max_consecutive_errors):
            h.loop.run_cycle()  # must never raise
        assert h.loop._kill.is_set()  # the emergency stop was reached anyway
        assert "Trading halted" in h.notifier.titles()
        h.loop.run_cycle()
        assert h.loop.stopped


class TestEmergencyStopTellsTheTruth:
    def test_a_flatten_that_cannot_even_read_the_balances_is_reported(
        self, cfg: Settings, rules: SymbolRules, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = dataclasses.replace(cfg, risk=dataclasses.replace(cfg.risk, flatten_on_kill=True))
        h = harness(s, rules, store)
        enter_long(h)

        def down() -> dict:  # type: ignore[type-arg]
            raise DataError("account endpoint unreachable")

        h.loop._kill.trip("operator")
        monkeypatch.setattr(h.paper, "balances", down)
        h.loop.run_cycle()
        assert h.loop.stopped
        titles = h.notifier.titles()
        assert "EMERGENCY STOP" in titles and "Flatten failed" in titles
        text = " ".join(m for _, t, m in h.notifier.items if t == "Flatten failed")
        assert "may still be OPEN" in text

    def test_the_error_driven_stop_lists_the_open_position_even_after_a_restart(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        enter_long(h)
        held = h.paper.balances()["BTC"].free
        h.loop._last_positions.clear()  # as after a restart: nothing remembered from a snapshot
        h.feed.quote_error = DataError("quotes down")
        for _ in range(h.loop._s.risk.max_consecutive_errors):
            h.loop.run_cycle()
        (halted,) = [m for _, t, m in h.notifier.items if t == "Trading halted"]
        assert "BTCUSDT" in halted and str(held)[:5] in halted

    def test_positions_that_cannot_be_read_are_called_unknown_not_none(
        self, cfg: Settings, rules: SymbolRules, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.loop._last_positions.clear()

        def down() -> dict:  # type: ignore[type-arg]
            raise DataError("down")

        monkeypatch.setattr(h.paper, "balances", down)
        assert h.loop._open_positions_text().startswith("unknown")


class TestNoTradingOnStaleSignals:
    def make_second_loop(
        self, h: Harness, cfg: Settings, rules: SymbolRules, store: Store
    ) -> TradingLoop:
        risk = RiskManager(cfg.risk, fee_rate=D("0.001"), slippage_bps=D(5))
        return TradingLoop(
            settings=cfg, feed=h.feed, broker=h.paper, order_manager=h.loop.order_manager,
            risk=risk, positions=h.loop.positions, store=store, kill=h.loop._kill,
            notifier=h.notifier, clock=h.clock, strategy=h.loop._strategy,
            rules={"BTCUSDT": rules},
        )  # fmt: skip

    def test_a_restarted_process_does_not_act_on_a_signal_that_closed_while_it_was_away(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()  # first process: warm start on candle 3 (kv lastc persists, as in live mode)
        assert store.kv_get("lastc:BTCUSDT")
        h.clock.advance(3600 + 2 * 60)  # the crossover candle (4) closed 2 minutes ago
        h.feed.mark_to_last_close()
        second = self.make_second_loop(h, cfg, rules, store)  # a restart
        second.run_cycle()
        assert h.orders() == []  # candle 4 was noted, not traded
        assert store.kv_get("lastc:BTCUSDT") == str(h.candles[4].open_time_ms)

    def test_a_restarted_process_still_acts_on_a_strategy_exit_that_closed_while_away(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        """Only ENTRIES are held back by the warm-up: an exit is state, never stale."""
        h = harness(cfg, rules, store)
        enter_long(h)  # bought on candle 4
        h.clock.advance(4 * 3600)  # the exit candle (8) has closed while the bot was away
        h.feed.mark_to_last_close()
        second = self.make_second_loop(h, cfg, rules, store)  # a restart: nothing warmed up
        second.run_cycle()
        assert any(o.intent_key.endswith(":SELL") for o in h.orders())

    def test_a_late_cycle_in_the_same_process_skips_an_old_entry_but_not_an_old_exit(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.clock.advance(3600 + 20 * 60)  # the BUY candle closed 20 minutes ago (limit: 10)
        h.feed.mark_to_last_close()
        h.loop.run_cycle()
        assert h.orders() == []

    def test_the_age_limit_scales_with_the_interval_and_the_poll_rate(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        assert h.loop._max_signal_age_ms() == 600_000  # 1h candles: 10 minutes
        fast = dataclasses.replace(cfg, market=MarketSettings(interval="1m", poll_seconds=30.0))
        h.loop._s = fast
        assert h.loop._max_signal_age_ms() == 90_000  # never less than three polls


class TestCadence:
    def test_the_pause_between_cycles_accounts_for_the_time_a_cycle_took(
        self, cfg: Settings, rules: SymbolRules, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = harness(cfg, rules, store)
        starts: list[float] = []
        real = h.loop.run_cycle

        def slow_cycle() -> None:
            starts.append(h.clock.monotonic())
            h.clock.advance(20)  # this cycle took 20 s
            real()

        monkeypatch.setattr(h.loop, "run_cycle", slow_cycle)
        h.loop.run(max_cycles=3)
        gaps = [b - a for a, b in itertools.pairwise(starts)]
        assert gaps and all(abs(g - 30.0) < 1.5 for g in gaps)  # a 30 s rhythm, not 50 s


def test_json_state_is_plain(store: Store) -> None:
    store.kv_set("x", json.dumps({"a": 1}))
    assert json.loads(store.kv_get("x") or "{}") == {"a": 1}


class TestUnresolvedBuysCountAsExposure:
    def test_an_unknown_buy_is_part_of_the_exposure_of_the_next_symbol(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        from botux.models import OrderType
        from botux.storage import OrderRecord

        h = two_symbol_harness(cfg, rules, store)
        h.cycle()
        store.insert_order(
            OrderRecord(
                "bx" + "a" * 30, "k", "BTCUSDT", Side.BUY, OrderType.MARKET, D("0.5"), None,
                OrderStatus.UNKNOWN, h.clock.now_ms(), h.clock.now_ms(),
            )
        )  # fmt: skip
        _, view = h.loop._snapshot({"BTCUSDT": h.feed.quote("BTCUSDT")})
        assert view.positions_notional["BTCUSDT"] >= D("0.5") * D("99")
        assert view.equity < D("1100")  # counted as exposure, NOT as equity


class TestCapitalChangeSurvivesACrash:
    def test_the_rebase_request_is_remembered_until_a_cycle_has_applied_it(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()  # the day reference exists and is saved
        smaller = dataclasses.replace(cfg, risk=dataclasses.replace(cfg.risk, bot_capital=D(400)))
        second = TestNoTradingOnStaleSignals().make_second_loop(h, smaller, rules, store)
        assert store.kv_get("risk:rebase") == "1"  # written the moment the change is seen
        third = TestNoTradingOnStaleSignals().make_second_loop(h, smaller, rules, store)
        assert third.risk.rebase_pending  # a crash before the first cycle forgot nothing
        second.run_cycle()
        assert not store.kv_get("risk:rebase")  # applied and saved
