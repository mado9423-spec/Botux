"""Regression tests for defects found by the independent review of the order lifecycle,
the live gate and the kill switch. Each class names the failure it prevents."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from botux.app import (
    KEY_MAX_LOOKUP_FAILURES,
    KEY_RECHECK_SECONDS,
    KEY_RETRY_SECONDS,
    bind_account,
    build_live_app,
    make_gate_check,
)
from botux.cli import main
from botux.clock import FakeClock
from botux.config import (
    ExecutionSettings,
    MarketSettings,
    RiskSettings,
    Settings,
    StrategySettings,
    parse_settings,
)
from botux.errors import ConfigError, ExchangeError, LiveTradingNotAllowed
from botux.exchange.client import BinanceSpotClient
from botux.exchange.rules import SymbolRules
from botux.execution.live import LiveBroker
from botux.execution.order_manager import LOST_REQUERY_MS, OrderIntent, outcome_of
from botux.execution.paper import FaultKind
from botux.killswitch import KillSwitch
from botux.ledger import build_ledger
from botux.models import Fill, OrderResult, OrderStatus, OrderType, Quote, Side
from botux.safety import ARM_PHRASE, LiveAuthorization, arm_live_interactive
from botux.secrets_guard import Credentials, Secret
from botux.storage import OrderRecord, Store
from tests.conftest import FakeExchange
from tests.test_live_gate import API_KEY, API_SECRET, ENV, GOOD_KEY, live, live_exchange
from tests.test_runner import Harness, harness

D = Decimal


@pytest.fixture
def cfg(settings: Settings) -> Settings:
    return dataclasses.replace(
        settings,
        strategy=StrategySettings("sma_cross", {"fast": 2, "slow": 3}),
        market=MarketSettings(poll_seconds=1.0),
        execution=ExecutionSettings(reconcile_attempts=2, reconcile_interval_seconds=0.0),
    )


def enter_long(h: Harness) -> None:
    h.cycle()
    h.cycle(hours=1)
    assert h.paper.balances()["BTC"].free > 0


def sells(h: Harness) -> list[OrderRecord]:
    return [o for o in h.orders() if o.side is Side.SELL]


def titles(h: Harness) -> list[str]:
    return h.notifier.titles()


class TestAnUnfilledExitIsNeverReportedAsDone:
    """Review: an answer of EXPIRED/0 filled was reported 'accepted', so a failed stop-loss or
    flatten raised no alert and the operator believed the account was flat."""

    def test_outcome_depends_on_what_really_happened(self) -> None:
        def rec(status: OrderStatus, executed: str) -> OrderRecord:
            return OrderRecord(
                "c", "k", "BTCUSDT", Side.SELL, OrderType.MARKET, D("1"), None, status, 1, 1,
                executed_qty=D(executed),
            )  # fmt: skip

        assert outcome_of(rec(OrderStatus.FILLED, "1")) == "accepted"
        assert outcome_of(rec(OrderStatus.NEW, "0")) == "accepted"  # resting on the book
        assert outcome_of(rec(OrderStatus.PARTIALLY_FILLED, "0.5")) == "accepted"  # still open
        for st in (OrderStatus.EXPIRED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            assert outcome_of(rec(st, "0")) == "not_filled"
            assert outcome_of(rec(st, "0.4")) == "partial"
        for st in (OrderStatus.UNKNOWN, OrderStatus.PENDING_SEND, OrderStatus.LOST):
            assert outcome_of(rec(st, "0")) == "uncertain"

    def test_a_stop_loss_the_venue_expired_unfilled_raises_a_critical_alert(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        enter_long(h)
        h.paper.fill_plan.push("0")  # the venue lets the next order lapse without a fill
        h.feed.bid, h.feed.ask = D("90.00"), D("90.01")
        h.loop.run_cycle()
        (stop,) = sells(h)
        assert stop.status is OrderStatus.EXPIRED and stop.executed_qty == 0
        assert "Exit order did not go through" in titles(h)
        assert h.paper.balances()["BTC"].free > 0  # still holding: the operator is told so
        h.clock.advance(61)  # the next minute the stop is attempted again, and works
        h.loop.run_cycle()
        assert sells(h)[-1].status is OrderStatus.FILLED

    def test_a_half_filled_stop_loss_is_reported_and_retried(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        enter_long(h)
        held = h.paper.balances()["BTC"].free
        h.paper.fill_plan.push("0.5")
        h.feed.bid, h.feed.ask = D("90.00"), D("90.01")
        h.loop.run_cycle()
        assert "Exit order did not go through" in titles(h)
        left = h.paper.balances()["BTC"].free
        assert 0 < left < held  # about half remains and is still protected
        assert h.loop.positions.get("BTCUSDT") is not None

    def test_a_flatten_that_did_not_fill_is_reported(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        s = dataclasses.replace(cfg, risk=dataclasses.replace(cfg.risk, flatten_on_kill=True))
        h = harness(s, rules, store)
        enter_long(h)
        h.paper.fill_plan.push("0")
        h.loop._kill.trip("test")
        h.loop.run_cycle()
        assert "Flatten failed" in titles(h) and h.loop.stopped
        assert h.paper.balances()["BTC"].free > 0


class TestALostSellKeepsExitsBlocked:
    """Review: after a SELL was declared LOST the exit block was lifted, so the stop-loss sent
    a second SELL for a position whose first SELL might still execute (or sell baseline coins)."""

    def lose_a_sell(self, h: Harness) -> None:
        enter_long(h)
        h.paper.faults.push(FaultKind.TIMEOUT_NOT_EXECUTED)
        h.feed.bid, h.feed.ask = D("90.00"), D("90.01")
        for _ in range(4):  # reconcile attempts (2) are used up across the cycles
            h.loop.run_cycle()
        (sell,) = sells(h)
        assert sell.status is OrderStatus.LOST

    def test_no_second_sell_until_a_human_decides(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        self.lose_a_sell(h)
        placed = h.paper.placed_count
        reason = h.loop.order_manager.is_symbol_blocked("BTCUSDT", for_exit=True)
        assert reason and "LOST" in reason
        h.clock.advance(61)  # a new stop-loss minute: still no second SELL
        h.loop.run_cycle()
        assert h.paper.placed_count == placed and len(sells(h)) == 1
        assert "Order LOST" in titles(h)
        assert any("SELLs on this symbol are blocked" in m for _, _, m in h.notifier.items)
        h.loop.order_manager.resolve_lost("BTCUSDT")  # the operator checked the exchange
        assert h.loop.order_manager.is_symbol_blocked("BTCUSDT", for_exit=True) is None

    def test_a_lost_buy_does_not_stop_an_exit(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        enter_long(h)
        store.insert_order(
            OrderRecord(
                "lostbuy", "kb", "BTCUSDT", Side.BUY, OrderType.MARKET, D("1"), None,
                OrderStatus.LOST, 1, 1,
            )
        )  # fmt: skip
        store.kv_set("lost:BTCUSDT", "lostbuy")
        assert h.loop.order_manager.is_symbol_blocked("BTCUSDT") is not None  # entries: blocked
        assert h.loop.order_manager.is_symbol_blocked("BTCUSDT", for_exit=True) is None


class TestLostOrdersAreLookedForAgain:
    """Review: an order that surfaced after being declared LOST was never recorded."""

    def test_a_late_fill_is_adopted_and_the_block_is_lifted(
        self, cfg: Settings, rules: SymbolRules, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        hidden = {"on": True}
        real_get = h.paper.get_order

        def get_order(symbol: str, cid: str) -> OrderResult | None:
            return None if hidden["on"] else real_get(symbol, cid)

        monkeypatch.setattr(h.paper, "get_order", get_order)
        h.paper.faults.push(FaultKind.TIMEOUT_EXECUTED)
        h.cycle(hours=1)  # BUY executes, but the venue "cannot find" it when asked
        for _ in range(3):
            h.loop.run_cycle()
        (buy,) = h.orders()
        assert buy.status is OrderStatus.LOST and store.kv_get("lost:BTCUSDT")
        assert buy.executed_qty == 0 and h.loop.positions.get("BTCUSDT") is None
        hidden["on"] = False  # the order becomes visible
        h.loop.run_cycle()  # too soon: it is only looked at every few minutes
        assert h.orders()[0].status is OrderStatus.LOST
        h.clock.advance(LOST_REQUERY_MS / 1000 + 1)
        h.loop.run_cycle()
        found = h.orders()[0]
        assert found.status is OrderStatus.FILLED and found.executed_qty > 0
        assert store.fills_for(found.client_order_id)
        assert not store.kv_get("lost:BTCUSDT")  # nothing left to resolve
        assert h.loop.positions.get("BTCUSDT") is not None  # the entry and stop are known now
        assert "Lost order found" in titles(h)

    def test_lost_orders_count_toward_the_daily_order_limit(self, store: Store) -> None:
        store.insert_order(
            OrderRecord(
                "l", "kl", "BTCUSDT", Side.BUY, OrderType.MARKET, D("1"), None,
                OrderStatus.LOST, 5, 5,
            )
        )  # fmt: skip
        store.insert_order(
            OrderRecord(
                "r", "kr", "BTCUSDT", Side.BUY, OrderType.MARKET, D("1"), None,
                OrderStatus.REJECTED, 5, 5,
            )
        )  # fmt: skip
        assert store.count_orders_since(0) == 1  # the LOST one may have executed; the rejected not


class TestOrderRowAndFillsAreWrittenTogether:
    """Review: a crash between the order update and its fills left a FILLED order without fills,
    which no reconciliation would ever repair."""

    def test_a_failure_while_recording_fills_rolls_the_order_update_back(
        self, cfg: Settings, rules: SymbolRules, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        real = store.add_fills
        calls = {"n": 0}

        def flaky(*a: object, **k: object) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("disk I/O error")
            real(*a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr(store, "add_fills", flaky)
        with pytest.raises(sqlite3.OperationalError):
            h.loop.order_manager.submit(
                OrderIntent("kx", "BTCUSDT", Side.BUY, OrderType.MARKET, D("0.5"), D("100.01"))
            )
        row = store.get_by_intent("kx")
        assert row is not None and row.status is OrderStatus.PENDING_SEND  # NOT terminal
        assert store.fills_for(row.client_order_id) == []
        # the next reconciliation repairs it: the order is fetched again and recorded in full
        h.loop.order_manager.reconcile_pending()
        row = store.get_by_intent("kx")
        assert row is not None and row.status is OrderStatus.FILLED
        assert len(store.fills_for(row.client_order_id)) == 1
        assert h.loop.positions.get("BTCUSDT") is not None

    def test_transactions_group_writes_and_roll_back_together(self, store: Store) -> None:
        def failing_transaction() -> None:
            with store.transaction():
                store.kv_set("a", "1")
                store.kv_set("b", "2")
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            failing_transaction()
        assert store.kv_get("a") is None and store.kv_get("b") is None
        with store.transaction():
            store.kv_set("a", "1")
            with store.transaction():  # joins the outer one
                store.kv_set("b", "2")
        assert store.kv_get("a") == "1" and store.kv_get("b") == "2"


class TestKillMeansStop:
    """Review: after `botux kill` a stop-loss or strategy exit could still be sent in the cycle
    in flight, although flatten_on_kill was false; and unresolved orders were not mentioned."""

    def test_no_exit_goes_out_once_killed_unless_flatten_is_configured(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        enter_long(h)
        placed = h.paper.placed_count
        h.loop._kill.trip("operator")
        h.feed.bid, h.feed.ask = D("90.00"), D("90.01")
        h.loop._cycle()  # the cycle that was already running when the stop arrived
        assert h.paper.placed_count == placed and sells(h) == []
        assert "Exit order did not go through" not in titles(h)  # not an error: it was a stop
        h.loop.run_cycle()
        assert h.loop.stopped and sells(h) == []
        assert "Open position left unprotected" in titles(h)

    def test_flatten_on_kill_still_sells(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        s = dataclasses.replace(cfg, risk=dataclasses.replace(cfg.risk, flatten_on_kill=True))
        h = harness(s, rules, store)
        enter_long(h)
        h.loop._kill.trip("operator")
        h.loop.run_cycle()
        assert len(sells(h)) == 1 and sells(h)[0].status is OrderStatus.FILLED

    def test_unresolved_orders_at_the_moment_of_the_stop_are_called_out(
        self, cfg: Settings, rules: SymbolRules, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patient = dataclasses.replace(
            cfg, execution=ExecutionSettings(reconcile_attempts=5, reconcile_interval_seconds=0.0)
        )
        h = harness(patient, rules, store)
        h.cycle()
        monkeypatch.setattr(h.paper, "get_order", lambda _s, _c: None)
        h.paper.faults.push(FaultKind.TIMEOUT_EXECUTED)
        h.cycle(hours=1)
        assert h.loop.order_manager.uncertain_orders()
        h.loop._kill.trip("operator")
        h.loop.run_cycle()
        assert "Orders with unknown outcome at emergency stop" in titles(h)

    def test_kill_command_fails_loudly_when_the_file_cannot_be_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)
        blocker = tmp_path / "a-regular-file"
        blocker.write_text("x")
        cfg_file = tmp_path / "botux.toml"
        cfg_file.write_text(f'[paths]\nkill_file = "{blocker / "KILL"}"\n')
        assert main(["-c", str(cfg_file), "kill"]) == 2
        err = capsys.readouterr().err
        assert "could NOT be written" in err and "will not see" in err

    def test_an_unwritable_kill_location_stops_the_start_up(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("x")
        with pytest.raises(ConfigError, match="not writable"):
            KillSwitch(blocker / "KILL").ensure_writable()
        (tmp_path / "dir").mkdir()
        with pytest.raises(ConfigError, match="directory"):
            KillSwitch(tmp_path / "dir").ensure_writable()
        KillSwitch(tmp_path / "ok" / "KILL").ensure_writable()  # creates the folder, leaves no file
        assert not (tmp_path / "ok" / "KILL").exists()

    def test_trip_reports_whether_the_file_exists(self, tmp_path: Path) -> None:
        assert KillSwitch(tmp_path / "x" / "KILL").trip("r") is True
        blocker = tmp_path / "f"
        blocker.write_text("x")
        assert KillSwitch(blocker / "KILL").trip("r") is False


class TestAGateHaltDoesNotOutliveTheProcess:
    """Review: a halt caused by an expired arming was persisted, so re-arming + restarting never
    re-enabled entries and no command could clear it."""

    def test_after_rearming_and_restarting_entries_are_possible_again(
        self, settings: Settings, clock: FakeClock, tmp_path: Path
    ) -> None:
        s = live(settings)
        arm_live_interactive(s, clock.now_ms(), lambda _p: ARM_PHRASE, lambda _t: None)
        db = tmp_path / "live.sqlite3"
        store = Store(db)
        app = build_live_app(s, ENV, http=live_exchange(clock).http(), clock=clock, store=store)
        clock.advance(25 * 3600)  # the arming expires
        app.loop.run_cycle()
        assert app.loop.risk.entries_allowed()[0] is False
        state = json.loads(store.kv_get("risk_state") or "{}")
        assert state.get("hard_halt") is None  # nothing sticky was written
        app.close()
        store.close()
        arm_live_interactive(s, clock.now_ms(), lambda _p: ARM_PHRASE, lambda _t: None)
        store2 = Store(db)
        app2 = build_live_app(s, ENV, http=live_exchange(clock).http(), clock=clock, store=store2)
        app2.loop.run_cycle()
        assert app2.loop.risk.blockers.get("live-gate") is None
        app2.close()
        store2.close()


class TestHostsMustBelongToOneEnvironment:
    """Review: testnet prices could size real mainnet orders (hosts were validated separately)."""

    def test_config_refuses_mixed_environments(self) -> None:
        for md, tr in (
            ("https://testnet.binance.vision", "https://api.binance.com"),
            ("https://data-api.binance.vision", "https://testnet.binance.vision"),
        ):
            with pytest.raises(ConfigError, match="same Binance environment"):
                parse_settings({"market": {"market_data_base_url": md, "trading_base_url": tr}})
        parse_settings(
            {
                "market": {
                    "market_data_base_url": "https://testnet.binance.vision",
                    "trading_base_url": "https://testnet.binance.vision",
                }
            }
        )

    def test_the_client_refuses_them_too(self, clock: FakeClock) -> None:
        with pytest.raises(ConfigError, match="same Binance environment"):
            BinanceSpotClient(
                market_data_base_url="https://testnet.binance.vision",
                trading_base_url="https://api.binance.com",
                clock=clock,
            )


class TestKeyPermissionLookups:
    """Review: a failed hourly lookup was skipped for another hour and never escalated."""

    def make(self, settings: Settings, clock: FakeClock, ex: FakeExchange):  # type: ignore[no-untyped-def]
        s = live(settings)
        arm_live_interactive(s, clock.now_ms(), lambda _p: ARM_PHRASE, lambda _t: None)
        client = BinanceSpotClient(
            credentials=Credentials(Secret(API_KEY), Secret(API_SECRET)),
            http=ex.http(),
            clock=clock,
        )
        return make_gate_check(s, KillSwitch(s.paths.kill_file, clock), clock, client)

    def test_a_failed_lookup_is_retried_after_a_minute_not_an_hour(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        ex = FakeExchange()
        state = {"ok": False}
        ex.on(
            "GET",
            "/sapi/v1/account/apiRestrictions",
            lambda _r: httpx.Response(200, json=GOOD_KEY) if state["ok"] else httpx.Response(503),
        )
        check = self.make(settings, clock, ex)
        clock.advance(KEY_RECHECK_SECONDS + 1)
        check()
        asked = ex.count("/sapi/v1/account/apiRestrictions")
        assert asked >= 1
        check()  # immediately again: not yet
        assert ex.count("/sapi/v1/account/apiRestrictions") == asked
        clock.advance(KEY_RETRY_SECONDS + 1)
        state["ok"] = True
        check()
        assert ex.count("/sapi/v1/account/apiRestrictions") == asked + 1
        check()  # success resets the schedule to the hourly rhythm
        assert ex.count("/sapi/v1/account/apiRestrictions") == asked + 1

    def test_too_many_failures_in_a_row_close_the_gate(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        ex = FakeExchange()
        ex.on("GET", "/sapi/v1/account/apiRestrictions", lambda _r: httpx.Response(503))
        check = self.make(settings, clock, ex)
        clock.advance(KEY_RECHECK_SECONDS + 1)

        def keep_checking() -> None:
            for _ in range(KEY_MAX_LOOKUP_FAILURES + 1):
                check()
                clock.advance(KEY_RETRY_SECONDS + 1)

        with pytest.raises(LiveTradingNotAllowed, match="could not be verified"):
            keep_checking()


class TestSignedClientIgnoresTheEnvironment:
    """Review: a stray HTTPS_PROXY / CA variable could route signed requests via a third party."""

    def test_only_credentialed_clients_distrust_the_environment(self) -> None:
        public = BinanceSpotClient()
        signed = BinanceSpotClient(credentials=Credentials(Secret(API_KEY), Secret(API_SECRET)))
        assert public._http.trust_env is True
        assert signed._http.trust_env is False
        public.close()
        signed.close()

    def test_a_proxy_variable_does_not_reach_a_signed_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
        signed = BinanceSpotClient(credentials=Credentials(Secret(API_KEY), Secret(API_SECRET)))
        assert not signed._http._mounts  # no env-derived proxy transports
        signed.close()


class TestADatabaseBelongsToOneAccount:
    """Review: pointing the key at another account made the bot inherit the old baselines."""

    def test_binding(self, store: Store) -> None:
        bind_account(store, "111")
        bind_account(store, "111")
        with pytest.raises(LiveTradingNotAllowed, match="different Binance account"):
            bind_account(store, "222")
        with pytest.raises(LiveTradingNotAllowed, match="no uid"):
            bind_account(store, None)

    def test_the_live_app_refuses_a_second_account_on_the_same_database(
        self, settings: Settings, clock: FakeClock, tmp_path: Path
    ) -> None:
        s = live(settings)
        arm_live_interactive(s, clock.now_ms(), lambda _p: ARM_PHRASE, lambda _t: None)
        db = tmp_path / "live.sqlite3"
        first_store = Store(db)
        first = build_live_app(
            s, ENV, http=live_exchange(clock).http(), clock=clock, store=first_store
        )
        first.close()
        first_store.close()
        other = live_exchange(clock)
        other.on("GET", "/api/v3/account", {"canTrade": True, "uid": 999, "balances": []})
        second_store = Store(db)
        try:
            with pytest.raises(LiveTradingNotAllowed, match="different Binance account"):
                build_live_app(s, ENV, http=other.http(), clock=clock, store=second_store)
        finally:
            second_store.close()

    def test_an_account_answer_without_uid_is_refused(
        self, settings: Settings, clock: FakeClock, store: Store
    ) -> None:
        s = live(settings)
        arm_live_interactive(s, clock.now_ms(), lambda _p: ARM_PHRASE, lambda _t: None)
        ex = live_exchange(clock)
        ex.on("GET", "/api/v3/account", {"canTrade": True, "balances": []})
        with pytest.raises(LiveTradingNotAllowed, match="no uid"):
            build_live_app(s, ENV, http=ex.http(), clock=clock, store=store)


class TestRealCommissionsFromMyTrades:
    """Review: orders adopted through GET /order got commission 0 / unknown asset."""

    def setup_broker(self, clock: FakeClock, rules: SymbolRules, ex: FakeExchange) -> LiveBroker:
        client = BinanceSpotClient(
            credentials=Credentials(Secret(API_KEY), Secret(API_SECRET)),
            trading_enabled=True,
            http=ex.http(),
            clock=clock,
        )
        return LiveBroker(client, {"BTCUSDT": rules}, LiveAuthorization("f", 0))

    def trades(self, qtys: tuple[str, str]) -> list[dict[str, object]]:
        return [
            {"id": 2, "price": "60010.00", "qty": qtys[1], "commission": "0.00001",
             "commissionAsset": "BNB"},
            {"id": 1, "price": "60000.00", "qty": qtys[0], "commission": "0.0000005",
             "commissionAsset": "BTC"},
        ]  # fmt: skip

    def order_answer(self, executed: str) -> dict[str, object]:
        return {
            "symbol": "BTCUSDT", "orderId": 77, "clientOrderId": "bxreal", "price": "0",
            "origQty": "0.001", "executedQty": executed, "cummulativeQuoteQty": "60.005",
            "status": "FILLED", "type": "MARKET", "side": "BUY", "transactTime": 1,
        }  # fmt: skip

    def test_exact_fills_replace_the_estimate_when_they_add_up(
        self, clock: FakeClock, rules: SymbolRules, store: Store
    ) -> None:
        from botux.alerts import MemoryNotifier
        from botux.execution.order_manager import OrderManager

        ex = FakeExchange()
        ex.on("GET", "/api/v3/order", self.order_answer("0.001"))
        ex.on("GET", "/api/v3/myTrades", self.trades(("0.0005", "0.0005")))
        broker = self.setup_broker(clock, rules, ex)
        om = OrderManager(broker, store, {"BTCUSDT": rules}, MemoryNotifier(), clock=clock)
        rec = OrderRecord(
            "bxreal", "kk", "BTCUSDT", Side.BUY, OrderType.MARKET, D("0.001"), None,
            OrderStatus.UNKNOWN, 1, 1,
        )  # fmt: skip
        store.insert_order(rec)
        om._reconcile_one(rec)
        fills = store.fills_for("bxreal")
        assert [(f.price, f.quantity, f.commission_asset) for f in fills] == [
            (D("60000.00"), D("0.0005"), "BTC"),  # ordered by trade id, not by arrival
            (D("60010.00"), D("0.0005"), "BNB"),
        ]
        weights = [r for r in ex.requests if r.url.path == "/api/v3/myTrades"]
        assert len(weights) == 1 and weights[0].url.params["orderId"] == "77"
        led = build_ledger(
            D(1000), [("BTCUSDT", Side.BUY, f) for f in fills], {"BTCUSDT": rules}, D("0.001")
        )
        assert led.base_qty("BTCUSDT") == D("0.0009995")  # BTC fee deducted, BNB fee is not base
        assert led.symbols["BTCUSDT"].third_asset_fees and led.symbols["BTCUSDT"].inexact_qty == 0

    def test_trades_that_do_not_add_up_are_not_trusted(
        self, clock: FakeClock, rules: SymbolRules, store: Store
    ) -> None:
        from botux.alerts import MemoryNotifier
        from botux.execution.order_manager import OrderManager

        ex = FakeExchange()
        ex.on("GET", "/api/v3/order", self.order_answer("0.001"))
        ex.on("GET", "/api/v3/myTrades", self.trades(("0.0005", "0.0003")))  # 0.0008 != 0.001
        om = OrderManager(
            self.setup_broker(clock, rules, ex), store, {"BTCUSDT": rules}, MemoryNotifier(),
            clock=clock,
        )  # fmt: skip
        rec = OrderRecord(
            "bxreal", "kk", "BTCUSDT", Side.BUY, OrderType.MARKET, D("0.001"), None,
            OrderStatus.UNKNOWN, 1, 1,
        )  # fmt: skip
        store.insert_order(rec)
        om._reconcile_one(rec)
        (fill,) = store.fills_for("bxreal")
        assert fill.quantity == D("0.001") and fill.commission_asset == ""  # honest estimate

    def test_a_failing_trades_lookup_falls_back_to_the_estimate(
        self, clock: FakeClock, rules: SymbolRules, store: Store
    ) -> None:
        from botux.alerts import MemoryNotifier
        from botux.execution.order_manager import OrderManager

        ex = FakeExchange()
        ex.on("GET", "/api/v3/order", self.order_answer("0.001"))
        ex.on(
            "GET",
            "/api/v3/myTrades",
            lambda _r: httpx.Response(400, json={"code": -1102, "msg": "x"}),
        )
        om = OrderManager(
            self.setup_broker(clock, rules, ex), store, {"BTCUSDT": rules}, MemoryNotifier(),
            clock=clock,
        )  # fmt: skip
        rec = OrderRecord(
            "bxreal", "kk", "BTCUSDT", Side.BUY, OrderType.MARKET, D("0.001"), None,
            OrderStatus.UNKNOWN, 1, 1,
        )  # fmt: skip
        store.insert_order(rec)
        try:
            om._reconcile_one(rec)
        except ExchangeError:  # pragma: no cover - the lookup error must be absorbed
            pytest.fail("a failed trades lookup must not break reconciliation")
        assert store.fills_for("bxreal")[0].commission_asset == ""

    def test_myTrades_is_a_read_only_signed_endpoint(self) -> None:
        from botux.exchange.client import ENDPOINTS, Security

        ep = ENDPOINTS["my_trades"]
        assert ep.method == "GET" and ep.security is Security.USER_DATA and not ep.changes_state


class TestStopRearmedFromTheRealCost:
    """Review: after a crash between a fill and its bookkeeping the stop was re-armed from the
    CURRENT price, which can be far below the real entry."""

    def test_average_cost_follows_buys_and_sells(self, rules: SymbolRules) -> None:
        rows = [
            ("BTCUSDT", Side.BUY, Fill(D(100), D(1), D(0), "BTC")),
            ("BTCUSDT", Side.BUY, Fill(D(110), D(1), D(0), "BTC")),
            ("BTCUSDT", Side.SELL, Fill(D(130), D(1), D(0), "USDT")),
        ]
        v = build_ledger(D(1000), rows, {"BTCUSDT": rules}, D("0.001"))
        assert v.average_cost["BTCUSDT"] == D("105")  # a sale does not change the average cost
        round_trip = [rows[0], ("BTCUSDT", Side.SELL, Fill(D(90), D(1), D(0), "USDT"))]
        flat = build_ledger(D(1000), round_trip, {"BTCUSDT": rules}, D("0.001"))
        assert "BTCUSDT" not in flat.average_cost

    def lose_the_bookkeeping(self, h: Harness) -> Decimal:
        enter_long(h)
        info = h.loop.positions.get("BTCUSDT")
        assert info is not None
        store = h.store
        store.kv_set("pos:BTCUSDT", "")  # the bookkeeping was lost in a crash
        h.loop.positions._cache.clear()
        return info.entry_price

    def test_the_loop_rearms_from_average_cost(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        entry = self.lose_the_bookkeeping(h)
        h.feed.bid, h.feed.ask = D("99.50"), D("99.52")  # below the entry, above its stop
        h.loop.run_cycle()
        rearmed = h.loop.positions.get("BTCUSDT")
        assert rearmed is not None
        assert abs(rearmed.entry_price - entry) < D("0.5")  # the real cost, not today's 99.51
        assert rearmed.stop_price == rearmed.entry_price * D("0.98")
        assert "Stop-loss re-armed" in titles(h) and sells(h) == []

    def test_a_position_already_below_its_real_stop_is_closed_at_once(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        self.lose_the_bookkeeping(h)
        h.feed.bid, h.feed.ask = D("95.00"), D("95.02")  # a stop re-armed from 95 would be 93.1
        h.loop.run_cycle()
        (stop,) = sells(h)
        assert stop.status is OrderStatus.FILLED and stop.intent_key.startswith("stop:")


class TestGitignoreCoversCommonSecretNames:
    @pytest.mark.parametrize(
        "path", [".envrc", "live.env", "binance.env", "keys.json", "api_keys.txt", "alerts.jsonl"]
    )
    def test_ignored(self, path: str) -> None:
        import subprocess

        root = Path(__file__).resolve().parents[1]
        if not (root / ".git").exists():
            pytest.skip("not a git checkout")
        r = subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "check-ignore", "-q", path],  # noqa: S607
            check=False,
        )
        assert r.returncode == 0, f"{path} is not ignored"


def test_risk_settings_still_accept_defaults() -> None:
    assert RiskSettings().bot_capital is None


class TestExpiryReasonIsKept:
    """Binance reports WHY an order expired (e.g. the execution price range rule); a not-filled
    exit is much easier to understand with that reason in the order's note."""

    def test_parse_and_record(self, store: Store) -> None:
        from botux.execution.live import parse_order

        payload = {
            "symbol": "BTCUSDT", "orderId": 5, "clientOrderId": "bxr", "price": "0",
            "origQty": "0.001", "executedQty": "0", "cummulativeQuoteQty": "0",
            "status": "EXPIRED", "type": "MARKET", "side": "SELL", "transactTime": 1,
            "expiryReason": "EXECUTION_RULE_PRICE_RANGE_EXCEEDED",
        }  # fmt: skip
        result = parse_order(payload)
        assert result.expiry_reason == "EXECUTION_RULE_PRICE_RANGE_EXCEEDED"
        assert parse_order(payload | {"expiryReason": "NONE"}).expiry_reason == ""
        assert (
            parse_order({k: v for k, v in payload.items() if k != "expiryReason"}).expiry_reason
            == ""
        )

    def test_the_order_manager_writes_the_reason_into_the_record(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        rec = OrderRecord(
            "bxr2", "kr2", "BTCUSDT", Side.SELL, OrderType.MARKET, D("0.1"), None,
            OrderStatus.PENDING_SEND, 1, 1,
        )  # fmt: skip
        store.insert_order(rec)
        answer = OrderResult(
            "bxr2", "BTCUSDT", Side.SELL, OrderType.MARKET, OrderStatus.EXPIRED, D("0.1"),
            expiry_reason="EXECUTION_RULE_PRICE_RANGE_EXCEEDED",
        )  # fmt: skip
        h.loop.order_manager._apply(rec, answer)
        saved = store.get_order("bxr2")
        assert saved is not None and saved.note == "expired: EXECUTION_RULE_PRICE_RANGE_EXCEEDED"
        assert outcome_of(saved) == "not_filled"


class TestCancelPathsAndRestarts:
    """Review: the cancel/restart branches of the order manager had no tests."""

    def order_manager(self, rules: SymbolRules, clock: FakeClock, store: Store, broker):  # type: ignore[no-untyped-def]
        from botux.alerts import MemoryNotifier
        from botux.execution.order_manager import OrderManager

        notifier = MemoryNotifier()
        om = OrderManager(
            broker, store, {"BTCUSDT": rules}, notifier, clock=clock, reconcile_interval_seconds=0.0
        )
        return om, notifier

    def paper(self, rules: SymbolRules, clock: FakeClock):  # type: ignore[no-untyped-def]
        from botux.execution.paper import PaperBroker
        from botux.models import Quote

        b = PaperBroker(
            {"BTCUSDT": rules}, {"USDT": D(100_000)}, fee_rate=D("0.001"), slippage_bps=D(5),
            clock=clock,
        )  # fmt: skip
        b.set_quote(Quote("BTCUSDT", D("60000.00"), D("60000.01"), clock.now_ms()))
        return b

    def resting(self, key: str) -> OrderIntent:
        return OrderIntent(
            key, "BTCUSDT", Side.BUY, OrderType.LIMIT, D("0.01"), D("60000.01"), D("59000.00")
        )

    def test_a_failing_cancel_is_reported_and_does_not_stop_the_others(
        self, rules: SymbolRules, clock: FakeClock, store: Store
    ) -> None:
        broker = self.paper(rules, clock)
        om, notifier = self.order_manager(rules, clock, store, broker)
        om.submit(self.resting("a"))
        om.submit(self.resting("b"))
        real_cancel = broker.cancel_order
        calls: list[str] = []

        def cancel(symbol: str, cid: str):  # type: ignore[no-untyped-def]
            calls.append(cid)
            if len(calls) == 1:
                raise ExchangeError("cancel endpoint down")
            return real_cancel(symbol, cid)

        broker.cancel_order = cancel
        assert om.cancel_all_open() == 1  # one failed, one cancelled
        assert len(calls) == 2
        assert "Cancel failed" in notifier.titles()

    def test_open_orders_found_in_the_database_after_a_restart_are_tracked_again(
        self, rules: SymbolRules, clock: FakeClock, store: Store
    ) -> None:
        broker = self.paper(rules, clock)
        om, _ = self.order_manager(rules, clock, store, broker)
        om.submit(self.resting("a"))
        rec = store.get_by_intent("a")
        assert rec is not None and rec.status is OrderStatus.NEW
        # the process restarts: a new manager, the same database and the same venue
        om2, _ = self.order_manager(rules, clock, store, broker)
        assert [o.client_order_id for o in om2.open_orders()] == [rec.client_order_id]
        broker.set_quote(Quote("BTCUSDT", D("58900.00"), D("58900.01"), clock.now_ms()))
        om2.reconcile_pending()  # the limit filled while we were away: the record catches up
        done = store.get_order(rec.client_order_id)
        assert done is not None and done.status is OrderStatus.FILLED
        assert len(store.fills_for(rec.client_order_id)) == 1
        assert om2.open_orders() == []

    def test_a_cancel_whose_first_attempt_succeeded_is_settled_by_a_query(
        self, rules: SymbolRules, clock: FakeClock, store: Store
    ) -> None:
        """DELETE times out (but worked); the retry answers -2011; the order is then queried."""
        state = {"deletes": 0}
        ex = FakeExchange()

        def delete(_r: httpx.Request) -> httpx.Response:
            state["deletes"] += 1
            if state["deletes"] == 1:
                return httpx.Response(503, text="gateway timeout")
            return httpx.Response(400, json={"code": -2011, "msg": "Unknown order sent."})

        canceled = {
            "symbol": "BTCUSDT", "orderId": 5, "clientOrderId": "bxopen", "price": "59000.00",
            "origQty": "0.01", "executedQty": "0", "cummulativeQuoteQty": "0",
            "status": "CANCELED", "type": "LIMIT", "side": "BUY", "updateTime": 1,
        }  # fmt: skip
        ex.on("DELETE", "/api/v3/order", delete)
        ex.on("GET", "/api/v3/order", canceled)
        ex.on("GET", "/api/v3/time", {"serverTime": clock.now_ms()})
        client = BinanceSpotClient(
            credentials=Credentials(Secret(API_KEY), Secret(API_SECRET)),
            trading_enabled=True,
            http=ex.http(),
            clock=clock,
        )
        broker = LiveBroker(client, {"BTCUSDT": rules}, LiveAuthorization("f", 0))
        om, _ = self.order_manager(rules, clock, store, broker)
        rec = OrderRecord(
            "bxopen", "kopen", "BTCUSDT", Side.BUY, OrderType.LIMIT, D("0.01"), D("59000"),
            OrderStatus.NEW, 1, 1,
        )  # fmt: skip
        store.insert_order(rec)
        assert om.cancel_all_open() == 0  # the venue says "not open" ...
        got = store.get_order("bxopen")
        assert got is not None and got.status is OrderStatus.CANCELED  # ... and a query settles it
        assert state["deletes"] == 2  # retried once (cancels are idempotent), never more

    def test_an_unresolved_sell_keeps_the_stop_loss_from_selling_twice(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        patient = dataclasses.replace(
            cfg, execution=ExecutionSettings(reconcile_attempts=5, reconcile_interval_seconds=0.0)
        )
        h = harness(patient, rules, store)
        enter_long(h)
        h.paper.faults.push(FaultKind.TIMEOUT_NOT_EXECUTED)
        h.feed.bid, h.feed.ask = D("90.00"), D("90.01")
        h.loop.run_cycle()
        (sell,) = sells(h)
        assert sell.status in (OrderStatus.UNKNOWN, OrderStatus.PENDING_SEND)
        placed = h.paper.placed_count
        h.clock.advance(61)  # a new stop-loss minute while the first SELL is unresolved
        h.paper.fill_plan.push("0")  # (would be visible if a second SELL were sent)
        h.loop.run_cycle()
        assert h.paper.placed_count == placed  # no second SELL while the first is unresolved
