"""Order lifecycle: duplicates, outages, retries, unconfirmed orders, crash recovery."""

from __future__ import annotations

import dataclasses
import re
from decimal import Decimal
from pathlib import Path

import pytest

from botux.alerts import MemoryNotifier
from botux.clock import FakeClock
from botux.exchange.rules import SymbolRules
from botux.execution.order_manager import OrderIntent, OrderManager, client_order_id_for
from botux.execution.paper import FaultKind, PaperBroker
from botux.killswitch import KillSwitch
from botux.models import Fill, OrderRequest, OrderResult, OrderStatus, OrderType, Quote, Side
from botux.storage import OrderRecord, Store

D = Decimal


class Crash(BaseException):
    """Simulates the process dying (not an Exception, so nothing in the bot can swallow it)."""


class Env:
    def __init__(self, rules: SymbolRules, clock: FakeClock, store: Store, tmp: Path) -> None:
        self.clock, self.store, self.rules = clock, store, rules
        self.notifier = MemoryNotifier()
        self.kill = KillSwitch(tmp / "KILL", clock)
        self.fills: list[tuple[OrderRecord, list[Fill]]] = []
        self.broker = PaperBroker(
            {"BTCUSDT": rules},
            {"USDT": D(1000)},
            fee_rate=D("0.001"),
            slippage_bps=D(5),
            clock=clock,
        )
        self.broker.set_quote(Quote("BTCUSDT", D("60000.00"), D("60000.01"), clock.now_ms()))
        self.om = self.manager(self.broker)

    def manager(self, broker: object, **kw: object) -> OrderManager:
        return OrderManager(
            broker,  # type: ignore[arg-type]
            self.store,
            {"BTCUSDT": self.rules},
            self.notifier,
            clock=self.clock,
            kill_switch=self.kill,
            reconcile_attempts=kw.get("attempts", 5),  # type: ignore[arg-type]
            reconcile_interval_seconds=1.0,
            on_fill=lambda rec, fills: self.fills.append((rec, fills)),
        )


@pytest.fixture
def env(rules: SymbolRules, clock: FakeClock, store: Store, tmp_path: Path) -> Env:
    return Env(rules, clock, store, tmp_path)


def buy(key: str = "k1", qty: str = "0.001", *, is_exit: bool = False) -> OrderIntent:
    return OrderIntent(
        key, "BTCUSDT", Side.BUY, OrderType.MARKET, D(qty), D("60000.01"), is_exit=is_exit
    )


def sell(key: str, qty: str) -> OrderIntent:
    return OrderIntent(
        key, "BTCUSDT", Side.SELL, OrderType.MARKET, D(qty), D("60000.00"), is_exit=True
    )


class TestClientOrderIds:
    def test_deterministic_and_within_binance_limits(self) -> None:
        a, b = (
            client_order_id_for("sma:BTCUSDT:123:BUY"),
            client_order_id_for("sma:BTCUSDT:123:BUY"),
        )
        assert a == b != client_order_id_for("sma:BTCUSDT:124:BUY")
        assert len(a) <= 36 and re.fullmatch(r"[A-Za-z0-9_-]+", a)


class TestHappyPath:
    def test_filled_order_is_persisted_with_fills(self, env: Env) -> None:
        r = env.om.submit(buy())
        assert r.outcome == "accepted" and r.record is not None
        rec = env.store.get_order(r.record.client_order_id)
        assert (
            rec is not None and rec.status is OrderStatus.FILLED and rec.executed_qty == D("0.001")
        )
        assert len(env.store.fills_for(rec.client_order_id)) == 1
        assert len(env.fills) == 1


class TestDuplicates:
    def test_same_intent_twice_sends_one_order(self, env: Env) -> None:
        assert env.om.submit(buy()).outcome == "accepted"
        again = env.om.submit(buy())
        assert again.outcome == "duplicate"
        assert env.broker.placed_count == 1

    def test_same_intent_after_a_restart_is_still_a_duplicate(
        self, env: Env, rules: SymbolRules
    ) -> None:
        env.om.submit(buy())
        restarted = env.manager(env.broker)  # new process, same database
        assert restarted.submit(buy()).outcome == "duplicate"
        assert env.broker.placed_count == 1

    def test_rejected_intent_is_not_hammered_again(self, env: Env) -> None:
        env.broker.faults.push(FaultKind.REJECT)
        assert env.om.submit(buy()).outcome == "rejected"
        assert env.om.submit(buy()).outcome == "duplicate"
        assert env.broker.placed_count == 0

    def test_different_intents_are_independent(self, env: Env) -> None:
        env.om.submit(buy("k1"))
        env.om.submit(buy("k2"))
        assert env.broker.placed_count == 2


class TestOutagesAndRejections:
    def test_network_down_before_sending_leaves_no_trace_and_can_be_retried(self, env: Env) -> None:
        env.broker.faults.push(FaultKind.NETWORK_DOWN)
        r = env.om.submit(buy())
        assert r.outcome == "retry_later" and "not sent" in r.reason
        assert env.store.list_orders() == []  # provably never left: the intent key is free again
        assert env.om.is_symbol_blocked("BTCUSDT") is None
        assert env.om.submit(buy()).outcome == "accepted"  # same intent, now it goes through
        assert env.broker.placed_count == 1

    def test_exchange_side_rejection(self, env: Env) -> None:
        env.broker.faults.push(FaultKind.REJECT)
        r = env.om.submit(buy())
        assert r.record is not None and r.record.status is OrderStatus.REJECTED

    def test_local_filter_violation_never_reaches_the_broker_or_the_database(
        self, env: Env
    ) -> None:
        r = env.om.submit(buy(qty="0.000001"))
        assert r.outcome == "blocked" and "LOT_SIZE" in r.reason
        assert env.store.list_orders() == [] and env.broker.placed_count == 0

    def test_refusals_before_the_engine_are_retryable_not_unknown(self, env: Env) -> None:
        from botux.errors import APIError, ClockSkewError, RateLimitError

        class Refusing:
            def __init__(self, exc: Exception) -> None:
                self.exc = exc

            def place_order(self, req: OrderRequest) -> OrderResult:
                raise self.exc

        for exc in (RateLimitError("429", retry_after=5), ClockSkewError("skew", code=-1021)):
            r = env.manager(Refusing(exc)).submit(buy())
            assert r.outcome == "retry_later" and env.store.list_orders() == []
        # while any other API error is a final refusal that consumes the intent
        r = env.manager(Refusing(APIError("nope", code=-2010, http_status=400))).submit(buy())
        assert r.outcome == "rejected" and r.record is not None
        assert r.record.status is OrderStatus.REJECTED
        assert env.om.submit(buy()).outcome == "duplicate"


class TestUnconfirmedOrders:
    def test_timeout_after_execution_is_adopted_not_resent(self, env: Env) -> None:
        env.broker.faults.push(FaultKind.TIMEOUT_EXECUTED)
        r = env.om.submit(buy())
        assert r.outcome == "accepted" and r.record is not None
        assert r.record.status is OrderStatus.FILLED
        assert env.broker.placed_count == 1  # reconciled by querying, NOT by sending again
        assert "Order outcome unknown" in env.notifier.titles()
        env.om.reconcile_pending()
        env.om.reconcile_pending()
        assert len(env.store.fills_for(r.record.client_order_id)) == 1  # fills never double counted
        assert len(env.fills) == 1  # position bookkeeping fired exactly once
        assert env.broker.placed_count == 1

    def test_timeout_without_execution_ends_as_lost_and_blocks_the_symbol(self, env: Env) -> None:
        env.broker.faults.push(FaultKind.TIMEOUT_NOT_EXECUTED)
        r = env.om.submit(buy())
        assert r.outcome == "uncertain" and r.record is not None
        assert r.record.status is OrderStatus.UNKNOWN
        # while uncertain, nothing else may be sent for the symbol
        assert env.om.submit(buy("other")).outcome == "blocked"
        env.om.reconcile_pending()
        env.om.reconcile_pending()  # attempts 4 and 5
        rec = env.store.get_order(r.record.client_order_id)
        assert rec is not None and rec.status is OrderStatus.LOST
        assert "Order LOST" in env.notifier.titles()
        # LOST is terminal but the symbol stays blocked until a human resolves it
        assert env.om.submit(buy("after")).outcome == "blocked"
        env.om.resolve_lost("BTCUSDT")
        assert env.om.submit(buy("after")).outcome == "accepted"
        assert (
            env.broker.placed_count == 1
        )  # only the later, deliberate order; never an auto-resend

    def test_a_failed_status_query_is_not_counted_as_not_found(self, env: Env) -> None:
        from botux.errors import NetworkError

        env.broker.faults.push(FaultKind.TIMEOUT_NOT_EXECUTED)
        r = env.om.submit(buy())
        assert r.record is not None

        class Flaky:
            def get_order(self, symbol: str, cid: str) -> OrderResult | None:
                raise NetworkError("down")

        before = env.store.get_order(r.record.client_order_id)
        assert before is not None
        attempts = before.reconcile_attempts
        flaky = env.manager(Flaky())
        flaky.reconcile_pending()
        after = env.store.get_order(r.record.client_order_id)
        assert after is not None and after.reconcile_attempts == attempts
        assert after.status is OrderStatus.UNKNOWN


class TestUnrecognisedVenueStatus:
    def test_an_unreadable_status_does_not_block_the_symbol_forever(self, env: Env) -> None:
        env.broker.faults.push(FaultKind.TIMEOUT_NOT_EXECUTED)
        r = env.om.submit(buy())
        assert r.record is not None

        class Weird:
            def get_order(self, symbol: str, cid: str) -> OrderResult:
                return OrderResult(
                    cid, symbol, Side.BUY, OrderType.MARKET, OrderStatus.UNKNOWN, D("0.001")
                )

        om = env.manager(Weird())
        for _ in range(5):
            om.reconcile_pending()
        rec = env.store.get_order(r.record.client_order_id)
        assert rec is not None and rec.status is OrderStatus.LOST
        assert "Order LOST" in env.notifier.titles()


class TestCrashRecovery:
    def test_write_ahead_record_survives_a_crash_and_is_reconciled_not_resent(
        self, env: Env
    ) -> None:
        class CrashAfterSend:
            def place_order(self, req: OrderRequest) -> OrderResult:
                env.broker.place_order(req)  # the exchange executed it...
                raise Crash  # ...but the process died before it could record the answer

        om = env.manager(CrashAfterSend())
        with pytest.raises(Crash):
            om.submit(buy())
        (rec,) = env.store.list_orders()
        assert rec.status is OrderStatus.PENDING_SEND  # persisted BEFORE the send

        restarted = env.manager(env.broker)  # new process
        assert restarted.submit(buy("new")).outcome == "blocked"  # must reconcile first
        assert restarted.submit(buy()).outcome == "blocked"  # same intent must not be re-sent
        restarted.reconcile_pending()
        done = env.store.get_order(rec.client_order_id)
        assert done is not None and done.status is OrderStatus.FILLED
        assert env.broker.placed_count == 1
        assert len(env.store.fills_for(rec.client_order_id)) == 1


class TestKillSwitchInteraction:
    def test_once_killed_only_a_flatten_order_may_still_go_out(self, env: Env) -> None:
        env.om.submit(buy("e0"))
        env.kill.trip("test")
        assert env.om.submit(buy("e1")).outcome == "blocked"
        qty = env.rules.round_qty(env.broker.balances()["BTC"].free)
        # an ordinary exit (stop-loss, strategy exit) is NOT sent any more: the stop means stop
        blocked = env.om.submit(sell("x1", str(qty)))
        assert blocked.outcome == "blocked" and "kill switch" in blocked.reason
        assert env.broker.placed_count == 1
        # the flatten policy (flatten_on_kill = true) is the one explicit exception
        flatten = dataclasses.replace(sell("x2", str(qty)), flatten=True)
        assert env.om.submit(flatten).outcome == "accepted"

    def test_cancel_all_open_orders(self, env: Env) -> None:
        limit = OrderIntent(
            "L1",
            "BTCUSDT",
            Side.BUY,
            OrderType.LIMIT,
            D("0.01"),
            D("60000.01"),
            price=D("59000.00"),
        )
        assert env.om.submit(limit).outcome == "accepted"
        assert len(env.om.open_orders()) == 1
        assert env.om.cancel_all_open() == 1
        assert env.om.open_orders() == []
        assert env.broker.balances()["USDT"].free == D(1000)


class TestFillBookkeeping:
    def test_applying_the_same_full_result_twice_records_fills_once(self, env: Env) -> None:
        r = env.om.submit(buy())
        assert r.record is not None
        result = env.broker.get_order("BTCUSDT", r.record.client_order_id)
        assert result is not None
        for _ in range(3):
            env.om._apply(r.record, result)
        assert len(env.store.fills_for(r.record.client_order_id)) == 1
        assert len(env.fills) == 1

    def test_totals_only_updates_become_incremental_estimated_fills(self, env: Env) -> None:
        """A live GET /order has no fills list: partial progress is turned into deltas."""
        rec = OrderRecord(
            "c1",
            "k",
            "BTCUSDT",
            Side.BUY,
            OrderType.LIMIT,
            D("1"),
            D("100"),
            OrderStatus.NEW,
            0,
            0,
        )
        assert env.store.insert_order(rec)

        def snapshot(executed: str, quote: str, status: OrderStatus) -> OrderResult:
            return OrderResult(
                "c1", "BTCUSDT", Side.BUY, OrderType.LIMIT, status, D("1"), D(executed), D(quote)
            )

        env.om._apply(rec, snapshot("0.4", "40", OrderStatus.PARTIALLY_FILLED))
        env.om._apply(rec, snapshot("0.4", "40", OrderStatus.PARTIALLY_FILLED))  # no progress
        env.om._apply(rec, snapshot("1", "101", OrderStatus.FILLED))
        fills = env.store.fills_for("c1")
        # second delta: 0.6 BTC for 61 USDT (101 - 40 cumulative)
        assert [(f.quantity, f.price) for f in fills] == [
            (D("0.4"), D("100")),
            (D("0.6"), D("61") / D("0.6")),
        ]
        assert sum((f.quantity for f in fills), D(0)) == D("1")
        assert len(env.fills) == 2


class TestExitsAreNotBlockedByUnresolvedBuys:
    def test_a_stop_loss_sell_goes_out_while_a_buy_is_unresolved(self, env: Env) -> None:
        env.om.submit(buy("e0"))  # we hold some BTC
        env.broker.faults.push(FaultKind.TIMEOUT_NOT_EXECUTED)
        assert (
            env.om.submit(buy("e1")).outcome == "uncertain"
        )  # an unresolved BUY now blocks entries
        assert env.om.submit(buy("e2")).outcome == "blocked"
        qty = env.rules.round_qty(env.broker.balances()["BTC"].free)
        assert env.om.submit(sell("x1", str(qty))).outcome == "accepted"

    def test_but_a_second_sell_waits_for_an_unresolved_sell(self, env: Env) -> None:
        env.om.submit(buy("e0"))
        qty = env.rules.round_qty(env.broker.balances()["BTC"].free)
        env.broker.faults.push(FaultKind.TIMEOUT_NOT_EXECUTED)
        assert env.om.submit(sell("x1", str(qty))).outcome == "uncertain"
        assert env.om.submit(sell("x2", str(qty))).outcome == "blocked"  # never sell twice
        assert env.broker.placed_count == 1
