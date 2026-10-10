"""Partial fills, expiry of market remainders, cancel-vs-fill races and no double counting."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from botux.alerts import MemoryNotifier
from botux.clock import FakeClock
from botux.exchange.rules import SymbolRules
from botux.execution.order_manager import OrderIntent, OrderManager
from botux.execution.paper import PaperBroker
from botux.killswitch import KillSwitch
from botux.ledger import build_ledger
from botux.models import Fill, OrderRequest, OrderStatus, OrderType, Quote, Side
from botux.storage import OrderRecord, Store

D = Decimal


class Env:
    def __init__(self, rules: SymbolRules, clock: FakeClock, store: Store, tmp: Path) -> None:
        self.clock, self.store, self.rules = clock, store, rules
        self.notifier = MemoryNotifier()
        self.fills: list[tuple[str, list[Fill]]] = []
        self.paper = PaperBroker(
            {"BTCUSDT": rules},
            {"USDT": D(1000)},
            fee_rate=D("0.001"),
            slippage_bps=D(5),
            clock=clock,
        )
        self.paper.set_quote(Quote("BTCUSDT", D("60000.00"), D("60000.01"), clock.now_ms()))
        self.om = OrderManager(
            self.paper,
            store,
            {"BTCUSDT": rules},
            self.notifier,
            clock=clock,
            kill_switch=KillSwitch(tmp / "KILL", clock),
            reconcile_interval_seconds=0.0,
            on_fill=lambda rec, fills: self.fills.append((rec.client_order_id, fills)),
        )

    def requote(self, bid: str, ask: str) -> None:
        self.paper.set_quote(Quote("BTCUSDT", D(bid), D(ask), self.clock.now_ms()))

    def record(self, key: str) -> OrderRecord:
        rec = self.store.get_by_intent(key)
        assert rec is not None
        return rec

    def ledger_qty(self) -> Decimal:
        rows = self.store.fill_rows_after(0)
        v = build_ledger(D(1000), rows, {"BTCUSDT": self.rules}, D("0.001"))
        return v.base_qty("BTCUSDT")


@pytest.fixture
def env(rules: SymbolRules, clock: FakeClock, store: Store, tmp_path: Path) -> Env:
    return Env(rules, clock, store, tmp_path)


def market(key: str, side: Side, qty: str) -> OrderIntent:
    ref = "60000.01" if side is Side.BUY else "60000.00"
    return OrderIntent(key, "BTCUSDT", side, OrderType.MARKET, D(qty), D(ref), is_exit=False)


def limit(key: str, side: Side, qty: str, price: str) -> OrderIntent:
    return OrderIntent(key, "BTCUSDT", side, OrderType.LIMIT, D(qty), D(price), D(price))


class TestSimulatorPartialFills:
    def test_a_partly_filled_market_order_expires_with_only_what_it_got(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = PaperBroker(
            {"BTCUSDT": rules},
            {"USDT": D(1000)},
            fee_rate=D("0.001"),
            slippage_bps=D(0),
            clock=clock,
        )
        b.set_quote(Quote("BTCUSDT", D("100.00"), D("100.01"), clock.now_ms()))
        b.fill_plan.push("0.4")
        res = b.place_order(OrderRequest("m1", "BTCUSDT", Side.BUY, OrderType.MARKET, D("1")))
        assert res.status is OrderStatus.EXPIRED  # as on Binance: the market remainder lapses
        assert res.executed_qty == D("0.4") and res.cumulative_quote_qty == D("0.4") * D("100.01")
        assert len(res.fills) == 1 and res.fills[0].quantity == D("0.4")
        bal = b.balances()
        assert bal["USDT"].free == D(1000) - D("0.4") * D("100.01")  # only the filled part paid
        assert bal["BTC"].free == D("0.4") - D("0.0004")  # fee on the executed part only
        assert b.open_orders("BTCUSDT") == []

    def test_a_partly_filled_limit_order_stays_open_with_the_rest_locked(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = PaperBroker(
            {"BTCUSDT": rules},
            {"USDT": D(1000)},
            fee_rate=D("0.001"),
            slippage_bps=D(0),
            clock=clock,
        )
        b.set_quote(Quote("BTCUSDT", D("99.99"), D("100.01"), clock.now_ms()))
        b.fill_plan.push("0.25")
        res = b.place_order(
            OrderRequest("l1", "BTCUSDT", Side.BUY, OrderType.LIMIT, D("2"), D("100.50"))
        )
        assert res.status is OrderStatus.PARTIALLY_FILLED and res.executed_qty == D("0.5")
        bal = b.balances()
        assert bal["USDT"].locked == D("1.5") * D("100.50")  # only the unfilled rest is locked
        assert bal["USDT"].free + bal["USDT"].locked == D(1000) - D("0.5") * D("100.01")
        assert [o.client_order_id for o in b.open_orders("BTCUSDT")] == ["l1"]

    def test_the_rest_of_a_resting_limit_fills_later_without_double_charging(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = PaperBroker(
            {"BTCUSDT": rules},
            {"USDT": D(1000)},
            fee_rate=D("0.001"),
            slippage_bps=D(0),
            clock=clock,
        )
        b.set_quote(Quote("BTCUSDT", D("99.99"), D("100.01"), clock.now_ms()))
        b.fill_plan.push("0.5")
        b.place_order(OrderRequest("l1", "BTCUSDT", Side.BUY, OrderType.LIMIT, D("2"), D("100.50")))
        b.set_quote(Quote("BTCUSDT", D("99.00"), D("100.00"), clock.now_ms()))  # still crossing
        res = b.get_order("BTCUSDT", "l1")
        assert res is not None and res.status is OrderStatus.FILLED
        assert res.executed_qty == D(2) and len(res.fills) == 2
        bal = b.balances()
        assert bal["USDT"].locked == 0
        spent = D("1") * D("100.01") + D("1") * D("100.50")  # first at the ask, rest at the limit
        assert bal["USDT"].free == D(1000) - spent

    def test_cancelling_a_partly_filled_limit_returns_only_the_unfilled_funds(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = PaperBroker(
            {"BTCUSDT": rules},
            {"USDT": D(1000)},
            fee_rate=D("0.001"),
            slippage_bps=D(0),
            clock=clock,
        )
        b.set_quote(Quote("BTCUSDT", D("99.99"), D("100.01"), clock.now_ms()))
        b.fill_plan.push("0.5")
        b.place_order(OrderRequest("l1", "BTCUSDT", Side.BUY, OrderType.LIMIT, D("2"), D("100.50")))
        res = b.cancel_order("BTCUSDT", "l1")
        assert res is not None and res.status is OrderStatus.CANCELED
        assert res.executed_qty == D(1)  # what was bought stays bought
        bal = b.balances()
        assert bal["USDT"].locked == 0 and bal["USDT"].free == D(1000) - D(1) * D("100.01")

    def test_a_fraction_that_rounds_to_nothing_leaves_a_limit_order_resting_unfilled(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = PaperBroker(
            {"BTCUSDT": rules},
            {"USDT": D(1000)},
            fee_rate=D("0.001"),
            slippage_bps=D(0),
            clock=clock,
        )
        b.set_quote(Quote("BTCUSDT", D("99.99"), D("100.01"), clock.now_ms()))
        b.fill_plan.push("0.0000001")
        res = b.place_order(
            OrderRequest("l1", "BTCUSDT", Side.BUY, OrderType.LIMIT, D("0.1"), D("100.50"))
        )
        assert res.status is OrderStatus.NEW and res.executed_qty == 0
        assert b.balances()["USDT"].locked == D("0.1") * D("100.50")
        m = b.place_order(OrderRequest("m1", "BTCUSDT", Side.BUY, OrderType.MARKET, D("0.1")))
        assert m.status is OrderStatus.FILLED  # the plan was consumed; later orders are normal

    def test_fractions_must_be_below_one(self) -> None:
        from botux.execution.paper import FillPlan

        with pytest.raises(ValueError):
            FillPlan().push("1")


class TestOrderManagerWithPartialFills:
    def test_a_partial_market_buy_is_recorded_once_and_the_books_agree(self, env: Env) -> None:
        env.paper.fill_plan.push("0.4")
        res = env.om.submit(market("k1", Side.BUY, "0.01"))
        assert res.outcome == "partial"  # the caller is told it was NOT fully done
        rec = env.record("k1")
        assert rec.status is OrderStatus.EXPIRED and rec.executed_qty == D("0.004")
        assert len(env.fills) == 1 and env.fills[0][1][0].quantity == D("0.004")
        # a later reconciliation of the same (terminal) order must not count the fill again
        env.om.reconcile_pending()
        env.om.reconcile_pending()
        assert len(env.fills) == 1 and len(env.store.fills_for(rec.client_order_id)) == 1
        # the bot's books equal the simulator's balances
        assert env.ledger_qty() == env.paper.balances()["BTC"].free

    def test_a_partially_filled_limit_is_tracked_and_only_new_fills_are_added(
        self, env: Env
    ) -> None:
        env.requote("60000.00", "60000.01")
        env.paper.fill_plan.push("0.5")
        res = env.om.submit(limit("k2", Side.BUY, "0.002", "60010.00"))
        assert res.outcome == "accepted"
        rec = env.record("k2")
        assert rec.status is OrderStatus.PARTIALLY_FILLED and rec.executed_qty == D("0.001")
        assert env.om.open_orders("BTCUSDT")[0].client_order_id == rec.client_order_id
        env.om.reconcile_pending()  # nothing new at the venue: no duplicate fill
        assert len(env.store.fills_for(rec.client_order_id)) == 1
        env.requote("60005.00", "60005.01")  # still crossing: the rest fills
        env.om.reconcile_pending()
        rec = env.record("k2")
        assert rec.status is OrderStatus.FILLED and rec.executed_qty == D("0.002")
        fills = env.store.fills_for(rec.client_order_id)
        assert [f.quantity for f in fills] == [D("0.001"), D("0.001")]
        assert env.om.open_orders("BTCUSDT") == []
        assert env.ledger_qty() == env.paper.balances()["BTC"].free

    def test_cancelling_a_partly_filled_order_keeps_the_executed_part(self, env: Env) -> None:
        env.paper.fill_plan.push("0.5")
        env.om.submit(limit("k3", Side.BUY, "0.002", "60010.00"))
        assert env.om.cancel_all_open() == 1
        rec = env.record("k3")
        assert rec.status is OrderStatus.CANCELED and rec.executed_qty == D("0.001")
        assert env.paper.balances()["USDT"].locked == 0
        assert env.ledger_qty() == env.paper.balances()["BTC"].free

    def test_a_fill_that_wins_the_race_against_the_cancel_is_still_recorded(self, env: Env) -> None:
        env.om.submit(limit("k4", Side.BUY, "0.002", "59000.00"))  # resting, far below the market
        rec = env.record("k4")
        assert rec.status is OrderStatus.NEW
        # the market falls through the limit: it fills at the venue while the bot is not looking
        env.requote("58990.00", "58990.01")
        assert env.paper.get_order("BTCUSDT", rec.client_order_id).status is OrderStatus.FILLED  # type: ignore[union-attr]
        # the bot, still believing the order is open, tries to cancel it
        assert env.om.cancel_all_open() == 0  # the venue says: not open any more
        rec = env.record("k4")
        assert rec.status is OrderStatus.FILLED and rec.executed_qty == D("0.002")
        assert len(env.store.fills_for(rec.client_order_id)) == 1
        assert len(env.fills) == 1  # the position bookkeeping was told exactly once
        assert env.ledger_qty() == env.paper.balances()["BTC"].free

    def test_an_order_cancelled_at_the_venue_behind_our_back_is_not_left_open(
        self, env: Env
    ) -> None:
        env.om.submit(limit("k5", Side.BUY, "0.002", "59000.00"))
        rec = env.record("k5")
        env.paper.cancel_order("BTCUSDT", rec.client_order_id)  # e.g. the user cancelled it
        env.om.reconcile_pending()
        assert env.record("k5").status is OrderStatus.CANCELED
        assert env.om.open_orders("BTCUSDT") == []

    def test_an_unknown_outcome_that_was_in_fact_a_partial_fill_is_adopted_not_resent(
        self, env: Env
    ) -> None:
        from botux.execution.paper import FaultKind

        env.paper.faults.push(FaultKind.TIMEOUT_EXECUTED)
        env.paper.fill_plan.push("0.5")
        res = env.om.submit(market("k6", Side.BUY, "0.002"))
        rec = env.record("k6")
        assert res.outcome == "partial"  # reconciled straight away: the venue ended it half-done
        assert rec.status is OrderStatus.EXPIRED and rec.executed_qty == D("0.001")
        assert env.paper.placed_count == 1  # never sent a second time
        assert len(env.store.fills_for(rec.client_order_id)) == 1

    def test_totals_only_answers_create_one_estimated_fill_per_increment(self, env: Env) -> None:
        # a venue answer without a fills list (GET /order): increments become estimated fills
        rec = OrderRecord(
            "bx1", "k7", "BTCUSDT", Side.BUY, OrderType.LIMIT, D("0.002"), D("60000"),
            OrderStatus.NEW, 1, 1,
        )  # fmt: skip
        env.store.insert_order(rec)
        from botux.models import OrderResult

        def answer(executed: str, status: OrderStatus) -> OrderResult:
            return OrderResult(
                "bx1", "BTCUSDT", Side.BUY, OrderType.LIMIT, status, D("0.002"),
                D(executed), D(executed) * D(60000), D(60000),
            )  # fmt: skip

        env.om._apply(rec, answer("0.001", OrderStatus.PARTIALLY_FILLED))
        env.om._apply(rec, answer("0.001", OrderStatus.PARTIALLY_FILLED))  # unchanged: no new fill
        env.om._apply(rec, answer("0.002", OrderStatus.FILLED))
        fills = env.store.fills_for("bx1")
        assert [f.quantity for f in fills] == [D("0.001"), D("0.001")]
        assert all(f.commission_asset == "" for f in fills)  # honestly marked as estimated
