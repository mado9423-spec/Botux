"""Property test: whatever the sequence of orders, partial fills and ambiguous replies, once the
order manager has reconciled, the bot's OWN books (ledger) agree with the venue's balances."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from botux.alerts import MemoryNotifier
from botux.clock import FakeClock
from botux.exchange.rules import SymbolRules
from botux.execution.order_manager import OrderIntent, OrderManager
from botux.execution.paper import FaultKind, PaperBroker
from botux.killswitch import KillSwitch
from botux.ledger import Ledger
from botux.models import OrderType, Quote, Side
from botux.storage import Store
from tests.conftest import EXCHANGE_INFO_BTCUSDT, T0

D = Decimal
FEE = D("0.001")

step = st.sampled_from(["buy", "sell", "limit", "reconcile"])
fault = st.sampled_from([None, None, None, FaultKind.TIMEOUT_EXECUTED, FaultKind.REJECT])
fraction = st.sampled_from([None, None, "0.5", "0.25", "0.0"])
qty = st.integers(min_value=1, max_value=40).map(lambda n: D(n) / D(1000))  # 0.001 .. 0.04 BTC
price = st.integers(min_value=50_000, max_value=70_000).map(lambda n: D(n))


@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(ops=st.lists(st.tuples(step, qty, fault, fraction, price), min_size=1, max_size=14))
def test_the_ledger_always_agrees_with_the_venue_after_reconciliation(
    ops: list[tuple[str, Decimal, FaultKind | None, str | None, Decimal]], tmp_path_factory: object
) -> None:
    rules = SymbolRules.from_exchange_info(EXCHANGE_INFO_BTCUSDT, "BTCUSDT")
    clock = FakeClock(T0)
    tmp = Path(str(tmp_path_factory.mktemp("books")))  # type: ignore[attr-defined]
    store = Store(":memory:")
    try:
        paper = PaperBroker(
            {"BTCUSDT": rules}, {"USDT": D(1_000_000)}, fee_rate=FEE, slippage_bps=D(5), clock=clock
        )  # fmt: skip
        om = OrderManager(
            paper, store, {"BTCUSDT": rules}, MemoryNotifier(), clock=clock,
            kill_switch=KillSwitch(tmp / "KILL", clock), reconcile_attempts=50,
            reconcile_interval_seconds=0.0,
        )  # fmt: skip
        ledger = Ledger(store, {"BTCUSDT": rules}, capital=D(1_000_000), fee_rate=FEE)
        for i, (kind, q, f, frac, px) in enumerate(ops):
            paper.set_quote(Quote("BTCUSDT", px - D("0.01"), px + D("0.01"), clock.now_ms()))
            if kind == "reconcile":
                om.reconcile_pending()
                continue
            if kind == "limit":  # a resting buy below the market; later quotes may fill it in parts
                limit_px = rules.round_price(px * D("0.999"))
                om.submit(
                    OrderIntent(
                        f"k{i}",
                        "BTCUSDT",
                        Side.BUY,
                        OrderType.LIMIT,
                        rules.round_qty(q),
                        px + D("0.01"),
                        limit_px,
                    )
                )
                continue
            if f is not None:
                paper.faults.push(f)
            if frac is not None:
                paper.fill_plan.push(frac)
            side = Side.BUY if kind == "buy" else Side.SELL
            held = paper.balances().get("BTC")
            quantity = rules.round_qty(
                q if side is Side.BUY else min(q, held.free if held else D(0))
            )
            if quantity <= 0:
                continue
            ref = px + D("0.01") if side is Side.BUY else px - D("0.01")
            om.submit(OrderIntent(f"k{i}", "BTCUSDT", side, OrderType.MARKET, quantity, ref))
        om.reconcile_pending()  # every ambiguous reply is settled by a query
        om.reconcile_pending()
        om.cancel_all_open()  # resting limits end: the unfilled rest returns to the account
        om.reconcile_pending()
        assert not om.uncertain_orders() and not om.open_orders()
        view = ledger.view()
        balances = paper.balances()
        btc = balances["BTC"].total if "BTC" in balances else D(0)
        usdt = balances["USDT"].total
        assert view.base_qty("BTCUSDT") == btc
        assert view.cash == usdt
    finally:
        store.close()
