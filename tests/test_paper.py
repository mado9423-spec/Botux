"""Paper simulator: fills, fees, filters, balances, duplicate ids, faults, and isolation."""

from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path

import pytest

import botux.execution.paper as paper_module
from botux.clock import FakeClock
from botux.errors import NetworkError, OrderRejected, OutcomeUnknownError
from botux.exchange.rules import SymbolRules
from botux.execution.paper import FaultKind, PaperBroker
from botux.models import OrderRequest, OrderStatus, OrderType, Quote, Side

D = Decimal


def broker(rules: SymbolRules, clock: FakeClock, usdt: str = "1000") -> PaperBroker:
    b = PaperBroker(
        {"BTCUSDT": rules}, {"USDT": D(usdt)}, fee_rate=D("0.001"), slippage_bps=D(5), clock=clock
    )
    b.set_quote(Quote("BTCUSDT", D("60000.00"), D("60000.01"), clock.now_ms()))
    return b


def market(cid: str, side: Side, qty: str) -> OrderRequest:
    return OrderRequest(cid, "BTCUSDT", side, OrderType.MARKET, D(qty))


def limit(cid: str, side: Side, qty: str, price: str) -> OrderRequest:
    return OrderRequest(cid, "BTCUSDT", side, OrderType.LIMIT, D(qty), D(price))


class TestFills:
    def test_market_buy_pays_slippage_and_charges_fee_in_base_asset(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = broker(rules, clock)
        res = b.place_order(market("a1", Side.BUY, "0.001"))
        # ask 60000.01 * 1.0005 = 60030.010005 -> rounded UP to the tick
        assert res.status is OrderStatus.FILLED and res.fills[0].price == D("60030.02")
        bal = b.balances()
        assert bal["USDT"].free == D("1000") - D("0.001") * D("60030.02")
        assert res.fills[0].commission_asset == "BTC"
        assert bal["BTC"].free == D("0.001") - D("0.000001")  # 0.1% fee taken from what we bought

    def test_selling_the_bought_amount_fails_because_the_fee_was_paid_in_base(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = broker(rules, clock)
        b.place_order(market("a1", Side.BUY, "0.001"))
        with pytest.raises(OrderRejected, match="insufficient balance"):
            b.place_order(market("a2", Side.SELL, "0.001"))

    def test_sell_what_the_balance_allows_and_charge_fee_in_quote(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = broker(rules, clock)
        b.place_order(market("a1", Side.BUY, "0.001"))
        qty = rules.round_qty(b.balances()["BTC"].free)  # 0.00099: step-rounded, as the bot does
        res = b.place_order(market("a2", Side.SELL, str(qty)))
        assert res.fills[0].commission_asset == "USDT"
        # bid 60000.00 * (1 - 0.0005) = 59970.00 (already on the tick)
        assert res.fills[0].price == D("59970.00")
        assert b.balances()["BTC"].free == D("0.000009")  # dust that can no longer be sold

    def test_round_trip_loses_money_to_costs_when_price_is_flat(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = broker(rules, clock)
        b.place_order(market("a1", Side.BUY, "0.01"))
        qty = rules.round_qty(b.balances()["BTC"].free)
        b.place_order(market("a2", Side.SELL, str(qty)))
        bal = b.balances()
        equity = bal["USDT"].total + bal["BTC"].total * D("60000")
        assert equity < D("1000")


class TestRejections:
    def test_filter_violations_are_rejected_like_the_exchange_does(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = broker(rules, clock)
        with pytest.raises(OrderRejected, match="Filter failure"):
            b.place_order(market("a1", Side.BUY, "0.000001"))
        assert b.placed_count == 0

    def test_insufficient_funds(self, rules: SymbolRules, clock: FakeClock) -> None:
        b = broker(rules, clock, usdt="10")
        with pytest.raises(OrderRejected, match="insufficient balance"):
            b.place_order(market("a1", Side.BUY, "0.01"))

    def test_unknown_symbol_and_missing_market_data(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = broker(rules, clock)
        with pytest.raises(OrderRejected):
            b.place_order(OrderRequest("a1", "ETHUSDT", Side.BUY, OrderType.MARKET, D("1")))
        fresh = PaperBroker({"BTCUSDT": rules}, {"USDT": D(100)}, fee_rate=D(0), slippage_bps=D(0))
        with pytest.raises(OrderRejected, match="no market data"):
            fresh.place_order(market("a1", Side.BUY, "0.001"))


class TestLimitOrdersAndIds:
    def test_resting_limit_order_locks_funds_then_fills_when_price_crosses(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = broker(rules, clock)
        res = b.place_order(limit("l1", Side.BUY, "0.01", "59000.00"))
        assert res.status is OrderStatus.NEW
        assert b.balances()["USDT"].locked == D("590.00")
        assert b.balances()["USDT"].free == D("410.00")
        b.set_quote(Quote("BTCUSDT", D("58890.00"), D("58900.00"), clock.now_ms()))
        filled = b.get_order("BTCUSDT", "l1")
        assert filled is not None and filled.status is OrderStatus.FILLED
        assert filled.fills[0].price == D("59000.00")
        assert b.balances()["USDT"].locked == 0

    def test_marketable_limit_fills_immediately_at_the_ask(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        res = broker(rules, clock).place_order(limit("l1", Side.BUY, "0.001", "60100.00"))
        assert res.status is OrderStatus.FILLED and res.fills[0].price == D("60000.01")

    def test_cancel_releases_the_locked_funds(self, rules: SymbolRules, clock: FakeClock) -> None:
        b = broker(rules, clock)
        b.place_order(limit("l1", Side.BUY, "0.01", "59000.00"))
        res = b.cancel_order("BTCUSDT", "l1")
        assert res is not None and res.status is OrderStatus.CANCELED
        assert b.balances()["USDT"].free == D("1000") and b.balances()["USDT"].locked == 0
        assert b.cancel_order("BTCUSDT", "l1") is None  # already closed
        assert b.open_orders("BTCUSDT") == []

    def test_duplicate_id_of_an_open_order_is_rejected(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = broker(rules, clock)
        b.place_order(limit("dup", Side.BUY, "0.001", "59000.00"))
        with pytest.raises(OrderRejected, match="Duplicate order sent"):
            b.place_order(limit("dup", Side.BUY, "0.001", "59000.00"))

    def test_id_of_a_filled_order_is_accepted_again_like_binance(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        # Documented Binance behaviour: a repeated newClientOrderId is only rejected while the
        # first order is still open. This is exactly why the bot never re-sends blindly.
        b = broker(rules, clock)
        b.place_order(market("dup", Side.BUY, "0.001"))
        b.place_order(market("dup", Side.BUY, "0.001"))
        assert b.placed_count == 2


class TestFaultInjection:
    def test_network_down_means_nothing_happened(
        self, rules: SymbolRules, clock: FakeClock
    ) -> None:
        b = broker(rules, clock)
        b.faults.push(FaultKind.NETWORK_DOWN)
        with pytest.raises(NetworkError) as e:
            b.place_order(market("a1", Side.BUY, "0.001"))
        assert e.value.not_sent and b.placed_count == 0

    def test_timeout_without_execution(self, rules: SymbolRules, clock: FakeClock) -> None:
        b = broker(rules, clock)
        b.faults.push(FaultKind.TIMEOUT_NOT_EXECUTED)
        with pytest.raises(OutcomeUnknownError):
            b.place_order(market("a1", Side.BUY, "0.001"))
        assert b.get_order("BTCUSDT", "a1") is None and b.placed_count == 0

    def test_timeout_after_execution(self, rules: SymbolRules, clock: FakeClock) -> None:
        b = broker(rules, clock)
        b.faults.push(FaultKind.TIMEOUT_EXECUTED)
        with pytest.raises(OutcomeUnknownError):
            b.place_order(market("a1", Side.BUY, "0.001"))
        got = b.get_order("BTCUSDT", "a1")
        assert got is not None and got.status is OrderStatus.FILLED and b.placed_count == 1

    def test_reject_fault(self, rules: SymbolRules, clock: FakeClock) -> None:
        b = broker(rules, clock)
        b.faults.push(FaultKind.REJECT)
        with pytest.raises(OrderRejected):
            b.place_order(market("a1", Side.BUY, "0.001"))


def test_paper_simulator_has_no_network_code() -> None:
    """The simulator must be structurally unable to reach the network."""
    tree = ast.parse(Path(paper_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned = {"httpx", "requests", "socket", "urllib", "aiohttp", "botux.exchange.client"}
    assert not (imported & banned), imported & banned
