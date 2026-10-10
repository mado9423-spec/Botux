"""Paper-trading simulator. It never touches the network.

Fill model (deliberately pessimistic and simple):

* MARKET BUY fills at ask * (1 + slippage), rounded UP to the tick; SELL at bid * (1 - slippage),
  rounded DOWN. LIMIT orders fill at their limit price when the quote crosses them.
* The fee is charged in the asset received, as Binance does without BNB discount: BUY pays it in
  the base asset (so you receive slightly less than you bought), SELL pays it in the quote asset.
* The same exchange filters as the real venue are applied, so a quantity that would be rejected
  live is rejected here too.
* Binance semantics are mimicked on purpose: an order id is rejected as a duplicate unless the
  previous order with that id was FILLED ("accepted only when the previous one is filled").
* ``FaultPlan`` injects rejections, timeouts (executed or not) and outages so that the order
  manager's handling of uncertain outcomes can be exercised without a real exchange.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from decimal import ROUND_CEILING, Decimal
from enum import StrEnum

from botux.clock import Clock, SystemClock
from botux.decimals import ZERO, ceil_to_step, floor_to_step
from botux.errors import NetworkError, OrderRejected, OutcomeUnknownError
from botux.exchange.rules import SymbolRules
from botux.models import (
    Balance,
    Fill,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Quote,
    Side,
)

_COMMISSION_PRECISION = Decimal("0.00000001")


class FaultKind(StrEnum):
    REJECT = "reject"
    TIMEOUT_EXECUTED = "timeout_executed"  # executes, then reports an unknown outcome
    TIMEOUT_NOT_EXECUTED = "timeout_not_executed"  # unknown outcome, nothing happened
    NETWORK_DOWN = "network_down"  # definitively not sent


class FaultPlan:
    """FIFO queue of faults applied to the next ``place_order`` calls."""

    def __init__(self) -> None:
        self._queue: deque[FaultKind] = deque()

    def push(self, kind: FaultKind, times: int = 1) -> None:
        self._queue.extend([kind] * times)

    def pop(self) -> FaultKind | None:
        return self._queue.popleft() if self._queue else None


class PaperBroker:
    def __init__(
        self,
        rules: Mapping[str, SymbolRules],
        balances: Mapping[str, Decimal],
        *,
        fee_rate: Decimal,
        slippage_bps: Decimal,
        clock: Clock | None = None,
        faults: FaultPlan | None = None,
    ) -> None:
        self._rules = dict(rules)
        self._free: dict[str, Decimal] = dict(balances)
        self._locked: dict[str, Decimal] = {}
        self._fee = fee_rate
        self._slip = slippage_bps / Decimal(10_000)
        self._clock = clock or SystemClock()
        self.faults = faults or FaultPlan()
        self._orders: dict[str, OrderResult] = {}
        self._quotes: dict[str, Quote] = {}
        self._next_id = 1
        self.placed_count = 0  # number of orders actually executed/accepted by the simulator

    # ------------------------------------------------------------------ market data

    def set_quote(self, quote: Quote) -> None:
        self._quotes[quote.symbol] = quote
        self._match_resting(quote.symbol)

    # ------------------------------------------------------------------ Broker API

    def place_order(self, request: OrderRequest) -> OrderResult:
        fault = self.faults.pop()
        if fault is FaultKind.NETWORK_DOWN:
            raise NetworkError("simulated outage", not_sent=True)
        if fault is FaultKind.REJECT:
            raise OrderRejected("simulated rejection", code=-2010)
        if fault is FaultKind.TIMEOUT_NOT_EXECUTED:
            raise OutcomeUnknownError("simulated timeout; order was NOT executed")
        result = self._execute(request)
        if fault is FaultKind.TIMEOUT_EXECUTED:
            raise OutcomeUnknownError("simulated timeout; order WAS executed")
        return result

    def get_order(self, symbol: str, client_order_id: str) -> OrderResult | None:
        order = self._orders.get(client_order_id)
        return order if order is not None and order.symbol == symbol else None

    def cancel_order(self, symbol: str, client_order_id: str) -> OrderResult | None:
        order = self._orders.get(client_order_id)
        if order is None or order.symbol != symbol or order.status.is_terminal:
            return None
        self._unlock(order)
        order.status = OrderStatus.CANCELED
        order.transact_time_ms = self._clock.now_ms()
        return order

    def open_orders(self, symbol: str) -> list[OrderResult]:
        return [o for o in self._orders.values() if o.symbol == symbol and not o.status.is_terminal]

    def balances(self) -> dict[str, Balance]:
        assets = set(self._free) | set(self._locked)
        return {a: Balance(self._free.get(a, ZERO), self._locked.get(a, ZERO)) for a in assets}

    # ------------------------------------------------------------------ internals

    def _reject(self, message: str, code: int) -> OrderRejected:
        return OrderRejected(message, code=code)

    def _execute(self, req: OrderRequest) -> OrderResult:
        rules = self._rules.get(req.symbol)
        if rules is None:
            raise self._reject("Invalid symbol.", -1121)
        existing = self._orders.get(req.client_order_id)
        if existing is not None and existing.status is not OrderStatus.FILLED:
            # Binance: an id is reusable only once the previous order is FILLED (not when it was
            # cancelled, expired or is still open).
            raise self._reject("Duplicate order sent.", -2010)
        quote = self._quotes.get(req.symbol)
        if quote is None:
            raise self._reject("no market data for symbol in simulator", -1013)

        ref = req.reference_price or (quote.ask if req.side is Side.BUY else quote.bid)
        problems = rules.violations(req.side, req.type, req.quantity, req.price, ref)
        if problems:
            raise self._reject("; ".join(problems), -1013)

        order = OrderResult(
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            side=req.side,
            type=req.type,
            status=OrderStatus.NEW,
            orig_qty=req.quantity,
            price=req.price,
            exchange_order_id=str(self._next_id),
            transact_time_ms=self._clock.now_ms(),
        )
        self._next_id += 1

        marketable_price: Decimal | None = None
        if req.type is OrderType.MARKET:
            raw = (
                quote.ask * (1 + self._slip)
                if req.side is Side.BUY
                else quote.bid * (1 - self._slip)
            )
            marketable_price = (
                ceil_to_step(raw, rules.tick_size)
                if req.side is Side.BUY
                else floor_to_step(raw, rules.tick_size)
            )
        else:
            assert req.price is not None
            if req.side is Side.BUY and req.price >= quote.ask:
                marketable_price = quote.ask
            elif req.side is Side.SELL and req.price <= quote.bid:
                marketable_price = quote.bid

        if marketable_price is not None:
            self._check_funds(rules, req.side, req.quantity, marketable_price)
            self._settle(order, rules, marketable_price)
        else:
            # resting LIMIT order: lock the funds it could consume
            assert req.price is not None
            self._check_funds(rules, req.side, req.quantity, req.price)
            self._lock(order, rules)
        self._orders[req.client_order_id] = order
        self.placed_count += 1
        return order

    def _check_funds(self, rules: SymbolRules, side: Side, qty: Decimal, price: Decimal) -> None:
        if side is Side.BUY:
            if self._free.get(rules.quote_asset, ZERO) < qty * price:
                raise self._reject("Account has insufficient balance for requested action.", -2010)
        elif self._free.get(rules.base_asset, ZERO) < qty:
            raise self._reject("Account has insufficient balance for requested action.", -2010)

    def _lock(self, order: OrderResult, rules: SymbolRules) -> None:
        if order.side is Side.BUY:
            assert order.price is not None
            amount, asset = order.orig_qty * order.price, rules.quote_asset
        else:
            amount, asset = order.orig_qty, rules.base_asset
        self._free[asset] = self._free.get(asset, ZERO) - amount
        self._locked[asset] = self._locked.get(asset, ZERO) + amount

    def _unlock(self, order: OrderResult) -> None:
        rules = self._rules[order.symbol]
        remaining = order.orig_qty - order.executed_qty
        if order.side is Side.BUY:
            assert order.price is not None
            amount, asset = remaining * order.price, rules.quote_asset
        else:
            amount, asset = remaining, rules.base_asset
        self._locked[asset] = self._locked.get(asset, ZERO) - amount
        self._free[asset] = self._free.get(asset, ZERO) + amount

    def _settle(self, order: OrderResult, rules: SymbolRules, price: Decimal) -> None:
        """Apply a full fill at ``price`` (funds must be free, not locked)."""
        qty = order.orig_qty - order.executed_qty
        notional = qty * price
        if order.side is Side.BUY:
            fee = (qty * self._fee).quantize(_COMMISSION_PRECISION, rounding=ROUND_CEILING)
            fee_asset = rules.base_asset
            self._free[rules.quote_asset] = self._free.get(rules.quote_asset, ZERO) - notional
            self._free[rules.base_asset] = self._free.get(rules.base_asset, ZERO) + qty - fee
        else:
            fee = (notional * self._fee).quantize(_COMMISSION_PRECISION, rounding=ROUND_CEILING)
            fee_asset = rules.quote_asset
            self._free[rules.base_asset] = self._free.get(rules.base_asset, ZERO) - qty
            self._free[rules.quote_asset] = self._free.get(rules.quote_asset, ZERO) + notional - fee
        order.fills.append(Fill(price, qty, fee, fee_asset))
        order.executed_qty += qty
        order.cumulative_quote_qty += notional
        order.status = OrderStatus.FILLED
        order.transact_time_ms = self._clock.now_ms()

    def _match_resting(self, symbol: str) -> None:
        quote = self._quotes[symbol]
        rules = self._rules.get(symbol)
        if rules is None:
            return
        for order in list(self._orders.values()):
            if (
                order.symbol != symbol
                or order.status.is_terminal
                or order.type is not OrderType.LIMIT
            ):
                continue
            assert order.price is not None
            crosses = (order.side is Side.BUY and order.price >= quote.ask) or (
                order.side is Side.SELL and order.price <= quote.bid
            )
            if crosses:
                self._unlock(order)
                self._settle(order, rules, order.price)
