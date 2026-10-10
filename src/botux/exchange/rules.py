"""Symbol trading rules parsed from ``/api/v3/exchangeInfo`` and local order validation.

Reference: https://github.com/binance/binance-spot-api-docs/blob/master/filters.md

We validate every order locally against PRICE_FILTER, LOT_SIZE, MARKET_LOT_SIZE,
MIN_NOTIONAL / NOTIONAL, PERCENT_PRICE(_BY_SIDE) and MAX_NUM_ORDERS *before* it is sent, so
that a violation never costs an API call, and quantities are always rounded DOWN.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from botux.decimals import ZERO, D, floor_to_step, is_multiple_of
from botux.errors import DataError, FilterViolation
from botux.models import OrderType, Side


@dataclass(frozen=True, slots=True)
class SymbolRules:
    symbol: str
    status: str
    base_asset: str
    quote_asset: str
    spot_trading_allowed: bool
    order_types: frozenset[str]
    # PRICE_FILTER (0 disables a sub-rule)
    min_price: Decimal = ZERO
    max_price: Decimal = ZERO
    tick_size: Decimal = ZERO
    # LOT_SIZE
    min_qty: Decimal = ZERO
    max_qty: Decimal = ZERO
    step_size: Decimal = ZERO
    # MARKET_LOT_SIZE (all zero = disabled)
    market_min_qty: Decimal = ZERO
    market_max_qty: Decimal = ZERO
    market_step_size: Decimal = ZERO
    # MIN_NOTIONAL / NOTIONAL
    min_notional: Decimal = ZERO
    apply_min_to_market: bool = True
    max_notional: Decimal = ZERO
    apply_max_to_market: bool = False
    # PERCENT_PRICE / PERCENT_PRICE_BY_SIDE (None = absent)
    percent_up_buy: Decimal | None = None
    percent_down_buy: Decimal | None = None
    percent_up_sell: Decimal | None = None
    percent_down_sell: Decimal | None = None
    # MAX_NUM_ORDERS (0 = absent)
    max_num_orders: int = 0

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_exchange_info(cls, payload: Mapping[str, Any], symbol: str) -> SymbolRules:
        """Parse one symbol out of an exchangeInfo response."""
        for entry in payload.get("symbols", []):
            if entry.get("symbol") == symbol:
                return cls._from_symbol_entry(entry)
        raise DataError(f"symbol {symbol} not present in exchangeInfo")

    @classmethod
    def _from_symbol_entry(cls, entry: Mapping[str, Any]) -> SymbolRules:
        filters = {f["filterType"]: f for f in entry.get("filters", [])}
        kw: dict[str, Any] = {}
        if (f := filters.get("PRICE_FILTER")) is not None:
            kw |= {
                "min_price": D(f["minPrice"]),
                "max_price": D(f["maxPrice"]),
                "tick_size": D(f["tickSize"]),
            }
        if (f := filters.get("LOT_SIZE")) is not None:
            kw |= {
                "min_qty": D(f["minQty"]),
                "max_qty": D(f["maxQty"]),
                "step_size": D(f["stepSize"]),
            }
        if (f := filters.get("MARKET_LOT_SIZE")) is not None:
            kw |= {
                "market_min_qty": D(f["minQty"]),
                "market_max_qty": D(f["maxQty"]),
                "market_step_size": D(f["stepSize"]),
            }
        min_notional = ZERO
        apply_min = True
        if (f := filters.get("MIN_NOTIONAL")) is not None:
            min_notional = D(f["minNotional"])
            apply_min = bool(f.get("applyToMarket", True))
        if (f := filters.get("NOTIONAL")) is not None:
            # Both may be present on some symbols: obey the stricter minimum.
            new_min = D(f["minNotional"])
            if new_min >= min_notional:
                min_notional = new_min
                apply_min = bool(f.get("applyMinToMarket", True))
            kw |= {
                "max_notional": D(f["maxNotional"]),
                "apply_max_to_market": bool(f.get("applyMaxToMarket", False)),
            }
        kw |= {"min_notional": min_notional, "apply_min_to_market": apply_min}
        if (f := filters.get("PERCENT_PRICE")) is not None:
            up, down = D(f["multiplierUp"]), D(f["multiplierDown"])
            kw |= {
                "percent_up_buy": up,
                "percent_down_buy": down,
                "percent_up_sell": up,
                "percent_down_sell": down,
            }
        if (f := filters.get("PERCENT_PRICE_BY_SIDE")) is not None:
            kw |= {
                "percent_up_buy": D(f["bidMultiplierUp"]),
                "percent_down_buy": D(f["bidMultiplierDown"]),
                "percent_up_sell": D(f["askMultiplierUp"]),
                "percent_down_sell": D(f["askMultiplierDown"]),
            }
        if (f := filters.get("MAX_NUM_ORDERS")) is not None:
            kw["max_num_orders"] = int(f["maxNumOrders"])
        return cls(
            symbol=entry["symbol"],
            status=entry.get("status", ""),
            base_asset=entry["baseAsset"],
            quote_asset=entry["quoteAsset"],
            spot_trading_allowed=bool(entry.get("isSpotTradingAllowed", False)),
            order_types=frozenset(entry.get("orderTypes", [])),
            **kw,
        )

    # ------------------------------------------------------------------ rounding

    def round_price(self, price: Decimal) -> Decimal:
        """Round DOWN to the tick size."""
        return floor_to_step(price, self.tick_size)

    def round_qty(self, qty: Decimal, order_type: OrderType = OrderType.MARKET) -> Decimal:
        """Round DOWN to a quantity valid for LOT_SIZE (and MARKET_LOT_SIZE for market orders).

        Binance applies LOT_SIZE to every order and MARKET_LOT_SIZE *in addition* for MARKET
        orders (a zero field in MARKET_LOT_SIZE means "not enforced"). Never rounds up.
        """
        steps = [self.step_size]
        if order_type is OrderType.MARKET and self.market_step_size:
            steps.append(self.market_step_size)
        rounded = qty
        for _ in range(4):  # converge when the two steps are not multiples of each other
            previous = rounded
            for step in steps:
                rounded = floor_to_step(rounded, step)
            if rounded == previous:
                break
        return rounded

    # ------------------------------------------------------------------ validation

    def violations(
        self,
        side: Side,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None,
        reference_price: Decimal | None,
        notional_buffer: Decimal = Decimal(1),
    ) -> list[str]:
        """Return human-readable filter violations (empty list = order passes locally).

        ``reference_price`` stands in for the exchange's "average price" on MARKET orders and
        for the PERCENT_PRICE check. ``notional_buffer`` (>= 1) demands extra headroom over
        MIN_NOTIONAL because the exchange evaluates it against its own price, not ours.
        """
        problems: list[str] = []
        if self.status != "TRADING":
            problems.append(f"symbol status is {self.status or 'unknown'}, not TRADING")
        if not self.spot_trading_allowed:
            problems.append("spot trading is not allowed for this symbol")
        if self.order_types and order_type.value not in self.order_types:
            problems.append(f"order type {order_type.value} not supported for {self.symbol}")
        if quantity <= 0:
            problems.append("quantity must be positive")
            return problems

        if self.min_qty and quantity < self.min_qty:
            problems.append(
                f"Filter failure: LOT_SIZE (quantity {quantity} < minQty {self.min_qty})"
            )
        if self.max_qty and quantity > self.max_qty:
            problems.append(
                f"Filter failure: LOT_SIZE (quantity {quantity} > maxQty {self.max_qty})"
            )
        if not is_multiple_of(quantity, self.step_size):
            problems.append(
                f"Filter failure: LOT_SIZE (quantity {quantity} not a multiple of {self.step_size})"
            )
        if order_type is OrderType.MARKET:
            if self.market_min_qty and quantity < self.market_min_qty:
                problems.append(
                    f"Filter failure: MARKET_LOT_SIZE (quantity {quantity} < "
                    f"minQty {self.market_min_qty})"
                )
            if self.market_max_qty and quantity > self.market_max_qty:
                problems.append(
                    f"Filter failure: MARKET_LOT_SIZE (quantity {quantity} > "
                    f"maxQty {self.market_max_qty})"
                )
            if not is_multiple_of(quantity, self.market_step_size):
                problems.append(
                    f"Filter failure: MARKET_LOT_SIZE (quantity {quantity} not a multiple of "
                    f"{self.market_step_size})"
                )

        eff_price = (
            price if (order_type is OrderType.LIMIT and price is not None) else reference_price
        )
        if order_type is OrderType.LIMIT:
            if price is None or price <= 0:
                problems.append("LIMIT order needs a positive price")
            else:
                if self.min_price and price < self.min_price:
                    problems.append(f"Filter failure: PRICE_FILTER (price {price} < minPrice)")
                if self.max_price and price > self.max_price:
                    problems.append(f"Filter failure: PRICE_FILTER (price {price} > maxPrice)")
                if not is_multiple_of(price, self.tick_size):
                    problems.append(
                        f"Filter failure: PRICE_FILTER (price {price} not a multiple of "
                        f"{self.tick_size})"
                    )
                problems += self._percent_price_problems(side, price, reference_price)

        if eff_price is None or eff_price <= 0:
            problems.append("no reference price available to evaluate notional filters")
        else:
            notional = quantity * eff_price
            check_min = order_type is OrderType.LIMIT or self.apply_min_to_market
            if self.min_notional and check_min and notional < self.min_notional * notional_buffer:
                problems.append(
                    f"Filter failure: NOTIONAL (notional {notional:.8f} < "
                    f"{self.min_notional * notional_buffer:.8f} incl. safety buffer)"
                )
            check_max = order_type is OrderType.LIMIT or self.apply_max_to_market
            if self.max_notional and check_max and notional > self.max_notional:
                problems.append(
                    f"Filter failure: NOTIONAL (notional {notional} > {self.max_notional})"
                )
        return problems

    def _percent_price_problems(
        self, side: Side, price: Decimal, reference_price: Decimal | None
    ) -> list[str]:
        if reference_price is None or reference_price <= 0:
            return []
        up, down = (
            (self.percent_up_buy, self.percent_down_buy)
            if side is Side.BUY
            else (self.percent_up_sell, self.percent_down_sell)
        )
        problems: list[str] = []
        if up is not None and price > reference_price * up:
            problems.append("Filter failure: PERCENT_PRICE (price too high vs. average)")
        if down is not None and price < reference_price * down:
            problems.append("Filter failure: PERCENT_PRICE (price too low vs. average)")
        return problems

    def validate(
        self,
        side: Side,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None,
        reference_price: Decimal | None,
        notional_buffer: Decimal = Decimal(1),
    ) -> None:
        """Raise :class:`FilterViolation` if the order would be rejected by the exchange."""
        problems = self.violations(
            side, order_type, quantity, price, reference_price, notional_buffer
        )
        if problems:
            raise FilterViolation("; ".join(problems))
