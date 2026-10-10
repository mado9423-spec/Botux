"""Candle-by-candle back-test sharing the paper simulator's cost model.

Conventions (chosen to avoid optimistic bias):

* No look-ahead: the strategy sees candles up to and including candle ``i`` (closed); any order
  it triggers executes at the OPEN of candle ``i + 1``.
* Costs: slippage against us on every fill (BUY price rounded up, SELL rounded down to the tick)
  plus the fee, charged in the received asset exactly like the paper simulator.
* Stop-loss: if a candle's low touches the stop, we exit at the stop price, or at the open if
  the candle gaps below it, then add slippage. If the entry candle itself touches the stop we
  assume it was hit (pessimistic).
* Sizing and limits use the same :class:`RiskManager` as the live loop.

Known limits: exchange filters are today's, not historical; fills are assumed complete (no
order-book depth); no latency; no outages. Real results will differ, usually for the worse.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal

from botux.clock import FakeClock
from botux.config import RiskSettings
from botux.decimals import ZERO, ceil_to_step, floor_to_step
from botux.exchange.rules import SymbolRules
from botux.models import Candle, OrderType, Side
from botux.risk.manager import PortfolioView, RiskManager
from botux.strategy.base import Signal, Strategy

_COMMISSION_PRECISION = Decimal("0.00000001")


@dataclass(frozen=True, slots=True)
class Trade:
    entry_time_ms: int
    exit_time_ms: int
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl: Decimal  # net of fees, in quote units
    fees: Decimal
    exit_reason: str  # "signal" | "stop" | "end"


@dataclass(slots=True)
class BacktestResult:
    equity_curve: list[tuple[int, Decimal]] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    initial_cash: Decimal = ZERO
    final_equity: Decimal = ZERO
    candles_in_position: int = 0
    candles_total: int = 0
    fees_paid: Decimal = ZERO
    buy_and_hold_return: float = 0.0
    interval_ms: int = 0


def run_backtest(
    candles: Sequence[Candle],
    strategy: Strategy,
    rules: SymbolRules,
    risk: RiskSettings,
    *,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_bps: Decimal,
    interval_ms: int,
) -> BacktestResult:
    slip = slippage_bps / Decimal(10_000)
    clock = FakeClock()
    rm = RiskManager(risk, fee_rate=fee_rate, slippage_bps=slippage_bps, clock=clock)
    sym = rules.symbol

    cash = initial_cash
    base = ZERO  # base asset held
    entry_price = ZERO
    entry_cost = ZERO  # quote spent (excl. fee, which is paid in base)
    entry_time = 0
    stop_price = ZERO
    trade_fees = ZERO
    pending: Signal = Signal.HOLD
    res = BacktestResult(
        initial_cash=initial_cash, interval_ms=interval_ms, candles_total=len(candles)
    )

    def holding() -> bool:
        """True while we hold a sellable amount (dust below minQty counts as flat)."""
        sellable = rules.round_qty(base, OrderType.MARKET)
        return sellable > 0 and sellable >= rules.min_qty

    def buy_fill(raw_open: Decimal) -> Decimal:
        return ceil_to_step(raw_open * (1 + slip), rules.tick_size)

    def sell_fill(raw: Decimal) -> Decimal:
        return floor_to_step(raw * (1 - slip), rules.tick_size)

    def close_position(t_ms: int, raw_price: Decimal, reason: str) -> None:
        nonlocal cash, base, entry_cost, trade_fees, entry_price
        qty = rules.round_qty(base, OrderType.MARKET)
        if qty <= 0:
            return
        price = sell_fill(raw_price)
        if rules.violations(Side.SELL, OrderType.MARKET, qty, None, price):
            return  # below an exchange minimum (dust): cannot be sold
        notional = qty * price
        fee = (notional * fee_rate).quantize(_COMMISSION_PRECISION, rounding=ROUND_CEILING)
        cash += notional - fee
        base -= qty
        trade_fees += fee
        pnl = notional - fee - entry_cost
        res.trades.append(Trade(entry_time, t_ms, entry_price, price, qty, pnl, trade_fees, reason))
        res.fees_paid += fee
        entry_cost = ZERO
        trade_fees = ZERO

    n = len(candles)
    for i, c in enumerate(candles):
        clock.set_ms(c.open_time_ms)
        # 1) execute the order decided at the previous close, at this candle's open
        if pending is Signal.ENTER_LONG and not holding():
            mark = c.open
            equity = cash + base * mark
            rm.update_equity(c.open_time_ms, equity)
            view = PortfolioView(equity, cash, {sym: ZERO}, {sym: base})
            sizing = rm.size_entry(rules, buy_fill(c.open), view)
            if sizing.quantity > 0:
                price = buy_fill(c.open)
                cost = sizing.quantity * price
                fee_base = (sizing.quantity * fee_rate).quantize(
                    _COMMISSION_PRECISION, rounding=ROUND_CEILING
                )
                if cost <= cash:
                    cash -= cost
                    base += sizing.quantity - fee_base
                    entry_price, entry_cost, entry_time = price, cost, c.open_time_ms
                    stop_price = rm.stop_price(price)
                    trade_fees = fee_base * price  # base-asset fee valued at entry price
                    res.fees_paid += fee_base * price
        elif pending is Signal.EXIT_LONG and holding():
            close_position(c.open_time_ms, c.open, "signal")
        pending = Signal.HOLD

        # 2) stop-loss inside this candle
        if holding() and c.low <= stop_price:
            close_position(c.open_time_ms, min(stop_price, c.open), "stop")

        # 3) mark to market at the close and decide for the next open
        equity = cash + base * c.close
        res.equity_curve.append((c.close_time_ms, equity))
        if holding():
            res.candles_in_position += 1
        rm.update_equity(c.close_time_ms, equity)
        if i + 1 < n:
            lo = max(0, i + 1 - strategy.warmup)
            pending = strategy.on_candles(candles[lo : i + 1], holding())

    if holding():  # close at the final close so every trade is accounted for
        last = candles[-1]
        close_position(last.close_time_ms, last.close, "end")
        res.equity_curve[-1] = (last.close_time_ms, cash + base * last.close)
    res.final_equity = cash + base * candles[-1].close if candles else initial_cash
    if candles:
        res.buy_and_hold_return = float(candles[-1].close / candles[0].open - 1)
    return res
