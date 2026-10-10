"""Performance statistics for a back-test. Floats are fine here: these are reporting numbers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

from botux.backtest.engine import BacktestResult

YEAR_MS = 365 * 86_400_000


@dataclass(frozen=True, slots=True)
class Metrics:
    total_return: float
    buy_and_hold_return: float
    buy_and_hold_same_capital: float
    max_drawdown: float
    trades: int
    win_rate: float | None
    profit_factor: float | None
    avg_trade_pnl: float | None
    exposure: float
    sharpe: float | None
    fees_paid: float
    final_equity: float


def compute_metrics(res: BacktestResult) -> Metrics:
    initial = float(res.initial_cash)
    equities = [float(e) for _, e in res.equity_curve]
    final = float(res.final_equity)
    total_return = final / initial - 1 if initial else 0.0

    peak = initial
    max_dd = 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)

    pnls = [float(t.pnl) for t in res.trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls) if pnls else None
    profit_factor = (sum(wins) / -sum(losses)) if losses else (math.inf if wins else None)
    avg = sum(pnls) / len(pnls) if pnls else None

    sharpe: float | None = None
    if len(equities) > 2 and res.interval_ms:
        rets = [b / a - 1 for a, b in pairwise(equities) if a > 0]
        if len(rets) > 1:
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
            if var > 0:
                sharpe = mean / math.sqrt(var) * math.sqrt(YEAR_MS / res.interval_ms)

    return Metrics(
        total_return=total_return,
        buy_and_hold_return=res.buy_and_hold_return,
        buy_and_hold_same_capital=res.buy_and_hold_same_capital,
        max_drawdown=max_dd,
        trades=len(pnls),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl=avg,
        exposure=res.candles_in_position / res.candles_total if res.candles_total else 0.0,
        sharpe=sharpe,
        fees_paid=float(res.fees_paid),
        final_equity=final,
    )


def format_metrics(m: Metrics) -> str:
    def signed(x: float) -> str:
        return f"{x:+.2%}"

    def plain(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.1%}"

    def num(x: float | None) -> str:
        return "n/a" if x is None else ("inf" if math.isinf(x) else f"{x:.2f}")

    return (
        f"  return {signed(m.total_return)} | buy&hold same capital "
        f"{signed(m.buy_and_hold_same_capital)} (100% invested: {signed(m.buy_and_hold_return)}) | "
        f"max drawdown {m.max_drawdown:.2%}\n"
        f"  trades {m.trades} | win rate {plain(m.win_rate)} | "
        f"profit factor {num(m.profit_factor)} | avg trade PnL {num(m.avg_trade_pnl)}\n"
        f"  exposure {m.exposure:.1%} | Sharpe(ann., rf=0) {num(m.sharpe)} | "
        f"fees {m.fees_paid:.2f} | final equity {m.final_equity:.2f}"
    )
