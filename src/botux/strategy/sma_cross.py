"""Simple moving-average crossover: an EDUCATIONAL example, not an investment recommendation.

* ENTER_LONG when the fast SMA crosses above the slow SMA on the latest closed candle.
* EXIT_LONG while in a position and the fast SMA is below the slow SMA.

Long-only (Spot cannot short). Such trend-following rules tend to lose money in sideways markets
because of repeated whipsaws and costs; nothing here suggests the rule is profitable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from botux.errors import ConfigError
from botux.models import Candle
from botux.strategy.base import Signal


class SmaCross:
    name = "sma_cross"

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if not (isinstance(fast, int) and isinstance(slow, int)):
            raise ConfigError("sma_cross: fast and slow must be integers")
        if not 2 <= fast < slow <= 1000:
            raise ConfigError("sma_cross: require 2 <= fast < slow <= 1000")
        self.fast = fast
        self.slow = slow
        self.warmup = slow + 1  # need slow+1 closes to compare the previous and current state

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> SmaCross:
        unknown = set(params) - {"fast", "slow"}
        if unknown:
            raise ConfigError(f"sma_cross: unknown parameter(s) {sorted(unknown)}")
        return cls(fast=params.get("fast", 20), slow=params.get("slow", 50))

    @staticmethod
    def _sma(candles: Sequence[Candle], n: int, end: int) -> Decimal:
        window = candles[end - n : end]
        return sum((c.close for c in window), Decimal(0)) / n

    def on_candles(self, candles: Sequence[Candle], in_position: bool) -> Signal:
        n = len(candles)
        if n < self.warmup:
            return Signal.HOLD
        fast_now, slow_now = self._sma(candles, self.fast, n), self._sma(candles, self.slow, n)
        fast_prev, slow_prev = (
            self._sma(candles, self.fast, n - 1),
            self._sma(candles, self.slow, n - 1),
        )
        if in_position:
            return Signal.EXIT_LONG if fast_now < slow_now else Signal.HOLD
        if fast_prev <= slow_prev and fast_now > slow_now:
            return Signal.ENTER_LONG
        return Signal.HOLD
