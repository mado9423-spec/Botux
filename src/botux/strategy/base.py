"""Strategy interface. A strategy only ever sees CLOSED candles and returns a Signal.

It knows nothing about money, orders or risk: sizing and limits live in the risk manager, so a
strategy can be swapped or tested in isolation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol

from botux.errors import ConfigError
from botux.models import Candle


class Signal(StrEnum):
    HOLD = "HOLD"
    ENTER_LONG = "ENTER_LONG"
    EXIT_LONG = "EXIT_LONG"


class Strategy(Protocol):
    name: str
    warmup: int  # minimum number of closed candles needed to produce a meaningful signal

    def on_candles(self, candles: Sequence[Candle], in_position: bool) -> Signal:
        """``candles`` ends with the most recently CLOSED candle (never the forming one)."""
        ...


def create_strategy(name: str, params: Mapping[str, Any]) -> Strategy:
    from botux.strategy.sma_cross import SmaCross  # local import: registry without cycles

    registry = {"sma_cross": SmaCross}
    try:
        cls = registry[name]
    except KeyError:
        raise ConfigError(f"unknown strategy {name!r}; available: {sorted(registry)}") from None
    return cls.from_params(params)
