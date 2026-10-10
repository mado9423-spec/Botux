"""Injectable time sources so that retries, cool-downs and daily limits are testable."""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """Wall-clock + monotonic time and sleeping, all in one injectable object."""

    def now_ms(self) -> int:
        """Wall-clock Unix time in milliseconds."""
        ...

    def monotonic(self) -> float:
        """Monotonic seconds (for intervals and cool-downs)."""
        ...

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds``."""
        ...


class SystemClock:
    """Real time."""

    def now_ms(self) -> int:
        return int(time.time() * 1000)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class FakeClock:
    """Deterministic clock: ``sleep`` advances time instantly. Used by tests and backtests."""

    def __init__(self, start_ms: int = 1_700_000_000_000) -> None:
        self._ms = float(start_ms)
        self.sleeps: list[float] = []

    def now_ms(self) -> int:
        return int(self._ms)

    def monotonic(self) -> float:
        return self._ms / 1000.0

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._ms += max(0.0, seconds) * 1000.0

    def advance(self, seconds: float) -> None:
        self._ms += seconds * 1000.0

    def set_ms(self, ms: int) -> None:
        self._ms = float(ms)
