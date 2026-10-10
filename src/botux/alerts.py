"""Alerting. Every message is redacted before it leaves the process."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from botux.clock import Clock, SystemClock
from botux.secrets_guard import GLOBAL_REDACTOR, Redactor

log = logging.getLogger(__name__)

LEVELS = ("INFO", "WARNING", "CRITICAL")


class Notifier(Protocol):
    def notify(self, level: str, title: str, message: str) -> None: ...


class LogNotifier:
    def __init__(self, redactor: Redactor | None = None) -> None:
        self._r = redactor or GLOBAL_REDACTOR

    def notify(self, level: str, title: str, message: str) -> None:
        lvl = {"INFO": logging.INFO, "WARNING": logging.WARNING}.get(level, logging.CRITICAL)
        log.log(lvl, "ALERT %s: %s", self._r.scrub(title), self._r.scrub(message))


class FileNotifier:
    """Append-only JSON lines. Easy to tail or ship; no network access."""

    def __init__(
        self, path: str | Path, clock: Clock | None = None, redactor: Redactor | None = None
    ) -> None:
        self._path = Path(path)
        self._clock = clock or SystemClock()
        self._r = redactor or GLOBAL_REDACTOR

    def notify(self, level: str, title: str, message: str) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(
                {
                    "ts_ms": self._clock.now_ms(),
                    "level": level,
                    "title": self._r.scrub(title),
                    "message": self._r.scrub(message),
                },
                sort_keys=True,
            )
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            log.exception("could not write alert file %s", self._path)


@dataclass
class MemoryNotifier:
    """Collects alerts in memory (tests)."""

    items: list[tuple[str, str, str]] = field(default_factory=list)

    def notify(self, level: str, title: str, message: str) -> None:
        self.items.append((level, title, message))

    def titles(self) -> list[str]:
        return [t for _, t, _ in self.items]


class CompositeNotifier:
    def __init__(self, *notifiers: Notifier) -> None:
        self._notifiers = notifiers

    def notify(self, level: str, title: str, message: str) -> None:
        for n in self._notifiers:
            try:
                n.notify(level, title, message)
            except Exception:
                log.exception("notifier %s failed", type(n).__name__)
