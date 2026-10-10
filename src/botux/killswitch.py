"""Emergency stop.

The switch is "tripped" when either the in-memory flag is set or the kill file exists. The file
makes it usable from another terminal even if the bot is unresponsive to signals:

    botux kill --reason "why"      # creates the file
    botux unkill                   # removes it (an explicit human decision)

A tripped switch is sticky: the bot never clears it by itself.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from botux.clock import Clock, SystemClock

log = logging.getLogger(__name__)


class KillSwitch:
    def __init__(self, path: str | Path, clock: Clock | None = None) -> None:
        self._path = Path(path)
        self._clock = clock or SystemClock()
        self._tripped = False
        self._reason = ""

    @property
    def path(self) -> Path:
        return self._path

    def is_set(self) -> bool:
        return self._tripped or self._path.exists()

    def reason(self) -> str:
        if self._reason:
            return self._reason
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "kill file present"
        if isinstance(data, dict):
            return str(data.get("reason", "kill file present"))
        return "kill file present"  # e.g. `echo 1 > KILL`: any content still means STOP

    def trip(self, reason: str) -> None:
        """Set the switch. The in-memory flag is set even if the file cannot be written."""
        self._tripped = True
        self._reason = reason
        log.critical("KILL SWITCH TRIPPED: %s", reason)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".kill-")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"reason": reason, "ts_ms": self._clock.now_ms()}, fh)
            os.replace(tmp, self._path)
        except OSError:
            log.exception("could not write kill file %s (in-memory flag is still set)", self._path)

    def clear(self) -> None:
        """Remove the kill file and reset the flag. Only a human-invoked command calls this."""
        self._tripped = False
        self._reason = ""
        self._path.unlink(missing_ok=True)
