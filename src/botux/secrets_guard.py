"""Keeping secrets out of logs, alerts and string representations.

Three layers:

1. ``Secret`` / ``Credentials``: objects whose ``repr``/``str`` never reveal the value.
2. ``Redactor``: scrubs registered secret values and well-known patterns (Binance-style 64-char
   keys, ``signature=...``, ``X-MBX-APIKEY: ...``, ``api_key=...``) from any text.
3. ``RedactingFilter`` / ``RedactingFormatter``: plug the redactor into ``logging`` so that even
   messages, arguments and tracebacks are scrubbed before they reach a handler.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

REDACTED = "***REDACTED***"

# Order matters: specific patterns first.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(signature=)[0-9A-Za-z%+/=_-]{16,}"),
    re.compile(r"(?i)(x-mbx-apikey['\"]?\s*[:=]\s*['\"]?)[^\s,'\"}&]+"),
    re.compile(
        r"(?i)((?:api[_-]?key|api[_-]?secret|secret[_-]?key|secret|token|password|passwd)"
        r"['\"]?\s*[:=]\s*['\"]?)[^\s,'\"}&]+"
    ),
    # Binance API keys / HMAC secrets are 64 alphanumeric characters.
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{64}(?![A-Za-z0-9])"),
)


class Redactor:
    """Thread-safe text scrubber with a runtime-extensible set of literal secrets."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._lock = threading.Lock()
        self._secrets: set[str] = set()
        for s in secrets:
            self.register(s)

    def register(self, secret: str) -> None:
        """Add a literal value that must never appear in any output."""
        if secret and len(secret) >= 4:
            with self._lock:
                self._secrets.add(secret)

    def scrub(self, text: str) -> str:
        with self._lock:
            literals = sorted(self._secrets, key=len, reverse=True)
        for lit in literals:
            text = text.replace(lit, REDACTED)
        for pat in _PATTERNS:
            if pat.groups:
                text = pat.sub(lambda m: m.group(1) + REDACTED, text)
            else:
                text = pat.sub(REDACTED, text)
        return text


# Process-wide redactor used by the logging filter and by notifiers.
GLOBAL_REDACTOR = Redactor()


class RedactingFilter(logging.Filter):
    """Scrub the fully formatted message (args merged) of every record."""

    def __init__(self, redactor: Redactor | None = None) -> None:
        super().__init__()
        self._redactor = redactor or GLOBAL_REDACTOR

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg = self._redactor.scrub(message)
        record.args = None
        return True


class RedactingFormatter(logging.Formatter):
    """Also scrub exception tracebacks and stack info, which Filters do not see."""

    def __init__(self, fmt: str | None = None, redactor: Redactor | None = None) -> None:
        super().__init__(fmt)
        self._redactor = redactor or GLOBAL_REDACTOR

    def format(self, record: logging.LogRecord) -> str:
        return self._redactor.scrub(super().format(record))


@dataclass(frozen=True, slots=True)
class Secret:
    """A string that refuses to be printed."""

    _value: str = field(repr=False)

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(***)"

    def __str__(self) -> str:
        return "***"


@dataclass(frozen=True, slots=True)
class Credentials:
    """Binance API key pair. Neither value appears in ``repr``/``str``."""

    api_key: Secret
    api_secret: Secret

    def __repr__(self) -> str:
        return "Credentials(***)"

    __str__ = __repr__


def setup_logging(level: int = logging.INFO, *, log_file: str | None = None) -> None:
    """Configure root logging with redaction on every handler."""
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        h.setFormatter(RedactingFormatter(fmt))
        h.addFilter(RedactingFilter())
        root.addHandler(h)
    root.setLevel(level)
    # httpx logs full request URLs at INFO, which would include query strings.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
