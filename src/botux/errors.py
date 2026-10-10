"""Exception hierarchy. Every error the bot raises on purpose derives from BotuxError."""

from __future__ import annotations


class BotuxError(Exception):
    """Base class for all intentional Botux errors."""


class ConfigError(BotuxError):
    """Invalid or unsafe configuration."""


class LiveTradingNotAllowed(BotuxError):
    """Live trading was requested but at least one safety gate is not satisfied."""


class FilterViolation(BotuxError):
    """An order violates the exchange's symbol filters (tick size, lot size, notional...)."""


class StorageError(BotuxError):
    """The local database cannot be trusted (failed integrity check, unwritable)."""


class DataError(BotuxError):
    """Market data is missing, inconsistent, stale or otherwise unusable."""


class OrderRejected(BotuxError):
    """The venue (or the paper simulator) definitively rejected the order. Nothing was executed."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


# --- exchange / transport errors ---------------------------------------------------------


class ExchangeError(BotuxError):
    """Base class for errors raised by the Binance connector."""


class EndpointNotAllowed(ExchangeError):
    """The requested endpoint is not on the connector's allow-list (or is not enabled)."""


class APIError(ExchangeError):
    """Binance answered with an error payload ({"code": ..., "msg": ...}) or an HTTP 4xx."""

    def __init__(self, message: str, *, code: int | None = None, http_status: int | None = None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status


class GeoRestrictedError(APIError):
    """HTTP 451: the service is not available from this location. Never worked around."""


class ClockSkewError(APIError):
    """Local clock is too far from the server clock (code -1021 or measured skew too large)."""


class RateLimitError(ExchangeError):
    """HTTP 429 (or a local cool-down after one). Back off; do not hammer the API."""

    def __init__(self, message: str, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class IPBannedError(RateLimitError):
    """HTTP 418: the IP was auto-banned for ignoring 429s. Nothing is sent until it expires."""


class NetworkError(ExchangeError):
    """Transport-level failure.

    ``not_sent`` is True only when we know the request never left the machine (e.g. the TCP
    connection could not be established), so retrying cannot create a duplicate.
    """

    def __init__(self, message: str, *, not_sent: bool = False) -> None:
        super().__init__(message)
        self.not_sent = not_sent


class OutcomeUnknownError(ExchangeError):
    """A state-changing request may or may not have been executed (timeout, 5XX, -1007).

    Never treat this as a failure and never blindly resend: reconcile by querying the order.
    """
