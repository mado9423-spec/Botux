"""Core value objects shared by the backtester, the paper simulator and the live connector."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from botux.decimals import ZERO


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(StrEnum):
    """Order lifecycle. NEW/PARTIALLY_FILLED/FILLED/CANCELED/REJECTED/EXPIRED mirror Binance.

    The other three are local states:

    * PENDING_SEND: persisted before sending; we do not know whether it left the machine.
    * UNKNOWN: sent, but the answer was a timeout/5XX/-1007. It may have been executed.
    * LOST: reconciliation found no such order after repeated checks. Terminal. Never auto-resent.
    """

    PENDING_SEND = "PENDING_SEND"
    UNKNOWN = "UNKNOWN"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    LOST = "LOST"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATUSES

    @property
    def is_uncertain(self) -> bool:
        return self in (OrderStatus.PENDING_SEND, OrderStatus.UNKNOWN)


TERMINAL_STATUSES = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
        OrderStatus.LOST,
    }
)


@dataclass(frozen=True, slots=True)
class Candle:
    """One closed kline. ``open_time_ms``/``close_time_ms`` are Unix milliseconds (UTC)."""

    open_time_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time_ms: int


@dataclass(frozen=True, slots=True)
class Quote:
    """Best bid/ask snapshot."""

    symbol: str
    bid: Decimal
    ask: Decimal
    ts_ms: int

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """What we want to send. ``client_order_id`` is chosen by us and is the idempotency key."""

    client_order_id: str
    symbol: str
    side: Side
    type: OrderType
    quantity: Decimal
    price: Decimal | None = None  # LIMIT only
    reference_price: Decimal | None = None  # for filter checks on MARKET orders


@dataclass(frozen=True, slots=True)
class Fill:
    price: Decimal
    quantity: Decimal
    commission: Decimal
    commission_asset: str


@dataclass(slots=True)
class OrderResult:
    """Venue's (or simulator's) view of an order."""

    client_order_id: str
    symbol: str
    side: Side
    type: OrderType
    status: OrderStatus
    orig_qty: Decimal
    executed_qty: Decimal = ZERO
    cumulative_quote_qty: Decimal = ZERO
    price: Decimal | None = None
    exchange_order_id: str | None = None
    transact_time_ms: int = 0
    fills: list[Fill] = field(default_factory=list)
    expiry_reason: str = ""  # Binance `expiryReason` of an expired order (e.g. price range rule)

    @property
    def avg_price(self) -> Decimal | None:
        if self.executed_qty <= 0:
            return None
        return self.cumulative_quote_qty / self.executed_qty


@dataclass(frozen=True, slots=True)
class Balance:
    free: Decimal
    locked: Decimal = ZERO

    @property
    def total(self) -> Decimal:
        return self.free + self.locked
