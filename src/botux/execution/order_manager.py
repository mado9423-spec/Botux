"""Order lifecycle: idempotent submission, uncertain outcomes, reconciliation.

The rules this module enforces (each is covered by a test):

1. Write-ahead: the order is stored as PENDING_SEND *before* it is sent.
2. Deterministic ``client_order_id`` derived from an intent key => the same intent can never be
   sent twice, even across restarts.
3. An order is NEVER re-sent automatically. If the outcome is unknown (timeout, 5XX, -1007...) we
   query the venue by client order id and adopt whatever state it reports.
4. While any order on a symbol is uncertain, no new order is sent for that symbol.
5. If reconciliation cannot find the order after N attempts it becomes LOST: we alert, keep the
   symbol blocked, and wait for a human (``botux resolve-lost``).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal

from botux.alerts import Notifier
from botux.clock import Clock, SystemClock
from botux.errors import ExchangeError, NetworkError, OrderRejected, OutcomeUnknownError
from botux.exchange.rules import SymbolRules
from botux.execution.broker import Broker
from botux.killswitch import KillSwitch
from botux.models import Fill, OrderRequest, OrderResult, OrderStatus, OrderType, Side
from botux.storage import OrderRecord, Store

log = logging.getLogger(__name__)

OPEN_STATUSES = (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)
UNCERTAIN_STATUSES = (OrderStatus.PENDING_SEND, OrderStatus.UNKNOWN)

FillCallback = Callable[[OrderRecord, list[Fill]], None]


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A decision to trade. ``intent_key`` makes it idempotent (strategy:symbol:candle:side)."""

    intent_key: str
    symbol: str
    side: Side
    type: OrderType
    quantity: Decimal
    reference_price: Decimal
    price: Decimal | None = None
    is_exit: bool = False  # exits stay possible during a daily-loss halt / flatten


@dataclass(frozen=True, slots=True)
class SubmitResult:
    outcome: str  # accepted | duplicate | blocked | rejected | uncertain
    record: OrderRecord | None = None
    reason: str = ""


def client_order_id_for(intent_key: str) -> str:
    """Deterministic id: 'bx' + 30 hex chars = 32 chars, within Binance's 36-char limit."""
    return "bx" + hashlib.sha256(intent_key.encode()).hexdigest()[:30]


class OrderManager:
    def __init__(
        self,
        broker: Broker,
        store: Store,
        rules: Mapping[str, SymbolRules],
        notifier: Notifier,
        *,
        clock: Clock | None = None,
        kill_switch: KillSwitch | None = None,
        reconcile_attempts: int = 5,
        reconcile_interval_seconds: float = 5.0,
        notional_buffer: Decimal = Decimal("1.05"),
        on_fill: FillCallback | None = None,
    ) -> None:
        self._broker = broker
        self._store = store
        self._rules = dict(rules)
        self._notifier = notifier
        self._clock = clock or SystemClock()
        self._kill = kill_switch
        self._max_attempts = reconcile_attempts
        self._interval = reconcile_interval_seconds
        self._buffer = notional_buffer
        self._on_fill = on_fill

    def set_fill_callback(self, callback: FillCallback) -> None:
        self._on_fill = callback

    # ------------------------------------------------------------------ queries

    def uncertain_orders(self, symbol: str | None = None) -> list[OrderRecord]:
        return self._store.list_orders(UNCERTAIN_STATUSES, symbol)

    def open_orders(self, symbol: str | None = None) -> list[OrderRecord]:
        return self._store.list_orders(OPEN_STATUSES, symbol)

    def is_symbol_blocked(self, symbol: str) -> str | None:
        """Reason string when no new order may be sent for ``symbol``, else None."""
        if self.uncertain_orders(symbol):
            return f"unresolved order(s) on {symbol}: reconciling first"
        if self._store.kv_get(f"lost:{symbol}"):
            return (
                f"an order on {symbol} was LOST; verify on the exchange then run "
                "'botux resolve-lost'"
            )
        return None

    def resolve_lost(self, symbol: str) -> None:
        """Human decision: the operator checked the exchange and accepts the current state."""
        self._store.kv_set(f"lost:{symbol}", "")
        self._store.log_event(self._clock.now_ms(), "WARNING", "resolve_lost", {"symbol": symbol})

    # ------------------------------------------------------------------ submit

    def submit(self, intent: OrderIntent) -> SubmitResult:
        now = self._clock.now_ms()
        if self._kill is not None and self._kill.is_set() and not intent.is_exit:
            return SubmitResult("blocked", None, "kill switch is set")
        blocked = self.is_symbol_blocked(intent.symbol)
        if blocked:
            return SubmitResult("blocked", None, blocked)
        existing = self._store.get_by_intent(intent.intent_key)
        if existing is not None:
            return SubmitResult("duplicate", existing, "intent already submitted")

        rules = self._rules.get(intent.symbol)
        if rules is None:
            return SubmitResult("blocked", None, f"no trading rules loaded for {intent.symbol}")
        buffer = self._buffer if intent.side is Side.BUY else Decimal(1)
        problems = rules.violations(
            intent.side, intent.type, intent.quantity, intent.price, intent.reference_price, buffer
        )
        if problems:
            return SubmitResult("blocked", None, "; ".join(problems))

        cid = client_order_id_for(intent.intent_key)
        record = OrderRecord(
            client_order_id=cid,
            intent_key=intent.intent_key,
            symbol=intent.symbol,
            side=intent.side,
            type=intent.type,
            quantity=intent.quantity,
            price=intent.price,
            status=OrderStatus.PENDING_SEND,
            created_ms=now,
            updated_ms=now,
        )
        if not self._store.insert_order(record):  # lost a race / duplicate key
            return SubmitResult("duplicate", self._store.get_by_intent(intent.intent_key), "")

        request = OrderRequest(
            client_order_id=cid,
            symbol=intent.symbol,
            side=intent.side,
            type=intent.type,
            quantity=intent.quantity,
            price=intent.price,
            reference_price=intent.reference_price,
        )
        try:
            result = self._broker.place_order(request)
        except OrderRejected as exc:
            return self._finish_rejected(record, f"rejected: {exc}")
        except NetworkError as exc:
            if exc.not_sent:
                return self._finish_rejected(record, f"not sent: {exc}")
            return self._finish_uncertain(record, str(exc))
        except OutcomeUnknownError as exc:
            return self._finish_uncertain(record, str(exc))
        except ExchangeError as exc:
            # 4XX / rate limit / geo block / clock skew on an order => it was refused, not executed.
            return self._finish_rejected(record, f"{type(exc).__name__}: {exc}")

        self._apply(record, result)
        return SubmitResult("accepted", record, "")

    def _finish_rejected(self, record: OrderRecord, note: str) -> SubmitResult:
        record.status = OrderStatus.REJECTED
        record.note = note[:500]
        record.updated_ms = self._clock.now_ms()
        self._store.update_order(record)
        self._store.log_event(
            record.updated_ms,
            "WARNING",
            "order_rejected",
            {"id": record.client_order_id, "note": record.note},
        )
        log.warning(
            "order %s %s %s rejected: %s", record.symbol, record.side, record.quantity, note
        )
        return SubmitResult("rejected", record, note)

    def _finish_uncertain(self, record: OrderRecord, note: str) -> SubmitResult:
        record.status = OrderStatus.UNKNOWN
        record.note = note[:500]
        record.updated_ms = self._clock.now_ms()
        self._store.update_order(record)
        self._store.log_event(
            record.updated_ms,
            "WARNING",
            "order_unknown",
            {"id": record.client_order_id, "note": record.note},
        )
        self._notifier.notify(
            "WARNING",
            "Order outcome unknown",
            f"{record.symbol} {record.side} {record.quantity}: {note}. "
            "Reconciling; NOT re-sending.",
        )
        # A few immediate attempts, spaced out, before returning control to the trading loop.
        for i in range(min(self._max_attempts, 3)):
            if i:
                self._clock.sleep(self._interval)
            self._reconcile_one(record)
            if record.status not in UNCERTAIN_STATUSES:
                break
        outcome = "uncertain" if record.status in UNCERTAIN_STATUSES else "accepted"
        return SubmitResult(outcome, record, note)

    # ------------------------------------------------------------------ results / reconcile

    def _apply(self, record: OrderRecord, result: OrderResult) -> None:
        """Adopt the venue's view of the order and record any fills not seen before."""
        new_fills: list[Fill] = []
        known = len(self._store.fills_for(record.client_order_id))
        if result.fills:
            new_fills = result.fills[known:]
        elif result.executed_qty > record.executed_qty:
            dq = result.executed_qty - record.executed_qty
            dquote = result.cumulative_quote_qty - record.cumulative_quote_qty
            new_fills = [Fill(dquote / dq, dq, Decimal(0), "")]  # estimated from totals
        record.status = result.status
        record.exchange_order_id = result.exchange_order_id or record.exchange_order_id
        record.executed_qty = result.executed_qty
        record.cumulative_quote_qty = result.cumulative_quote_qty
        record.updated_ms = self._clock.now_ms()
        self._store.update_order(record)
        if new_fills:
            self._store.add_fills(record.client_order_id, new_fills, record.updated_ms)
            if self._on_fill is not None:
                self._on_fill(record, new_fills)

    def _reconcile_one(self, record: OrderRecord) -> None:
        try:
            result = self._broker.get_order(record.symbol, record.client_order_id)
        except ExchangeError as exc:
            log.warning("reconcile query for %s failed: %s", record.client_order_id, exc)
            return  # try again next cycle; do not count a failed query as 'not found'
        if result is not None:
            self._apply(record, result)
            return
        if record.status in UNCERTAIN_STATUSES or record.status in OPEN_STATUSES:
            record.reconcile_attempts += 1
            record.updated_ms = self._clock.now_ms()
            if record.reconcile_attempts >= self._max_attempts:
                record.status = OrderStatus.LOST
                record.note = (record.note + " | not found on venue after reconciliation")[:500]
                self._store.update_order(record)
                self._store.kv_set(f"lost:{record.symbol}", record.client_order_id)
                self._notifier.notify(
                    "CRITICAL",
                    "Order LOST",
                    f"{record.symbol} {record.side} {record.quantity} id={record.client_order_id} "
                    "could not be found on the venue. It will NOT be re-sent. Check the exchange, "
                    "then run 'botux resolve-lost'.",
                )
            else:
                self._store.update_order(record)

    def reconcile_pending(self) -> None:
        """One reconciliation pass over every uncertain or open order (call every cycle)."""
        for record in self._store.list_orders((*UNCERTAIN_STATUSES, *OPEN_STATUSES)):
            self._reconcile_one(record)

    # ------------------------------------------------------------------ cancel

    def cancel_all_open(self) -> int:
        """Cancel every open order we know about. Returns how many were cancelled."""
        cancelled = 0
        for record in self.open_orders():
            try:
                result = self._broker.cancel_order(record.symbol, record.client_order_id)
            except ExchangeError as exc:
                self._notifier.notify(
                    "CRITICAL", "Cancel failed", f"{record.symbol} {record.client_order_id}: {exc}"
                )
                continue
            if result is not None:
                self._apply(record, result)
                cancelled += 1
            else:
                self._reconcile_one(record)  # already filled/cancelled meanwhile
        return cancelled
