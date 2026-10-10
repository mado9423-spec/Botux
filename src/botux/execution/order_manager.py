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
from botux.errors import (
    ClockSkewError,
    ExchangeError,
    NetworkError,
    OrderRejected,
    OutcomeUnknownError,
    RateLimitError,
)
from botux.exchange.rules import SymbolRules
from botux.execution.broker import Broker
from botux.killswitch import KillSwitch
from botux.models import Fill, OrderRequest, OrderResult, OrderStatus, OrderType, Side
from botux.storage import OrderRecord, Store

log = logging.getLogger(__name__)

OPEN_STATUSES = (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)
UNCERTAIN_STATUSES = (OrderStatus.PENDING_SEND, OrderStatus.UNKNOWN)

FillCallback = Callable[[OrderRecord, list[Fill]], None]
# Asked right before every NEW (non-exit) order is written/sent; a returned string vetoes it.
EntryGuard = Callable[["OrderIntent"], str | None]


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
    flatten: bool = False  # the one kind of order still allowed once the kill switch is set


@dataclass(frozen=True, slots=True)
class SubmitResult:
    # accepted   - placed and not (yet) known to have failed (filled, or resting on the book)
    # not_filled - the venue answered but executed nothing (REJECTED/EXPIRED/CANCELED, 0 filled)
    # partial    - the venue answered, executed only part of it and ended the order
    # duplicate | blocked | rejected | retry_later | uncertain (outcome unknown: reconcile)
    outcome: str
    record: OrderRecord | None = None
    reason: str = ""


def outcome_of(record: OrderRecord) -> str:
    """How an order that the venue has answered about should be reported to the caller."""
    if record.status in UNCERTAIN_STATUSES or record.status is OrderStatus.LOST:
        return "uncertain"
    if record.status in (OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.CANCELED):
        if record.executed_qty <= 0:
            return "not_filled"
        if record.executed_qty < record.quantity:
            return "partial"
    return "accepted"


LOST_REQUERY_MS = 300_000  # how often an order we gave up on is looked up again


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
        entry_guard: EntryGuard | None = None,
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
        self._entry_guard = entry_guard

    def set_entry_guard(self, guard: EntryGuard) -> None:
        self._entry_guard = guard

    def update_rules(self, rules: Mapping[str, SymbolRules]) -> None:
        self._rules.update(rules)

    def set_fill_callback(self, callback: FillCallback) -> None:
        self._on_fill = callback

    # ------------------------------------------------------------------ queries

    def uncertain_orders(self, symbol: str | None = None) -> list[OrderRecord]:
        return self._store.list_orders(UNCERTAIN_STATUSES, symbol)

    def open_orders(self, symbol: str | None = None) -> list[OrderRecord]:
        return self._store.list_orders(OPEN_STATUSES, symbol)

    def lost_orders(self, symbol: str) -> list[OrderRecord]:
        """Orders given up on and not yet resolved by a human (or found again by a re-query)."""
        if not self._store.kv_get(f"lost:{symbol}"):
            return []
        return self._store.list_orders((OrderStatus.LOST,), symbol)

    def is_symbol_blocked(self, symbol: str, *, for_exit: bool = False) -> str | None:
        """Reason string when no new order may be sent for ``symbol``, else None.

        Entries are blocked while ANY order on the symbol is unresolved or was LOST. Exits are
        blocked only while a SELL is unresolved or LOST (to avoid selling twice, possibly coins
        that are not the bot's): an unresolved BUY must never stop a stop-loss, because sells
        are sized from the bot's own holdings.
        """
        uncertain = self.uncertain_orders(symbol)
        lost = self.lost_orders(symbol)
        if for_exit:
            if any(o.side is Side.SELL for o in uncertain):
                return f"unresolved SELL on {symbol}: reconciling first"
            if any(o.side is Side.SELL for o in lost):
                return (
                    f"a SELL on {symbol} was LOST (outcome unknown): the position may still be "
                    "open or already sold. Check the exchange, close it by hand if needed, then "
                    "run 'botux resolve-lost'"
                )
            return None
        if uncertain:
            return f"unresolved order(s) on {symbol}: reconciling first"
        if lost:
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
        if self._kill is not None and self._kill.is_set() and not intent.flatten:
            return SubmitResult("blocked", None, "kill switch is set")
        if self._entry_guard is not None and not intent.is_exit:
            veto = self._entry_guard(intent)  # e.g. the live gate, evaluated at the last moment
            if veto:
                return SubmitResult("blocked", None, veto)
        blocked = self.is_symbol_blocked(intent.symbol, for_exit=intent.is_exit)
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
        except (ClockSkewError, RateLimitError) as exc:
            # Refused before the matching engine (timestamp window / rate limit / local cool-down):
            # provably not executed, and worth retrying shortly.
            return self._finish_retry_later(record, f"{type(exc).__name__}: {exc}")
        except NetworkError as exc:
            if exc.not_sent:
                return self._finish_retry_later(record, f"not sent: {exc}")
            return self._finish_uncertain(record, str(exc))
        except OutcomeUnknownError as exc:
            return self._finish_uncertain(record, str(exc))
        except ExchangeError as exc:
            # Other 4XX / geo block / WAF block on an order => it was refused, not executed.
            return self._finish_rejected(record, f"{type(exc).__name__}: {exc}")

        self._apply(record, result)
        return SubmitResult(outcome_of(record), record, "")

    def _finish_retry_later(self, record: OrderRecord, note: str) -> SubmitResult:
        """Forget an order that provably never left, so the same intent can be tried again."""
        self._store.delete_unsent_order(record.client_order_id)
        self._store.log_event(
            self._clock.now_ms(),
            "WARNING",
            "order_not_sent",
            {"id": record.client_order_id, "note": note},
        )
        log.warning("order %s %s not sent (will retry): %s", record.symbol, record.side, note)
        return SubmitResult("retry_later", None, note)

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
        return SubmitResult(outcome_of(record), record, note)

    # ------------------------------------------------------------------ results / reconcile

    def _apply(self, record: OrderRecord, result: OrderResult) -> None:
        """Adopt the venue's view of the order and record any fills not seen before.

        The order row and its new fills are written in ONE transaction, so a crash can never
        leave a FILLED order without the fills that explain it.
        """
        new_fills: list[Fill] = []
        known = len(self._store.fills_for(record.client_order_id))
        fills = result.fills
        if not fills and result.executed_qty > record.executed_qty:
            fills = self._exact_fills(record, result) or []
        if fills:
            new_fills = fills[known:]
        elif result.executed_qty > record.executed_qty:
            dq = result.executed_qty - record.executed_qty
            dquote = result.cumulative_quote_qty - record.cumulative_quote_qty
            new_fills = [Fill(dquote / dq, dq, Decimal(0), "")]  # estimated from totals
        record.status = result.status
        record.exchange_order_id = result.exchange_order_id or record.exchange_order_id
        record.executed_qty = result.executed_qty
        record.cumulative_quote_qty = result.cumulative_quote_qty
        record.updated_ms = self._clock.now_ms()
        if result.expiry_reason:  # e.g. EXECUTION_RULE_PRICE_RANGE_EXCEEDED: keep the venue's why
            record.note = f"expired: {result.expiry_reason}"
        with self._store.transaction():
            self._store.update_order(record)
            if new_fills:
                self._store.add_fills(record.client_order_id, new_fills, record.updated_ms)
        if new_fills and self._on_fill is not None:
            self._on_fill(record, new_fills)

    def _exact_fills(self, record: OrderRecord, result: OrderResult) -> list[Fill] | None:
        """Real per-trade fills (with commissions) when the venue's order answer carries none.

        Only trusted when they add up to the executed quantity the order itself reports.
        """
        if not result.exchange_order_id:
            return None
        try:
            fills = self._broker.order_fills(record.symbol, result.exchange_order_id)
        except ExchangeError as exc:
            log.warning("could not fetch trades of %s: %s", record.client_order_id, exc)
            return None
        if fills and sum((f.quantity for f in fills), Decimal(0)) == result.executed_qty:
            return fills
        return None

    def _reconcile_one(self, record: OrderRecord) -> None:
        try:
            result = self._broker.get_order(record.symbol, record.client_order_id)
        except ExchangeError as exc:
            log.warning("reconcile query for %s failed: %s", record.client_order_id, exc)
            return  # try again next cycle; do not count a failed query as 'not found'
        if result is not None:
            self._apply(record, result)
            # The venue answered, but with a status we cannot interpret: that is not progress.
            if record.status in UNCERTAIN_STATUSES:
                self._count_failed_attempt(record, "venue reports an unrecognised status")
            return
        if record.status in UNCERTAIN_STATUSES or record.status in OPEN_STATUSES:
            self._count_failed_attempt(record, "not found on venue after reconciliation")

    def _requery_lost(self) -> None:
        """Look again, slowly, at orders we gave up on: one may have surfaced at the venue."""
        now = self._clock.now_ms()
        for record in self._store.list_orders((OrderStatus.LOST,)):
            if not self._store.kv_get(f"lost:{record.symbol}"):
                continue  # a human already decided about this symbol
            if now - record.updated_ms < LOST_REQUERY_MS:
                continue
            try:
                result = self._broker.get_order(record.symbol, record.client_order_id)
            except ExchangeError:
                continue
            record.updated_ms = now
            if result is None or result.status in UNCERTAIN_STATUSES:
                self._store.update_order(record)  # remember when we last looked
                continue
            self._apply(record, result)
            self._notifier.notify(
                "WARNING",
                "Lost order found",
                f"{record.symbol} {record.side} id={record.client_order_id} now reports "
                f"{record.status} (executed {record.executed_qty}). Its fills were recorded.",
            )
            if not self._store.list_orders((OrderStatus.LOST,), record.symbol):
                self._store.kv_set(f"lost:{record.symbol}", "")

    def _count_failed_attempt(self, record: OrderRecord, why: str) -> None:
        record.reconcile_attempts += 1
        record.updated_ms = self._clock.now_ms()
        if record.reconcile_attempts < self._max_attempts:
            self._store.update_order(record)
            return
        record.status = OrderStatus.LOST
        record.note = f"{record.note} | {why}"[:500]
        with self._store.transaction():  # the record and the block on the symbol, or neither
            self._store.update_order(record)
            self._store.kv_set(f"lost:{record.symbol}", record.client_order_id)
        self._notifier.notify(
            "CRITICAL",
            "Order LOST",
            f"{record.symbol} {record.side} {record.quantity} id={record.client_order_id}: {why}. "
            "It will NOT be re-sent"
            + (
                "; further SELLs on this symbol are blocked too, so the position may be unprotected"
                if record.side is Side.SELL
                else ""
            )
            + ". Check the exchange, then run 'botux resolve-lost'. The bot keeps looking "
            "for the order every few minutes.",
        )

    def reconcile_pending(self) -> None:
        """One reconciliation pass over every uncertain or open order (call every cycle)."""
        for record in self._store.list_orders((*UNCERTAIN_STATUSES, *OPEN_STATUSES)):
            self._reconcile_one(record)
        self._requery_lost()

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
