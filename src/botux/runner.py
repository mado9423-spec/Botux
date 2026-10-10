"""The trading loop. One ``run_cycle`` = reconcile, fetch data, check risk, act on signals.

The loop is broker-agnostic: with a :class:`PaperBroker` nothing leaves the machine except
public market-data GETs; with a :class:`LiveBroker` (only obtainable through the safety gate) the
same code path sends real orders.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Protocol

from botux.alerts import Notifier
from botux.clock import Clock
from botux.config import SUPPORTED_INTERVALS, Settings
from botux.data.market import fetch_latest_closed, fetch_quote, validate_live_candles
from botux.decimals import ZERO
from botux.errors import (
    BotuxError,
    ConfigError,
    DataError,
    ExchangeError,
    LiveTradingNotAllowed,
    StorageError,
)
from botux.exchange.client import BinanceSpotClient
from botux.exchange.rules import SymbolRules
from botux.execution.broker import Broker
from botux.execution.order_manager import OrderIntent, OrderManager, SubmitResult
from botux.execution.paper import PaperBroker
from botux.execution.positions import PositionBook
from botux.killswitch import KillSwitch
from botux.ledger import Ledger, LedgerView, balance_discrepancy, owned_base
from botux.models import Balance, Candle, Fill, OrderType, Quote, Side
from botux.risk.manager import PortfolioView, RiskManager
from botux.storage import OrderRecord, Store
from botux.strategy.base import Signal, Strategy

log = logging.getLogger(__name__)

DAY_MS = 86_400_000
ACK_KEY = "ack:balances"  # set by `botux ack-balances`, consumed by the loop
ACK_DAILY_KEY = "ack:daily_loss"  # set by `botux reset-daily-loss`, consumed by the loop
BLOCKERS_KEY = "entry_blockers"  # JSON, for `botux status`
QUOTE_TOLERANCE = Decimal("0.01")


class MarketFeed(Protocol):
    def closed_candles(self, symbol: str, count: int) -> list[Candle]: ...

    def quote(self, symbol: str) -> Quote: ...

    def now_ms(self) -> int: ...


class BinanceMarketFeed:
    """Public Binance market data (no keys). ``now_ms`` includes the measured clock offset."""

    def __init__(self, client: BinanceSpotClient, interval: str, clock: Clock) -> None:
        self._client = client
        self._interval = interval
        self._clock = clock

    def now_ms(self) -> int:
        return self._clock.now_ms() + self._client.clock_offset_ms

    def closed_candles(self, symbol: str, count: int) -> list[Candle]:
        return fetch_latest_closed(self._client, symbol, self._interval, count, self.now_ms())

    def quote(self, symbol: str) -> Quote:
        return fetch_quote(self._client, symbol, self.now_ms())


class TradingLoop:
    ALERT_REPEAT_SECONDS = 600.0
    MAX_RETRIES_PER_CANDLE = 3

    def __init__(
        self,
        *,
        settings: Settings,
        feed: MarketFeed,
        broker: Broker,
        order_manager: OrderManager,
        risk: RiskManager,
        positions: PositionBook,
        store: Store,
        kill: KillSwitch,
        notifier: Notifier,
        clock: Clock,
        strategy: Strategy,
        rules: Mapping[str, SymbolRules],
        paper_broker: PaperBroker | None = None,
        act_on_current_candle: bool = False,
        gate_check: Callable[[], None] | None = None,
        rules_loader: Callable[[], Mapping[str, SymbolRules]] | None = None,
    ) -> None:
        self._s = settings
        self._feed = feed
        self._broker = broker
        self._om = order_manager
        self._risk = risk
        self._positions = positions
        self._store = store
        self._kill = kill
        self._notifier = notifier
        self._clock = clock
        self._strategy = strategy
        self._rules = dict(rules)
        self._paper = paper_broker
        self._act_current = act_on_current_candle
        self._gate_check = gate_check
        self._rules_loader = rules_loader
        self._rules_loaded_at = clock.monotonic()
        self._rules_next_try = self._rules_loaded_at + settings.market.rules_refresh_hours * 3600
        self._stop_requested = False
        self._kill_handled = False
        self._gate_closed = False
        order_manager.set_entry_guard(self._entry_guard)
        self._halt_alerted_day: int | None = None
        self._alerted_at: dict[str, float] = {}
        self._retries: dict[tuple[str, int], int] = {}
        self._last_positions: dict[str, Decimal] = {}
        self._last_mid: dict[str, Decimal] = {}
        self._warmed: set[str] = set()
        self._data_failures: dict[str, int] = {}
        self._quote_failures: dict[str, int] = {}
        self._baseline: dict[str, Decimal] = {}
        self.stopped = False
        self.cycles = 0
        raw = store.kv_get("risk_state")
        if raw:
            risk.restore_state(json.loads(raw))
        if not store.integrity_ok():
            raise StorageError("the database failed its integrity check; refusing to start")
        capital = settings.bot_capital
        if capital is None:
            raise ConfigError(
                "risk.bot_capital must be set: the bot never treats the whole account as its own"
            )
        self._ledger = Ledger(
            store,
            rules,
            capital=capital,
            fee_rate=settings.execution.fee_rate,
        )
        if self._ledger.capital_changed_from is not None:
            store.log_event(
                clock.now_ms(),
                "WARNING",
                "capital_changed",
                {"from": str(self._ledger.capital_changed_from), "to": str(capital)},
            )
            risk.rebase_day_start()

    @property
    def order_manager(self) -> OrderManager:
        return self._om

    @property
    def risk(self) -> RiskManager:
        return self._risk

    @property
    def positions(self) -> PositionBook:
        return self._positions

    # ------------------------------------------------------------------ fills callback

    def on_fill(self, record: OrderRecord, fills: list[Fill]) -> None:
        """Wired into the OrderManager: keeps entry price / stop level in sync."""
        if record.side is Side.BUY:
            if record.executed_qty > 0:
                # volume-weighted entry across ALL fills of the order (not just the first one)
                price = record.cumulative_quote_qty / record.executed_qty
                self._positions.on_buy(record.symbol, ZERO, record.executed_qty, price)
            return
        # SELL: forget the entry only once (almost) nothing is left to protect
        try:
            held = self._position_qty(record.symbol, self._broker.balances())
        except Exception:
            log.exception("could not read balances after a sell; keeping the stop in place")
            return
        rules = self._rules[record.symbol]
        if rules.round_qty(held, OrderType.MARKET) < max(rules.min_qty, rules.step_size):
            self._positions.clear(record.symbol)

    # ------------------------------------------------------------------ control

    def request_stop(self) -> None:
        """Graceful stop (SIGINT/SIGTERM): finish the current cycle, cancel open orders, exit."""
        self._stop_requested = True

    def run(self, max_cycles: int | None = None) -> None:
        log.info("trading loop started (%s)", ", ".join(self._s.market.symbols))
        while not (self._stop_requested or self.stopped):
            started = self._clock.monotonic()
            self.run_cycle()
            if self.stopped or (max_cycles is not None and self.cycles >= max_cycles):
                break
            # a fixed cadence: the time the cycle itself took counts toward the pause
            elapsed = self._clock.monotonic() - started
            self._sleep_interruptibly(max(0.0, self._s.market.poll_seconds - elapsed))
        if not self.stopped:
            self._shutdown("stop requested" if self._stop_requested else "cycle limit reached")

    def _sleep_interruptibly(self, seconds: float) -> None:
        remaining = seconds
        while remaining > 0 and not self._stop_requested and not self._kill.is_set():
            step = min(1.0, remaining)
            self._clock.sleep(step)
            remaining -= step

    def _shutdown(self, why: str) -> None:
        try:
            cancelled = self._om.cancel_all_open()
        except Exception:
            log.exception("failed to cancel open orders during shutdown")
            cancelled = -1
        self._safe_event("INFO", "shutdown", {"why": why, "cancelled_open_orders": str(cancelled)})
        log.info("trading loop stopped: %s", why)
        self.stopped = True

    def _alert_throttled(self, key: str, level: str, title: str, message: str) -> None:
        """Alert at most once per ALERT_REPEAT_SECONDS for the same key (avoid spamming)."""
        now = self._clock.monotonic()
        last = self._alerted_at.get(key)
        if last is None or now - last >= self.ALERT_REPEAT_SECONDS:
            self._alerted_at[key] = now
            self._notifier.notify(level, title, message)

    def _open_positions_text(self) -> str:
        """Positions the bot holds, read afresh when possible (alerts must never omit one)."""
        try:
            balances = self._broker.balances()
            held = {s: self._position_qty(s, balances) for s in self._s.market.symbols}
        except Exception:
            held = dict(self._last_positions)
            if not held:
                return "unknown (balances could not be read)"
        held = {s: q for s, q in held.items() if q > 0}
        return ", ".join(f"{s}={q}" for s, q in held.items()) or "none"

    # ------------------------------------------------------------------ one cycle

    def run_cycle(self) -> None:
        if self._kill.is_set():
            self._on_kill()
            return
        self.cycles += 1
        try:
            self._check_gate()
            self._cycle()
            self._risk.record_success()
        except Exception as exc:
            log.error("cycle %d failed: %s: %s", self.cycles, type(exc).__name__, exc)
            try:
                self._store.log_event(
                    self._clock.now_ms(),
                    "ERROR",
                    "cycle_error",
                    {"type": type(exc).__name__, "message": str(exc)[:300]},
                )
            except Exception:  # a broken database must not turn one error into a dead loop
                log.exception("could not record the cycle error (is the database usable?)")
            if self._risk.record_error():
                reason = (
                    f"{self._risk.consecutive_errors} consecutive errors; "
                    f"last: {type(exc).__name__}"
                )
                self._kill.trip(reason)
                held = self._open_positions_text()
                suffix = ""
                if held != "none" and not self._s.risk.flatten_on_kill:
                    suffix = f" OPEN POSITIONS LEFT WITHOUT STOP PROTECTION: {held}."
                self._notifier.notify("CRITICAL", "Trading halted", reason + suffix)

    def _check_gate(self) -> None:
        """Re-verify the live gate (arming expiry, disarm, settings drift, key rights).

        A closed gate halts NEW entries for good (until restart); exits stay possible so that
        existing positions can still be protected. Called every cycle AND right before every new
        order (see :meth:`_entry_guard`).
        """
        if self._gate_check is None or self._gate_closed:
            return
        try:
            self._gate_check()
        except LiveTradingNotAllowed as exc:
            self._gate_closed = True
            # a blocker, NOT a persisted halt: after the operator re-arms and restarts, the gate
            # is evaluated afresh instead of an old verdict being restored from the database
            self._block("live-gate", f"live gate closed: {exc}")
            self._notifier.notify(
                "CRITICAL",
                "Live gate closed",
                f"{exc}. New entries are halted; exits still work. Run 'botux kill' to stop "
                "completely, or re-arm and restart.",
            )

    def _entry_guard(self, intent: OrderIntent) -> str | None:
        """Last-moment veto for any NEW order: gate and risk state are looked at again NOW."""
        self._check_gate()
        ok, why = self._risk.entries_allowed()
        return None if ok else why

    def _on_kill(self) -> None:
        """Emergency stop: cancel open orders, optionally flatten, then stop for good."""
        if self._kill_handled:
            return
        self._kill_handled = True
        try:
            try:
                reason = self._kill.reason()
            except Exception:
                reason = "kill switch set"
            self._notifier.notify("CRITICAL", "EMERGENCY STOP", reason)
            self._safe_event("CRITICAL", "kill_switch", {"reason": reason})
            try:
                self._om.reconcile_pending()  # learn what happened to anything in flight
            except Exception:
                log.exception("could not reconcile orders during emergency stop")
            try:
                self._om.cancel_all_open()
            except Exception:
                log.exception("could not cancel open orders during emergency stop")
            unresolved = self._om.uncertain_orders()
            if unresolved:
                ids = ", ".join(o.client_order_id for o in unresolved)
                self._notifier.notify(
                    "CRITICAL",
                    "Orders with unknown outcome at emergency stop",
                    f"{len(unresolved)} order(s) could not be resolved ({ids}) and the bot is "
                    "stopping: check the exchange by hand; a position may exist that nothing "
                    "manages.",
                )
            if self._s.risk.flatten_on_kill:
                for sym in self._s.market.symbols:
                    self._flatten_with_retries(sym)
            else:
                held = self._open_positions_text()
                if held != "none":
                    self._notifier.notify(
                        "CRITICAL",
                        "Open position left unprotected",
                        f"Emergency stop with open positions ({held}) and flatten_on_kill=false: "
                        "nothing manages the stop-loss any more. Close them manually if needed.",
                    )
        finally:
            self.stopped = True

    FLATTEN_ATTEMPTS = 3

    def _flatten_with_retries(self, symbol: str) -> None:
        """Try to sell the position at the emergency stop. If reading balances or the quote keeps
        failing, SAY so: the stop must never end with the operator believing the account is flat."""
        for attempt in range(1, self.FLATTEN_ATTEMPTS + 1):
            try:
                self._flatten(symbol, "flatten_on_kill")
                return
            except Exception as exc:
                log.exception("flatten of %s failed (attempt %d)", symbol, attempt)
                if attempt == self.FLATTEN_ATTEMPTS:
                    self._notifier.notify(
                        "CRITICAL",
                        "Flatten failed",
                        f"{symbol}: could not sell the position at the emergency stop "
                        f"({type(exc).__name__}). It may still be OPEN and nothing manages it: "
                        "close it by hand.",
                    )
                else:
                    self._clock.sleep(2.0)

    def _flatten(self, symbol: str, why: str) -> None:
        rules = self._rules[symbol]
        qty = rules.round_qty(self._position_qty(symbol, self._broker.balances()), OrderType.MARKET)
        if qty <= 0:
            return
        quote = self._feed.quote(symbol)
        result = self._om.submit(
            OrderIntent(
                intent_key=f"flatten:{symbol}:{self._clock.now_ms() // 60_000}:{why}",
                symbol=symbol,
                side=Side.SELL,
                type=OrderType.MARKET,
                quantity=qty,
                reference_price=quote.bid,
                is_exit=True,
                flatten=True,
            )
        )
        if result.outcome != "accepted":
            self._notifier.notify(
                "CRITICAL",
                "Flatten failed",
                f"{symbol}: {result.outcome} {result.reason}. Close the position manually.",
            )

    # ------------------------------------------------------------------ helpers

    def _now(self) -> int:
        return self._feed.now_ms()

    def _baseline_for(self, base: str, total: Decimal) -> Decimal:
        """Holding of ``base`` when the bot first looked at the account (free + locked)."""
        if base not in self._baseline:
            stored = self._store.kv_get(f"baseline:{base}")
            if stored is None:
                self._store.kv_set(f"baseline:{base}", str(total))
                self._baseline[base] = total
            else:
                self._baseline[base] = Decimal(stored)
        return self._baseline[base]

    def _owned(
        self, symbol: str, balances: Mapping[str, Balance], view: LedgerView | None = None
    ) -> tuple[Decimal, Decimal]:
        """(base amount that is the bot's, part of it that is free to sell).

        The bot's share is what ITS OWN fills explain (the ledger), further capped by what is
        really on the account above the baseline (coins the user already held, including coins
        locked in the user's own orders, are never the bot's to sell).
        """
        base = self._rules[symbol].base_asset
        bal = balances.get(base)
        free = bal.free if bal is not None else ZERO
        total = bal.total if bal is not None else ZERO
        led = view if view is not None else self._ledger.view()
        owned = owned_base(
            total, self._baseline_for(base, total), led.base_qty(symbol), self._ledger.offset(base)
        )
        return owned, min(free, owned)

    def _position_qty(self, symbol: str, balances: Mapping[str, Balance]) -> Decimal:
        """Sellable amount that belongs to the bot (only FREE coins can be sold)."""
        return self._owned(symbol, balances)[1]

    def _safe_event(self, level: str, kind: str, detail: dict[str, str]) -> None:
        """Record an event; a failing database must never cost us the rest of the cycle."""
        try:
            self._store.log_event(self._now(), level, kind, detail)
        except Exception:
            log.exception("could not record event %s", kind)

    def _block(self, key: str, reason: str, *, alert: str | None = None) -> None:
        """Block new entries (exits stay possible) while ``reason`` holds."""
        if self._risk.set_blocker(key, reason):
            log.warning("entries blocked [%s]: %s", key, reason)
            self._safe_event("CRITICAL", "entries_blocked", {key: reason})
        if alert is not None:
            self._alert_throttled(f"block:{key}", "CRITICAL", alert, reason)

    def _unblock(self, key: str) -> None:
        if self._risk.clear_blocker(key):
            log.warning("entries unblocked [%s]", key)
            self._safe_event("INFO", "entries_unblocked", {"key": key})

    def _check_ownership(self, balances: Mapping[str, Balance], led: LedgerView) -> Decimal:
        """Compare real balances with what the bot's own fills explain.

        Returns the bot's cash (ledger cash, reduced by any acknowledged quote deficit). Any
        unexplained difference (a deposit, a withdrawal, a manual trade, a fill we never
        recorded) blocks NEW entries until a human has looked and run ``botux ack-balances``.
        Nothing is ever sold or cancelled because of a mismatch.
        """
        ack = bool(self._store.kv_get(ACK_KEY))
        quote_asset = self._s.risk.quote_asset
        fee = self._s.execution.fee_rate
        settled = not self._om.uncertain_orders()  # unresolved orders explain mismatches
        acknowledged: list[str] = []

        for sym in self._s.market.symbols:
            rules = self._rules[sym]
            base = rules.base_asset
            bal = balances.get(base)
            total = bal.total if bal is not None else ZERO
            offset = self._ledger.offset(base)
            baseline = self._baseline_for(base, total)
            ledger_qty = led.base_qty(sym)
            disc = balance_discrepancy(total, baseline, ledger_qty, offset)
            if ack and settled and disc != 0:
                # A human has looked: whatever lies beyond the bot's own records is the user's
                # (baseline); if the account holds LESS than the records say, the bot owns less.
                new_baseline = max(ZERO, total - ledger_qty)
                new_offset = min(ZERO, total - ledger_qty)
                self._store.kv_set(f"baseline:{base}", str(new_baseline))
                self._baseline[base] = new_baseline
                self._ledger.set_offset(base, new_offset)
                acknowledged.append(f"{base}: {disc:+f}")
                disc = ZERO
            if not settled:
                continue
            if abs(disc) > self._ledger.tolerance(sym, led):
                unprotected = ledger_qty > owned_base(total, baseline, ledger_qty, offset)
                self._block(
                    f"balance:{base}",
                    f"{base} on the account differs from the bot's own records by {disc:+f} "
                    "(deposit, withdrawal, manual trade or an unrecorded fill?). Verify on the "
                    "exchange, then run 'botux ack-balances'. Nothing was sold or cancelled."
                    + (
                        f" The bot's recorded position ({ledger_qty} {base}) is NOT covered by "
                        "the stop-loss until this is resolved, because the bot cannot prove which "
                        "coins are its own."
                        if unprotected
                        else ""
                    ),
                    alert="Unexpected balance change",
                )
            else:
                self._unblock(f"balance:{base}")

        qbal = balances.get(quote_asset)
        total_q = qbal.total if qbal is not None else ZERO
        q_offset = self._ledger.offset(quote_asset)
        cash = led.cash + min(ZERO, q_offset)
        deficit = cash - total_q
        if ack and settled and deficit > 0:
            self._ledger.set_offset(quote_asset, q_offset - deficit)
            acknowledged.append(f"{quote_asset}: {-deficit:+f}")
            cash -= deficit
            deficit = ZERO
        if settled:
            inexact = any(sl.inexact_qty > 0 for sl in led.symbols.values())
            tol = QUOTE_TOLERANCE + (self._ledger.capital * fee * 2 if inexact else ZERO)
            if deficit > tol:
                self._block(
                    f"balance:{quote_asset}",
                    f"{quote_asset} on the account is {deficit:f} below the bot's own cash "
                    "records (withdrawal, manual trade or an unrecorded fill?). Verify on the "
                    "exchange, then run 'botux ack-balances' (the missing amount then counts as "
                    "a loss of bot capital).",
                    alert="Unexpected balance change",
                )
            else:
                self._unblock(f"balance:{quote_asset}")
        if ack and settled:
            self._store.kv_set(ACK_KEY, "")
            self._store.log_event(
                self._now(), "WARNING", "balances_acknowledged", {"offsets": acknowledged}
            )
            self._notifier.notify(
                "WARNING",
                "Balances acknowledged",
                "A human accepted the current balances: "
                + (", ".join(acknowledged) or "no change"),
            )
        return cash

    def _check_open_orders(self) -> None:
        """Entries stop while the exchange shows an open order that this bot did not place.

        Foreign orders may hold funds or fill at any time; the bot cannot tell whether they are
        the user's own or a leftover of a lost record, so it does not touch them (no cancel) and
        simply does not trade until they are gone.
        """
        for sym in self._s.market.symbols:
            key = f"orders:{sym}"
            try:
                live_orders = self._broker.open_orders(sym)
            except ExchangeError as exc:
                self._block(key, f"could not list open orders for {sym}: {type(exc).__name__}")
                continue
            foreign = [o for o in live_orders if self._store.get_order(o.client_order_id) is None]
            if foreign:
                ids = ", ".join(o.client_order_id[:12] for o in foreign[:5])
                self._block(
                    key,
                    f"{len(foreign)} open order(s) on {sym} were not placed by this bot "
                    f"({ids}). They are left untouched; trading on {sym} resumes when they are "
                    "gone and balances are acknowledged.",
                    alert="Foreign open order",
                )
            else:
                self._unblock(key)

    def _snapshot(self, quotes: Mapping[str, Quote]) -> tuple[Decimal, PortfolioView]:
        balances = self._broker.balances()
        quote_asset = self._s.risk.quote_asset
        qbal = balances.get(quote_asset)
        account_free = ZERO if qbal is None else qbal.free
        led = self._ledger.view()
        cash = self._check_ownership(balances, led)
        free_quote = max(ZERO, min(cash, account_free))
        equity = cash  # the BOT's equity: its own cash plus the coins its own fills explain
        notional: dict[str, Decimal] = {}
        qty: dict[str, Decimal] = {}
        for sym in self._s.market.symbols:
            owned_total, sellable = self._owned(sym, balances, led)
            qty[sym] = sellable
            notional[sym] = owned_total * self._mid(sym, quotes, needed=owned_total > 0)
            equity += notional[sym]
        self._last_positions = dict(qty)
        day_start = (self._now() // DAY_MS) * DAY_MS
        open_orders = len(self._om.open_orders()) + len(self._om.uncertain_orders())
        view = PortfolioView(
            equity=equity,
            free_quote=free_quote,
            positions_notional=notional,
            position_qty=qty,
            open_orders=open_orders,
            orders_today=self._store.count_orders_since(day_start),
        )
        self._persist_blockers()
        return equity, view

    def _mid(self, sym: str, quotes: Mapping[str, Quote], *, needed: bool) -> Decimal:
        """Mid price of ``sym``: this cycle's quote, else the last one seen (a symbol whose quote
        failed is still valued so the other symbols' sizing is not distorted)."""
        quote = quotes.get(sym)
        if quote is not None:
            self._last_mid[sym] = quote.mid
            return quote.mid
        known = self._last_mid.get(sym)
        if known is None:
            if needed:
                raise DataError(f"no price known for {sym}, which holds a position")
            return ZERO
        return known

    def _persist_blockers(self) -> None:
        raw = json.dumps(self._risk.blockers, sort_keys=True)
        try:
            if self._store.kv_get(BLOCKERS_KEY) != raw:
                self._store.kv_set(BLOCKERS_KEY, raw)
        except Exception:
            log.exception("could not persist the entry blockers")

    # ------------------------------------------------------------------ the cycle

    RULES_RETRY_SECONDS = 300.0

    def _maybe_refresh_rules(self) -> None:
        """Re-read exchangeInfo now and then: filters and symbol status can change under us."""
        if self._rules_loader is None:
            return
        now = self._clock.monotonic()
        if now < self._rules_next_try:
            return
        refresh = self._s.market.rules_refresh_hours * 3600
        try:
            new = dict(self._rules_loader())
        except BotuxError as exc:
            self._rules_next_try = now + self.RULES_RETRY_SECONDS
            log.warning("could not refresh exchange rules: %s: %s", type(exc).__name__, exc)
            if now - self._rules_loaded_at > 2 * refresh:
                self._block(
                    "rules-stale",
                    f"exchange rules could not be refreshed for "
                    f"{(now - self._rules_loaded_at) / 3600:.1f} h; not entering on stale filters",
                    alert="Exchange rules are stale",
                )
            return
        self._rules_next_try = now + refresh
        self._rules_loaded_at = now
        self._unblock("rules-stale")
        changed = [sym for sym, r in new.items() if self._rules.get(sym) != r]
        if changed:
            self._rules.update(new)
            self._om.update_rules(new)
            self._broker.update_rules(new)
            self._ledger.update_rules(new)
            self._store.log_event(self._now(), "WARNING", "rules_changed", {"symbols": changed})
            self._notifier.notify(
                "WARNING",
                "Exchange rules changed",
                f"Filters of {', '.join(changed)} differ from the ones in use; the new ones apply "
                "from now on (positions that no longer satisfy them may need manual handling).",
            )
        for sym, r in self._rules.items():
            if r.status != "TRADING" or not r.spot_trading_allowed:
                self._block(
                    f"symbol-status:{sym}",
                    f"{sym} is not open for spot trading (status {r.status})",
                    alert="Symbol not tradable",
                )
            else:
                self._unblock(f"symbol-status:{sym}")

    def _health_probe(self, now: int) -> None:
        """The write-ahead log must be writable before any order is considered."""
        try:
            self._store.kv_set("health:ts", str(now))
        except Exception as exc:
            raise StorageError(f"database is not writable ({type(exc).__name__})") from None

    def _cycle(self) -> None:
        now = self._now()
        self._health_probe(now)
        self._maybe_refresh_rules()
        errors: list[Exception] = []
        try:
            self._om.reconcile_pending()
        except Exception as exc:  # protection must not depend on reconciliation succeeding
            log.error("reconciliation failed: %s: %s", type(exc).__name__, exc)
            errors.append(exc)

        # One symbol's broken quote must not take the others (or their stop-losses) down.
        quotes: dict[str, Quote] = {}
        failed: dict[str, Exception] = {}
        for sym in self._s.market.symbols:
            try:
                quotes[sym] = self._feed.quote(sym)
            except Exception as exc:
                log.error("%s quote failed: %s: %s", sym, type(exc).__name__, exc)
                failed[sym] = exc
                continue
            if self._paper is not None:
                self._paper.set_quote(quotes[sym])
        if not quotes:
            raise next(iter(failed.values()), DataError("no quotes at all"))

        self._check_open_orders()
        equity, view = self._snapshot(quotes)
        self._risk.update_equity(now, equity)
        self._apply_daily_loss_reset(equity)
        day = now // DAY_MS
        if self._risk.daily_loss_halted and self._halt_alerted_day != day:
            self._halt_alerted_day = day
            self._notifier.notify(
                "CRITICAL",
                "Daily loss limit reached",
                f"PnL today {self._risk.daily_pnl(equity):.2f} {self._s.risk.quote_asset}; "
                "no new entries until a human decides: 'botux reset-daily-loss' (this halt does "
                "NOT end at midnight).",
            )

        # 1) Protection first. It needs only quotes and balances, never candle data, so a broken
        #    kline feed cannot disable the stop-loss. One failing symbol must not starve others.
        acted = False
        for sym in self._s.market.symbols:
            if self._kill.is_set():
                return  # the next run_cycle performs the emergency-stop policy
            if sym not in quotes:
                continue
            try:
                acted |= self._protect(sym, quotes[sym], view)
            except Exception as exc:
                log.error("%s protection step failed: %s: %s", sym, type(exc).__name__, exc)
                errors.append(exc)  # protection failures always count: a position may be exposed
        if acted:
            equity, view = self._snapshot(quotes)

        # Bookkeeping comes AFTER protection: a full disk must not be able to skip a stop-loss.
        try:
            self._store.kv_set("risk_state", json.dumps(self._risk.export_state()))
            self._store.add_equity(now, equity, view.free_quote)
        except Exception as exc:
            log.error("could not persist the cycle state: %s: %s", type(exc).__name__, exc)
            errors.append(exc)

        # 2) Signals, which do need validated candles. A candle problem (a gap, stale or missing
        #    data) means "no signal for this symbol now": protection above does not need candles,
        #    so it must not count toward the emergency stop, only be reported.
        for sym in self._s.market.symbols:
            if self._kill.is_set():
                return
            if sym not in quotes:
                continue
            try:
                candles = self._feed.closed_candles(sym, self._strategy.warmup + 2)
                validate_live_candles(
                    candles, self._s.market.interval, now, self._s.market.max_candle_age_factor
                )
                if self._signals(sym, candles, quotes[sym], view):
                    equity, view = self._snapshot(quotes)  # sizing for the next symbol sees it
                self._data_failures[sym] = 0
            except DataError as exc:
                self._data_failures[sym] = self._data_failures.get(sym, 0) + 1
                log.error("%s no signal this cycle: %s", sym, exc)
                self._safe_event("WARNING", "no_signal", {"symbol": sym, "why": str(exc)[:200]})
                if self._data_failures[sym] >= 3:
                    self._alert_throttled(
                        f"data:{sym}",
                        "WARNING",
                        "No usable market data",
                        f"{sym}: no signal for {self._data_failures[sym]} cycles in a row ({exc}). "
                        "Open positions keep their stop-loss; no new entries or strategy exits "
                        "until the data is healthy again.",
                    )
            except Exception as exc:
                log.error("%s signal step failed: %s: %s", sym, type(exc).__name__, exc)
                errors.append(exc)

        # A failed quote counts as an error only when it can matter: the symbol is held (its
        # stop-loss cannot run) or nothing works at all. A bad quote for a symbol we are flat in
        # is reported but must not stop the bot, nor the stops of the others.
        for sym, qerr in failed.items():
            self._quote_failures[sym] = self._quote_failures.get(sym, 0) + 1
            if self._last_positions.get(sym, ZERO) > 0:
                errors.append(qerr)
            elif self._quote_failures[sym] >= 3:
                self._alert_throttled(
                    f"quote:{sym}",
                    "WARNING",
                    "No usable quote",
                    f"{sym}: quote failing for {self._quote_failures[sym]} cycles "
                    f"({type(qerr).__name__}); no trading on it meanwhile.",
                )
        for sym in quotes:
            self._quote_failures[sym] = 0
        if errors:
            raise errors[0]

    def _max_signal_age_ms(self) -> int:
        """An entry is only taken soon after its candle closed: half an interval at most (10
        minutes cap), but never less than three polls so a normal cycle always qualifies."""
        return max(min(self._s.interval_ms // 2, 600_000), int(self._s.market.poll_seconds * 3000))

    def _apply_daily_loss_reset(self, equity: Decimal) -> None:
        """Consume a `botux reset-daily-loss` request (a human decision, never automatic)."""
        if not self._store.kv_get(ACK_DAILY_KEY):
            return
        self._store.kv_set(ACK_DAILY_KEY, "")
        lifted = self._risk.reset_daily_loss(equity)
        self._halt_alerted_day = None
        self._store.log_event(
            self._now(),
            "WARNING",
            "daily_loss_reset",
            {"lifted": lifted, "new_reference_equity": str(equity)},
        )
        self._notifier.notify(
            "WARNING",
            "Daily loss limit reset by a human",
            f"{'Halt lifted' if lifted else 'No halt was active'}; the new daily reference is "
            f"{equity:.2f} {self._s.risk.quote_asset}.",
        )

    def _in_position(self, sym: str, view: PortfolioView) -> tuple[Decimal, bool]:
        rules = self._rules[sym]
        held = rules.round_qty(view.position_qty.get(sym, ZERO), OrderType.MARKET)
        return held, held > 0 and held >= rules.min_qty

    def _protect(self, sym: str, quote: Quote, view: PortfolioView) -> bool:
        """Software stop-loss: only effective while the bot runs and is connected."""
        held, in_position = self._in_position(sym, view)
        if not in_position:
            return False
        info = self._positions.get(sym)
        if info is None:
            # e.g. after a crash between the fill and its bookkeeping. The bot's own fills still
            # say what the position cost, so the stop is re-armed from that, not from today's
            # price (which could be far below the real entry).
            cost = self._ledger.view().average_cost.get(sym)
            basis = cost if cost is not None and cost > 0 else quote.mid
            self._positions.on_buy(sym, ZERO, held, basis)
            info = self._positions.get(sym)
            self._notifier.notify(
                "WARNING",
                "Stop-loss re-armed",
                f"{sym}: position {held} had no recorded entry; stop re-armed from "
                f"{'its average cost' if basis != quote.mid else 'the current price'} {basis:f}.",
            )
        if info is not None and quote.bid <= info.stop_price:
            log.warning("%s stop-loss: bid %s <= stop %s", sym, quote.bid, info.stop_price)
            result = self._submit_exit(
                sym, held, quote, f"stop:{sym}:{self._now() // 60_000}", view, "stop-loss"
            )
            return result is not None
        return False

    def _signals(self, sym: str, candles: list[Candle], quote: Quote, view: PortfolioView) -> bool:
        """Act on a newly closed candle. Returns True when an order was attempted."""
        rules = self._rules[sym]
        held, in_position = self._in_position(sym, view)
        last = candles[-1]
        key = f"lastc:{sym}"
        seen = self._store.kv_get(key)
        if sym not in self._warmed and not self._act_current:
            # Every process start begins with a warm-up: the newest closed candle is noted and
            # not traded, however long ago the previous run stopped, so a restart can never act
            # on a stale signal.
            self._warmed.add(sym)
            if seen in (None, "") or last.open_time_ms > int(seen):
                self._store.kv_set(key, str(last.open_time_ms))
                log.info("%s: warm start, waiting for the next candle before trading", sym)
            return False
        self._warmed.add(sym)
        previous = int(seen) if seen not in (None, "") else None
        if previous is not None and last.open_time_ms <= previous:
            return False  # same candle again, or a stale answer from the API: never go backwards
        self._store.kv_set(key, str(last.open_time_ms))  # at most one decision per candle

        signal = self._strategy.on_candles(candles, in_position)
        log.info("%s signal=%s in_position=%s equity=%.2f", sym, signal, in_position, view.equity)
        result: SubmitResult | None = None
        if signal is Signal.ENTER_LONG and not in_position:
            age_ms = self._now() - last.close_time_ms
            if age_ms > self._max_signal_age_ms():
                log.warning("%s entry skipped: the signal is %d s old", sym, age_ms // 1000)
                return False
            blocked = self._om.is_symbol_blocked(sym)
            if blocked:
                log.warning("%s entry skipped: %s", sym, blocked)
                return False
            result = self._submit_entry(sym, rules, quote, view, last)
        elif signal is Signal.EXIT_LONG and in_position:
            result = self._submit_exit(
                sym,
                held,
                quote,
                f"{self._strategy.name}:{sym}:{last.open_time_ms}:SELL",
                view,
                "strategy exit",
            )
        if result is not None and result.outcome == "retry_later":
            n = self._retries.get((sym, last.open_time_ms), 0) + 1
            self._retries[(sym, last.open_time_ms)] = n
            if n < self.MAX_RETRIES_PER_CANDLE:  # provably not sent: look at this candle again
                self._store.kv_set(key, "" if previous is None else str(previous))
        return result is not None

    def _submit_entry(
        self, sym: str, rules: SymbolRules, quote: Quote, view: PortfolioView, last: Candle
    ) -> SubmitResult | None:
        spread = (quote.ask - quote.bid) / quote.mid
        if spread > self._s.risk.max_spread_pct:
            log.warning(
                "%s entry skipped: spread %.4f%% is wider than the limit", sym, spread * 100
            )
            self._alert_throttled(
                f"spread:{sym}",
                "WARNING",
                "Entry skipped: wide spread",
                f"{sym}: spread {spread:.3%} > max_spread_pct {self._s.risk.max_spread_pct:.3%}; "
                "not entering on a thin or broken order book.",
            )
            return None
        sizing = self._risk.size_entry(rules, quote.ask, view, OrderType.MARKET)
        if sizing.quantity <= 0:
            log.info("%s entry skipped by risk sizing: %s", sym, "; ".join(sizing.reasons))
            return None
        decision = self._risk.check_order(
            rules, Side.BUY, OrderType.MARKET, sizing.quantity, None, quote.ask, view
        )
        if not decision.allowed:
            log.info("%s entry rejected by risk: %s", sym, "; ".join(decision.reasons))
            return None
        result = self._om.submit(
            OrderIntent(
                intent_key=f"{self._strategy.name}:{sym}:{last.open_time_ms}:BUY",
                symbol=sym,
                side=Side.BUY,
                type=OrderType.MARKET,
                quantity=sizing.quantity,
                reference_price=quote.ask,
            )
        )
        log.info("%s BUY %s -> %s %s", sym, sizing.quantity, result.outcome, result.reason)
        return result

    def _submit_exit(
        self,
        sym: str,
        held: Decimal,
        quote: Quote,
        intent_key: str,
        view: PortfolioView,
        why: str,
    ) -> SubmitResult | None:
        rules = self._rules[sym]
        decision = self._risk.check_order(
            rules, Side.SELL, OrderType.MARKET, held, None, quote.bid, view
        )
        if not decision.allowed:
            self._alert_throttled(
                f"exit-risk:{sym}",
                "CRITICAL",
                "Exit refused by local checks",
                f"{sym}: cannot {why} {held}: {'; '.join(decision.reasons)}. The position is "
                "still open.",
            )
            return None
        result = self._om.submit(
            OrderIntent(
                intent_key=intent_key,
                symbol=sym,
                side=Side.SELL,
                type=OrderType.MARKET,
                quantity=held,
                reference_price=quote.bid,
                is_exit=True,
            )
        )
        log.info("%s SELL %s (%s) -> %s %s", sym, held, why, result.outcome, result.reason)
        if result.outcome not in ("accepted", "duplicate") and not self._kill.is_set():
            self._alert_throttled(
                f"exit-fail:{sym}",
                "CRITICAL",
                "Exit order did not go through",
                f"{sym}: {why} of {held} -> {result.outcome}: {result.reason}. The position may "
                "still be open.",
            )
        return result


def interval_ms_of(settings: Settings) -> int:
    return SUPPORTED_INTERVALS[settings.market.interval]
