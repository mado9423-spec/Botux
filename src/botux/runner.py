"""The trading loop. One ``run_cycle`` = reconcile, fetch data, check risk, act on signals.

The loop is broker-agnostic: with a :class:`PaperBroker` nothing leaves the machine except
public market-data GETs; with a :class:`LiveBroker` (only obtainable through the safety gate) the
same code path sends real orders.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from decimal import Decimal
from typing import Protocol

from botux.alerts import Notifier
from botux.clock import Clock
from botux.config import SUPPORTED_INTERVALS, Settings
from botux.data.market import fetch_latest_closed, fetch_quote, validate_live_candles
from botux.decimals import ZERO
from botux.exchange.client import BinanceSpotClient
from botux.exchange.rules import SymbolRules
from botux.execution.broker import Broker
from botux.execution.order_manager import OrderIntent, OrderManager
from botux.execution.paper import PaperBroker
from botux.execution.positions import PositionBook
from botux.killswitch import KillSwitch
from botux.models import Balance, Candle, Fill, OrderType, Quote, Side
from botux.risk.manager import PortfolioView, RiskManager
from botux.storage import OrderRecord, Store
from botux.strategy.base import Signal, Strategy

log = logging.getLogger(__name__)

DAY_MS = 86_400_000


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
        self._stop_requested = False
        self._kill_handled = False
        self._halt_alerted_day: int | None = None
        self.stopped = False
        self.cycles = 0
        raw = store.kv_get("risk_state")
        if raw:
            risk.restore_state(json.loads(raw))
        self._baseline: dict[str, Decimal] = {}

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
            first = fills[0]
            self._positions.on_buy(
                record.symbol, ZERO, sum((f.quantity for f in fills), ZERO), first.price
            )
        else:
            self._positions.clear(record.symbol)

    # ------------------------------------------------------------------ control

    def request_stop(self) -> None:
        """Graceful stop (SIGINT/SIGTERM): finish the current cycle, cancel open orders, exit."""
        self._stop_requested = True

    def run(self, max_cycles: int | None = None) -> None:
        log.info("trading loop started (%s)", ", ".join(self._s.market.symbols))
        while not (self._stop_requested or self.stopped):
            self.run_cycle()
            if self.stopped or (max_cycles is not None and self.cycles >= max_cycles):
                break
            self._sleep_interruptibly(self._s.market.poll_seconds)
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
        self._store.log_event(
            self._clock.now_ms(),
            "INFO",
            "shutdown",
            {"why": why, "cancelled_open_orders": cancelled},
        )
        log.info("trading loop stopped: %s", why)
        self.stopped = True

    # ------------------------------------------------------------------ one cycle

    def run_cycle(self) -> None:
        if self._kill.is_set():
            self._on_kill()
            return
        self.cycles += 1
        try:
            self._cycle()
            self._risk.record_success()
        except Exception as exc:
            log.error("cycle %d failed: %s: %s", self.cycles, type(exc).__name__, exc)
            self._store.log_event(
                self._clock.now_ms(),
                "ERROR",
                "cycle_error",
                {"type": type(exc).__name__, "message": str(exc)[:300]},
            )
            if self._risk.record_error():
                reason = (
                    f"{self._risk.consecutive_errors} consecutive errors; "
                    f"last: {type(exc).__name__}"
                )
                self._kill.trip(reason)
                self._notifier.notify("CRITICAL", "Trading halted", reason)

    def _on_kill(self) -> None:
        """Emergency stop: cancel open orders, optionally flatten, then stop for good."""
        if self._kill_handled:
            return
        self._kill_handled = True
        reason = self._kill.reason()
        self._notifier.notify("CRITICAL", "EMERGENCY STOP", reason)
        self._store.log_event(self._clock.now_ms(), "CRITICAL", "kill_switch", {"reason": reason})
        try:
            self._om.cancel_all_open()
        except Exception:
            log.exception("could not cancel open orders during emergency stop")
        if self._s.risk.flatten_on_kill:
            for sym in self._s.market.symbols:
                try:
                    self._flatten(sym, "flatten_on_kill")
                except Exception:
                    log.exception("flatten failed for %s", sym)
        self.stopped = True

    def _flatten(self, symbol: str, why: str) -> None:
        rules = self._rules[symbol]
        qty = rules.round_qty(self._position_qty(symbol, self._broker.balances()), OrderType.MARKET)
        if qty <= 0:
            return
        quote = self._feed.quote(symbol)
        self._om.submit(
            OrderIntent(
                intent_key=f"flatten:{symbol}:{self._clock.now_ms() // 60_000}:{why}",
                symbol=symbol,
                side=Side.SELL,
                type=OrderType.MARKET,
                quantity=qty,
                reference_price=quote.bid,
                is_exit=True,
            )
        )

    # ------------------------------------------------------------------ helpers

    def _now(self) -> int:
        return self._feed.now_ms()

    def _position_qty(self, symbol: str, balances: Mapping[str, Balance]) -> Decimal:
        """Free base balance above the baseline recorded at first start (never touches pre-existing
        holdings of the same asset)."""
        base = self._rules[symbol].base_asset
        bal = balances.get(base)
        total = bal.free if bal is not None else ZERO
        if base not in self._baseline:
            stored = self._store.kv_get(f"baseline:{base}")
            if stored is None:
                self._store.kv_set(f"baseline:{base}", str(total))
                self._baseline[base] = total
            else:
                self._baseline[base] = Decimal(stored)
        return max(ZERO, total - self._baseline[base])

    def _snapshot(self, quotes: Mapping[str, Quote]) -> tuple[Decimal, PortfolioView]:
        balances = self._broker.balances()
        quote_asset = self._s.risk.quote_asset
        qbal = balances.get(quote_asset)
        free_quote = ZERO if qbal is None else qbal.free
        equity = ZERO if qbal is None else qbal.total
        notional: dict[str, Decimal] = {}
        qty: dict[str, Decimal] = {}
        for sym in self._s.market.symbols:
            base = self._rules[sym].base_asset
            qty[sym] = self._position_qty(sym, balances)  # also records the baseline once
            bal = balances.get(base)
            owned_total = max(ZERO, (bal.total if bal else ZERO) - self._baseline[base])
            notional[sym] = owned_total * quotes[sym].mid
            equity += notional[sym]
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
        return equity, view

    # ------------------------------------------------------------------ the cycle

    def _cycle(self) -> None:
        now = self._now()
        self._om.reconcile_pending()

        candles: dict[str, list[Candle]] = {}
        quotes: dict[str, Quote] = {}
        for sym in self._s.market.symbols:
            cs = self._feed.closed_candles(sym, self._strategy.warmup + 2)
            validate_live_candles(
                cs, self._s.market.interval, now, self._s.market.max_candle_age_factor
            )
            candles[sym] = cs
            quotes[sym] = self._feed.quote(sym)
            if self._paper is not None:
                self._paper.set_quote(quotes[sym])

        equity, view = self._snapshot(quotes)
        self._risk.update_equity(now, equity)
        self._store.kv_set("risk_state", json.dumps(self._risk.export_state()))
        self._store.add_equity(now, equity, view.free_quote)
        day = now // DAY_MS
        if self._risk.daily_loss_halted and self._halt_alerted_day != day:
            self._halt_alerted_day = day
            self._notifier.notify(
                "CRITICAL",
                "Daily loss limit reached",
                f"PnL today {self._risk.daily_pnl(equity):.2f} {self._s.risk.quote_asset}; "
                "no new entries until the next UTC day.",
            )

        for sym in self._s.market.symbols:
            self._manage_symbol(sym, candles[sym], quotes[sym], view)

    def _manage_symbol(
        self, sym: str, candles: list[Candle], quote: Quote, view: PortfolioView
    ) -> None:
        reason = self._om.is_symbol_blocked(sym)
        if reason:
            log.warning("%s skipped: %s", sym, reason)
            return
        rules = self._rules[sym]
        held = rules.round_qty(view.position_qty.get(sym, ZERO), OrderType.MARKET)
        in_position = held > 0 and held >= rules.min_qty
        now = self._now()

        # Software stop-loss: only effective while the bot is running and connected.
        info = self._positions.get(sym)
        if in_position and info is None:
            log.warning(
                "%s: holding a position without a recorded entry; stop-loss is INACTIVE", sym
            )
        if in_position and info is not None and quote.bid <= info.stop_price:
            log.warning("%s stop-loss: bid %s <= stop %s", sym, quote.bid, info.stop_price)
            self._submit_exit(sym, held, quote, f"stop:{sym}:{now // 60_000}", view)
            return

        last = candles[-1]
        key = f"lastc:{sym}"
        seen = self._store.kv_get(key)
        if seen is None and not self._act_current:
            self._store.kv_set(key, str(last.open_time_ms))
            log.info("%s: warm start, waiting for the next candle before trading", sym)
            return
        if seen == str(last.open_time_ms):
            return
        self._store.kv_set(key, str(last.open_time_ms))  # at most one decision per candle

        signal = self._strategy.on_candles(candles, in_position)
        log.info("%s signal=%s in_position=%s equity=%.2f", sym, signal, in_position, view.equity)
        if signal is Signal.ENTER_LONG and not in_position:
            self._submit_entry(sym, rules, quote, view, last)
        elif signal is Signal.EXIT_LONG and in_position:
            self._submit_exit(
                sym, held, quote, f"{self._strategy.name}:{sym}:{last.open_time_ms}:SELL", view
            )

    def _submit_entry(
        self, sym: str, rules: SymbolRules, quote: Quote, view: PortfolioView, last: Candle
    ) -> None:
        sizing = self._risk.size_entry(rules, quote.ask, view, OrderType.MARKET)
        if sizing.quantity <= 0:
            log.info("%s entry skipped by risk sizing: %s", sym, "; ".join(sizing.reasons))
            return
        decision = self._risk.check_order(
            rules, Side.BUY, OrderType.MARKET, sizing.quantity, None, quote.ask, view
        )
        if not decision.allowed:
            log.info("%s entry rejected by risk: %s", sym, "; ".join(decision.reasons))
            return
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

    def _submit_exit(
        self, sym: str, held: Decimal, quote: Quote, intent_key: str, view: PortfolioView
    ) -> None:
        rules = self._rules[sym]
        decision = self._risk.check_order(
            rules, Side.SELL, OrderType.MARKET, held, None, quote.bid, view
        )
        if not decision.allowed:
            log.warning("%s exit rejected by risk: %s", sym, "; ".join(decision.reasons))
            return
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
        log.info("%s SELL %s -> %s %s", sym, held, result.outcome, result.reason)


def interval_ms_of(settings: Settings) -> int:
    return SUPPORTED_INTERVALS[settings.market.interval]
