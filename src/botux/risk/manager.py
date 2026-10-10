"""Position sizing and risk limits.

Principles:

* Sizing is driven by the amount we are willing to lose if the stop-loss is hit, then reduced
  by hard caps (per order, per symbol, total exposure, available cash). It is never increased
  to reach an exchange minimum: if the result is below MIN_NOTIONAL / minQty we do not trade.
* Quantities are rounded DOWN to the lot step.
* Exits (SELL of an existing position) are always allowed by the limits; entries are not.
* A stop-loss is a software rule evaluated when the bot looks at the market. It is NOT a
  guarantee: gaps, slippage, outages and a stopped bot can all produce larger losses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from botux.clock import Clock, SystemClock
from botux.config import RiskSettings
from botux.decimals import ZERO, D
from botux.exchange.rules import SymbolRules
from botux.models import OrderType, Side

DAY_MS = 86_400_000


@dataclass(frozen=True, slots=True)
class PortfolioView:
    """What the risk manager needs to know about the account right now (quote-asset units)."""

    equity: Decimal
    free_quote: Decimal
    positions_notional: Mapping[str, Decimal]
    position_qty: Mapping[str, Decimal]
    open_orders: int = 0
    orders_today: int = 0


@dataclass(frozen=True, slots=True)
class SizingResult:
    quantity: Decimal
    binding: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allowed: bool
    reasons: tuple[str, ...] = ()


class RiskManager:
    def __init__(
        self,
        settings: RiskSettings,
        *,
        fee_rate: Decimal,
        slippage_bps: Decimal,
        clock: Clock | None = None,
    ) -> None:
        self.s = settings
        self._fee = fee_rate
        self._slip = slippage_bps / Decimal(10_000)
        self._clock = clock or SystemClock()
        self._day: int | None = None
        self._day_start_equity: Decimal | None = None
        self._daily_loss_halted = False
        self._consecutive_errors = 0
        self._hard_halt: str | None = None
        # Conditions recomputed from facts every cycle (foreign orders, balance mismatches, failed
        # health checks). Unlike ``_hard_halt`` they are never persisted: they clear themselves
        # when the fact goes away, or after a human acknowledges it.
        self._blockers: dict[str, str] = {}
        self._rebase_pending = False

    # ------------------------------------------------------------------ state

    @property
    def daily_loss_halted(self) -> bool:
        return self._daily_loss_halted

    @property
    def hard_halt_reason(self) -> str | None:
        return self._hard_halt

    @property
    def day_start_equity(self) -> Decimal | None:
        return self._day_start_equity

    def halt(self, reason: str) -> None:
        """Stop all new entries until a human restarts the bot (sticky)."""
        self._hard_halt = reason

    @property
    def blockers(self) -> dict[str, str]:
        return dict(self._blockers)

    def set_blocker(self, key: str, reason: str) -> bool:
        """Block new entries while ``reason`` holds. Returns True when it is newly raised."""
        new = self._blockers.get(key) != reason
        self._blockers[key] = reason
        return new

    def clear_blocker(self, key: str) -> bool:
        """Lift one blocker. Returns True when it was set."""
        return self._blockers.pop(key, None) is not None

    def rebase_day_start(self) -> None:
        """Use the next equity reading as today's reference (e.g. after the bot capital changed).

        The daily-loss halt, if tripped, stays tripped: only a human reset lifts it.
        """
        self._rebase_pending = True

    def update_equity(self, now_ms: int, equity: Decimal) -> None:
        """Track the UTC-day reference and trip the daily-loss halt when the limit is reached.

        "Daily loss" = the BOT's equity at the day's first reading minus the current equity, in
        the quote asset: realized and unrealized, after fees and slippage already paid. Once
        tripped the halt does NOT end at midnight; only :meth:`reset_daily_loss` (a human
        decision) lifts it.
        """
        day = now_ms // DAY_MS
        if self._rebase_pending and self._day_start_equity is not None:
            self._rebase_pending = False
            self._day_start_equity = equity
        if self._day != day:
            self._day = day
            self._day_start_equity = equity
        assert self._day_start_equity is not None
        if equity - self._day_start_equity <= -self.s.max_daily_loss:
            self._daily_loss_halted = True

    def reset_daily_loss(self, equity: Decimal) -> bool:
        """Human decision: lift the daily-loss halt and start a fresh reference at ``equity``.

        Returns True when a halt was actually lifted.
        """
        was = self._daily_loss_halted
        self._daily_loss_halted = False
        self._day_start_equity = equity
        return was

    def daily_pnl(self, equity: Decimal) -> Decimal:
        return ZERO if self._day_start_equity is None else equity - self._day_start_equity

    def record_error(self) -> bool:
        """Count a consecutive failure. Returns True when the configured limit is reached."""
        self._consecutive_errors += 1
        return self._consecutive_errors >= self.s.max_consecutive_errors

    def record_success(self) -> None:
        self._consecutive_errors = 0

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors

    def export_state(self) -> dict[str, Any]:
        return {
            "day": self._day,
            "day_start_equity": None
            if self._day_start_equity is None
            else str(self._day_start_equity),
            "daily_loss_halted": self._daily_loss_halted,
            "hard_halt": self._hard_halt,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self._day = state.get("day")
        dse = state.get("day_start_equity")
        self._day_start_equity = None if dse is None else D(dse)
        self._daily_loss_halted = bool(state.get("daily_loss_halted", False))
        self._hard_halt = state.get("hard_halt")

    # ------------------------------------------------------------------ queries

    def entries_allowed(self) -> tuple[bool, str]:
        if self._hard_halt:
            return False, f"trading halted: {self._hard_halt}"
        if self._daily_loss_halted:
            return (
                False,
                "daily loss limit reached; no new entries until a human runs "
                "'botux reset-daily-loss'",
            )
        if self._blockers:
            return False, "entries blocked: " + "; ".join(
                f"{k}: {v}" for k, v in sorted(self._blockers.items())
            )
        return True, ""

    def stop_price(self, entry_price: Decimal) -> Decimal:
        return entry_price * (Decimal(1) - self.s.stop_loss_pct)

    def _unit_cost_factor(self) -> Decimal:
        """Cost multiplier per unit bought: price + slippage + fee."""
        return Decimal(1) + self._slip + self._fee

    # ------------------------------------------------------------------ sizing

    def size_entry(
        self,
        rules: SymbolRules,
        price: Decimal,
        view: PortfolioView,
        order_type: OrderType = OrderType.MARKET,
    ) -> SizingResult:
        """Quantity to BUY, rounded down; ``quantity == 0`` means "do not trade" (see reasons)."""
        if price <= 0:
            return SizingResult(ZERO, "invalid_price", ("price must be positive",))
        ok, why = self.entries_allowed()
        if not ok:
            return SizingResult(ZERO, "halted", (why,))
        if view.equity <= 0:
            return SizingResult(ZERO, "no_equity", ("equity must be positive",))

        sym = rules.symbol
        pos_notional = view.positions_notional.get(sym, ZERO)
        total_exposure = sum(view.positions_notional.values(), ZERO)
        risk_budget = view.equity * self.s.risk_per_trade_pct
        # Loss if the stop is hit: the stop distance plus slippage and fees on BOTH legs.
        loss_per_unit = price * (self.s.stop_loss_pct + 2 * self._slip + 2 * self._fee)
        # The hard caps are measured at the WORST expected fill (ask plus modelled slippage), so
        # the executed notional of a market order cannot exceed them.
        worst = price * (1 + self._slip) if order_type is OrderType.MARKET else price
        candidates: dict[str, Decimal] = {
            "risk_per_trade": risk_budget / loss_per_unit,
            "max_order_notional": self.s.max_order_notional / worst,
            "max_position_notional": (self.s.max_position_notional - pos_notional) / worst,
            "max_total_exposure": (self.s.max_total_exposure - total_exposure) / worst,
            "available_cash": view.free_quote / (price * self._unit_cost_factor()),
        }
        binding, raw_qty = min(candidates.items(), key=lambda kv: kv[1])
        if raw_qty <= 0:
            return SizingResult(ZERO, binding, (f"no headroom under limit '{binding}'",))
        qty = rules.round_qty(raw_qty, order_type)
        problems = (
            rules.violations(
                Side.BUY,
                order_type,
                qty,
                price if order_type is OrderType.LIMIT else None,
                price,
                self.s.min_notional_buffer,
            )
            if qty > 0
            else ["quantity rounds down to zero"]
        )
        if problems:
            return SizingResult(
                ZERO,
                binding,
                ("sized quantity is below exchange minimums and will NOT be rounded up", *problems),
            )
        # The position must be closable: after the buy fee, what is left to sell, valued at the
        # stop level, still has to satisfy the exchange minimums (otherwise a stop-loss could
        # never be executed). Dust positions are refused rather than opened.
        sellable = rules.round_qty(qty * (1 - self._fee), OrderType.MARKET)
        exit_problems = rules.violations(
            Side.SELL, OrderType.MARKET, sellable, None, price * (1 - self.s.stop_loss_pct)
        )
        if sellable <= 0 or exit_problems:
            return SizingResult(
                ZERO,
                binding,
                (
                    "the position could not be sold again at the stop level (exchange minimums)",
                    *exit_problems,
                ),
            )
        return SizingResult(qty, binding)

    # ------------------------------------------------------------------ order gate

    def check_order(
        self,
        rules: SymbolRules,
        side: Side,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None,
        reference_price: Decimal,
        view: PortfolioView,
    ) -> RiskDecision:
        """Final gate. Every violated rule is reported, not just the first."""
        reasons: list[str] = []
        sym = rules.symbol
        eff_price = (
            price if (order_type is OrderType.LIMIT and price is not None) else reference_price
        )

        if side is Side.SELL:
            held = view.position_qty.get(sym, ZERO)
            if quantity > held:
                reasons.append(
                    f"cannot sell {quantity} {sym}: only {held} held (no shorting on Spot)"
                )
        else:
            ok, why = self.entries_allowed()
            if not ok:
                reasons.append(why)
            if view.orders_today >= self.s.max_orders_per_day:
                reasons.append(f"max_orders_per_day ({self.s.max_orders_per_day}) reached")
            notional = (
                quantity * eff_price * (1 + self._slip if order_type is OrderType.MARKET else 1)
            )
            if notional > self.s.max_order_notional:
                reasons.append(
                    f"order notional {notional:.2f} > max_order_notional "
                    f"{self.s.max_order_notional}"
                )
            pos_after = view.positions_notional.get(sym, ZERO) + notional
            if pos_after > self.s.max_position_notional:
                reasons.append(
                    f"position notional after fill {pos_after:.2f} > max_position_notional "
                    f"{self.s.max_position_notional}"
                )
            exposure_after = sum(view.positions_notional.values(), ZERO) + notional
            if exposure_after > self.s.max_total_exposure:
                reasons.append(
                    f"total exposure after fill {exposure_after:.2f} > max_total_exposure "
                    f"{self.s.max_total_exposure}"
                )
            if notional * self._unit_cost_factor() > view.free_quote:
                reasons.append(
                    "insufficient free quote balance (including fee and slippage reserve)"
                )

        if view.open_orders >= self.s.max_open_orders and side is Side.BUY:
            reasons.append(f"max_open_orders ({self.s.max_open_orders}) reached")

        if order_type is OrderType.LIMIT and price is not None and reference_price > 0:
            deviation = abs(price - reference_price) / reference_price
            if deviation > self.s.max_price_deviation_pct:
                reasons.append(
                    f"limit price deviates {deviation:.2%} from reference "
                    f"(max {self.s.max_price_deviation_pct:.2%})"
                )

        buffer = self.s.min_notional_buffer if side is Side.BUY else Decimal(1)
        reasons += rules.violations(side, order_type, quantity, price, reference_price, buffer)
        return RiskDecision(not reasons, tuple(reasons))
