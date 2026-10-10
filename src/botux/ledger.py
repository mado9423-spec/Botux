"""What the bot owns, derived from the bot's OWN recorded fills (never from account totals).

Why this exists: an exchange account holds balances the bot did not create (a user's earlier
coins, deposits, funds locked in the user's own orders). Treating the whole quote balance as bot
capital would size trades from, and risk, money that is not the bot's. The ledger therefore
tracks

* ``capital``  - the amount the operator allotted to the bot (``risk.bot_capital``),
* the net quote-asset flow of the bot's fills (spent on BUYs, received on SELLs, quote fees),
* the base-asset quantity the bot acquired and still owns (after fees paid in the base asset).

``cash = capital + net quote flow`` and ``equity = cash + owned base * price``. Everything outside
the ledger (baseline holdings, deposits, withdrawals, foreign orders) is invisible to sizing and to
the daily-loss limit; differences between the ledger and the real balances are *detected* and
stop new entries until a human acknowledges them (see :func:`balance_discrepancy`).

Honest limits: fees paid in a third asset (e.g. BNB) are not valued, so reported equity ignores
them; fills whose commission the venue did not report (a reconciled order has only totals) use
the configured fee rate as an estimate and widen the tolerance accordingly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal

from botux.decimals import ZERO
from botux.exchange.rules import SymbolRules
from botux.models import Fill, Side
from botux.storage import Store

EPOCH_KEY = "ledger:epoch"
CAPITAL_KEY = "ledger:capital"
OFFSET_PREFIX = "ledger:offset:"


@dataclass(frozen=True, slots=True)
class SymbolLedger:
    symbol: str
    base_qty: Decimal = ZERO  # base asset the bot acquired and still owns, per its own fills
    quote_flow: Decimal = ZERO  # net quote asset: negative = spent, positive = received
    inexact_qty: Decimal = ZERO  # filled quantity whose commission had to be estimated
    third_asset_fees: bool = False  # some commission was paid in an asset we do not value


@dataclass(frozen=True, slots=True)
class LedgerView:
    capital: Decimal
    symbols: Mapping[str, SymbolLedger]

    @property
    def cash(self) -> Decimal:
        return self.capital + sum((s.quote_flow for s in self.symbols.values()), ZERO)

    def base_qty(self, symbol: str) -> Decimal:
        s = self.symbols.get(symbol)
        return ZERO if s is None else s.base_qty


@dataclass(slots=True)
class _Acc:
    base: Decimal = ZERO
    flow: Decimal = ZERO
    inexact: Decimal = ZERO
    third: bool = False


def build_ledger(
    capital: Decimal,
    rows: Iterable[tuple[str, Side, Fill]],
    rules: Mapping[str, SymbolRules],
    fee_rate: Decimal,
) -> LedgerView:
    """Fold fills into per-symbol positions and the bot's cash."""
    acc = {sym: _Acc() for sym in rules}
    for symbol, side, fill in rows:
        r = rules.get(symbol)
        if r is None:
            continue
        a = acc[symbol]
        notional = fill.price * fill.quantity
        asset = fill.commission_asset
        base_in = fill.quantity if side is Side.BUY else ZERO
        base_out = fill.quantity if side is Side.SELL else ZERO
        quote_in = notional if side is Side.SELL else ZERO
        quote_out = notional if side is Side.BUY else ZERO
        if not asset:  # the venue gave totals only: assume the configured fee on the received side
            a.inexact += fill.quantity
            if side is Side.BUY:
                base_in -= fill.quantity * fee_rate
            else:
                quote_in -= notional * fee_rate
        elif asset == r.base_asset:
            if side is Side.BUY:
                base_in -= fill.commission
            else:
                base_out += fill.commission
        elif asset == r.quote_asset:
            if side is Side.BUY:
                quote_out += fill.commission
            else:
                quote_in -= fill.commission
        else:
            a.third = True
        a.base += base_in - base_out
        a.flow += quote_in - quote_out
    return LedgerView(
        capital,
        {
            sym: SymbolLedger(
                sym,
                base_qty=a.base,
                quote_flow=a.flow,
                inexact_qty=a.inexact,
                third_asset_fees=a.third,
            )
            for sym, a in acc.items()
        },
    )


def tolerance(rules: SymbolRules, ledger: SymbolLedger, fee_rate: Decimal) -> Decimal:
    """How far real balances may drift from the ledger before it counts as a discrepancy:
    rounding (two lot steps) plus twice the fee estimated on fills without a reported fee."""
    return rules.step_size * 2 + ledger.inexact_qty * fee_rate * 2


def balance_discrepancy(
    total_base: Decimal, baseline: Decimal, ledger_qty: Decimal, offset: Decimal
) -> Decimal:
    """Real base held above/below what baseline + ledger + acknowledged offset explain."""
    return (total_base - baseline) - ledger_qty - offset


def owned_base(
    total_base: Decimal, baseline: Decimal, ledger_qty: Decimal, offset: Decimal
) -> Decimal:
    """Base asset the bot may regard as its own, conservatively.

    Never more than the ledger says, never more than what is really there above the user's
    baseline. An acknowledged *deficit* shrinks it (we cannot tell whose coins went missing); an
    acknowledged *surplus* is external and does not enlarge it.
    """
    return max(ZERO, min(ledger_qty + min(ZERO, offset), total_base - baseline))


class Ledger:
    """Store-backed facade used by the trading loop."""

    def __init__(
        self,
        store: Store,
        rules: Mapping[str, SymbolRules],
        *,
        capital: Decimal,
        fee_rate: Decimal,
    ) -> None:
        self._store = store
        self._rules = dict(rules)
        self._fee = fee_rate
        self.capital = capital
        # The epoch (id of the last fill that existed when the ledger began) separates this
        # ledger's fills from older sessions: a paper session starts fresh, while a live database
        # keeps its epoch for ever.
        raw = store.kv_get(EPOCH_KEY)
        if not raw:
            raw = str(store.last_fill_id())
            store.kv_set(EPOCH_KEY, raw)
        self._epoch = int(raw)
        self.capital_changed_from: Decimal | None = None
        stored = store.kv_get(CAPITAL_KEY)
        if stored and Decimal(stored) != capital:
            self.capital_changed_from = Decimal(stored)
        store.kv_set(CAPITAL_KEY, str(capital))

    def view(self) -> LedgerView:
        rows = self._store.fill_rows_after(self._epoch)
        return build_ledger(self.capital, rows, self._rules, self._fee)

    def tolerance(self, symbol: str, view: LedgerView) -> Decimal:
        return tolerance(self._rules[symbol], view.symbols[symbol], self._fee)

    def offset(self, base_asset: str) -> Decimal:
        raw = self._store.kv_get(f"{OFFSET_PREFIX}{base_asset}")
        return Decimal(raw) if raw else ZERO

    def set_offset(self, base_asset: str, value: Decimal) -> None:
        self._store.kv_set(f"{OFFSET_PREFIX}{base_asset}", str(value))
