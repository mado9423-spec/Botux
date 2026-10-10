"""Entry price / stop level bookkeeping for the positions this bot opened.

Quantities always come from the broker's balances (the source of truth, and they already
reflect fees paid in the base asset). This book only remembers the entry price and the
software stop-loss level, persisted so that a restart does not forget them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal

from botux.decimals import D
from botux.storage import Store


@dataclass(frozen=True, slots=True)
class PositionInfo:
    entry_price: Decimal
    stop_price: Decimal


class PositionBook:
    def __init__(self, store: Store, stop_loss_pct: Decimal) -> None:
        self._store = store
        self._stop_pct = stop_loss_pct
        self._cache: dict[str, PositionInfo] = {}

    def get(self, symbol: str) -> PositionInfo | None:
        if symbol in self._cache:
            return self._cache[symbol]
        raw = self._store.kv_get(f"pos:{symbol}")
        if not raw:
            return None
        data = json.loads(raw)
        info = PositionInfo(D(data["entry"]), D(data["stop"]))
        self._cache[symbol] = info
        return info

    def _save(self, symbol: str, info: PositionInfo | None) -> None:
        if info is None:
            self._cache.pop(symbol, None)
            self._store.kv_set(f"pos:{symbol}", "")
            return
        self._cache[symbol] = info
        self._store.kv_set(
            f"pos:{symbol}",
            json.dumps({"entry": str(info.entry_price), "stop": str(info.stop_price)}),
        )

    def on_buy(
        self, symbol: str, held_before: Decimal, fill_qty: Decimal, fill_price: Decimal
    ) -> None:
        """Blend the new fill into the average entry; the stop follows the average entry."""
        prev = self.get(symbol)
        if prev is None or held_before <= 0:
            entry = fill_price
        else:
            entry = (prev.entry_price * held_before + fill_price * fill_qty) / (
                held_before + fill_qty
            )
        self._save(symbol, PositionInfo(entry, entry * (Decimal(1) - self._stop_pct)))

    def clear(self, symbol: str) -> None:
        self._save(symbol, None)
