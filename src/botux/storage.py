"""SQLite persistence: orders (write-ahead), fills, events, key/value state, equity curve.

Orders are inserted with status PENDING_SEND *before* anything is sent. After a crash the
bot therefore knows about every order that might exist on the exchange and reconciles it
instead of blindly creating a second one. All SQL uses bound parameters.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from botux.decimals import ZERO, D
from botux.errors import ConfigError
from botux.models import Fill, OrderStatus, OrderType, Side
from botux.secrets_guard import GLOBAL_REDACTOR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    client_order_id      TEXT PRIMARY KEY,
    intent_key           TEXT NOT NULL UNIQUE,
    symbol               TEXT NOT NULL,
    side                 TEXT NOT NULL,
    type                 TEXT NOT NULL,
    quantity             TEXT NOT NULL,
    price                TEXT,
    status               TEXT NOT NULL,
    exchange_order_id    TEXT,
    executed_qty         TEXT NOT NULL DEFAULT '0',
    cumulative_quote_qty TEXT NOT NULL DEFAULT '0',
    created_ms           INTEGER NOT NULL,
    updated_ms           INTEGER NOT NULL,
    reconcile_attempts   INTEGER NOT NULL DEFAULT 0,
    note                 TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS fills (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    client_order_id  TEXT NOT NULL REFERENCES orders(client_order_id),
    price            TEXT NOT NULL,
    quantity         TEXT NOT NULL,
    commission       TEXT NOT NULL,
    commission_asset TEXT NOT NULL,
    ts_ms            INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms  INTEGER NOT NULL,
    level  TEXT NOT NULL,
    kind   TEXT NOT NULL,
    detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS equity (
    ts_ms INTEGER NOT NULL, equity TEXT NOT NULL, cash TEXT NOT NULL
);
"""


@dataclass(slots=True)
class OrderRecord:
    client_order_id: str
    intent_key: str
    symbol: str
    side: Side
    type: OrderType
    quantity: Decimal
    price: Decimal | None
    status: OrderStatus
    created_ms: int
    updated_ms: int
    exchange_order_id: str | None = None
    executed_qty: Decimal = ZERO
    cumulative_quote_qty: Decimal = ZERO
    reconcile_attempts: int = 0
    note: str = ""


class InstanceLock:
    """Advisory single-instance lock (``flock`` on ``<database>.lock``).

    Two bots on the same database would race on deterministic client order ids. In-memory
    databases and platforms without ``fcntl`` get a no-op lock.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = None if str(db_path) == ":memory:" else Path(f"{db_path}.lock")
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._path is None:
            return
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX platforms
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise ConfigError(
                f"another Botux instance is already using {self._path.with_suffix('')}"
            ) from None
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)  # closing the descriptor drops the flock
            self._fd = None


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._depth = 0
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _write(self) -> Iterator[None]:
        """One commit per write, unless an enclosing :meth:`transaction` groups several."""
        if self._depth:
            yield
            return
        with self._conn:
            yield

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Make several writes atomic: all are committed together or none is (single thread)."""
        if self._depth:  # already inside one: join it
            yield
            return
        self._depth += 1
        try:
            with self._conn:
                yield
        finally:
            self._depth -= 1

    def integrity_ok(self) -> bool:
        """SQLite's own quick consistency check ("ok" when the file is sound)."""
        try:
            row = self._conn.execute("PRAGMA quick_check").fetchone()
        except sqlite3.Error:
            return False
        return row is not None and str(row[0]) == "ok"

    def claim_mode(self, mode: str) -> None:
        """Bind this database to one trading mode; refuse to be opened by the other one.

        Paper sessions reset position/halt state, which would wipe the safety state of a live
        session if both used the same file.
        """
        existing = self.kv_get("db_mode")
        if existing and existing != mode:
            raise ConfigError(
                f"this database belongs to '{existing}' mode and must not be opened in '{mode}' "
                "mode; give each mode its own database file"
            )
        self.kv_set("db_mode", mode)

    # ------------------------------------------------------------------ orders

    def insert_order(self, rec: OrderRecord) -> bool:
        """Insert; returns False (and changes nothing) if the intent key already exists."""
        try:
            with self._write():
                self._conn.execute(
                    "INSERT INTO orders (client_order_id, intent_key, symbol, side, type, quantity,"
                    " price, status, exchange_order_id, executed_qty, cumulative_quote_qty,"
                    " created_ms, updated_ms, reconcile_attempts, note)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        rec.client_order_id,
                        rec.intent_key,
                        rec.symbol,
                        rec.side.value,
                        rec.type.value,
                        str(rec.quantity),
                        None if rec.price is None else str(rec.price),
                        rec.status.value,
                        rec.exchange_order_id,
                        str(rec.executed_qty),
                        str(rec.cumulative_quote_qty),
                        rec.created_ms,
                        rec.updated_ms,
                        rec.reconcile_attempts,
                        GLOBAL_REDACTOR.scrub(rec.note),
                    ),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def delete_unsent_order(self, client_order_id: str) -> bool:
        """Delete a PENDING_SEND row that provably never reached the venue (only those)."""
        with self._write():
            cur = self._conn.execute(
                "DELETE FROM orders WHERE client_order_id=? AND status=?",
                (client_order_id, OrderStatus.PENDING_SEND.value),
            )
        return cur.rowcount == 1

    def update_order(self, rec: OrderRecord) -> None:
        with self._write():
            self._conn.execute(
                "UPDATE orders SET status=?, exchange_order_id=?, executed_qty=?,"
                " cumulative_quote_qty=?, updated_ms=?, reconcile_attempts=?, note=?"
                " WHERE client_order_id=?",
                (
                    rec.status.value,
                    rec.exchange_order_id,
                    str(rec.executed_qty),
                    str(rec.cumulative_quote_qty),
                    rec.updated_ms,
                    rec.reconcile_attempts,
                    GLOBAL_REDACTOR.scrub(rec.note),
                    rec.client_order_id,
                ),
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> OrderRecord:
        return OrderRecord(
            client_order_id=row["client_order_id"],
            intent_key=row["intent_key"],
            symbol=row["symbol"],
            side=Side(row["side"]),
            type=OrderType(row["type"]),
            quantity=D(row["quantity"]),
            price=None if row["price"] is None else D(row["price"]),
            status=OrderStatus(row["status"]),
            created_ms=row["created_ms"],
            updated_ms=row["updated_ms"],
            exchange_order_id=row["exchange_order_id"],
            executed_qty=D(row["executed_qty"]),
            cumulative_quote_qty=D(row["cumulative_quote_qty"]),
            reconcile_attempts=row["reconcile_attempts"],
            note=row["note"],
        )

    def get_order(self, client_order_id: str) -> OrderRecord | None:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE client_order_id=?", (client_order_id,)
        ).fetchone()
        return None if row is None else self._row_to_record(row)

    def get_by_intent(self, intent_key: str) -> OrderRecord | None:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE intent_key=?", (intent_key,)
        ).fetchone()
        return None if row is None else self._row_to_record(row)

    def list_orders(
        self, statuses: Iterable[OrderStatus] | None = None, symbol: str | None = None
    ) -> list[OrderRecord]:
        clauses: list[str] = []
        args: list[Any] = []
        if statuses is not None:
            sts = [s.value for s in statuses]
            clauses.append("status IN (" + ",".join("?" * len(sts)) + ")")
            args += sts
        if symbol is not None:
            clauses.append("symbol=?")
            args.append(symbol)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        # `where` is assembled only from fixed fragments and "?" placeholders; values are bound.
        rows = self._conn.execute(
            "SELECT * FROM orders" + where + " ORDER BY created_ms, client_order_id",  # noqa: S608
            args,
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count_orders_since(self, since_ms: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM orders WHERE created_ms>=? AND status<>?",
            (since_ms, OrderStatus.REJECTED.value),  # a LOST order may well have executed
        ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------ fills

    def add_fills(self, client_order_id: str, fills: Iterable[Fill], ts_ms: int) -> None:
        with self._write():
            self._conn.executemany(
                "INSERT INTO fills (client_order_id, price, quantity, commission, commission_asset,"
                " ts_ms) VALUES (?,?,?,?,?,?)",
                [
                    (
                        client_order_id,
                        str(f.price),
                        str(f.quantity),
                        str(f.commission),
                        f.commission_asset,
                        ts_ms,
                    )
                    for f in fills
                ],
            )

    def last_fill_id(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(id), 0) AS n FROM fills").fetchone()
        return int(row["n"])

    def fill_rows_after(self, fill_id: int) -> list[tuple[str, Side, Fill]]:
        """Every fill recorded after row ``fill_id`` (exclusive), oldest first."""
        rows = self._conn.execute(
            "SELECT o.symbol AS symbol, o.side AS side, f.price, f.quantity, f.commission,"
            " f.commission_asset FROM fills f JOIN orders o ON o.client_order_id=f.client_order_id"
            " WHERE f.id>? ORDER BY f.id",
            (fill_id,),
        ).fetchall()
        return [
            (
                str(r["symbol"]),
                Side(r["side"]),
                Fill(D(r["price"]), D(r["quantity"]), D(r["commission"]), r["commission_asset"]),
            )
            for r in rows
        ]

    def fills_for(self, client_order_id: str) -> list[Fill]:
        rows = self._conn.execute(
            "SELECT * FROM fills WHERE client_order_id=? ORDER BY id", (client_order_id,)
        ).fetchall()
        return [
            Fill(D(r["price"]), D(r["quantity"]), D(r["commission"]), r["commission_asset"])
            for r in rows
        ]

    # ------------------------------------------------------------------ events / kv / equity

    def log_event(self, ts_ms: int, level: str, kind: str, detail: dict[str, Any] | str) -> None:
        raw = detail if isinstance(detail, str) else json.dumps(detail, default=str, sort_keys=True)
        text = GLOBAL_REDACTOR.scrub(raw)
        with self._write():
            self._conn.execute(
                "INSERT INTO events (ts_ms, level, kind, detail) VALUES (?,?,?,?)",
                (ts_ms, level, kind, text),
            )

    def events(self, kind: str | None = None) -> list[sqlite3.Row]:
        if kind is None:
            return self._conn.execute("SELECT * FROM events ORDER BY id").fetchall()
        return self._conn.execute(
            "SELECT * FROM events WHERE kind=? ORDER BY id", (kind,)
        ).fetchall()

    def kv_get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def kv_set(self, key: str, value: str) -> None:
        with self._write():
            self._conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def kv_clear_prefix(self, prefix: str) -> None:
        with self._write():
            self._conn.execute(
                "DELETE FROM kv WHERE key LIKE ? ESCAPE '\\'",
                (prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%",),
            )

    def expire_stale_orders(self, now_ms: int, note: str) -> int:
        """Mark every non-terminal order EXPIRED (used when a paper session starts fresh)."""
        terminal = [s.value for s in OrderStatus if s.is_terminal]
        with self._write():
            placeholders = ",".join("?" * len(terminal))  # only "?" characters, values are bound
            cur = self._conn.execute(
                f"UPDATE orders SET status=?, note=?, updated_ms=? "  # noqa: S608
                f"WHERE status NOT IN ({placeholders})",
                (OrderStatus.EXPIRED.value, note, now_ms, *terminal),
            )
        return cur.rowcount

    def add_equity(self, ts_ms: int, equity: Decimal, cash: Decimal) -> None:
        with self._write():
            self._conn.execute(
                "INSERT INTO equity (ts_ms, equity, cash) VALUES (?,?,?)",
                (ts_ms, str(equity), str(cash)),
            )

    def equity_curve(self) -> list[tuple[int, Decimal]]:
        rows = self._conn.execute("SELECT ts_ms, equity FROM equity ORDER BY ts_ms").fetchall()
        return [(int(r["ts_ms"]), D(r["equity"])) for r in rows]
