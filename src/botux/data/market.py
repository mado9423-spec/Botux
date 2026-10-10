"""Market-data retrieval and validation (public endpoints only, no keys involved)."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from botux.config import SUPPORTED_INTERVALS
from botux.decimals import D
from botux.errors import DataError
from botux.exchange.client import BinanceSpotClient
from botux.models import Candle, Quote

CSV_HEADER = ["open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms"]


def parse_klines(raw: Sequence[Sequence[Any]]) -> list[Candle]:
    """Convert Binance kline arrays to candles. Malformed rows raise DataError."""
    out: list[Candle] = []
    for row in raw:
        try:
            out.append(
                Candle(
                    open_time_ms=int(row[0]),
                    open=D(str(row[1])),
                    high=D(str(row[2])),
                    low=D(str(row[3])),
                    close=D(str(row[4])),
                    volume=D(str(row[5])),
                    close_time_ms=int(row[6]),
                )
            )
        except (IndexError, ValueError, TypeError) as exc:
            raise DataError(f"malformed kline row: {row!r}") from exc
    return out


@dataclass(frozen=True, slots=True)
class DataReport:
    count: int
    duplicates: int
    out_of_order: int
    gaps: int
    bad_ohlc: int

    @property
    def ok(self) -> bool:
        return not (self.duplicates or self.out_of_order or self.gaps or self.bad_ohlc)

    @property
    def fatal(self) -> bool:
        """Problems that make the series unusable even for a back-test."""
        return bool(self.duplicates or self.out_of_order or self.bad_ohlc)

    def describe(self) -> str:
        return (
            f"{self.count} candles, duplicates={self.duplicates}, "
            f"out_of_order={self.out_of_order}, gaps={self.gaps}, bad_ohlc={self.bad_ohlc}"
        )


def check_candles(candles: Sequence[Candle], interval_ms: int) -> DataReport:
    dup = ooo = gaps = bad = 0
    prev: Candle | None = None
    for c in candles:
        if not (
            c.low > 0
            and c.low <= min(c.open, c.close)
            and c.high >= max(c.open, c.close)
            and c.volume >= 0
            and c.close_time_ms >= c.open_time_ms
        ):
            bad += 1
        if prev is not None:
            if c.open_time_ms == prev.open_time_ms:
                dup += 1
            elif c.open_time_ms < prev.open_time_ms:
                ooo += 1
            elif c.open_time_ms - prev.open_time_ms != interval_ms:
                gaps += 1
        prev = c
    return DataReport(len(candles), dup, ooo, gaps, bad)


def validate_live_candles(
    candles: Sequence[Candle], interval: str, now_ms: int, max_age_factor: float
) -> None:
    """Strict checks for candles that will drive a trading decision. Raises DataError."""
    interval_ms = SUPPORTED_INTERVALS[interval]
    if not candles:
        raise DataError("no candles received")
    report = check_candles(candles, interval_ms)
    if not report.ok:
        raise DataError(f"inconsistent candle data: {report.describe()}")
    age = now_ms - candles[-1].close_time_ms
    if age < -2000:
        raise DataError("latest 'closed' candle closes in the future: clock skew or bad data")
    if age > max_age_factor * interval_ms:
        raise DataError(f"market data is stale: last closed candle is {age / 1000:.0f}s old")


CLOSE_GRACE_MS = 2_000


def only_closed(
    candles: Sequence[Candle], now_ms: int, grace_ms: int = CLOSE_GRACE_MS
) -> list[Candle]:
    """Drop the still-forming candle, plus one that closed less than ``grace_ms`` ago.

    REST klines carry no "closed" flag and tolerated clock skew can reach one second, so a
    candle is trusted only after a short grace period (its final values could still change).
    """
    return [c for c in candles if c.close_time_ms < now_ms - grace_ms]


def fetch_range(
    client: BinanceSpotClient, symbol: str, interval: str, start_ms: int, end_ms: int
) -> list[Candle]:
    """Download klines in [start_ms, end_ms] with pagination (1000 per request)."""
    interval_ms = SUPPORTED_INTERVALS[interval]
    out: list[Candle] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = parse_klines(
            client.klines(symbol, interval, start_ms=cursor, end_ms=end_ms, limit=1000)
        )
        if not batch:
            break
        out.extend(batch)
        next_cursor = batch[-1].open_time_ms + interval_ms
        if next_cursor <= cursor:
            raise DataError("pagination made no progress")
        cursor = next_cursor
    return out


def fetch_latest_closed(
    client: BinanceSpotClient, symbol: str, interval: str, count: int, now_ms: int
) -> list[Candle]:
    raw = client.klines(symbol, interval, limit=min(1000, count + 1))
    return only_closed(parse_klines(raw), now_ms)[-count:]


def fetch_quote(client: BinanceSpotClient, symbol: str, now_ms: int) -> Quote:
    data = client.book_ticker(symbol)
    bid, ask = D(data["bidPrice"]), D(data["askPrice"])
    if bid <= 0 or ask <= 0 or bid > ask:
        raise DataError(f"invalid book ticker for {symbol}: bid={bid} ask={ask}")
    return Quote(symbol, bid, ask, now_ms)


def save_csv(path: Path, candles: Sequence[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for c in candles:
            w.writerow([c.open_time_ms, c.open, c.high, c.low, c.close, c.volume, c.close_time_ms])


def load_csv(path: Path) -> list[Candle]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != CSV_HEADER:
            raise DataError(f"{path}: unexpected CSV header {header!r}")
        return parse_klines(list(reader))
