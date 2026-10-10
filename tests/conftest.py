"""Shared fixtures: realistic symbol rules, fake clock, fake exchange over httpx.MockTransport."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterator, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from botux.clock import FakeClock
from botux.config import LiveSettings, PathSettings, Settings
from botux.exchange.rules import SymbolRules
from botux.models import Candle
from botux.secrets_guard import GLOBAL_REDACTOR
from botux.storage import Store

HOUR_MS = 3_600_000
T0 = 1_700_000_000_000 // HOUR_MS * HOUR_MS  # an hour-aligned start

# Shape and magnitudes follow a real BTCUSDT /exchangeInfo answer.
EXCHANGE_INFO_BTCUSDT: dict[str, Any] = {
    "timezone": "UTC",
    "serverTime": 1_700_000_000_000,
    "rateLimits": [],
    "exchangeFilters": [],
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "baseAsset": "BTC",
            "baseAssetPrecision": 8,
            "quoteAsset": "USDT",
            "quotePrecision": 8,
            "quoteAssetPrecision": 8,
            "orderTypes": ["LIMIT", "LIMIT_MAKER", "MARKET", "STOP_LOSS_LIMIT"],
            "isSpotTradingAllowed": True,
            "isMarginTradingAllowed": True,
            "filters": [
                {
                    "filterType": "PRICE_FILTER",
                    "minPrice": "0.01000000",
                    "maxPrice": "1000000.00000000",
                    "tickSize": "0.01000000",
                },
                {
                    "filterType": "LOT_SIZE",
                    "minQty": "0.00001000",
                    "maxQty": "9000.00000000",
                    "stepSize": "0.00001000",
                },
                {
                    "filterType": "MARKET_LOT_SIZE",
                    "minQty": "0.00000000",
                    "maxQty": "107.00000000",
                    "stepSize": "0.00000000",
                },
                {
                    "filterType": "NOTIONAL",
                    "minNotional": "5.00000000",
                    "applyMinToMarket": True,
                    "maxNotional": "9000000.00000000",
                    "applyMaxToMarket": False,
                    "avgPriceMins": 5,
                },
                {
                    "filterType": "PERCENT_PRICE_BY_SIDE",
                    "bidMultiplierUp": "5",
                    "bidMultiplierDown": "0.2",
                    "askMultiplierUp": "5",
                    "askMultiplierDown": "0.2",
                    "avgPriceMins": 5,
                },
                {"filterType": "MAX_NUM_ORDERS", "maxNumOrders": 200},
            ],
        }
    ],
}


@pytest.fixture
def exchange_info() -> dict[str, Any]:
    info: dict[str, Any] = json.loads(json.dumps(EXCHANGE_INFO_BTCUSDT))
    return info


@pytest.fixture
def rules(exchange_info: dict[str, Any]) -> SymbolRules:
    return SymbolRules.from_exchange_info(exchange_info, "BTCUSDT")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(T0 + 10 * HOUR_MS + 5_000)


@pytest.fixture
def store() -> Iterator[Store]:
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Default (paper) settings with every file path redirected into tmp_path."""
    return dataclasses.replace(
        Settings(),
        paths=PathSettings(
            kill_file=str(tmp_path / "KILL"),
            database=str(tmp_path / "botux.sqlite3"),
            data_dir=str(tmp_path / "cache"),
            alerts_file=str(tmp_path / "alerts.jsonl"),
        ),
        live=LiveSettings(arm_file=str(tmp_path / "live_armed.json")),
    )


@pytest.fixture(autouse=True)
def _isolate_redactor() -> Iterator[None]:
    """Secrets registered by one test must not leak into the next."""
    before = set(GLOBAL_REDACTOR._secrets)
    yield
    GLOBAL_REDACTOR._secrets.clear()
    GLOBAL_REDACTOR._secrets.update(before)


Num = float | str | Decimal


def make_candle(i: int, o: Num, h: Num, low: Num, c: Num, *, start: int = T0) -> Candle:
    open_time = start + i * HOUR_MS
    return Candle(
        open_time_ms=open_time,
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(low)),
        close=Decimal(str(c)),
        volume=Decimal("10"),
        close_time_ms=open_time + HOUR_MS - 1,
    )


def candles_from_closes(
    closes: Sequence[float | str], *, start: int = T0, spread: str = "0.5"
) -> list[Candle]:
    """Candles whose open equals the previous close (continuous), tiny high/low wicks."""
    out: list[Candle] = []
    prev = Decimal(str(closes[0]))
    w = Decimal(spread)
    for i, close in enumerate(closes):
        c = Decimal(str(close))
        o = prev
        out.append(make_candle(i, o, max(o, c) + w, min(o, c) - w, c, start=start))
        prev = c
    return out


def kline_rows(candles: Sequence[Candle]) -> list[list[Any]]:
    return [
        [
            c.open_time_ms,
            str(c.open),
            str(c.high),
            str(c.low),
            str(c.close),
            str(c.volume),
            c.close_time_ms,
            "0",
            1,
            "0",
            "0",
            "0",
        ]
        for c in candles
    ]


Route = Callable[[httpx.Request], httpx.Response]


class FakeExchange:
    """MockTransport handler that records every request and answers from a route table."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Route] = {}
        self.requests: list[httpx.Request] = []

    def on(self, method: str, path: str, handler: Route | dict[str, Any] | list[Any]) -> None:
        if callable(handler):
            self.routes[(method, path)] = handler
        else:
            payload = handler
            self.routes[(method, path)] = lambda _req: httpx.Response(200, json=payload)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        route = self.routes.get((request.method, request.url.path))
        if route is None:
            return httpx.Response(404, json={"code": -1, "msg": f"no route {request.url.path}"})
        return route(request)

    def http(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def paths(self, method: str | None = None) -> list[str]:
        return [r.url.path for r in self.requests if method is None or r.method == method]

    def count(self, path: str) -> int:
        return sum(1 for r in self.requests if r.url.path == path)
