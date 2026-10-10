"""The whole live path against a STATEFUL fake exchange (no network): signed requests, real
Binance-shaped answers, the trading loop, the order manager, the ledger and the safety checks.

Nothing here contacts Binance; the live key used is a made-up placeholder string."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

from botux.app import App, build_live_app
from botux.clock import FakeClock
from botux.config import MarketSettings, Settings, StrategySettings
from botux.models import Candle, OrderStatus, Side
from botux.safety import ARM_PHRASE, arm_live_interactive
from botux.storage import Store
from tests.conftest import (
    EXCHANGE_INFO_BTCUSDT,
    HOUR_MS,
    T0,
    FakeExchange,
    candles_from_closes,
    kline_rows,
)
from tests.test_live_gate import API_KEY, API_SECRET, ENV, GOOD_KEY, live

D = Decimal
SCENARIO = [100, 100, 100, 100, 101, 103, 104, 104, 100, 96, 96]
ALLOWED_PATHS = {
    "/api/v3/time",
    "/api/v3/exchangeInfo",
    "/api/v3/klines",
    "/api/v3/ticker/bookTicker",
    "/api/v3/account",
    "/api/v3/order",
    "/api/v3/openOrders",
    "/api/v3/myTrades",
    "/sapi/v1/account/apiRestrictions",
}


class VenueState:
    """A tiny matching engine: market orders fill at the quote, fees in the received asset."""

    def __init__(self, clock: FakeClock, candles: list[Candle]) -> None:
        self.clock, self.candles = clock, candles
        self.bid, self.ask = D("99.99"), D("100.01")
        self.free = {"USDT": D(1000), "BTC": D(0)}
        self.orders: dict[str, dict[str, object]] = {}
        self.foreign_open: list[dict[str, object]] = []
        self.next_id = 1000
        self.posts = 0

    def mark(self) -> None:
        closed = [c for c in self.candles if c.close_time_ms < self.clock.now_ms()]
        self.bid = closed[-1].close - D("0.01")
        self.ask = closed[-1].close + D("0.01")

    def exchange(self) -> FakeExchange:
        ex = FakeExchange()
        ex.on(
            "GET",
            "/api/v3/time",
            lambda _r: httpx.Response(200, json={"serverTime": self.clock.now_ms()}),
        )
        ex.on("GET", "/sapi/v1/account/apiRestrictions", dict(GOOD_KEY))
        ex.on("GET", "/api/v3/exchangeInfo", EXCHANGE_INFO_BTCUSDT)
        ex.on("GET", "/api/v3/klines", self.klines)
        ex.on("GET", "/api/v3/ticker/bookTicker", self.book_ticker)
        ex.on("GET", "/api/v3/account", self.account)
        ex.on("GET", "/api/v3/openOrders", lambda _r: httpx.Response(200, json=self.foreign_open))
        ex.on("POST", "/api/v3/order", self.new_order)
        ex.on("GET", "/api/v3/order", self.get_order)
        ex.on("GET", "/api/v3/myTrades", lambda _r: httpx.Response(200, json=[]))
        return ex

    def klines(self, _r: httpx.Request) -> httpx.Response:
        shown = [c for c in self.candles if c.open_time_ms <= self.clock.now_ms()]
        return httpx.Response(200, json=kline_rows(shown))

    def book_ticker(self, _r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"symbol": "BTCUSDT", "bidPrice": str(self.bid), "askPrice": str(self.ask)}
        )

    def account(self, _r: httpx.Request) -> httpx.Response:
        balances = [
            {"asset": a, "free": str(v), "locked": "0"} for a, v in self.free.items() if v > 0
        ]
        return httpx.Response(200, json={"canTrade": True, "uid": 4242, "balances": balances})

    def new_order(self, req: httpx.Request) -> httpx.Response:
        self.posts += 1
        body = parse_qs(req.content.decode())
        assert "signature" in body and "timestamp" in body and req.url.query == b""
        side, qty = body["side"][0], D(body["quantity"][0])
        cid = body["newClientOrderId"][0]
        price = self.ask if side == "BUY" else self.bid
        if side == "BUY":
            fee, asset = qty * D("0.001"), "BTC"
            self.free["USDT"] -= qty * price
            self.free["BTC"] += qty - fee
        else:
            fee, asset = qty * price * D("0.001"), "USDT"
            self.free["BTC"] -= qty
            self.free["USDT"] += qty * price - fee
        self.next_id += 1
        order = {
            "symbol": "BTCUSDT", "orderId": self.next_id, "clientOrderId": cid,
            "transactTime": self.clock.now_ms(), "price": "0.00000000",
            "origQty": str(qty), "executedQty": str(qty),
            "cummulativeQuoteQty": str(qty * price), "status": "FILLED",
            "type": "MARKET", "side": side,
            "fills": [{"price": str(price), "qty": str(qty), "commission": str(fee),
                       "commissionAsset": asset, "tradeId": self.next_id}],
        }  # fmt: skip
        self.orders[cid] = order
        return httpx.Response(200, json=order)

    def get_order(self, req: httpx.Request) -> httpx.Response:
        cid = req.url.params["origClientOrderId"]
        if cid not in self.orders:
            return httpx.Response(400, json={"code": -2013, "msg": "Order does not exist."})
        answer = dict(self.orders[cid])
        answer.pop("fills", None)  # GET /order carries no fills
        return httpx.Response(200, json=answer)


@dataclasses.dataclass
class Rig:
    settings: Settings
    clock: FakeClock
    venue: VenueState
    ex: FakeExchange
    store: Store
    app: App


@pytest.fixture
def setup(settings: Settings, tmp_path: Path) -> Iterator[Rig]:
    clock = FakeClock(T0 + 4 * HOUR_MS + 5_000)
    s = dataclasses.replace(
        live(settings),
        strategy=StrategySettings("sma_cross", {"fast": 2, "slow": 3}),
        market=MarketSettings(poll_seconds=30.0),
    )
    arm_live_interactive(s, clock.now_ms(), lambda _p: ARM_PHRASE, lambda _t: None)
    venue = VenueState(clock, candles_from_closes(SCENARIO))
    venue.mark()
    ex = venue.exchange()
    store = Store(tmp_path / "live.sqlite3")
    app = build_live_app(s, ENV, http=ex.http(), clock=clock, store=store)
    yield Rig(s, clock, venue, ex, store, app)
    app.close()
    store.close()


def cycle(clock: FakeClock, venue: VenueState, app: App, hours: int = 0) -> None:
    clock.advance(hours * 3600)
    venue.mark()
    app.loop.run_cycle()


def test_signal_to_fill_to_stop_through_the_live_path(
    setup: Rig, caplog: pytest.LogCaptureFixture
) -> None:
    clock, venue, ex, store, app = setup.clock, setup.venue, setup.ex, setup.store, setup.app
    with caplog.at_level(logging.DEBUG):
        cycle(clock, venue, app)  # warm start: nothing is traded
        assert venue.posts == 0
        cycle(clock, venue, app, hours=1)  # candle 4 closed: BUY
        (buy,) = store.list_orders()
        assert buy.side is Side.BUY and buy.status is OrderStatus.FILLED and venue.posts == 1
        assert buy.quantity * D("100.02") <= D(100) and app.loop.positions.get("BTCUSDT")
        # the bot's books agree with the venue, so nothing is flagged
        assert app.loop.risk.blockers == {}
        led = app.loop._ledger.view()
        assert led.base_qty("BTCUSDT") == venue.free["BTC"]
        assert led.cash == 500 - buy.cumulative_quote_qty
        # a crash through the stop: the SELL goes out once, from the bot's own coins
        venue.bid, venue.ask = D("90.00"), D("90.01")
        app.loop.run_cycle()
        stops = [o for o in store.list_orders() if o.intent_key.startswith("stop:")]
        assert len(stops) == 1 and stops[0].status is OrderStatus.FILLED
        assert venue.free["BTC"] < D("0.00001")  # flat (only fee dust is left)
        assert app.loop.risk.blockers == {}

    # every request stayed inside the allow-list; every order was signed and sent by POST body
    assert {r.url.path for r in ex.requests} <= ALLOWED_PATHS
    posts = [r for r in ex.requests if r.method == "POST"]
    assert [p.url.path for p in posts] == ["/api/v3/order"] * 2
    assert all(p.headers["X-MBX-APIKEY"] == API_KEY and p.url.query == b"" for p in posts)
    assert not [r for r in ex.requests if r.method == "DELETE"]
    # no secret anywhere durable
    dump = "\n".join(store._conn.iterdump())
    assert API_KEY not in caplog.text + dump and API_SECRET not in caplog.text + dump
    assert API_SECRET not in "".join(str(r.url) + r.content.decode() for r in ex.requests)
    # the database is bound to this account
    assert store.kv_get("account_uid") == "4242"


def test_a_foreign_open_order_stops_new_entries_and_is_never_cancelled(setup: Rig) -> None:
    clock, venue, ex, store, app = setup.clock, setup.venue, setup.ex, setup.store, setup.app
    cycle(clock, venue, app)
    venue.foreign_open = [
        {"symbol": "BTCUSDT", "orderId": 9, "clientOrderId": "web_abc123", "price": "50000",
         "origQty": "0.1", "executedQty": "0", "cummulativeQuoteQty": "0", "status": "NEW",
         "type": "LIMIT", "side": "BUY", "transactTime": 1},
    ]  # fmt: skip
    cycle(clock, venue, app, hours=1)  # the BUY signal candle
    assert store.list_orders() == [] and venue.posts == 0
    assert "orders:BTCUSDT" in app.loop.risk.blockers
    assert not [r for r in ex.requests if r.method == "DELETE"]


def test_coins_arriving_on_the_account_are_not_the_bots_to_sell(setup: Rig) -> None:
    clock, venue, app = setup.clock, setup.venue, setup.app
    cycle(clock, venue, app)
    venue.free["BTC"] = D("0.7")  # a deposit
    cycle(clock, venue, app, hours=1)
    assert venue.posts == 0 and "balance:BTC" in app.loop.risk.blockers
    venue.bid, venue.ask = D("50.00"), D("50.01")
    app.loop.run_cycle()
    assert venue.posts == 0 and venue.free["BTC"] == D("0.7")  # nothing was sold


def test_a_restart_on_the_same_database_keeps_the_books_and_does_not_retrade(
    setup: Rig, tmp_path: Path
) -> None:
    s, clock, venue, store, app = setup.settings, setup.clock, setup.venue, setup.store, setup.app
    cycle(clock, venue, app)
    cycle(clock, venue, app, hours=1)  # BUY
    assert venue.posts == 1
    app.close()
    store.close()
    store2 = Store(tmp_path / "live.sqlite3")
    app2 = build_live_app(s, ENV, http=venue.exchange().http(), clock=clock, store=store2)
    try:
        app2.loop.run_cycle()  # the same candle, after a restart
        assert venue.posts == 1
        assert app2.loop.risk.blockers == {}
        assert app2.loop._ledger.view().base_qty("BTCUSDT") == venue.free["BTC"]
    finally:
        app2.close()
        store2.close()
