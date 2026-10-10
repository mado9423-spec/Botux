"""Paper mode must never send real orders: proven over the REAL feed/client path.

The only thing faked is the HTTP transport, which records every request the bot makes.
"""

from __future__ import annotations

import dataclasses
import socket
from decimal import Decimal

import httpx
import pytest

from botux.app import build_paper_app
from botux.clock import FakeClock
from botux.config import MarketSettings, Settings, StrategySettings
from botux.errors import EndpointNotAllowed
from botux.models import OrderStatus, Side
from botux.storage import Store
from tests.conftest import (
    EXCHANGE_INFO_BTCUSDT,
    HOUR_MS,
    T0,
    FakeExchange,
    candles_from_closes,
    kline_rows,
)

SCENARIO = [100, 100, 100, 100, 101, 103, 104, 104, 100, 96, 96]
PUBLIC_PATHS = {
    "/api/v3/time",
    "/api/v3/exchangeInfo",
    "/api/v3/klines",
    "/api/v3/ticker/bookTicker",
}


@pytest.fixture
def cfg(settings: Settings) -> Settings:
    return dataclasses.replace(
        settings,
        strategy=StrategySettings("sma_cross", {"fast": 2, "slow": 3}),
        market=MarketSettings(poll_seconds=1.0),
    )


def public_only_exchange(clock: FakeClock) -> FakeExchange:
    """Serves ONLY public market data. Any other path would get a 404 and be recorded."""
    candles = candles_from_closes(SCENARIO)
    ex = FakeExchange()
    ex.on(
        "GET", "/api/v3/time", lambda _r: httpx.Response(200, json={"serverTime": clock.now_ms()})
    )
    ex.on("GET", "/api/v3/exchangeInfo", EXCHANGE_INFO_BTCUSDT)
    ex.on(
        "GET",
        "/api/v3/klines",
        lambda _r: httpx.Response(
            200, json=kline_rows([c for c in candles if c.open_time_ms <= clock.now_ms()])
        ),
    )
    ex.on(
        "GET",
        "/api/v3/ticker/bookTicker",
        lambda _r: httpx.Response(
            200,
            json={
                "symbol": "BTCUSDT",
                "bidPrice": "100.99",
                "askPrice": "101.01",
                "bidQty": "1",
                "askQty": "1",
            },
        ),
    )
    return ex


class TestPaperModeSendsNoRealOrders:
    def test_the_whole_flow_uses_only_public_get_requests(
        self, cfg: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Poison the credential loader and put keys in the environment: neither may matter.
        monkeypatch.setenv("BINANCE_API_KEY", "E" * 64)
        monkeypatch.setenv("BINANCE_API_SECRET", "F" * 64)
        monkeypatch.setattr(
            "botux.config.load_credentials", lambda *a, **k: pytest.fail("keys read")
        )
        monkeypatch.setattr(
            "botux.execution.live.load_credentials", lambda *a, **k: pytest.fail("keys read")
        )

        clock = FakeClock(T0 + 5 * HOUR_MS + 5_000)  # candle idx 4 (the BUY signal) just closed
        ex = public_only_exchange(clock)
        app = build_paper_app(
            cfg, http=ex.http(), clock=clock, store=store, act_on_current_candle=True
        )
        app.loop.run(max_cycles=3)

        # The simulator really did "trade" ...
        assert app.paper_broker is not None and app.paper_broker.placed_count == 1
        (order,) = store.list_orders()
        assert order.side is Side.BUY and order.status is OrderStatus.FILLED
        assert app.paper_broker.balances()["BTC"].free > 0
        # ... yet the network saw nothing but public market-data GETs.
        assert ex.requests, "the bot should have fetched market data"
        assert {r.method for r in ex.requests} == {"GET"}
        assert {r.url.path for r in ex.requests} <= PUBLIC_PATHS
        for r in ex.requests:
            assert "X-MBX-APIKEY" not in r.headers
            assert "signature" not in r.url.query.decode() and r.content == b""
            assert r.url.host == "data-api.binance.vision"
        app.close()

    def test_the_paper_client_is_structurally_unable_to_sign_or_trade(
        self, cfg: Settings, store: Store
    ) -> None:
        clock = FakeClock(T0 + 5 * HOUR_MS + 5_000)
        ex = public_only_exchange(clock)
        app = build_paper_app(cfg, http=ex.http(), clock=clock, store=store)
        assert not app.client.has_credentials and not app.client.trading_enabled
        with pytest.raises(EndpointNotAllowed):
            app.client.new_order(
                symbol="BTCUSDT",
                side="BUY",
                order_type="MARKET",
                quantity=Decimal("0.001"),
                client_order_id="x",
            )
        with pytest.raises(EndpointNotAllowed):
            app.client.account()
        assert {r.url.path for r in ex.requests} <= PUBLIC_PATHS
        app.close()

    def test_simulated_fills_need_no_socket_at_all(
        self, cfg: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With market data supplied locally, a full trading cycle opens no network connection."""
        from tests.test_runner import FakeFeed

        def no_network(*_a: object, **_k: object) -> None:
            raise AssertionError("a network connection was attempted")

        monkeypatch.setattr(socket.socket, "connect", no_network)
        monkeypatch.setattr(socket, "create_connection", no_network)
        clock = FakeClock(T0 + 5 * HOUR_MS + 5_000)
        feed = FakeFeed(clock, candles_from_closes(SCENARIO))
        feed.mark_to_last_close()
        from botux.exchange.rules import SymbolRules

        app = build_paper_app(
            cfg,
            clock=clock,
            store=store,
            feed=feed,
            act_on_current_candle=True,
            rules={"BTCUSDT": SymbolRules.from_exchange_info(EXCHANGE_INFO_BTCUSDT, "BTCUSDT")},
        )
        app.loop.run_cycle()
        assert app.paper_broker is not None and app.paper_broker.placed_count == 1
        app.close()
