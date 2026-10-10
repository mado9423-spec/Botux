"""Binance connector: allow-list, signing, rate limits, retries, unknown outcomes, clock skew."""

from __future__ import annotations

import random
from decimal import Decimal
from urllib.parse import parse_qs

import httpx
import pytest

from botux.clock import FakeClock
from botux.errors import (
    APIError,
    ClockSkewError,
    ConfigError,
    EndpointNotAllowed,
    ExchangeError,
    GeoRestrictedError,
    IPBannedError,
    NetworkError,
    OutcomeUnknownError,
    RateLimitError,
)
from botux.exchange.client import (
    ENDPOINTS,
    TIMESTAMP_SAFETY_MS,
    BinanceSpotClient,
    hmac_signature,
)
from botux.secrets_guard import Credentials, Secret
from tests.conftest import FakeExchange

API_KEY = "K" * 64
API_SECRET = "S" * 64
CREDS = Credentials(Secret(API_KEY), Secret(API_SECRET))


def make_client(
    ex: FakeExchange,
    clock: FakeClock,
    *,
    creds: bool = True,
    trading: bool = False,
    **kw: object,
) -> BinanceSpotClient:
    return BinanceSpotClient(
        credentials=CREDS if creds else None,
        trading_enabled=trading,
        http=ex.http(),
        clock=clock,
        rng=random.Random(0),
        **kw,  # type: ignore[arg-type]
    )


def order_kwargs() -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "order_type": "MARKET",
        "quantity": Decimal("0.001"),
        "client_order_id": "bxtest1",
    }


class TestSigning:
    def test_matches_official_documentation_vector(self) -> None:
        # Test vector from Binance's rest-api.md ("SIGNED Endpoint Examples for POST /api/v3/order")
        secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP" + "1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
        payload = (
            "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1"
            "&recvWindow=5000&timestamp=1499827319559"
        )
        expected = "c8db56825ae71d6d79447849e617115f" + "4a920fa2acdcab2b053c4b2838bd6b71"
        assert hmac_signature(secret, payload) == expected

    def test_signed_post_uses_body_header_key_and_valid_signature(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on("POST", "/api/v3/order", {"orderId": 1})
        client = make_client(ex, clock, trading=True)
        client.new_order(**order_kwargs())  # type: ignore[arg-type]
        req = ex.requests[0]
        body = req.content.decode()
        assert req.url.query == b""  # signature never travels in the URL
        assert req.headers["X-MBX-APIKEY"] == API_KEY
        assert API_KEY not in body and API_SECRET not in body
        fields = parse_qs(body)
        assert fields["recvWindow"] == ["5000"] and fields["quantity"] == ["0.001"]
        assert fields["newClientOrderId"] == ["bxtest1"]
        unsigned, _, sig = body.rpartition("&signature=")
        assert hmac_signature(API_SECRET, unsigned) == sig

    def test_signed_get_puts_signature_in_query(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on("GET", "/api/v3/account", {"balances": []})
        make_client(ex, clock).account()
        q = parse_qs(ex.requests[0].url.query.decode())
        assert "signature" in q and "timestamp" in q
        assert API_SECRET not in ex.requests[0].url.query.decode()

    def test_timestamp_uses_measured_clock_offset(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on(
            "GET",
            "/api/v3/time",
            lambda _r: httpx.Response(200, json={"serverTime": clock.now_ms() + 400}),
        )
        ex.on("GET", "/api/v3/account", {"balances": []})
        c = make_client(ex, clock)
        assert c.sync_time() == 400
        c.account()
        sent = int(parse_qs(ex.requests[-1].url.query.decode())["timestamp"][0])
        assert sent == clock.now_ms() + 400 - TIMESTAMP_SAFETY_MS


class TestAllowList:
    def test_no_money_movement_endpoints_exist(self) -> None:
        forbidden = ("withdraw", "transfer", "margin", "futures", "capital", "loan", "convert")
        for ep in ENDPOINTS.values():
            assert not any(word in ep.path.lower() for word in forbidden), ep
            assert ep.method in {"GET", "POST", "DELETE"}
            if ep.method != "GET":
                assert ep.path in {"/api/v3/order", "/api/v3/order/test"}
        sapi = [ep.path for ep in ENDPOINTS.values() if ep.path.startswith("/sapi")]
        assert sapi == ["/sapi/v1/account/apiRestrictions"]  # the read-only permission check

    def test_unknown_endpoint_name_is_refused(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        c = make_client(ex, clock)
        with pytest.raises(EndpointNotAllowed):
            c._call("withdraw", {})
        assert ex.requests == []

    def test_signed_endpoints_need_credentials(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        c = make_client(ex, clock, creds=False)
        with pytest.raises(EndpointNotAllowed):
            c.account()
        with pytest.raises(EndpointNotAllowed):
            c.api_restrictions()
        assert ex.requests == []

    def test_trading_needs_explicit_flag(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        c = make_client(ex, clock, trading=False)  # credentials present, trading NOT enabled
        with pytest.raises(EndpointNotAllowed):
            c.new_order(**order_kwargs())  # type: ignore[arg-type]
        with pytest.raises(EndpointNotAllowed):
            c.cancel_order("BTCUSDT", client_order_id="x")
        with pytest.raises(EndpointNotAllowed):
            c.test_order(**order_kwargs())  # type: ignore[arg-type]
        assert ex.requests == []

    def test_trading_flag_requires_credentials(self, clock: FakeClock) -> None:
        with pytest.raises(ConfigError):
            BinanceSpotClient(credentials=None, trading_enabled=True, clock=clock)

    @pytest.mark.parametrize(
        ("kw"),
        [
            {"market_data_base_url": "https://evil.example"},
            {"market_data_base_url": "http://data-api.binance.vision"},
            {"trading_base_url": "https://data-api.binance.vision"},
            {"trading_base_url": "https://api.binance.com.evil.example"},
        ],
    )
    def test_hosts_must_be_allow_listed_https(self, kw: dict[str, str], clock: FakeClock) -> None:
        with pytest.raises(ConfigError):
            BinanceSpotClient(clock=clock, **kw)  # type: ignore[arg-type]

    def test_recv_window_is_capped(self, clock: FakeClock) -> None:
        with pytest.raises(ConfigError):
            BinanceSpotClient(clock=clock, recv_window_ms=60_000)

    def test_floats_never_reach_the_wire(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        c = make_client(ex, clock, creds=False)
        with pytest.raises(TypeError):
            c._call("klines", {"symbol": "BTCUSDT", "interval": "1h", "limit": 1.5})

    def test_public_requests_go_to_market_data_host_signed_to_trading_host(
        self, clock: FakeClock
    ) -> None:
        ex = FakeExchange()
        ex.on("GET", "/api/v3/ticker/bookTicker", {"bidPrice": "1", "askPrice": "2"})
        ex.on("GET", "/api/v3/account", {"balances": []})
        c = make_client(ex, clock)
        c.book_ticker("BTCUSDT")
        c.account()
        assert ex.requests[0].url.host == "data-api.binance.vision"
        assert ex.requests[1].url.host == "api.binance.com"
        assert "X-MBX-APIKEY" not in ex.requests[0].headers  # key never sent with public calls


class TestRateLimitsAndBans:
    def test_429_starts_cooldown_and_blocks_further_requests(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on(
            "GET",
            "/api/v3/time",
            lambda _r: httpx.Response(
                429, json={"code": -1003, "msg": "x"}, headers={"Retry-After": "7"}
            ),
        )
        c = make_client(ex, clock, creds=False)
        with pytest.raises(RateLimitError) as e:
            c.server_time_ms()
        assert e.value.retry_after == 7
        n = len(ex.requests)
        with pytest.raises(RateLimitError):  # still cooling down: nothing is sent
            c.server_time_ms()
        assert len(ex.requests) == n
        clock.advance(8)
        ex.on("GET", "/api/v3/time", {"serverTime": 1})
        assert c.server_time_ms() == 1

    def test_418_ban_is_respected(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on(
            "GET",
            "/api/v3/time",
            lambda _r: httpx.Response(
                418, json={"code": -1003, "msg": "banned"}, headers={"Retry-After": "120"}
            ),
        )
        c = make_client(ex, clock, creds=False)
        with pytest.raises(IPBannedError):
            c.server_time_ms()
        with pytest.raises(IPBannedError):
            c.server_time_ms()
        assert len(ex.requests) == 1

    def test_local_weight_budget_makes_the_client_wait(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on("GET", "/api/v3/klines", [])
        c = make_client(ex, clock, creds=False, max_weight_per_minute=5)
        for _ in range(3):  # klines weigh 2 each: the third would exceed 5 within a minute
            c.klines("BTCUSDT", "1h", limit=1)
        assert clock.sleeps and sum(clock.sleeps) > 0

    def test_used_weight_header_is_recorded(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on(
            "GET",
            "/api/v3/time",
            lambda _r: httpx.Response(
                200, json={"serverTime": 1}, headers={"x-mbx-used-weight-1m": "42"}
            ),
        )
        c = make_client(ex, clock, creds=False)
        c.server_time_ms()
        assert c.last_used_weight_1m == 42

    def test_451_is_reported_and_never_retried_or_bypassed(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on("GET", "/api/v3/time", lambda _r: httpx.Response(451, text="restricted"))
        c = make_client(ex, clock, creds=False)
        with pytest.raises(GeoRestrictedError):
            c.server_time_ms()
        assert len(ex.requests) == 1 and not clock.sleeps


class TestRetriesOnSafeRequests:
    def test_get_is_retried_on_5xx_with_growing_backoff(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        replies = [
            httpx.Response(500),
            httpx.Response(502),
            httpx.Response(200, json={"serverTime": 9}),
        ]
        ex.on("GET", "/api/v3/time", lambda _r: replies.pop(0))
        c = make_client(ex, clock, creds=False)
        assert c.server_time_ms() == 9
        assert len(ex.requests) == 3
        assert len(clock.sleeps) == 2 and clock.sleeps[1] > clock.sleeps[0]

    def test_retries_are_bounded(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on("GET", "/api/v3/time", lambda _r: httpx.Response(503))
        c = make_client(ex, clock, creds=False, max_retries=2)
        with pytest.raises(ExchangeError):
            c.server_time_ms()
        assert len(ex.requests) == 3

    def test_get_retried_after_connection_errors(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        calls = {"n": 0}

        def flaky(_r: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.ConnectError("boom")
            return httpx.Response(200, json={"serverTime": 5})

        ex.on("GET", "/api/v3/time", flaky)
        assert make_client(ex, clock, creds=False).server_time_ms() == 5

    def test_cancel_is_retry_safe(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        replies = [httpx.Response(500), httpx.Response(200, json={"status": "CANCELED"})]
        ex.on("DELETE", "/api/v3/order", lambda _r: replies.pop(0))
        c = make_client(ex, clock, trading=True)
        assert c.cancel_order("BTCUSDT", client_order_id="x")["status"] == "CANCELED"
        assert len(ex.requests) == 2

    def test_4xx_api_error_is_not_retried(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on(
            "GET",
            "/api/v3/exchangeInfo",
            lambda _r: httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."}),
        )
        c = make_client(ex, clock, creds=False)
        with pytest.raises(APIError) as e:
            c.exchange_info("NOPE")
        assert e.value.code == -1121 and len(ex.requests) == 1


class TestOrdersAreNeverBlindlyRetried:
    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(500, text="oops"),
            httpx.Response(503, json={"code": -1007, "msg": "Timeout waiting for response"}),
            httpx.Response(504, text="gateway timeout"),
            httpx.Response(400, json={"code": -1006, "msg": "unexpected response"}),
            httpx.Response(500, json={"code": -1001, "msg": "Internal error"}),
            httpx.Response(408, text="request timeout"),
            httpx.Response(499, text="client closed request"),
            httpx.Response(400, text="<html>proxy error page</html>"),
            httpx.Response(302, headers={"location": "https://elsewhere.example"}),
            httpx.Response(200, text="not json at all"),
        ],
    )
    def test_ambiguous_answers_raise_unknown_outcome_after_one_request(
        self, response: httpx.Response, clock: FakeClock
    ) -> None:
        ex = FakeExchange()
        ex.on("POST", "/api/v3/order", lambda _r: response)
        c = make_client(ex, clock, trading=True)
        with pytest.raises(OutcomeUnknownError):
            c.new_order(**order_kwargs())  # type: ignore[arg-type]
        assert len(ex.requests) == 1 and not clock.sleeps

    def test_read_timeout_means_unknown_outcome(self, clock: FakeClock) -> None:
        ex = FakeExchange()

        def timeout(_r: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow")

        ex.on("POST", "/api/v3/order", timeout)
        c = make_client(ex, clock, trading=True)
        with pytest.raises(OutcomeUnknownError):
            c.new_order(**order_kwargs())  # type: ignore[arg-type]
        assert len(ex.requests) == 1

    def test_connection_failure_is_definitely_not_sent(self, clock: FakeClock) -> None:
        ex = FakeExchange()

        def refuse(_r: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        ex.on("POST", "/api/v3/order", refuse)
        c = make_client(ex, clock, trading=True)
        with pytest.raises(NetworkError) as e:
            c.new_order(**order_kwargs())  # type: ignore[arg-type]
        assert e.value.not_sent is True and len(ex.requests) == 1

    def test_definitive_rejection_is_a_plain_api_error(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on(
            "POST",
            "/api/v3/order",
            lambda _r: httpx.Response(400, json={"code": -1013, "msg": "Filter failure: NOTIONAL"}),
        )
        c = make_client(ex, clock, trading=True)
        with pytest.raises(APIError) as e:
            c.new_order(**order_kwargs())  # type: ignore[arg-type]
        assert not isinstance(e.value, OutcomeUnknownError) and e.value.code == -1013

    def test_limit_order_requires_price(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        c = make_client(ex, clock, trading=True)
        kw = order_kwargs() | {"order_type": "LIMIT"}
        with pytest.raises(ValueError):
            c.new_order(**kw)  # type: ignore[arg-type]


class TestClockSkew:
    def test_large_skew_is_refused(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on(
            "GET",
            "/api/v3/time",
            lambda _r: httpx.Response(200, json={"serverTime": clock.now_ms() + 5000}),
        )
        with pytest.raises(ClockSkewError):
            make_client(ex, clock, creds=False).sync_time()

    def test_minus_1021_triggers_one_resync_then_succeeds(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        replies = [
            httpx.Response(400, json={"code": -1021, "msg": "Timestamp outside recvWindow"}),
            httpx.Response(200, json={"balances": []}),
        ]
        ex.on("GET", "/api/v3/account", lambda _r: replies.pop(0))
        ex.on(
            "GET",
            "/api/v3/time",
            lambda _r: httpx.Response(200, json={"serverTime": clock.now_ms() + 300}),
        )
        make_client(ex, clock).account()
        assert ex.count("/api/v3/time") == 3  # one resync = 3 samples
        assert ex.count("/api/v3/account") == 2

    def test_minus_1021_on_an_order_is_not_retried(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on(
            "POST",
            "/api/v3/order",
            lambda _r: httpx.Response(400, json={"code": -1021, "msg": "ts"}),
        )
        ex.on(
            "GET",
            "/api/v3/time",
            lambda _r: httpx.Response(200, json={"serverTime": clock.now_ms() + 200}),
        )
        c = make_client(ex, clock, trading=True)
        with pytest.raises(ClockSkewError):
            c.new_order(**order_kwargs())  # type: ignore[arg-type]
        assert ex.count("/api/v3/order") == 1  # the order itself is never re-sent
        assert ex.count("/api/v3/time") == 3  # but the clock is re-measured for the next request
        assert c.clock_offset_ms == 200

    def test_clock_is_resynchronised_periodically_before_signed_requests(
        self, clock: FakeClock
    ) -> None:
        ex = FakeExchange()
        ex.on(
            "GET",
            "/api/v3/time",
            lambda _r: httpx.Response(200, json={"serverTime": clock.now_ms()}),
        )
        ex.on("GET", "/api/v3/account", {"balances": []})
        c = make_client(ex, clock)
        c.sync_time()
        before = ex.count("/api/v3/time")
        c.account()
        assert ex.count("/api/v3/time") == before  # fresh enough
        clock.advance(700)
        c.account()
        assert ex.count("/api/v3/time") == before + 1

    def test_sync_prefers_the_lowest_latency_sample(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        delays = iter(
            [2000, 40, 1500]
        )  # round trips in ms; the 40 ms sample is the trustworthy one
        offsets = iter([900, 20, -800])

        def slow_time(_r: httpx.Request) -> httpx.Response:
            clock.advance(next(delays) / 1000)
            return httpx.Response(200, json={"serverTime": clock.now_ms() + next(offsets) - 20})

        ex.on("GET", "/api/v3/time", slow_time)
        c = make_client(ex, clock, creds=False, max_clock_skew_ms=2500)
        assert abs(c.sync_time() - 0) <= 25  # taken from the 40 ms round trip, not the others


class TestQueries:
    def test_get_order_needs_exactly_one_identifier(self, clock: FakeClock) -> None:
        c = make_client(FakeExchange(), clock)
        with pytest.raises(ValueError):
            c.get_order("BTCUSDT")
        with pytest.raises(ValueError):
            c.get_order("BTCUSDT", client_order_id="a", order_id=1)

    def test_klines_query_parameters(self, clock: FakeClock) -> None:
        ex = FakeExchange()
        ex.on("GET", "/api/v3/klines", [])
        make_client(ex, clock, creds=False).klines("BTCUSDT", "1h", start_ms=1, end_ms=2, limit=10)
        q = parse_qs(ex.requests[0].url.query.decode())
        assert q == {
            "symbol": ["BTCUSDT"],
            "interval": ["1h"],
            "limit": ["10"],
            "startTime": ["1"],
            "endTime": ["2"],
        }
