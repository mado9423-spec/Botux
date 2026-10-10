"""Thin, allow-listed Binance Spot REST connector.

Reference: https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md

Safety properties (each one has a test):

* Only the endpoints in :data:`ENDPOINTS` can be called. There is deliberately no withdrawal,
  transfer, margin, futures or wallet-write endpoint: the single ``/sapi`` call is the read-only
  API-key-permission check.
* Signed endpoints need credentials; placing/cancelling orders additionally needs
  ``trading_enabled=True``, which only the guarded live path sets.
* Public data goes to the market-data host; signed requests only go to an allow-listed trading
  host, so a typo cannot send the API key somewhere else.
* Order-creating requests (``new_order``, ``test_order``) are NEVER retried here. A timeout, a 5XX,
  or error codes -1000/-1001/-1006/-1007/-1008 raise :class:`OutcomeUnknownError` (the order may
  have been executed). Cancels and reads ARE retried (they are idempotent); a cancel whose first
  attempt succeeded answers -2011 on the retry, which the live broker reports as "not open" and
  the order manager then settles by querying the order.
* A client that carries an API key ignores proxy / CA environment variables (``trust_env=False``).
* HTTP 429/418 start a local cool-down honouring ``Retry-After``; nothing is sent during it.
* HTTP 451 (restricted location) is reported as is. It is never worked around.
* Nothing sensitive is logged: only method, path, status and latency.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import random
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any
from urllib.parse import urlencode

import httpx

from botux.clock import Clock, SystemClock
from botux.config import MARKET_DATA_HOSTS, TRADING_HOSTS, check_same_environment
from botux.decimals import fmt
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
from botux.secrets_guard import Credentials

log = logging.getLogger(__name__)

# Binance rejects a timestamp more than 1 s AHEAD of its clock, but tolerates one up to recvWindow
# behind. We therefore deliberately stamp requests slightly in the past.
TIMESTAMP_SAFETY_MS = 300
RESYNC_INTERVAL_S = 600.0

# Error codes whose meaning is "we do not know whether the request was executed".
UNKNOWN_OUTCOME_CODES = frozenset({-1000, -1001, -1006, -1007, -1008})


class Security(StrEnum):
    NONE = "NONE"
    TRADE = "TRADE"
    USER_DATA = "USER_DATA"


@dataclass(frozen=True, slots=True)
class Endpoint:
    name: str
    method: str
    path: str
    security: Security
    weight: int
    safe_to_retry: bool
    changes_state: bool = False


def _ep(
    name: str,
    method: str,
    path: str,
    sec: Security,
    weight: int,
    *,
    retry: bool,
    state: bool = False,
) -> tuple[str, Endpoint]:
    return name, Endpoint(name, method, path, sec, weight, retry, state)


# Weights from the official docs (rest-api.md). The response header X-MBX-USED-WEIGHT-1M is
# the source of truth; these are used for local budgeting only.
ENDPOINTS: dict[str, Endpoint] = dict(
    [
        _ep("ping", "GET", "/api/v3/ping", Security.NONE, 1, retry=True),
        _ep("time", "GET", "/api/v3/time", Security.NONE, 1, retry=True),
        _ep("exchange_info", "GET", "/api/v3/exchangeInfo", Security.NONE, 20, retry=True),
        _ep("klines", "GET", "/api/v3/klines", Security.NONE, 2, retry=True),
        _ep("book_ticker", "GET", "/api/v3/ticker/bookTicker", Security.NONE, 2, retry=True),
        _ep("account", "GET", "/api/v3/account", Security.USER_DATA, 20, retry=True),
        _ep("get_order", "GET", "/api/v3/order", Security.USER_DATA, 4, retry=True),
        _ep("open_orders", "GET", "/api/v3/openOrders", Security.USER_DATA, 6, retry=True),
        _ep("my_trades", "GET", "/api/v3/myTrades", Security.USER_DATA, 5, retry=True),
        _ep(
            "api_restrictions",
            "GET",
            "/sapi/v1/account/apiRestrictions",
            Security.USER_DATA,
            1,
            retry=True,
        ),
        _ep("test_order", "POST", "/api/v3/order/test", Security.TRADE, 1, retry=False),
        _ep("new_order", "POST", "/api/v3/order", Security.TRADE, 1, retry=False, state=True),
        _ep("cancel_order", "DELETE", "/api/v3/order", Security.TRADE, 1, retry=True, state=True),
    ]
)


def hmac_signature(secret: str, payload: str) -> str:
    """HMAC-SHA256 hex digest of the exact query/body string (Binance 'SIGNED' endpoints)."""
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _param(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        return fmt(value)
    if isinstance(value, float):
        raise TypeError("floats are not allowed in API parameters; use Decimal")
    return str(value)


class _WeightWindow:
    """Sliding 60-second window of request weight used by this process."""

    def __init__(self, clock: Clock, budget: int) -> None:
        self._clock = clock
        self._budget = budget
        self._events: deque[tuple[float, int]] = deque()

    def _prune(self) -> None:
        cutoff = self._clock.monotonic() - 60.0
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()

    def used(self) -> int:
        self._prune()
        return sum(w for _, w in self._events)

    def reserve(self, weight: int) -> None:
        """Block (via the injected clock) until ``weight`` fits in the budget, then record it."""
        while True:
            self._prune()
            if self.used() + weight <= self._budget or not self._events:
                break
            wait = self._events[0][0] + 60.0 - self._clock.monotonic()
            self._clock.sleep(max(wait, 0.05))
        self._events.append((self._clock.monotonic(), weight))


class BinanceSpotClient:
    """Allow-listed Binance Spot REST client (synchronous)."""

    def __init__(
        self,
        *,
        market_data_base_url: str = "https://data-api.binance.vision",
        trading_base_url: str = "https://api.binance.com",
        credentials: Credentials | None = None,
        trading_enabled: bool = False,
        order_test_enabled: bool = False,
        http: httpx.Client | None = None,
        clock: Clock | None = None,
        recv_window_ms: int = 5000,
        max_retries: int = 3,
        max_weight_per_minute: int = 1000,
        max_clock_skew_ms: int = 1000,
        rng: random.Random | None = None,
    ) -> None:
        for url, allowed, what in (
            (market_data_base_url, MARKET_DATA_HOSTS, "market data"),
            (trading_base_url, TRADING_HOSTS, "trading"),
        ):
            host = httpx.URL(url).host
            if not url.startswith("https://") or host not in allowed:
                raise ConfigError(f"{what} base URL {url!r} is not an allow-listed https host")
        check_same_environment(market_data_base_url, trading_base_url)
        if recv_window_ms > 5000 or recv_window_ms < 500:
            raise ConfigError("recv_window_ms must be within [500, 5000] (Binance advises <= 5000)")
        if (trading_enabled or order_test_enabled) and credentials is None:
            raise ConfigError("trading_enabled / order_test_enabled require credentials")
        self._md_base = market_data_base_url.rstrip("/")
        self._tr_base = trading_base_url.rstrip("/")
        self._credentials = credentials
        self._trading_enabled = trading_enabled
        self._order_test_enabled = order_test_enabled
        # A client that carries an API key ignores proxy / CA environment variables
        # (HTTPS_PROXY, SSL_CERT_FILE...): a forgotten variable must not be able to route signed
        # requests through a third party. Public market-data clients keep the usual behaviour.
        self._http = http or httpx.Client(
            timeout=httpx.Timeout(15.0, connect=5.0),
            follow_redirects=False,
            trust_env=credentials is None,
            verify=True,
            headers={"User-Agent": "botux/0.1 (+educational)"},
        )
        self._clock = clock or SystemClock()
        self._recv_window_ms = recv_window_ms
        self._max_retries = max_retries
        self._window = _WeightWindow(self._clock, max_weight_per_minute)
        self._max_skew_ms = max_clock_skew_ms
        self._offset_ms = 0
        self._last_sync: float | None = None
        self._cooldown_until = 0.0
        self._cooldown_exc: type[RateLimitError] = RateLimitError
        self._rng = rng or random.Random()  # noqa: S311 - jitter only, not security-sensitive
        self.last_used_weight_1m: int | None = None

    # ------------------------------------------------------------------ properties

    @property
    def has_credentials(self) -> bool:
        return self._credentials is not None

    @property
    def trading_enabled(self) -> bool:
        return self._trading_enabled

    @property
    def order_test_enabled(self) -> bool:
        return self._order_test_enabled

    @property
    def clock_offset_ms(self) -> int:
        return self._offset_ms

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ guards

    def _check_allowed(self, ep: Endpoint) -> None:
        if ep.security is not Security.NONE and self._credentials is None:
            raise EndpointNotAllowed(f"{ep.name}: signed endpoints are disabled (no credentials)")
        # POST /order/test never reaches the matching engine, so it may be enabled on its own
        test_only = ep.name == "test_order" and self._order_test_enabled
        if ep.security is Security.TRADE and not (self._trading_enabled or test_only):
            raise EndpointNotAllowed(f"{ep.name}: trading is not enabled on this client")

    def _check_cooldown(self) -> None:
        remaining = self._cooldown_until - self._clock.monotonic()
        if remaining > 0:
            raise self._cooldown_exc(
                f"cooling down for another {remaining:.0f}s after a rate-limit response",
                retry_after=remaining,
            )

    def _start_cooldown(self, seconds: float, exc: type[RateLimitError]) -> None:
        self._cooldown_until = max(self._cooldown_until, self._clock.monotonic() + seconds)
        self._cooldown_exc = exc

    # ------------------------------------------------------------------ transport

    def _timestamp(self) -> int:
        return self._clock.now_ms() + self._offset_ms - TIMESTAMP_SAFETY_MS

    def _sign(self, params: Mapping[str, Any]) -> str:
        assert self._credentials is not None  # guarded by _check_allowed
        merged = {
            **{k: _param(v) for k, v in params.items()},
            "timestamp": str(self._timestamp()),
            "recvWindow": str(self._recv_window_ms),
        }
        query = urlencode(merged)
        digest = hmac_signature(self._credentials.api_secret.reveal(), query)
        return f"{query}&signature={digest}"

    def _send_once(self, ep: Endpoint, params: Mapping[str, Any]) -> Any:
        signed = ep.security is not Security.NONE
        base = self._tr_base if signed else self._md_base
        headers: dict[str, str] = {}
        content: str | None = None
        query: str | None = None
        if signed:
            assert self._credentials is not None
            headers["X-MBX-APIKEY"] = self._credentials.api_key.reveal()
            payload = self._sign(params)
            if ep.method == "GET":
                query = payload
            else:
                content = payload  # keep signature out of URLs
                headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif params:
            query = urlencode({k: _param(v) for k, v in params.items()})
        url = base + ep.path + (f"?{query}" if query else "")
        started = self._clock.monotonic()
        try:
            resp = self._http.request(ep.method, url, headers=headers, content=content)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise NetworkError(
                f"{ep.name}: could not connect ({type(exc).__name__})", not_sent=True
            ) from None
        except httpx.TransportError as exc:
            if not ep.safe_to_retry:
                raise OutcomeUnknownError(
                    f"{ep.name}: transport failure after sending ({type(exc).__name__}); "
                    "execution status unknown"
                ) from None
            raise NetworkError(f"{ep.name}: {type(exc).__name__}") from None
        log.debug(
            "%s %s -> %s (%.0f ms)",
            ep.method,
            ep.path,
            resp.status_code,
            (self._clock.monotonic() - started) * 1000,
        )
        return self._handle(ep, resp)

    def _handle(self, ep: Endpoint, resp: httpx.Response) -> Any:
        used = resp.headers.get("x-mbx-used-weight-1m")
        if used and used.isdigit():
            self.last_used_weight_1m = int(used)
        status = resp.status_code
        if status == 200:
            try:
                return resp.json()
            except ValueError:
                if ep.safe_to_retry:
                    raise _ServerError(f"{ep.name}: unreadable 200 response") from None
                raise OutcomeUnknownError(
                    f"{ep.name}: unreadable 200 response; execution status unknown"
                ) from None

        code: int | None = None
        msg = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                code = int(body["code"]) if "code" in body else None
                msg = str(body.get("msg", ""))
        except (ValueError, TypeError):
            msg = resp.text[:200]
        retry_after = _retry_after(resp)

        if status == 418:
            self._start_cooldown(retry_after or 120.0, IPBannedError)
            raise IPBannedError(f"IP banned (418): {msg}", retry_after=retry_after)
        if status == 429:
            self._start_cooldown(retry_after or 10.0, RateLimitError)
            raise RateLimitError(f"rate limited (429): {msg}", retry_after=retry_after)
        if status == 451:
            raise GeoRestrictedError(
                "HTTP 451: Binance refuses service from this location. This is not retried or "
                "worked around.",
                http_status=451,
            )
        if status == 403:
            self._start_cooldown(retry_after or 30.0, RateLimitError)
            raise APIError(f"HTTP 403 (WAF block or rate-limit violation): {msg}", http_status=403)
        # On a state-changing endpoint only a 4xx that carries a parsed Binance {"code": ...} body
        # (and is not a timeout-ish 408/499) proves the request was refused. Anything else - a
        # 5XX, a proxy's HTML page, a redirect, an unparsable body - leaves the outcome unknown.
        refused = 400 <= status < 500 and code is not None and status not in (408, 499)
        transient = status >= 500 or status in (408, 499) or code in UNKNOWN_OUTCOME_CODES
        if transient or (not ep.safe_to_retry and not refused):
            if ep.safe_to_retry:
                raise _ServerError(f"{ep.name}: server error {status} code={code} {msg}")
            raise OutcomeUnknownError(
                f"{ep.name}: server answered {status} (code={code}); execution status unknown"
            )
        if code == -1021:
            raise ClockSkewError(f"{ep.name}: {msg}", code=code, http_status=status)
        raise APIError(f"{ep.name}: {msg or 'HTTP ' + str(status)}", code=code, http_status=status)

    def _call(self, name: str, params: Mapping[str, Any] | None = None) -> Any:
        ep = ENDPOINTS.get(name)
        if ep is None:
            raise EndpointNotAllowed(f"unknown endpoint {name!r}")
        self._check_allowed(ep)
        params = params or {}
        attempts = (self._max_retries + 1) if ep.safe_to_retry else 1
        resynced = False
        for attempt in range(attempts):
            self._check_cooldown()
            self._maybe_resync(ep)
            self._window.reserve(ep.weight)
            try:
                return self._send_once(ep, params)
            except ClockSkewError:
                if ep.security is Security.NONE or resynced:
                    raise
                resynced = True
                self.sync_time()  # heal the offset for the next request
                if not ep.safe_to_retry:
                    raise  # an order is never re-sent here; the caller decides what to do
            except (NetworkError, _ServerError) as exc:
                if attempt + 1 >= attempts:
                    if isinstance(exc, _ServerError):
                        raise ExchangeError(str(exc)) from None
                    raise
                delay = min(30.0, 0.5 * (2**attempt)) + self._rng.uniform(0, 0.25)
                log.warning(
                    "%s failed (%s); retry %d/%d in %.1fs",
                    name,
                    type(exc).__name__,
                    attempt + 1,
                    attempts - 1,
                    delay,
                )
                self._clock.sleep(delay)
        raise ExchangeError(f"{name}: exhausted retries")  # pragma: no cover

    # ------------------------------------------------------------------ public API

    def ping(self) -> None:
        self._call("ping")

    def server_time_ms(self) -> int:
        return int(self._call("time")["serverTime"])

    def sync_time(self, samples: int = 3) -> int:
        """Measure local-vs-server clock offset (ms) and use it for signed requests.

        Several samples are taken and the one with the smallest round trip wins (its midpoint
        assumption is the most accurate). Too large an offset is refused outright.
        """
        best: tuple[int, int] | None = None  # (round trip ms, offset ms)
        for _ in range(max(1, samples)):
            t0 = self._clock.now_ms()
            server = self.server_time_ms()
            t1 = self._clock.now_ms()
            candidate = (t1 - t0, server - (t0 + t1) // 2)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        self._offset_ms = best[1]
        self._last_sync = self._clock.monotonic()
        if abs(self._offset_ms) > self._max_skew_ms:
            raise ClockSkewError(
                f"local clock is off by {self._offset_ms} ms (limit {self._max_skew_ms} ms); "
                "fix NTP before trading"
            )
        return self._offset_ms

    def _maybe_resync(self, ep: Endpoint) -> None:
        """Refresh the clock offset every RESYNC_INTERVAL_S before a signed request."""
        if (
            ep.security is not Security.NONE
            and self._last_sync is not None
            and self._clock.monotonic() - self._last_sync > RESYNC_INTERVAL_S
        ):
            self.sync_time(samples=1)

    def exchange_info(self, symbol: str) -> dict[str, Any]:
        return dict(self._call("exchange_info", {"symbol": symbol}))

    def klines(
        self,
        symbol: str,
        interval: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list[Any]]:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        result = self._call("klines", params)
        return list(result)

    def book_ticker(self, symbol: str) -> dict[str, Any]:
        return dict(self._call("book_ticker", {"symbol": symbol}))

    # ------------------------------------------------------------------ signed reads

    def account(self) -> dict[str, Any]:
        return dict(self._call("account", {"omitZeroBalances": "true"}))

    def api_restrictions(self) -> dict[str, Any]:
        return dict(self._call("api_restrictions"))

    def get_order(
        self, symbol: str, *, client_order_id: str | None = None, order_id: int | None = None
    ) -> dict[str, Any]:
        if (client_order_id is None) == (order_id is None):
            raise ValueError("pass exactly one of client_order_id / order_id")
        params: dict[str, Any] = {"symbol": symbol}
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        else:
            params["orderId"] = order_id
        return dict(self._call("get_order", params))

    def open_orders(self, symbol: str) -> list[dict[str, Any]]:
        return list(self._call("open_orders", {"symbol": symbol}))

    def my_trades(self, symbol: str, order_id: int) -> list[dict[str, Any]]:
        """GET /api/v3/myTrades for ONE order (weight 5): the real per-trade fills."""
        return list(self._call("my_trades", {"symbol": symbol, "orderId": order_id}))

    # ------------------------------------------------------------------ trading (guarded)

    def new_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
        client_order_id: str,
        time_in_force: str = "GTC",
    ) -> dict[str, Any]:
        """POST /api/v3/order. Never retried; may raise :class:`OutcomeUnknownError`."""
        return dict(
            self._call(
                "new_order",
                _order_params(
                    symbol, side, order_type, quantity, price, client_order_id, time_in_force
                ),
            )
        )

    def test_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
        client_order_id: str,
        time_in_force: str = "GTC",
    ) -> dict[str, Any]:
        """POST /api/v3/order/test: validates the request, signature and permissions only.

        The order is NOT sent to the matching engine, so a success proves nothing about
        execution, fills, liquidity or the strategy. Needs a client built with
        ``order_test_enabled`` (or ``trading_enabled``).
        """
        return dict(
            self._call(
                "test_order",
                _order_params(
                    symbol, side, order_type, quantity, price, client_order_id, time_in_force
                ),
            )
        )

    def cancel_order(self, symbol: str, *, client_order_id: str) -> dict[str, Any]:
        return dict(
            self._call("cancel_order", {"symbol": symbol, "origClientOrderId": client_order_id})
        )


def _order_params(
    symbol: str,
    side: str,
    order_type: str,
    quantity: Decimal,
    price: Decimal | None,
    client_order_id: str,
    time_in_force: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": quantity,
        "newClientOrderId": client_order_id,
        "newOrderRespType": "FULL",
    }
    if order_type == "LIMIT":
        if price is None:
            raise ValueError("LIMIT order requires a price")
        params["price"] = price
        params["timeInForce"] = time_in_force
    return params


class _ServerError(ExchangeError):
    """Internal: a 5XX on a retry-safe endpoint."""


def _retry_after(resp: httpx.Response) -> float:
    raw = resp.headers.get("retry-after", "")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0
