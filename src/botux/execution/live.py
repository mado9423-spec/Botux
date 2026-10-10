"""Guarded live broker (EXPERIMENTAL: never exercised against the real exchange by its author).

It can only be constructed through :func:`build_live_broker`, which runs the whole safety gate
before an API key is even read. Nothing in this module is reachable in paper mode.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from botux.clock import Clock, SystemClock
from botux.config import Settings, load_credentials
from botux.decimals import ZERO, D
from botux.errors import (
    APIError,
    ClockSkewError,
    ExchangeError,
    LiveTradingNotAllowed,
    OrderRejected,
    OutcomeUnknownError,
)
from botux.exchange.client import BinanceSpotClient
from botux.exchange.rules import SymbolRules
from botux.killswitch import KillSwitch
from botux.models import (
    Balance,
    Fill,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Side,
)
from botux.safety import LiveAuthorization, api_key_problems, check_live_gate

log = logging.getLogger(__name__)

_STATUS_MAP = {
    "NEW": OrderStatus.NEW,
    "PENDING_NEW": OrderStatus.NEW,
    "PENDING_CANCEL": OrderStatus.NEW,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED,
    "EXPIRED_IN_MATCH": OrderStatus.EXPIRED,
}

NO_SUCH_ORDER = -2013
CANCEL_REJECTED = -2011


def parse_order(payload: Mapping[str, Any]) -> OrderResult:
    """Map a Binance order JSON (ACK/RESULT/FULL, GET /order, DELETE /order) to OrderResult."""
    raw_price = payload.get("price")
    price = D(raw_price) if raw_price not in (None, "") else None
    if price is not None and price <= 0:
        price = None  # MARKET orders report 0.00000000
    fills = [
        Fill(
            D(f["price"]),
            D(f["qty"]),
            D(f.get("commission", "0")),
            str(f.get("commissionAsset", "")),
        )
        for f in payload.get("fills", [])
    ]
    status = _STATUS_MAP.get(str(payload.get("status", "")), OrderStatus.UNKNOWN)
    cumulative = D(payload.get("cummulativeQuoteQty", "0"))  # sic: Binance's spelling
    if cumulative < 0:
        cumulative = ZERO  # docs: negative means "not available" for some historical orders
    order_type = str(payload.get("type", ""))
    # For a cancel response `clientOrderId` is the CANCEL's own id; `origClientOrderId` is ours.
    client_id = payload.get("origClientOrderId") or payload.get("clientOrderId")
    return OrderResult(
        client_order_id=str(client_id),
        symbol=str(payload["symbol"]),
        side=Side(payload["side"]),
        type=OrderType.LIMIT if order_type.startswith("LIMIT") else OrderType.MARKET,
        status=status,
        orig_qty=D(payload.get("origQty", "0")),
        executed_qty=D(payload.get("executedQty", "0")),
        cumulative_quote_qty=cumulative,
        price=price,
        exchange_order_id=None if payload.get("orderId") is None else str(payload["orderId"]),
        transact_time_ms=int(payload.get("transactTime") or payload.get("updateTime") or 0),
        fills=fills,
    )


_PARSE_ERRORS = (KeyError, ValueError, TypeError, InvalidOperation)


class LiveBroker:
    def __init__(
        self,
        client: BinanceSpotClient,
        rules: Mapping[str, SymbolRules],
        authorization: LiveAuthorization,
    ) -> None:
        if not isinstance(authorization, LiveAuthorization):
            raise LiveTradingNotAllowed("a LiveAuthorization from the safety gate is required")
        if not client.trading_enabled:
            raise LiveTradingNotAllowed("client was not created with trading enabled")
        self._client = client
        self._rules = dict(rules)
        self.account_uid: str | None = None  # set by build_live_broker from GET /api/v3/account

    def place_order(self, request: OrderRequest) -> OrderResult:
        rules = self._rules.get(request.symbol)
        if rules is None:
            raise OrderRejected(f"no rules loaded for {request.symbol}")
        problems = rules.violations(
            request.side, request.type, request.quantity, request.price, request.reference_price
        )
        if problems:
            raise OrderRejected("; ".join(problems))
        try:
            payload = self._client.new_order(
                symbol=request.symbol,
                side=request.side.value,
                order_type=request.type.value,
                quantity=request.quantity,
                price=request.price,
                client_order_id=request.client_order_id,
            )
        except ClockSkewError:
            raise  # rejected before the matching engine; the order manager may retry later
        except APIError as exc:  # definitive 4XX answer from the API: not executed
            raise OrderRejected(f"{exc.code}: {exc}", code=exc.code) from exc
        try:
            return parse_order(payload)
        except _PARSE_ERRORS as exc:
            # The venue answered but we cannot read it: the order may well exist.
            raise OutcomeUnknownError(
                f"unreadable order response ({type(exc).__name__}); execution status unknown"
            ) from None

    def get_order(self, symbol: str, client_order_id: str) -> OrderResult | None:
        try:
            return parse_order(self._client.get_order(symbol, client_order_id=client_order_id))
        except APIError as exc:
            if exc.code == NO_SUCH_ORDER:
                return None
            raise
        except _PARSE_ERRORS as exc:
            raise ExchangeError(f"unreadable order answer ({type(exc).__name__})") from None

    def cancel_order(self, symbol: str, client_order_id: str) -> OrderResult | None:
        try:
            return parse_order(self._client.cancel_order(symbol, client_order_id=client_order_id))
        except APIError as exc:
            if exc.code in (CANCEL_REJECTED, NO_SUCH_ORDER):
                return None
            raise
        except _PARSE_ERRORS as exc:
            raise ExchangeError(f"unreadable cancel answer ({type(exc).__name__})") from None

    def open_orders(self, symbol: str) -> list[OrderResult]:
        try:
            return [parse_order(o) for o in self._client.open_orders(symbol)]
        except _PARSE_ERRORS as exc:
            raise ExchangeError(f"unreadable open-orders answer ({type(exc).__name__})") from None

    def update_rules(self, rules: Mapping[str, SymbolRules]) -> None:
        self._rules.update(rules)

    def order_fills(self, symbol: str, exchange_order_id: str) -> list[Fill]:
        try:
            trades = self._client.my_trades(symbol, int(exchange_order_id))
            ordered = sorted(trades, key=lambda t: int(t["id"]))
            return [
                Fill(
                    D(t["price"]),
                    D(t["qty"]),
                    D(t["commission"]),
                    str(t["commissionAsset"]),
                )
                for t in ordered
            ]
        except (*_PARSE_ERRORS, ValueError) as exc:
            raise ExchangeError(f"unreadable trades answer ({type(exc).__name__})") from None

    def balances(self) -> dict[str, Balance]:
        out: dict[str, Balance] = {}
        for b in self._client.account().get("balances", []):
            free, locked = D(b["free"]), D(b["locked"])
            if free or locked:
                out[str(b["asset"])] = Balance(free, locked)
        return out


def build_live_broker(
    settings: Settings,
    env: Mapping[str, str],
    kill: KillSwitch,
    *,
    clock: Clock | None = None,
    http: httpx.Client | None = None,
) -> tuple[LiveBroker, BinanceSpotClient]:
    """Run the full gate, THEN read the key, THEN verify the key's permissions.

    Order matters: in paper mode, or whenever any gate condition fails, this raises before
    ``load_credentials`` is called, so keys sitting in the environment are never even read.
    """
    clock = clock or SystemClock()
    auth = check_live_gate(settings, kill, clock.now_ms())  # raises LiveTradingNotAllowed
    credentials = load_credentials(env)
    client = BinanceSpotClient(
        market_data_base_url=settings.market.market_data_base_url,
        trading_base_url=settings.market.trading_base_url,
        credentials=credentials,
        trading_enabled=True,
        http=http,
        clock=clock,
        max_clock_skew_ms=settings.live.max_clock_skew_ms,
    )
    client.sync_time()
    problems = api_key_problems(client.api_restrictions(), settings.live.require_ip_restriction)
    if problems:
        client.close()
        raise LiveTradingNotAllowed("API key refused: " + "; ".join(problems))
    account = client.account()
    if account.get("canTrade") is not True:
        client.close()
        raise LiveTradingNotAllowed("the account reports canTrade != true; refusing to start")
    rules = {
        sym: SymbolRules.from_exchange_info(client.exchange_info(sym), sym)
        for sym in settings.market.symbols
    }
    log.warning("LIVE TRADING AUTHORISED for %s", ", ".join(settings.market.symbols))
    broker = LiveBroker(client, rules, auth)
    raw_uid = account.get("uid")
    broker.account_uid = None if raw_uid is None else str(raw_uid)
    return broker, client


def total_balance(balances: Mapping[str, Balance], asset: str) -> Decimal:
    b = balances.get(asset)
    return ZERO if b is None else b.total
