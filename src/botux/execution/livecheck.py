"""``botux live-check``: a read-only rehearsal of the live connection. NO order is ever sent.

What it does, with the operator's own key (only when the operator runs it, and only after the
config says ``mode = "live"`` and ``[live].enabled = true``):

* measures the clock offset against Binance,
* reads the API key's restrictions and applies the same rules as the live gate,
* reads ``canTrade`` and the balances of the configured assets,
* reads the exchange rules and the open orders of the configured symbols,
* asks ``POST /api/v3/order/test`` to validate a minimal MARKET BUY for each symbol.

``order/test`` validates syntax, signature, permissions and filters ONLY. It never reaches the
matching engine, so a success proves nothing about execution, fills, price, liquidity or whether
a strategy works. The client used here cannot call ``POST /api/v3/order`` (trading is not enabled
on it), and ``live-check`` does not need the arming file because it cannot trade.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal

import httpx

from botux.clock import Clock, SystemClock
from botux.config import Settings, load_credentials
from botux.decimals import ceil_to_step
from botux.errors import BotuxError, LiveTradingNotAllowed
from botux.exchange.client import BinanceSpotClient
from botux.exchange.rules import SymbolRules
from botux.models import OrderType, Side
from botux.safety import api_key_problems
from botux.secrets_guard import GLOBAL_REDACTOR

TEST_ORDER_ID = "bxlivecheck"
NOTIONAL_HEADROOM = Decimal("1.2")


def run_live_check(
    settings: Settings,
    env: Mapping[str, str],
    *,
    http: httpx.Client | None = None,
    clock: Clock | None = None,
    out: Callable[[str], None] = print,
) -> int:
    """Returns 0 when every check passed, 1 otherwise. Raises before touching any key when the
    configuration does not select live mode."""
    if settings.mode != "live" or not settings.live.enabled:
        raise LiveTradingNotAllowed(
            "live-check refused: it contacts Binance with YOUR API key, so the config must say "
            'mode = "live" and [live].enabled = true first (no order is sent either way)'
        )
    clock = clock or SystemClock()
    credentials = load_credentials(env)
    client = BinanceSpotClient(
        market_data_base_url=settings.market.market_data_base_url,
        trading_base_url=settings.market.trading_base_url,
        credentials=credentials,
        trading_enabled=False,  # POST /order is impossible on this client
        order_test_enabled=True,  # ... only POST /order/test
        http=http,
        clock=clock,
        max_clock_skew_ms=settings.live.max_clock_skew_ms,
    )
    failures = 0

    def report(ok: bool, text: str) -> None:
        nonlocal failures
        if not ok:
            failures += 1
        out(f"[{'ok' if ok else 'FAIL'}] {GLOBAL_REDACTOR.scrub(text)}")

    def attempt(label: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except BotuxError as exc:
            report(False, f"{label}: {type(exc).__name__}: {exc}")

    try:
        out("live-check: read-only rehearsal. No order is sent to the matching engine.")
        attempt(
            "clock", lambda: report(True, f"clock offset {client.sync_time()} ms (within limit)")
        )

        def key_rules() -> None:
            problems = api_key_problems(
                client.api_restrictions(), settings.live.require_ip_restriction
            )
            if problems:
                for p in problems:
                    report(False, p)
            else:
                report(
                    True,
                    "API key restrictions satisfy the live gate (no withdrawal, transfer,"
                    " margin, futures or options; Spot trading + reading; IP restricted)",
                )

        attempt("API key restrictions", key_rules)

        assets = {settings.risk.quote_asset}

        def account() -> None:
            info = client.account()
            report(info.get("canTrade") is True, f"account canTrade = {info.get('canTrade')}")

        attempt("account", account)

        rules: dict[str, SymbolRules] = {}
        for sym in settings.market.symbols:

            def load(sym: str = sym) -> None:
                r = SymbolRules.from_exchange_info(client.exchange_info(sym), sym)
                rules[sym] = r
                assets.add(r.base_asset)
                ok = r.status == "TRADING" and r.spot_trading_allowed
                report(
                    ok, f"{sym}: status {r.status}, spot trading allowed = {r.spot_trading_allowed}"
                )
                if r.max_position:
                    report(
                        True,
                        f"{sym}: MAX_POSITION filter {r.max_position} {r.base_asset} (cap on your "
                        "TOTAL holding): the bot does not check it locally; the exchange rejects "
                        "a BUY beyond it",
                    )

            attempt(f"{sym} rules", load)

        def balances() -> None:
            wanted = {
                str(b["asset"]): (b["free"], b["locked"])
                for b in client.account().get("balances", [])
                if str(b["asset"]) in assets
            }
            for asset in sorted(assets):
                free, locked = wanted.get(asset, ("0", "0"))
                report(
                    True,
                    f"{asset}: free {free}, locked {locked} (the bot uses only risk.bot_capital)",
                )

        attempt("balances", balances)

        for sym, r in rules.items():

            def open_orders(sym: str = sym) -> None:
                n = len(client.open_orders(sym))
                report(
                    n == 0,
                    f"{sym}: {n} open order(s) on the account"
                    + (
                        ""
                        if n == 0
                        else " - the bot treats any order it did not place as foreign"
                        " and will not enter while they exist"
                    ),
                )

            attempt(f"{sym} open orders", open_orders)

            def rehearse(sym: str = sym, r: SymbolRules = r) -> None:
                ask = Decimal(client.book_ticker(sym)["askPrice"])
                qty = ceil_to_step(
                    max(r.min_qty, r.min_notional * NOTIONAL_HEADROOM / ask), r.step_size
                )
                problems = r.violations(Side.BUY, OrderType.MARKET, qty, None, ask)
                if problems:
                    report(
                        False,
                        f"{sym}: cannot build a valid rehearsal quantity: {'; '.join(problems)}",
                    )
                    return
                client.test_order(
                    symbol=sym,
                    side=Side.BUY.value,
                    order_type=OrderType.MARKET.value,
                    quantity=qty,
                    client_order_id=TEST_ORDER_ID,
                )
                report(True, f"{sym}: order/test accepted a MARKET BUY of {qty} (validation only)")

            attempt(f"{sym} order/test", rehearse)
    finally:
        client.close()
    out(
        "Reminder: order/test checks the request format, signature, permissions and filters. It "
        "does NOT prove that an order would be filled, at what price, or that any strategy works."
    )
    out("RESULT: " + ("all checks passed" if failures == 0 else f"{failures} check(s) FAILED"))
    return 0 if failures == 0 else 1
