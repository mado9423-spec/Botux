"""Randomised whole-loop test: whatever mix of deposits, withdrawals, foreign orders, ambiguous
replies, partial fills and price crashes happens, the loop never crashes, never sells coins that
belong to someone else, and never buys while an entry blocker is up."""

from __future__ import annotations

import contextlib
import dataclasses
from decimal import Decimal

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from botux.config import MarketSettings, Settings, StrategySettings
from botux.exchange.rules import SymbolRules
from botux.execution.paper import FaultKind
from botux.models import OrderRequest, OrderType, Side
from botux.storage import Store
from tests.conftest import EXCHANGE_INFO_BTCUSDT
from tests.test_runner import Harness

D = Decimal

EVENT = st.sampled_from(
    [
        "none", "none", "none", "deposit", "withdraw", "foreign_order", "cancel_foreign",
        "timeout_executed", "timeout_not_executed", "partial", "crash", "ack", "kill_free",
    ]
)  # fmt: skip


def _settings() -> Settings:
    import tempfile
    from pathlib import Path

    from botux.config import LiveSettings, PathSettings

    tmp = Path(tempfile.mkdtemp(prefix="botux-inv-"))
    return dataclasses.replace(
        Settings(),
        paths=PathSettings(
            kill_file=str(tmp / "KILL"), database=str(tmp / "db.sqlite3"),
            data_dir=str(tmp / "cache"), alerts_file=str(tmp / "alerts.jsonl"),
        ),
        live=LiveSettings(arm_file=str(tmp / "armed.json")),
    )  # fmt: skip


def wavy(n: int) -> list[float]:
    out, price = [], 100.0
    for i in range(n):
        price *= 1 + (0.012 if (i // 7) % 2 == 0 else -0.012)
        out.append(round(price, 2))
    return out


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(events=st.lists(EVENT, min_size=8, max_size=40))
def test_loop_invariants_under_random_events(events: list[str]) -> None:
    settings_fixture = _settings()
    rules = SymbolRules.from_exchange_info(EXCHANGE_INFO_BTCUSDT, "BTCUSDT")
    cfg = dataclasses.replace(
        settings_fixture,
        strategy=StrategySettings("sma_cross", {"fast": 2, "slow": 5}),
        market=MarketSettings(poll_seconds=1.0),
    )
    store = Store(":memory:")
    try:
        h = Harness(cfg, rules, store, closed=8, closes=wavy(len(events) + 20))
        external_btc = D(0)
        foreign_seq = 0
        h.cycle()  # warm start
        for ev in events:
            h.clock.advance(3600)
            h.feed.mark_to_last_close()
            if ev == "deposit":
                h.paper._free["BTC"] = h.paper._free.get("BTC", D(0)) + D("0.3")
                external_btc += D("0.3")
            elif ev == "withdraw":
                h.paper._free["USDT"] = max(D(0), h.paper._free.get("USDT", D(0)) - D(50))
            elif ev == "foreign_order":
                foreign_seq += 1
                with contextlib.suppress(Exception):  # the venue may refuse it; that is fine
                    h.paper.place_order(
                        OrderRequest(
                            f"web_{foreign_seq}",
                            "BTCUSDT",
                            Side.BUY,
                            OrderType.LIMIT,
                            D("0.1"),
                            D("50"),
                            h.feed.ask,
                        )
                    )
            elif ev == "cancel_foreign":
                for o in list(h.paper.open_orders("BTCUSDT")):
                    if o.client_order_id.startswith("web_"):
                        h.paper.cancel_order("BTCUSDT", o.client_order_id)
            elif ev == "timeout_executed":
                h.paper.faults.push(FaultKind.TIMEOUT_EXECUTED)
            elif ev == "timeout_not_executed":
                h.paper.faults.push(FaultKind.TIMEOUT_NOT_EXECUTED)
            elif ev == "partial":
                h.paper.fill_plan.push("0.5")
            elif ev == "crash":
                h.feed.bid, h.feed.ask = h.feed.bid * D("0.9"), h.feed.ask * D("0.9")
            elif ev == "ack":
                store.kv_set("ack:balances", "1")
            before_buys = sum(1 for o in h.orders() if o.side is Side.BUY)
            h.loop.run_cycle()  # invariant 1: never raises
            if h.loop.stopped:
                break
            after_buys = sum(1 for o in h.orders() if o.side is Side.BUY)
            if h.loop.risk.blockers and not store.kv_get("ack:balances"):
                # invariant 2: nothing is bought in a cycle that ended with an entry blocker up
                # (a blocker raised mid-cycle may follow a buy of that same cycle only if the
                # buy preceded the blocker, which the order of checks makes impossible)
                assert after_buys == before_buys, h.loop.risk.blockers
            # invariant 3: coins that someone else put on the account are never sold
            btc_total = h.paper.balances().get("BTC")
            assert btc_total is None or btc_total.total >= external_btc
    finally:
        store.close()
