"""Exchange filters and symbol status are re-read while the bot runs."""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from botux.app import build_paper_app
from botux.clock import FakeClock
from botux.config import MarketSettings, Settings, StrategySettings, parse_settings
from botux.errors import ConfigError, ExchangeError
from botux.exchange.rules import SymbolRules
from botux.execution.order_manager import OrderIntent
from botux.models import OrderType, Side
from botux.storage import Store
from tests.conftest import EXCHANGE_INFO_BTCUSDT, FakeExchange
from tests.test_runner import Harness, harness

D = Decimal
HOURS = 3600


@pytest.fixture
def cfg(settings: Settings) -> Settings:
    return dataclasses.replace(
        settings,
        strategy=StrategySettings("sma_cross", {"fast": 2, "slow": 3}),
        market=MarketSettings(poll_seconds=1.0, rules_refresh_hours=6.0),
    )


class Loader:
    def __init__(self, rules: SymbolRules) -> None:
        self.rules = rules
        self.calls = 0
        self.error: Exception | None = None

    def __call__(self) -> dict[str, SymbolRules]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {"BTCUSDT": self.rules}


def setup(h: Harness, rules: SymbolRules) -> Loader:
    loader = Loader(rules)
    h.loop._rules_loader = loader
    return loader


class TestRefresh:
    def test_nothing_happens_before_the_interval_and_nothing_changes_when_equal(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        loader = setup(h, rules)
        h.cycle()
        assert loader.calls == 0  # not due yet
        h.clock.advance(6 * HOURS + 1)
        h.feed.mark_to_last_close()
        h.loop.run_cycle()
        assert loader.calls == 1
        assert (
            store.events("rules_changed") == []
            and "Exchange rules changed" not in h.notifier.titles()
        )

    def test_changed_filters_reach_every_component(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        loader = setup(h, dataclasses.replace(rules, step_size=D("0.01"), min_qty=D("0.01")))
        h.clock.advance(6 * HOURS + 1)
        h.feed.mark_to_last_close()
        h.loop.run_cycle()
        assert "Exchange rules changed" in h.notifier.titles()
        (event,) = store.events("rules_changed")
        assert "BTCUSDT" in event["detail"]
        # the order manager now validates against the NEW lot size
        res = h.loop.order_manager.submit(
            OrderIntent("kk", "BTCUSDT", Side.BUY, OrderType.MARKET, D("0.123"), D("100.01"))
        )
        assert res.outcome == "blocked" and "LOT_SIZE" in res.reason
        assert h.paper._rules["BTCUSDT"].step_size == D("0.01")
        assert h.loop._ledger._rules["BTCUSDT"].step_size == D("0.01")
        assert loader.calls == 1
        h.loop.run_cycle()  # next refresh only after another interval
        assert loader.calls == 1

    def test_a_failing_refresh_is_retried_soon_and_goes_stale_after_two_intervals(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        loader = setup(h, rules)
        loader.error = ExchangeError("exchange unreachable")
        h.clock.advance(6 * HOURS + 1)
        h.feed.mark_to_last_close()
        h.loop.run_cycle()
        assert loader.calls == 1 and "rules-stale" not in h.loop.risk.blockers
        h.loop.run_cycle()
        assert loader.calls == 1  # retried only every few minutes, not every cycle
        h.clock.advance(301)
        h.loop.run_cycle()
        assert loader.calls == 2
        h.clock.advance(7 * HOURS)  # now older than two intervals
        h.feed.mark_to_last_close()
        h.loop.run_cycle()
        assert "rules-stale" in h.loop.risk.blockers
        assert "Exchange rules are stale" in h.notifier.titles()
        loader.error = None  # the exchange is back
        h.clock.advance(301)
        h.feed.mark_to_last_close()
        h.loop.run_cycle()
        assert "rules-stale" not in h.loop.risk.blockers

    def test_a_symbol_that_stops_trading_blocks_entries_until_it_resumes(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        loader = setup(h, dataclasses.replace(rules, status="BREAK"))
        h.clock.advance(6 * HOURS + 1)
        h.feed.mark_to_last_close()
        h.loop.run_cycle()
        assert "symbol-status:BTCUSDT" in h.loop.risk.blockers
        assert "Symbol not tradable" in h.notifier.titles()
        loader.rules = rules
        h.clock.advance(6 * HOURS + 1)
        h.feed.mark_to_last_close()
        h.loop.run_cycle()
        assert "symbol-status:BTCUSDT" not in h.loop.risk.blockers


class TestWiring:
    def build(self, settings: Settings, clock: FakeClock, **kw: object):  # type: ignore[no-untyped-def]
        ex = FakeExchange()
        ex.on("GET", "/api/v3/exchangeInfo", EXCHANGE_INFO_BTCUSDT)
        ex.on("GET", "/api/v3/time", {"serverTime": clock.now_ms()})
        return build_paper_app(settings, http=ex.http(), clock=clock, store=Store(":memory:"), **kw)  # type: ignore[arg-type]

    def test_rules_read_from_the_exchange_are_refreshed_but_injected_ones_are_not(
        self, settings: Settings, clock: FakeClock, rules: SymbolRules
    ) -> None:
        class Feed:
            def now_ms(self) -> int:
                return clock.now_ms()

        fetched = self.build(settings, clock, feed=Feed())
        assert fetched.loop._rules_loader is not None
        fetched.store.close()
        injected = self.build(settings, clock, feed=Feed(), rules={"BTCUSDT": rules})
        assert injected.loop._rules_loader is None
        injected.store.close()

    def test_the_refresh_interval_is_validated(self) -> None:
        for bad in (0, 0.01, 1000):
            with pytest.raises(ConfigError, match="rules_refresh_hours"):
                parse_settings({"market": {"rules_refresh_hours": bad}})
        assert (
            parse_settings({"market": {"rules_refresh_hours": 2}}).market.rules_refresh_hours == 2
        )
