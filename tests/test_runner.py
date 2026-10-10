"""The trading loop end to end (paper broker, fake feed): signals, stops, kill switch, errors."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import pytest

from botux.alerts import MemoryNotifier
from botux.app import App, build_paper_app
from botux.clock import FakeClock
from botux.config import MarketSettings, Settings, StrategySettings
from botux.errors import DataError
from botux.exchange.rules import SymbolRules
from botux.execution.order_manager import OrderIntent
from botux.execution.paper import FaultKind
from botux.killswitch import KillSwitch
from botux.models import Balance, Candle, Fill, OrderStatus, OrderType, Quote, Side
from botux.risk.manager import PortfolioView
from botux.storage import OrderRecord, Store
from botux.strategy.base import Signal
from tests.conftest import HOUR_MS, T0, FakeExchange, candles_from_closes

D = Decimal
SCENARIO = [
    100,
    100,
    100,
    100,
    101,
    103,
    104,
    104,
    100,
    96,
    96,
]  # BUY signal at idx 4, SELL at idx 8


class FakeFeed:
    """Serves the candles that are closed 'now' according to the fake clock."""

    def __init__(self, clock: FakeClock, candles: list[Candle]) -> None:
        self.clock, self.all = clock, candles
        self.bid, self.ask = D("99.99"), D("100.01")
        self.error: Exception | None = None

    def now_ms(self) -> int:
        return self.clock.now_ms()

    def closed_candles(self, symbol: str, count: int) -> list[Candle]:
        if self.error is not None:
            raise self.error
        return [c for c in self.all if c.close_time_ms < self.clock.now_ms()][-count:]

    def quote(self, symbol: str) -> Quote:
        return Quote(symbol, self.bid, self.ask, self.clock.now_ms())

    def mark_to_last_close(self) -> None:
        last = self.closed_candles("BTCUSDT", 1)[-1].close
        self.bid, self.ask = last - D("0.01"), last + D("0.01")


class Harness:
    def __init__(
        self,
        settings: Settings,
        rules: SymbolRules,
        store: Store,
        *,
        closed: int = 4,
        act_current: bool = False,
        closes: list[float] | None = None,
    ) -> None:
        self.clock = FakeClock(T0 + closed * HOUR_MS + 5_000)  # `closed` candles are closed now
        self.candles = candles_from_closes(closes or SCENARIO)
        self.feed = FakeFeed(self.clock, self.candles)
        self.feed.mark_to_last_close()
        self.http = FakeExchange()  # any accidental network request is recorded here
        self.app: App = build_paper_app(
            settings,
            http=self.http.http(),
            clock=self.clock,
            store=store,
            feed=self.feed,
            rules={"BTCUSDT": rules},
            act_on_current_candle=act_current,
        )
        self.loop = self.app.loop
        self.notifier = MemoryNotifier()
        self.loop._notifier = self.notifier
        self.loop._om._notifier = self.notifier
        self.store = store

    @property
    def paper(self):  # type: ignore[no-untyped-def]
        assert self.app.paper_broker is not None
        return self.app.paper_broker

    def cycle(self, hours: int = 0) -> None:
        self.clock.advance(hours * 3600)
        self.feed.mark_to_last_close()
        self.loop.run_cycle()

    def orders(self) -> list[OrderRecord]:
        return self.store.list_orders()


@pytest.fixture
def cfg(settings: Settings) -> Settings:
    return dataclasses.replace(
        settings,
        strategy=StrategySettings("sma_cross", {"fast": 2, "slow": 3}),
        market=MarketSettings(poll_seconds=1.0),
    )


def harness(cfg: Settings, rules: SymbolRules, store: Store, **kw) -> Harness:  # type: ignore[no-untyped-def]
    return Harness(cfg, rules, store, **kw)


class TestSignalsToOrders:
    def test_entry_then_exit_end_to_end(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()  # idx 0..3 closed: warm start, nothing happens
        assert h.orders() == []
        h.cycle(hours=1)  # idx 4 closed: fast crosses above slow -> BUY
        (buy,) = h.orders()
        assert buy.side is Side.BUY and buy.status is OrderStatus.FILLED
        assert buy.intent_key == f"sma_cross:BTCUSDT:{h.candles[4].open_time_ms}:BUY"
        assert buy.quantity * D("101.01") <= D(100)  # hard per-order cap
        info = h.loop.positions.get("BTCUSDT")
        assert info is not None and info.stop_price == info.entry_price * D("0.98")
        for _ in range(3):  # idx 5,6,7: still long, no new orders
            h.cycle(hours=1)
        assert len(h.orders()) == 1
        h.cycle(hours=1)  # idx 8 closed: fast < slow -> SELL
        sell = h.orders()[1]
        assert sell.side is Side.SELL and sell.status is OrderStatus.FILLED
        assert sell.intent_key == f"sma_cross:BTCUSDT:{h.candles[8].open_time_ms}:SELL"
        assert h.paper.balances()["BTC"].free < rules.min_qty  # only unsellable dust remains
        assert h.loop.positions.get("BTCUSDT") is None
        assert len(store.equity_curve()) >= 6
        assert h.http.requests == []  # no network at all
        assert "CRITICAL" not in {lvl for lvl, _, _ in h.notifier.items}

    def test_a_losing_round_trip_costs_money_never_creates_it(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        for hours in (0, 1, 1, 1, 1, 1):
            h.cycle(hours)
        bal = h.paper.balances()
        equity = bal["USDT"].total + bal["BTC"].total * D("100")
        assert equity < cfg.paper.starting_balance

    def test_warm_start_does_not_trade_on_a_signal_that_already_happened(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store, closed=5)  # idx 4 (the crossover candle) is already closed
        h.cycle()
        assert h.orders() == []

    def test_demo_flag_acts_on_the_current_candle(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store, closed=5, act_current=True)
        h.cycle()
        assert [o.side for o in h.orders()] == [Side.BUY]

    def test_each_candle_is_decided_at_most_once(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store, closed=5, act_current=True)
        for _ in range(4):
            h.cycle()  # same closed candle polled repeatedly
        assert len(h.orders()) == 1

    def test_strategy_is_evaluated_once_per_closed_candle(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store, closed=5, act_current=True)
        calls: list[int] = []
        inner = h.loop._strategy

        class Counting:
            name = inner.name
            warmup = inner.warmup

            def on_candles(self, candles: Sequence[Candle], in_position: bool) -> Signal:
                calls.append(candles[-1].open_time_ms)
                return inner.on_candles(candles, in_position)

        h.loop._strategy = Counting()
        for _ in range(5):
            h.cycle()  # the same candle polled again and again
        h.cycle(hours=1)  # a new candle closes
        assert calls == [h.candles[4].open_time_ms, h.candles[5].open_time_ms]

    def test_stop_loss_exits_when_the_bid_falls_through_the_stop(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.cycle(hours=1)
        assert h.loop.positions.get("BTCUSDT") is not None
        h.feed.bid, h.feed.ask = D("97.00"), D("97.01")  # far below the 2% stop (~99.04)
        h.loop.run_cycle()
        stop_orders = [o for o in h.orders() if o.intent_key.startswith("stop:BTCUSDT:")]
        assert len(stop_orders) == 1 and stop_orders[0].status is OrderStatus.FILLED
        assert h.loop.positions.get("BTCUSDT") is None

    def test_pre_existing_holdings_of_the_base_asset_are_never_touched(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.paper._free["BTC"] = D(1)  # the user already owns 1 BTC outside the bot
        for hours in (0, 1, 1, 1, 1, 1):
            h.cycle(hours)
        sells = [o for o in h.orders() if o.side is Side.SELL]
        assert sells and all(o.quantity < D(1) for o in sells)
        assert h.paper.balances()["BTC"].free >= D(1)  # the original BTC is still there


class TestRiskInTheLoop:
    def test_daily_loss_limit_halts_entries_and_alerts_once(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.cycle(hours=1)  # long
        h.feed.bid, h.feed.ask = D("40.00"), D("40.01")  # crash: equity drops ~60 USDT > 20 limit
        h.loop.run_cycle()
        h.loop.run_cycle()
        assert h.loop.risk.daily_loss_halted
        assert h.notifier.titles().count("Daily loss limit reached") == 1
        ok, _ = h.loop.risk.entries_allowed()
        assert ok is False

    def test_risk_state_is_persisted(self, cfg: Settings, rules: SymbolRules, store: Store) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        state = json.loads(store.kv_get("risk_state") or "{}")
        assert D(state["day_start_equity"]) == D(1000)


class TestEmergencyStop:
    def test_kill_file_cancels_open_orders_and_stops_the_bot(
        self, cfg: Settings, rules: SymbolRules, store: Store, tmp_path: Path
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        limit = OrderIntent(
            "L", "BTCUSDT", Side.BUY, OrderType.LIMIT, D("0.5"), D("100"), price=D("90.00")
        )
        assert h.loop.order_manager.submit(limit).outcome == "accepted"
        assert h.paper.balances()["USDT"].locked == D("45.00")
        KillSwitch(cfg.paths.kill_file).trip("operator pressed the button")  # e.g. `botux kill`
        h.loop.run_cycle()
        assert h.loop.stopped
        assert h.paper.balances()["USDT"].locked == 0 and h.loop.order_manager.open_orders() == []
        assert "EMERGENCY STOP" in h.notifier.titles()
        # nothing can be started afterwards
        assert (
            h.loop.order_manager.submit(
                OrderIntent("after", "BTCUSDT", Side.BUY, OrderType.MARKET, D("0.1"), D("100"))
            ).outcome
            == "blocked"
        )
        h.loop.run()  # returns immediately
        assert len([o for o in h.orders() if o.status is OrderStatus.FILLED]) == 0

    def test_flatten_on_kill_sells_the_position(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        cfg2 = dataclasses.replace(cfg, risk=dataclasses.replace(cfg.risk, flatten_on_kill=True))
        h = harness(cfg2, rules, store)
        h.cycle()
        h.cycle(hours=1)
        assert h.paper.balances()["BTC"].free > 0
        KillSwitch(cfg.paths.kill_file).trip("flatten please")
        h.loop.run_cycle()
        assert h.paper.balances()["BTC"].free < rules.min_qty
        assert any(o.intent_key.startswith("flatten:") for o in h.orders())

    def test_kill_without_flatten_leaves_positions_alone(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.cycle(hours=1)
        KillSwitch(cfg.paths.kill_file).trip("stop")
        h.loop.run_cycle()
        assert h.paper.balances()["BTC"].free > 0 and h.loop.stopped

    def test_graceful_stop_request(self, cfg: Settings, rules: SymbolRules, store: Store) -> None:
        h = harness(cfg, rules, store)
        h.loop.request_stop()
        h.loop.run()
        assert h.loop.stopped and h.loop.cycles == 0
        assert [e["kind"] for e in store.events()] == ["shutdown"]

    def test_max_cycles_and_polling_pause(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.loop.run(max_cycles=3)
        assert h.loop.cycles == 3 and sum(h.clock.sleeps) >= 2.0


class TestErrorHandling:
    def test_consecutive_errors_trip_the_kill_switch(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        cfg2 = dataclasses.replace(
            cfg, risk=dataclasses.replace(cfg.risk, max_consecutive_errors=3)
        )
        h = harness(cfg2, rules, store)
        h.feed.error = DataError("feed down")
        for _ in range(3):
            h.loop.run_cycle()
        assert h.app.kill.is_set() and "Trading halted" in h.notifier.titles()
        h.loop.run_cycle()
        assert h.loop.stopped

    def test_a_successful_cycle_resets_the_error_counter(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        cfg2 = dataclasses.replace(
            cfg, risk=dataclasses.replace(cfg.risk, max_consecutive_errors=3)
        )
        h = harness(cfg2, rules, store)
        for _ in range(2):
            h.feed.error = DataError("blip")
            h.loop.run_cycle()
            h.loop.run_cycle()
            h.feed.error = None
            h.loop.run_cycle()
        assert not h.app.kill.is_set()

    def test_stale_market_data_blocks_trading(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.clock.advance(6 * 3600)  # no new closed candle for 6 hours
        h.feed.all = h.feed.all[:4]
        h.feed.mark_to_last_close()
        h.loop.run_cycle()
        assert h.loop.risk.consecutive_errors == 1 and h.orders() == []
        assert any(e["kind"] == "cycle_error" and "stale" in e["detail"] for e in store.events())


class TestUnconfirmedOrdersInTheLoop:
    def test_unknown_outcome_blocks_the_symbol_then_resolves_without_resending(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.paper.faults.push(FaultKind.TIMEOUT_NOT_EXECUTED)
        h.cycle(hours=1)  # BUY signal; the venue times out and nothing was executed
        (buy,) = h.orders()
        assert buy.status in (OrderStatus.UNKNOWN, OrderStatus.LOST)
        for _ in range(4):
            h.loop.run_cycle()  # reconciliation passes; never a second BUY
        (buy,) = h.orders()
        assert buy.status is OrderStatus.LOST and h.paper.placed_count == 0
        assert "Order LOST" in h.notifier.titles()
        # the symbol stays blocked: a later signal does not trade
        for _ in range(3):
            h.cycle(hours=1)
        assert len(h.orders()) == 1 and h.paper.placed_count == 0

    def test_unknown_but_executed_order_is_adopted_and_the_position_tracked(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.paper.faults.push(FaultKind.TIMEOUT_EXECUTED)
        h.cycle(hours=1)
        (buy,) = h.orders()
        assert buy.status is OrderStatus.FILLED and h.paper.placed_count == 1
        assert h.loop.positions.get("BTCUSDT") is not None


def uncertain_buy(store: Store, now_ms: int) -> None:
    """An unresolved BUY on the symbol: blocks entries (but must never block exits)."""
    store.insert_order(
        OrderRecord(
            "unresolved1",
            "k-unresolved",
            "BTCUSDT",
            Side.BUY,
            OrderType.MARKET,
            D("0.1"),
            None,
            OrderStatus.UNKNOWN,
            now_ms,
            now_ms,
        )
    )


def enter_long(h: Harness) -> None:
    h.cycle()
    h.cycle(hours=1)  # candle 4 closes -> BUY
    assert h.paper.balances()["BTC"].free > 0


class TestProtectionNeverDependsOnTheRestOfTheLoop:
    def test_stop_loss_fires_while_an_unresolved_buy_blocks_entries(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        enter_long(h)
        uncertain_buy(store, h.clock.now_ms())
        h.feed.bid, h.feed.ask = D("90.00"), D("90.01")
        h.loop.run_cycle()
        assert any(
            o.intent_key.startswith("stop:") and o.status is OrderStatus.FILLED for o in h.orders()
        )
        assert h.paper.balances()["BTC"].free < rules.min_qty

    def test_stop_loss_fires_even_when_the_candle_feed_is_broken(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        enter_long(h)
        h.feed.error = DataError("klines unavailable")
        h.feed.bid, h.feed.ask = D("90.00"), D("90.01")
        h.loop.run_cycle()  # the cycle is reported as failed ...
        assert h.loop.risk.consecutive_errors == 1
        assert any(o.intent_key.startswith("stop:") for o in h.orders())  # ... but the stop ran
        assert h.paper.balances()["BTC"].free < rules.min_qty

    def test_flatten_on_kill_works_with_an_unresolved_buy(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        cfg2 = dataclasses.replace(cfg, risk=dataclasses.replace(cfg.risk, flatten_on_kill=True))
        h = harness(cfg2, rules, store)
        enter_long(h)
        uncertain_buy(store, h.clock.now_ms())
        KillSwitch(cfg.paths.kill_file).trip("flatten")
        h.loop.run_cycle()
        assert h.paper.balances()["BTC"].free < rules.min_qty and h.loop.stopped

    def test_a_garbage_kill_file_still_stops_the_bot(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        cfg2 = dataclasses.replace(cfg, risk=dataclasses.replace(cfg.risk, flatten_on_kill=True))
        h = harness(cfg2, rules, store)
        enter_long(h)
        Path(cfg.paths.kill_file).write_text("1")  # `echo 1 > KILL`
        h.loop.run_cycle()
        assert h.loop.stopped and h.paper.balances()["BTC"].free < rules.min_qty

    def test_a_position_without_recorded_entry_gets_a_stop_re_armed(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        enter_long(h)
        h.loop.positions.clear("BTCUSDT")  # e.g. lost between the fill and its bookkeeping
        h.loop.run_cycle()
        assert h.loop.positions.get("BTCUSDT") is not None
        assert "Stop-loss re-armed" in h.notifier.titles()

    def test_failed_exits_raise_a_critical_alert_without_spamming(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        enter_long(h)
        h.paper.faults.push(FaultKind.REJECT, times=3)
        h.feed.bid, h.feed.ask = D("90.00"), D("90.01")
        h.loop.run_cycle()
        h.clock.advance(61)  # next minute bucket -> a fresh stop order is attempted
        h.loop.run_cycle()
        assert h.notifier.titles().count("Exit order did not go through") == 1  # throttled
        assert h.paper.balances()["BTC"].free > 0  # still long: the alert is the safety net

    def test_error_driven_halt_with_an_open_position_says_so(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        cfg2 = dataclasses.replace(
            cfg, risk=dataclasses.replace(cfg.risk, max_consecutive_errors=2)
        )
        h = harness(cfg2, rules, store)
        enter_long(h)
        h.feed.error = DataError("down")
        h.loop.run_cycle()
        h.loop.run_cycle()
        msg = next(m for _, t, m in h.notifier.items if t == "Trading halted")
        assert "WITHOUT STOP PROTECTION" in msg and "BTCUSDT" in msg


class TestBookkeepingAndBaseline:
    def test_locked_coins_of_the_user_are_never_sold(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.paper._free["BTC"] = D("0.5")
        h.paper._locked["BTC"] = D("0.5")  # 0.5 BTC sit in the user's own resting order
        enter_long(h)
        assert h.loop.positions.get("BTCUSDT") is not None
        h.paper._free["BTC"] += D("0.5")  # the user cancels his order: coins become free
        h.paper._locked["BTC"] = D(0)
        for _ in range(4):
            h.cycle(hours=1)  # the strategy exit happens at candle 8
        sells = [o for o in h.orders() if o.side is Side.SELL]
        assert sells and all(o.quantity < D("1") for o in sells)  # only the bot's own ~0.99 BTC
        assert h.paper.balances()["BTC"].free >= D(1)  # the user's 1 BTC is untouched

    def test_the_baseline_survives_a_restart_and_still_includes_locked_coins(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        balances = {"BTC": Balance(free=D("0.5"), locked=D("0.5"))}
        assert h.loop._position_qty("BTCUSDT", balances) == 0  # records the baseline (1.0 total)
        assert store.kv_get("baseline:BTC") == "1.0"
        h.loop._baseline.clear()  # a new process starts: the baseline must be read back from disk
        assert h.loop._position_qty("BTCUSDT", balances) == 0
        # only coins above the baseline are the bot's, and only FREE ones can be sold
        more = {"BTC": Balance(free=D("1.2"), locked=D("0.5"))}
        assert h.loop._position_qty("BTCUSDT", more) == D("0.7")
        locked_up = {"BTC": Balance(free=D("0.1"), locked=D("1.6"))}
        assert h.loop._position_qty("BTCUSDT", locked_up) == D("0.1")

    def test_entry_price_is_the_volume_weighted_average_of_all_fills(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        rec = OrderRecord(
            "c1",
            "k1",
            "BTCUSDT",
            Side.BUY,
            OrderType.MARKET,
            D("2"),
            None,
            OrderStatus.FILLED,
            0,
            0,
            executed_qty=D("2"),
            cumulative_quote_qty=D("210"),
        )
        h.loop.on_fill(
            rec, [Fill(D("100"), D("1"), D(0), "BTC"), Fill(D("110"), D("1"), D(0), "BTC")]
        )
        info = h.loop.positions.get("BTCUSDT")
        assert info is not None and info.entry_price == D("105")  # not 100 (the first fill)

    def test_a_partial_sell_keeps_the_stop_on_the_remainder(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        enter_long(h)
        rec = OrderRecord(
            "c2",
            "k2",
            "BTCUSDT",
            Side.SELL,
            OrderType.MARKET,
            D("0.9"),
            None,
            OrderStatus.EXPIRED,
            0,
            0,
            executed_qty=D("0.4"),
            cumulative_quote_qty=D("40"),
        )
        h.loop.on_fill(rec, [Fill(D("100"), D("0.4"), D(0), "USDT")])
        assert h.loop.positions.get("BTCUSDT") is not None  # most of the position is still open

    def test_a_stale_kline_answer_does_not_trigger_trades(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        enter_long(h)
        n = len(h.orders())
        h.feed.all = h.candles[:4]  # one poll returns an OLDER set of candles (candle 3 is last)
        h.feed.mark_to_last_close()
        h.clock.advance(60)
        h.loop.run_cycle()
        assert len(h.orders()) == n  # no spurious SELL decided on old data

    def test_a_provably_unsent_entry_is_retried_on_the_same_candle(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.paper.faults.push(FaultKind.NETWORK_DOWN)
        h.cycle(hours=1)  # BUY signal, but the network is down: nothing was sent
        assert h.orders() == [] and h.paper.placed_count == 0
        h.loop.run_cycle()  # same candle again: the decision was not burned
        assert [o.side for o in h.orders()] == [Side.BUY] and h.paper.placed_count == 1

    def test_retries_per_candle_are_bounded(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.paper.faults.push(FaultKind.NETWORK_DOWN, times=10)
        h.cycle(hours=1)
        for _ in range(6):
            h.loop.run_cycle()
        assert h.orders() == [] and h.paper.placed_count == 0
        assert (
            len(h.paper.faults._queue) == 10 - 3
        )  # attempted exactly MAX_RETRIES_PER_CANDLE times


class TestLiveGateWhileRunning:
    def test_closing_the_gate_halts_entries_but_keeps_exits(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        enter_long(h)
        from botux.errors import LiveTradingNotAllowed

        def closed() -> None:
            raise LiveTradingNotAllowed("arming expired")

        h.loop._gate_check = closed
        h.loop.run_cycle()
        assert h.loop.risk.hard_halt_reason and "arming expired" in h.loop.risk.hard_halt_reason
        assert "Live gate closed" in h.notifier.titles()
        # entries are impossible now ...
        assert h.loop.risk.size_entry(rules, D(100), _view()).quantity == 0
        # ... yet an existing position is still protected
        h.feed.bid, h.feed.ask = D("90.00"), D("90.01")
        h.loop.run_cycle()
        assert any(o.intent_key.startswith("stop:") for o in h.orders())
        h.loop.run_cycle()
        assert h.notifier.titles().count("Live gate closed") == 1


def _view() -> PortfolioView:
    return PortfolioView(D(1000), D(1000), {"BTCUSDT": D(0)}, {"BTCUSDT": D(0)})
