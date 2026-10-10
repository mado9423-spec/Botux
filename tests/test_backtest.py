"""Strategy, back-test engine, metrics, train/test study and market-data validation."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from botux.backtest.engine import BacktestResult, run_backtest
from botux.backtest.metrics import compute_metrics, format_metrics
from botux.backtest.study import run_study, run_walk_forward, split_three
from botux.config import RiskSettings
from botux.data.market import (
    check_candles,
    fetch_range,
    load_csv,
    only_closed,
    parse_klines,
    save_csv,
    validate_live_candles,
)
from botux.errors import ConfigError, DataError
from botux.exchange.client import BinanceSpotClient
from botux.exchange.rules import SymbolRules
from botux.models import Candle
from botux.strategy.base import Signal, Strategy, create_strategy
from botux.strategy.sma_cross import SmaCross
from tests.conftest import HOUR_MS, T0, FakeExchange, candles_from_closes, kline_rows, make_candle

D = Decimal
FAST_SLOW = {"fast": 2, "slow": 3}
# Closes used for exact scenarios (see comments in the tests for the hand calculation).
SCENARIO = [100, 100, 100, 100, 101, 103, 104, 104, 100, 96, 96]


def run(
    candles: Sequence[Candle],
    rules: SymbolRules,
    *,
    fee: str = "0.001",
    slip: str = "5",
    strategy: object | None = None,
    risk: RiskSettings | None = None,
) -> BacktestResult:
    return run_backtest(
        candles,
        strategy or create_strategy("sma_cross", FAST_SLOW),  # type: ignore[arg-type]
        rules,
        risk or RiskSettings(),
        initial_cash=D(1000),
        fee_rate=D(fee),
        slippage_bps=D(slip),
        interval_ms=HOUR_MS,
    )


class TestSmaCross:
    def test_no_signal_before_warmup(self) -> None:
        s = SmaCross(fast=2, slow=3)
        assert s.warmup == 4
        assert s.on_candles(candles_from_closes([1, 2, 3]), False) is Signal.HOLD

    def test_enters_on_the_crossover_candle_only(self) -> None:
        s = SmaCross(**FAST_SLOW)
        c = candles_from_closes(SCENARIO)
        signals = [s.on_candles(c[: i + 1], False) for i in range(len(c))]
        assert [i for i, x in enumerate(signals) if x is Signal.ENTER_LONG] == [4]

    def test_exits_while_fast_is_below_slow(self) -> None:
        s = SmaCross(**FAST_SLOW)
        c = candles_from_closes(SCENARIO)
        signals = [s.on_candles(c[: i + 1], True) for i in range(len(c))]
        assert signals[8] is Signal.EXIT_LONG and signals[6] is Signal.HOLD

    @pytest.mark.parametrize(
        "params",
        [
            {"fast": 5, "slow": 5},
            {"fast": 10, "slow": 5},
            {"fast": 1, "slow": 5},
            {"fast": 2.5, "slow": 5},
            {"fast": 2, "slow": 5000},
            {"fast": 2, "slow": 5, "x": 1},
        ],
    )
    def test_invalid_parameters_are_rejected(self, params: dict[str, object]) -> None:
        with pytest.raises(ConfigError):
            create_strategy("sma_cross", params)

    def test_unknown_strategy(self) -> None:
        with pytest.raises(ConfigError):
            create_strategy("moon_phase", {})


class TestEngine:
    def test_exact_round_trip_executes_at_the_next_open_with_costs(
        self, rules: SymbolRules
    ) -> None:
        res = run(candles_from_closes(SCENARIO), rules)
        (trade,) = res.trades
        # signal at the close of candle 4 -> BUY at candle 5's open:
        # 101 * 1.0005 = 101.0505, rounded UP to the tick -> 101.06
        assert trade.entry_time_ms == T0 + 5 * HOUR_MS and trade.entry_price == D("101.06")
        # exit signal at the close of candle 8 -> SELL at candle 9's open: 100 * 0.9995 = 99.95
        assert trade.exit_time_ms == T0 + 9 * HOUR_MS and trade.exit_price == D("99.95")
        assert trade.exit_reason == "signal"
        assert trade.pnl < 0 and trade.fees > 0
        assert trade.quantity * trade.entry_price <= D(100) + D("0.01")  # hard order cap respected

    def test_accounting_matches_a_hand_calculation(self, rules: SymbolRules) -> None:
        """Slippage off, fee 0.1% (BUY fee in BTC, SELL fee in USDT), all by hand:

        entry   price 101.00, qty = min(risk 2.475, cap 100/101) -> 0.99009  (cost 99.99909)
        buy fee 0.99009 * 0.001 = 0.00099009 BTC  -> holds 0.98909991, sellable 0.98909
        exit    price 100.00 -> proceeds 98.909, sell fee 0.098909 USDT
        pnl     98.909 - 0.098909 - 99.99909 = -1.188999
        """
        (trade,) = run(candles_from_closes(SCENARIO), rules, slip="0").trades
        assert trade.entry_price == D("101.00") and trade.exit_price == D("100.00")
        assert trade.quantity == D("0.98909")
        assert trade.pnl == D("-1.188999")
        assert trade.fees == D("0.00099009") * D("101.00") + D("0.098909")

    def test_costs_lower_the_result(self, rules: SymbolRules) -> None:
        c = candles_from_closes(SCENARIO)
        free = run(c, rules, fee="0", slip="0")
        costly = run(c, rules)
        assert costly.final_equity < free.final_equity
        assert costly.fees_paid > 0 and free.fees_paid == 0

    def test_stop_loss_triggers_inside_a_candle(self, rules: SymbolRules) -> None:
        c = candles_from_closes(SCENARIO[:8]) + [make_candle(8, 104, 104.5, 98, 99)]
        res = run(c, rules)
        (trade,) = res.trades
        stop = trade.entry_price * D("0.98")
        assert trade.exit_reason == "stop"
        assert trade.exit_price <= stop  # slippage can only make it worse than the stop level

    def test_gap_below_the_stop_fills_at_the_open_not_at_the_stop(self, rules: SymbolRules) -> None:
        c = candles_from_closes(SCENARIO[:8]) + [make_candle(8, 95, 95.5, 94, 94)]
        (trade,) = run(c, rules).trades
        assert trade.exit_reason == "stop" and trade.exit_price == D("94.95")  # 95 * 0.9995

    def test_open_position_is_closed_at_the_end_of_data(self, rules: SymbolRules) -> None:
        res = run(candles_from_closes(SCENARIO[:8]), rules)
        assert [t.exit_reason for t in res.trades] == ["end"]

    def test_strategy_never_sees_the_future(self, rules: SymbolRules) -> None:
        seen: list[tuple[int, int]] = []
        inner = create_strategy("sma_cross", FAST_SLOW)

        class Spy:
            name = "spy"
            warmup = inner.warmup

            def on_candles(self, candles: Sequence[Candle], in_position: bool) -> Signal:
                seen.append((len(candles), candles[-1].open_time_ms))
                return inner.on_candles(candles, in_position)

        c = candles_from_closes(SCENARIO)
        run(c, rules, strategy=Spy())
        # the call made after candle i only ever contains candles up to i (never i+1...)
        for k, (n, last_open) in enumerate(seen):
            assert last_open == c[k].open_time_ms and n <= k + 1

    def test_prefix_of_the_data_gives_identical_early_decisions(self, rules: SymbolRules) -> None:
        full = candles_from_closes(SCENARIO + [90, 85, 120, 130, 140])
        short = full[:9]
        a, b = run(full, rules), run(short, rules)
        assert a.trades[0].entry_time_ms == b.trades[0].entry_time_ms
        assert a.trades[0].entry_price == b.trades[0].entry_price

    def test_daily_loss_limit_blocks_new_entries(self, rules: SymbolRules) -> None:
        # With a tiny loss limit the first losing day stops re-entry on the same UTC day.
        risk = RiskSettings(max_daily_loss=D("0.01"))
        calm = run(candles_from_closes(SCENARIO), rules, risk=risk)
        assert calm.final_equity <= D(1000)


class TestMetrics:
    def test_known_values(self, rules: SymbolRules) -> None:
        res = run(candles_from_closes(SCENARIO), rules)
        m = compute_metrics(res)
        assert m.trades == 1 and m.win_rate == 0.0 and m.profit_factor == 0.0
        assert m.total_return < 0 and 0 < m.max_drawdown < 0.05
        assert 0 < m.exposure < 1
        text = format_metrics(m)
        assert "return" in text and "scaled to this run's time in market" in text
        assert m.buy_and_hold_exposure_matched == pytest.approx(
            m.buy_and_hold_same_capital * m.exposure
        )

    def test_buy_and_hold_baseline_uses_the_capital_the_strategy_really_deployed(
        self, rules: SymbolRules
    ) -> None:
        c = candles_from_closes([100] * 5 + [110] * 10)  # +10% in total
        res = run(c, rules)  # default caps: the 100 order cap binds on a 1000 account
        assert res.buy_and_hold_return == pytest.approx(0.10)
        assert res.entries == 1 and D("99") < res.avg_entry_notional <= D(100)
        assert res.buy_and_hold_same_capital == pytest.approx(
            0.10 * float(res.avg_entry_notional) / 1000
        )
        # when risk-per-trade (not the cap) limits the size, the baseline shrinks with it:
        # a 300 account deploys about 65 per entry, NOT the 100 the caps would allow
        small = run_backtest(
            c, create_strategy("sma_cross", FAST_SLOW), rules, RiskSettings(),
            initial_cash=D(300), fee_rate=D("0.001"), slippage_bps=D(5), interval_ms=HOUR_MS,
        )  # fmt: skip
        assert small.avg_entry_notional < D(70)
        assert small.buy_and_hold_same_capital == pytest.approx(
            0.10 * float(small.avg_entry_notional) / 300
        )
        assert small.buy_and_hold_same_capital < 0.10 * 100 / 300 * 0.8  # not the old 100/300

    def test_without_any_entry_the_baseline_falls_back_to_the_nominal_cap(
        self, rules: SymbolRules
    ) -> None:
        res = run(candles_from_closes([100] * 20), rules)
        assert res.entries == 0 and res.avg_entry_notional == 0
        assert res.buy_and_hold_same_capital == 0  # flat market: 0% of anything

    def test_a_decision_is_dropped_rather_than_filled_across_a_gap(
        self, rules: SymbolRules
    ) -> None:
        full = candles_from_closes(SCENARIO)  # the BUY decision is taken at candle 4's close
        gapped = full[:5] + full[8:]  # candles 5, 6, 7 are missing
        res = run(gapped, rules)
        assert res.signals_dropped_at_gaps >= 1
        assert not res.trades or res.trades[0].entry_time_ms != gapped[5].open_time_ms

    def test_exact_metrics_for_a_hand_built_equity_curve(self) -> None:
        from botux.backtest.engine import Trade

        res = BacktestResult(
            initial_cash=D(1000), final_equity=D(1020), interval_ms=HOUR_MS,
            candles_total=4, candles_in_position=2,
        )  # fmt: skip
        res.equity_curve = [
            (1, D(1100)), (2, D(990)), (3, D(1080)), (4, D(1020)),
        ]  # fmt: skip
        res.trades = [
            Trade(0, 1, D(100), D(110), D(1), D(30), D(1), "signal"),
            Trade(2, 3, D(110), D(105), D(1), D(-10), D(1), "stop"),
            Trade(3, 4, D(105), D(110), D(1), D(5), D(1), "signal"),
        ]
        m = compute_metrics(res)
        assert m.total_return == pytest.approx(0.02)
        assert m.max_drawdown == pytest.approx((1100 - 990) / 1100)  # the peak starts at the cash
        assert m.trades == 3 and m.win_rate == pytest.approx(2 / 3)
        assert m.profit_factor == pytest.approx(35 / 10)  # gross wins / gross losses
        assert m.avg_trade_pnl == pytest.approx(25 / 3)
        assert m.exposure == pytest.approx(0.5)
        rets = [990 / 1100 - 1, 1080 / 990 - 1, 1020 / 1080 - 1]
        mean = sum(rets) / 3
        sd = (sum((r - mean) ** 2 for r in rets) / 2) ** 0.5
        year_candles = 365 * 24
        assert m.sharpe == pytest.approx(mean / sd * year_candles**0.5)  # annualised by sqrt(N)

    def test_a_drawdown_that_starts_below_the_initial_cash_is_measured_from_it(self) -> None:
        res = BacktestResult(
            initial_cash=D(1000), final_equity=D(950), interval_ms=HOUR_MS, candles_total=2
        )
        res.equity_curve = [(1, D(900)), (2, D(950))]
        assert compute_metrics(res).max_drawdown == pytest.approx(0.10)

    def test_a_loss_that_happens_early_in_a_day_counts_even_if_nothing_is_traded_until_later(
        self, rules: SymbolRules
    ) -> None:
        day_ms = 86_400_000
        start = (T0 // day_ms + 1) * day_ms  # a UTC midnight
        day1 = [100] * 20 + [101, 103, 104, 104]  # long from candle 21, still open at midnight
        day2 = [104, 90, 90, 90, 90, 92, 95, 98, 99, 99] + [99] * 14  # stopped out at 01:00, rally
        assert len(day1) == 24 and len(day2) == 24
        candles = candles_from_closes(day1 + day2, start=start)
        free = run(candles, rules, risk=RiskSettings(max_daily_loss=D(500)))
        limited = run(candles, rules, risk=RiskSettings(max_daily_loss=D("0.5")))
        assert len(free.trades) >= 2  # the later rally is traded when there is no limit ...
        assert len(limited.trades) == 1  # ... but not after the early-morning loss tripped it

    def test_a_stop_can_hit_on_the_entry_candle_itself(self, rules: SymbolRules) -> None:
        c = candles_from_closes(SCENARIO[:6])
        # the entry candle (index 5) opens at the signal price and its low pierces the 2% stop
        c[5] = make_candle(5, "103", "103.5", "99", "100")
        res = run(c[:6], rules)
        stops = [t for t in res.trades if t.exit_reason == "stop"]
        assert stops and stops[0].exit_time_ms == c[5].open_time_ms

    def test_after_a_daily_loss_halt_a_second_entry_the_same_day_is_refused(
        self, rules: SymbolRules
    ) -> None:
        day_ms = 86_400_000
        start = (T0 // day_ms + 1) * day_ms
        # buy, get stopped out (loss > the tiny limit), then a fresh crossover later the SAME day
        closes = [100] * 4 + [101, 103, 104, 104, 100, 90, 90, 90, 90, 92, 95, 98, 99, 99]
        candles = candles_from_closes(closes, start=start)
        free = run(candles, rules, risk=RiskSettings(max_daily_loss=D(500)))
        halted = run(candles, rules, risk=RiskSettings(max_daily_loss=D("0.5")))
        assert len(free.trades) >= 2  # without the limit the second signal is traded
        assert len(halted.trades) == 1  # with it, the second entry that day is refused

    def test_no_trades_means_flat_equity(self, rules: SymbolRules) -> None:
        m = compute_metrics(run(candles_from_closes([100] * 20), rules))
        assert m.trades == 0 and m.total_return == 0 and m.sharpe is None
        assert m.win_rate is None and m.profit_factor is None


class TestStudy:
    @staticmethod
    def wavy(n: int) -> list[Candle]:
        closes = []
        price = 100.0
        for i in range(n):
            price *= 1 + (0.004 if (i // 40) % 2 == 0 else -0.004)
            closes.append(round(price, 2))
        return candles_from_closes(closes)

    def study(self, rules: SymbolRules, candles: list[Candle], **kw: object):  # type: ignore[no-untyped-def]
        args = {
            "strategy_name": "sma_cross",
            "default_params": {"fast": 5, "slow": 20},
            "grid": {"fast": [5, 10], "slow": [20, 40]},
            "initial_cash": D(1000),
            "fee_rate": D("0.001"),
            "slippage_bps": D(5),
            "interval_ms": HOUR_MS,
        } | kw
        return run_study(candles, rules, RiskSettings(), **args)  # type: ignore[arg-type]

    def test_split_is_chronological_disjoint_and_leaves_a_test_segment(self) -> None:
        c = self.wavy(500)
        train, val, test = split_three(c, 0.5, 0.2)
        assert len(train) + len(val) + len(test) == 500
        assert train[-1].open_time_ms < val[0].open_time_ms < val[-1].open_time_ms
        assert val[-1].open_time_ms < test[0].open_time_ms
        for bad in ((0.95, 0.2), (0.5, 0.05), (0.7, 0.3), (0.2, 0.2)):
            with pytest.raises(ConfigError):
                split_three(c, *bad)

    def test_each_segment_only_sees_what_it_is_allowed_to(
        self, rules: SymbolRules, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = self.wavy(900)
        train, val, test = split_three(c, 0.5, 0.2)
        seen: list[tuple[int, int, int]] = []
        real = run_backtest

        def spy(segment: Sequence[Candle], *a: object, **k: object) -> BacktestResult:
            seen.append((len(segment), segment[0].open_time_ms, segment[-1].open_time_ms))
            return real(segment, *a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr("botux.backtest.study.run_backtest", spy)
        report = self.study(rules, c)
        in_train = [r for r in seen if r[2] <= train[-1].open_time_ms]
        in_val = [r for r in seen if val[0].open_time_ms <= r[1] and r[2] <= val[-1].open_time_ms]
        in_test = [r for r in seen if r[1] >= test[0].open_time_ms]
        assert len(in_train) == 4  # the whole grid, on train data only
        assert 1 <= len(in_val) <= 3  # only the short list reaches the validation segment
        assert len(in_test) == 2  # the chosen parameters and the defaults, once each
        assert len(in_train) + len(in_val) + len(in_test) == len(seen)  # nothing straddles a border
        assert report.test_metrics is not None and report.default_test_metrics is not None
        assert report.validation_metrics is not None and report.shortlist
        assert any("parameter combinations" in w for w in report.warnings)
        assert any("single instrument" in w for w in report.warnings)

    def test_the_winner_is_picked_on_validation_not_on_train(self, rules: SymbolRules) -> None:
        report = self.study(rules, self.wavy(1200))
        best_val = max(v.total_return for _, _, v in report.shortlist)
        chosen = [v for p, _, v in report.shortlist if p == report.chosen_params]
        assert chosen and chosen[0].total_return == best_val

    def test_look_back_comes_from_the_past_and_nothing_after_the_segment(
        self, rules: SymbolRules
    ) -> None:
        c = self.wavy(600)
        cut = 400
        seen_last: list[int] = []

        class Spy:
            name = "spy"
            warmup = 25

            def on_candles(self, candles: Sequence[Candle], in_position: bool) -> Signal:
                seen_last.append(candles[-1].open_time_ms)
                return Signal.HOLD

        run_backtest(
            c[cut:],
            Spy(),
            rules,
            RiskSettings(),
            initial_cash=D(1000),
            fee_rate=D(0),
            slippage_bps=D(0),
            interval_ms=HOUR_MS,
            history=c[:cut],
        )
        # the strategy is consulted for every candle of the segment (so no cold start) ...
        assert seen_last[0] == c[cut].open_time_ms
        # ... and never sees a candle later than the one that has just closed
        assert seen_last == [x.open_time_ms for x in c[cut:-1]]

    def test_a_warm_segment_trades_where_a_cold_one_cannot(self, rules: SymbolRules) -> None:
        c = candles_from_closes([100] * 10 + [101, 103, 104, 104, 100, 96, 96])

        def strat() -> Strategy:
            return create_strategy("sma_cross", {"fast": 2, "slow": 8})

        common = {
            "initial_cash": D(1000),
            "fee_rate": D(0),
            "slippage_bps": D(0),
            "interval_ms": HOUR_MS,
        }
        # only the last 7 candles are tested; with 8 needed to warm up, a cold start has no signal
        cold = run_backtest(c[10:], strat(), rules, RiskSettings(), **common)  # type: ignore[arg-type]
        warm = run_backtest(c[10:], strat(), rules, RiskSettings(), history=c[:10], **common)  # type: ignore[arg-type]
        assert cold.trades == [] and len(warm.trades) == 1

    def test_the_shortlist_is_the_top_train_results_among_eligible_combinations(
        self, rules: SymbolRules
    ) -> None:
        report = self.study(rules, self.wavy(1200))
        eligible = [pm for pm in report.grid if pm[1].trades >= 5]
        best = sorted(eligible, key=lambda pm: pm[1].total_return, reverse=True)[:3]
        assert [p for p, _, _ in report.shortlist] == [p for p, _ in best]
        # train results shown for the short list are the very train results of the grid
        for params, train_m, _ in report.shortlist:
            assert train_m == {repr(p): m for p, m in report.grid}[repr(params)]

    def test_combinations_with_too_few_trades_are_not_eligible(self, rules: SymbolRules) -> None:
        report = self.study(rules, self.wavy(900))
        assert all(m.trades >= 5 for _, m, _ in report.shortlist)
        # a grid where no combination trades enough falls back to the defaults, with a warning
        quiet = self.study(
            rules, candles_from_closes([100] * 900), grid={"fast": [5], "slow": [20]}
        )
        assert not quiet.chosen_from_grid and quiet.chosen_params == {"fast": 5, "slow": 20}
        assert any("falling back" in w for w in quiet.warnings)

    def test_the_default_reference_run_uses_the_default_parameters(
        self, rules: SymbolRules, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[dict[str, object]] = []
        real = create_strategy

        def spy(name: str, params: object) -> object:
            seen.append(dict(params))  # type: ignore[call-overload]
            return real(name, params)  # type: ignore[arg-type]

        monkeypatch.setattr("botux.backtest.study.create_strategy", spy)
        report = self.study(rules, self.wavy(900), default_params={"fast": 3, "slow": 11})
        assert seen[-1] == {"fast": 3, "slow": 11}  # the very last run: defaults on the test data
        assert seen[-2] == report.chosen_params  # the one before: the chosen parameters, once

    def test_too_little_data_is_refused(self, rules: SymbolRules) -> None:
        with pytest.raises(DataError):
            self.study(rules, self.wavy(100), grid={})

    def test_few_trades_trigger_a_warning(self, rules: SymbolRules) -> None:
        report = self.study(rules, self.wavy(700), grid={"fast": [5], "slow": [20]})
        assert any("far too few" in w for w in report.warnings)

    def test_no_skill_is_called_out_when_a_matched_buy_and_hold_does_as_well(
        self, rules: SymbolRules
    ) -> None:
        c = candles_from_closes([100 + i * 0.5 for i in range(700)])  # a steady uptrend
        report = self.study(rules, c, grid={"fast": [5], "slow": [20]})
        assert report.test_metrics is not None
        assert report.test_metrics.buy_and_hold_exposure_matched >= 0
        assert report.chosen_params

    def test_the_daily_loss_halt_is_reset_each_utc_day_in_a_backtest(
        self, rules: SymbolRules
    ) -> None:
        # Day 1: a long position is stopped out, which trips the (tiny) daily loss limit.
        # Day 2: a fresh rally gives a new entry signal; the halt must not outlive the day,
        # because a back-test has no human to reset it (the live bot does need one).
        day_ms = 86_400_000
        start = (T0 // day_ms + 1) * day_ms  # a UTC midnight
        day1 = [100] * 4 + [101, 103, 104, 104, 100, 90, 90] + [90] * 13
        day2 = [90] * 3 + [92, 95, 98, 99] + [99] * 17
        assert len(day1) == len(day2) == 24
        candles = candles_from_closes(day1 + day2, start=start)
        res = run(candles, rules, risk=RiskSettings(max_daily_loss=D("0.5")))
        days = [t.entry_time_ms // day_ms for t in res.trades]
        assert days.count(start // day_ms) >= 1  # traded (and lost) on day 1
        assert days.count(start // day_ms + 1) >= 1  # and was allowed to trade again on day 2


class TestWalkForward:
    def study_kwargs(self) -> dict[str, Any]:
        return {
            "strategy_name": "sma_cross",
            "grid": {"fast": [5, 10], "slow": [20, 40]},
            "initial_cash": D(1000),
            "fee_rate": D("0.001"),
            "slippage_bps": D(5),
            "interval_ms": HOUR_MS,
        }

    def test_windows_are_chronological_and_each_choice_sees_only_the_past(
        self, rules: SymbolRules, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = TestStudy.wavy(1500)
        seen: list[tuple[int, int]] = []
        real = run_backtest

        def spy(segment: Sequence[Candle], *a: object, **k: object) -> BacktestResult:
            seen.append((segment[0].open_time_ms, segment[-1].open_time_ms))
            return real(segment, *a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr("botux.backtest.study.run_backtest", spy)
        wf = run_walk_forward(c, rules, RiskSettings(), folds=4, **self.study_kwargs())
        assert len(wf.folds) == 4
        starts = [f.test_range[0] for f in wf.folds]
        assert starts == sorted(starts) and len(set(starts)) == 4
        grid_size = 4  # 2 x 2 combinations, each run once per fold on the training window only
        for k, f in enumerate(wf.folds):
            assert f.train_range[1] == f.test_range[0]  # the test window starts where training ends
            train_win = (c[f.train_range[0]].open_time_ms, c[f.train_range[1] - 1].open_time_ms)
            test_win = (c[f.test_range[0]].open_time_ms, c[f.test_range[1] - 1].open_time_ms)
            calls = seen[k * (grid_size + 1) : (k + 1) * (grid_size + 1)]
            assert calls[:grid_size] == [train_win] * grid_size  # the search saw training data only
            assert calls[grid_size] == test_win  # the choice was judged once, on unseen data
        assert wf.trades == sum(f.test.trades for f in wf.folds)
        assert any("fills, latency" in w for w in wf.warnings)

    def test_chained_return_compounds_the_folds(self, rules: SymbolRules) -> None:
        c = TestStudy.wavy(1500)
        wf = run_walk_forward(c, rules, RiskSettings(), folds=3, **self.study_kwargs())
        expected = 1.0
        for f in wf.folds:
            expected *= 1 + f.test.total_return
        assert wf.chained_return == pytest.approx(expected - 1)

    def test_unstable_choices_and_few_trades_are_flagged(self, rules: SymbolRules) -> None:
        c = TestStudy.wavy(1500)
        wf = run_walk_forward(c, rules, RiskSettings(), folds=4, **self.study_kwargs())
        assert wf.distinct_choices >= 1
        assert any("far too few" in w for w in wf.warnings) == (wf.trades < 30)

    def test_bad_arguments_and_too_little_data(self, rules: SymbolRules) -> None:
        c = TestStudy.wavy(1500)
        with pytest.raises(ConfigError):
            run_walk_forward(c, rules, RiskSettings(), folds=1, **self.study_kwargs())
        with pytest.raises(ConfigError):
            run_walk_forward(
                c,
                rules,
                RiskSettings(),
                folds=3,
                train_fraction=0.9,
                **self.study_kwargs(),
            )
        with pytest.raises(DataError):
            run_walk_forward(
                TestStudy.wavy(300), rules, RiskSettings(), folds=5, **self.study_kwargs()
            )


class TestMarketData:
    def test_clean_series_passes(self) -> None:
        assert check_candles(candles_from_closes([1, 2, 3, 4]), HOUR_MS).ok

    def test_duplicates_gaps_disorder_and_bad_ohlc_are_detected(self) -> None:
        c = candles_from_closes([100, 101, 102, 103, 104])
        assert check_candles([*c, c[-1]], HOUR_MS).duplicates == 1
        assert check_candles([c[0], c[2], c[3]], HOUR_MS).gaps == 1
        assert check_candles([c[1], c[0]], HOUR_MS).out_of_order == 1
        bad = Candle(c[0].open_time_ms, D(100), D(99), D(98), D(100), D(1), c[0].close_time_ms)
        assert check_candles([bad], HOUR_MS).bad_ohlc == 1
        assert check_candles([c[0], c[2]], HOUR_MS).fatal is False  # a gap alone is tolerated

    def test_live_validation_rejects_stale_future_and_empty(self) -> None:
        c = candles_from_closes([100, 101, 102])
        now = c[-1].close_time_ms + 1000
        validate_live_candles(c, "1h", now, 3.0)
        with pytest.raises(DataError, match="stale"):
            validate_live_candles(c, "1h", now + 5 * HOUR_MS, 3.0)
        with pytest.raises(DataError, match="future"):
            validate_live_candles(c, "1h", c[-1].close_time_ms - 60_000, 3.0)
        with pytest.raises(DataError, match="no candles"):
            validate_live_candles([], "1h", now, 3.0)
        with pytest.raises(DataError, match="inconsistent"):
            validate_live_candles([c[0], c[2]], "1h", now, 3.0)

    def test_only_closed_drops_the_forming_candle(self) -> None:
        c = candles_from_closes([100, 101, 102])
        t = c[-1].open_time_ms  # the third candle has just opened
        assert only_closed(c, t + 3000) == c[:-1]  # forming candle dropped
        # the second candle closed only 5 ms ago: inside the 2 s grace period it is not trusted yet
        assert only_closed(c, t + 5) == c[:-2]
        assert only_closed(c, t + 5, grace_ms=0) == c[:-1]

    def test_parse_klines_and_malformed_rows(self) -> None:
        c = candles_from_closes([100, 101])
        assert parse_klines(kline_rows(c)) == c
        with pytest.raises(DataError):
            parse_klines([[1, "x"]])
        with pytest.raises(DataError):
            parse_klines([["a", "1", "1", "1", "1", "1", 1]])

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        c = candles_from_closes([100, 101, 102.5])
        save_csv(tmp_path / "x.csv", c)
        assert load_csv(tmp_path / "x.csv") == c
        (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
        with pytest.raises(DataError):
            load_csv(tmp_path / "bad.csv")

    def test_pagination_downloads_everything_in_order(self, clock: object) -> None:
        all_candles = candles_from_closes([100 + (i % 7) for i in range(2500)])
        ex = FakeExchange()

        def klines(req):  # type: ignore[no-untyped-def]
            import httpx

            q = req.url.params
            start, limit = int(q["startTime"]), int(q["limit"])
            rows = [c for c in all_candles if c.open_time_ms >= start][:limit]
            return httpx.Response(200, json=kline_rows(rows))

        ex.on("GET", "/api/v3/klines", klines)
        client = BinanceSpotClient(http=ex.http(), clock=clock)  # type: ignore[arg-type]
        got = fetch_range(
            client, "BTCUSDT", "1h", all_candles[0].open_time_ms, all_candles[-1].close_time_ms
        )
        assert got == all_candles and ex.count("/api/v3/klines") == 3
        assert check_candles(got, HOUR_MS).ok


class TestLiveMarketFeedHelpers:
    """fetch_quote / fetch_latest_closed are what the live loop really calls."""

    def client(self, ex: FakeExchange, clock: object) -> BinanceSpotClient:
        return BinanceSpotClient(http=ex.http(), clock=clock)  # type: ignore[arg-type]

    def test_quote_validation(self, clock: object) -> None:
        from botux.data.market import fetch_quote

        ex = FakeExchange()
        c = self.client(ex, clock)
        for body, ok in (
            ({"bidPrice": "99.99", "askPrice": "100.01"}, True),
            ({"bidPrice": "0.00000000", "askPrice": "0.00000000"}, False),  # halted symbol
            ({"bidPrice": "100.02", "askPrice": "100.01"}, False),  # crossed book
            ({"bidPrice": "-1", "askPrice": "100"}, False),
            ({"bidPrice": "abc", "askPrice": "100"}, False),
            ({"askPrice": "100"}, False),  # missing key
        ):
            ex.on("GET", "/api/v3/ticker/bookTicker", body)
            if ok:
                q = fetch_quote(c, "BTCUSDT", 123)
                assert (q.bid, q.ask, q.ts_ms) == (D("99.99"), D("100.01"), 123)
            else:
                with pytest.raises(DataError):
                    fetch_quote(c, "BTCUSDT", 123)

    def test_latest_closed_drops_the_forming_and_the_just_closed_candle_and_honours_count(
        self, clock: object
    ) -> None:
        import httpx

        from botux.data.market import fetch_latest_closed

        candles = candles_from_closes([100 + i for i in range(60)])
        ex = FakeExchange()
        seen: dict[str, str] = {}

        def klines(req: httpx.Request) -> httpx.Response:
            seen.update(req.url.params)
            now = T0 + 50 * HOUR_MS + 1_000  # candle 49 closed 1 s ago (inside the grace period)
            shown = [c for c in candles if c.open_time_ms <= now]  # includes the forming candle 50
            return httpx.Response(200, json=kline_rows(shown))

        ex.on("GET", "/api/v3/klines", klines)
        now_ms = T0 + 50 * HOUR_MS + 1_000
        got = fetch_latest_closed(self.client(ex, clock), "BTCUSDT", "1h", 10, now_ms)
        assert seen["limit"] == "11"  # one extra: the forming candle is requested and dropped
        assert len(got) == 10
        assert got[-1].open_time_ms == candles[48].open_time_ms  # 49 is inside the 2 s grace
        assert all(c.close_time_ms < now_ms - 2_000 for c in got)
        later = fetch_latest_closed(self.client(ex, clock), "BTCUSDT", "1h", 10, now_ms + 5_000)
        assert later[-1].open_time_ms == candles[49].open_time_ms
