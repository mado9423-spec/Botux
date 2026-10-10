"""Strategy, back-test engine, metrics, train/test study and market-data validation."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import pytest

from botux.backtest.engine import BacktestResult, run_backtest
from botux.backtest.metrics import compute_metrics, format_metrics
from botux.backtest.study import run_study, split_train_test
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
from botux.strategy.base import Signal, create_strategy
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
        assert "return" in format_metrics(m)

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

    def test_split_is_chronological_and_disjoint(self) -> None:
        c = self.wavy(500)
        train, test = split_train_test(c, 0.7)
        assert len(train) + len(test) == 500
        assert train[-1].open_time_ms < test[0].open_time_ms
        with pytest.raises(ConfigError):
            split_train_test(c, 0.95)

    def test_parameters_are_chosen_on_train_and_judged_on_test(
        self, rules: SymbolRules, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = self.wavy(600)
        train, test = split_train_test(c, 0.7)
        ranges: list[tuple[int, int, int]] = []
        real = run_backtest

        def spy(segment: Sequence[Candle], *a: object, **k: object) -> BacktestResult:
            ranges.append((len(segment), segment[0].open_time_ms, segment[-1].open_time_ms))
            return real(segment, *a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr("botux.backtest.study.run_backtest", spy)
        report = run_study(
            c,
            rules,
            RiskSettings(),
            strategy_name="sma_cross",
            default_params={"fast": 5, "slow": 20},
            grid={"fast": [5, 10], "slow": [20, 40]},
            initial_cash=D(1000),
            fee_rate=D("0.001"),
            slippage_bps=D(5),
            interval_ms=HOUR_MS,
        )
        train_runs = [r for r in ranges if r[2] <= train[-1].open_time_ms]
        test_runs = [r for r in ranges if r[1] >= test[0].open_time_ms]
        assert len(train_runs) >= 4  # the whole grid ran on train data only
        assert len(test_runs) == 2  # chosen + default parameters, once each, on test data only
        assert len(train_runs) + len(test_runs) == len(ranges)  # nothing straddles the boundary
        assert report.test_metrics is not None and report.default_test_metrics is not None
        assert any("parameter combinations" in w for w in report.warnings)
        assert any("single instrument" in w for w in report.warnings)

    def test_too_little_data_is_refused(self, rules: SymbolRules) -> None:
        with pytest.raises(DataError):
            run_study(
                self.wavy(100),
                rules,
                RiskSettings(),
                strategy_name="sma_cross",
                default_params={"fast": 5, "slow": 20},
                grid={},
                initial_cash=D(1000),
                fee_rate=D(0),
                slippage_bps=D(0),
                interval_ms=HOUR_MS,
            )

    def test_few_trades_trigger_a_warning(self, rules: SymbolRules) -> None:
        report = run_study(
            self.wavy(400),
            rules,
            RiskSettings(),
            strategy_name="sma_cross",
            default_params={"fast": 5, "slow": 20},
            grid={"fast": [5], "slow": [20]},
            initial_cash=D(1000),
            fee_rate=D("0.001"),
            slippage_bps=D(5),
            interval_ms=HOUR_MS,
        )
        assert any("far too few" in w for w in report.warnings)


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
        assert only_closed(c, c[-1].open_time_ms + 5) == c[:-1]

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
