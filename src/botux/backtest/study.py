"""Train / validation / test study and walk-forward evaluation.

The data are split chronologically (never shuffled) into three contiguous segments:

* TRAIN        - the whole parameter grid is run here and a short list is kept;
* VALIDATION   - only the short list is run here and the winner is picked;
* TEST         - the winner is evaluated exactly once. Nothing is ever tuned on it.

The strategy's look-back for a later segment is taken from the candles just before it (the past,
so no leakage); only the segment itself is traded and valued. Walk-forward repeats the "choose on
the past, judge on the next unseen window" cycle over several windows, which is a harder test than
a single split because the choice has to keep working.

None of this removes the risk of over-fitting: with a single instrument and one stretch of
history, a good out-of-sample number can still be luck, and a bad one can be an unlucky stretch.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from botux.backtest.engine import run_backtest
from botux.backtest.metrics import Metrics, compute_metrics
from botux.config import RiskSettings
from botux.errors import ConfigError, DataError
from botux.exchange.rules import SymbolRules
from botux.models import Candle
from botux.strategy.base import create_strategy

MIN_TRAIN_TRADES = 5
MIN_TRADES_FOR_STATISTICS = 30
SHORTLIST = 3
MIN_TRAIN, MIN_VALIDATION, MIN_TEST = 200, 100, 100


@dataclass(slots=True)
class StudyReport:
    train_candles: int
    validation_candles: int
    test_candles: int
    grid: list[tuple[dict[str, Any], Metrics]] = field(default_factory=list)
    shortlist: list[tuple[dict[str, Any], Metrics, Metrics]] = field(default_factory=list)
    chosen_params: dict[str, Any] = field(default_factory=dict)
    chosen_from_grid: bool = False
    train_metrics: Metrics | None = None
    validation_metrics: Metrics | None = None
    test_metrics: Metrics | None = None
    default_params: dict[str, Any] = field(default_factory=dict)
    default_test_metrics: Metrics | None = None
    warnings: list[str] = field(default_factory=list)


def split_three(
    candles: Sequence[Candle], train_fraction: float, validation_fraction: float
) -> tuple[list[Candle], list[Candle], list[Candle]]:
    if not 0.3 <= train_fraction <= 0.8:
        raise ConfigError("train_fraction must be within [0.3, 0.8]")
    if not 0.1 <= validation_fraction <= 0.4:
        raise ConfigError("validation_fraction must be within [0.1, 0.4]")
    if train_fraction + validation_fraction > 0.9:
        raise ConfigError("train + validation must leave at least 10% of the data for the test")
    a = int(len(candles) * train_fraction)
    b = int(len(candles) * (train_fraction + validation_fraction))
    return list(candles[:a]), list(candles[a:b]), list(candles[b:])


def _runner(
    strategy_name: str,
    rules: SymbolRules,
    risk: RiskSettings,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_bps: Decimal,
    interval_ms: int,
) -> Any:
    def run(
        segment: Sequence[Candle], params: Mapping[str, Any], history: Sequence[Candle] = ()
    ) -> Metrics:
        strategy = create_strategy(strategy_name, params)
        result = run_backtest(
            segment,
            strategy,
            rules,
            risk,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            interval_ms=interval_ms,
            history=history,
        )
        return compute_metrics(result)

    return run


def _search(
    run: Any,
    train: Sequence[Candle],
    grid: Mapping[str, Sequence[Any]],
) -> list[tuple[dict[str, Any], Metrics]]:
    keys = list(grid)
    out: list[tuple[dict[str, Any], Metrics]] = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values, strict=True))
        try:
            out.append((params, run(train, params)))
        except ConfigError:
            continue  # e.g. fast >= slow
    return out


def run_study(
    candles: Sequence[Candle],
    rules: SymbolRules,
    risk: RiskSettings,
    *,
    strategy_name: str,
    default_params: Mapping[str, Any],
    grid: Mapping[str, Sequence[Any]],
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_bps: Decimal,
    interval_ms: int,
    train_fraction: float = 0.5,
    validation_fraction: float = 0.2,
) -> StudyReport:
    train, validation, test = split_three(candles, train_fraction, validation_fraction)
    if len(train) < MIN_TRAIN or len(validation) < MIN_VALIDATION or len(test) < MIN_TEST:
        raise DataError(
            "not enough data for a meaningful split: "
            f"train={len(train)} validation={len(validation)} test={len(test)}"
        )
    report = StudyReport(
        len(train), len(validation), len(test), default_params=dict(default_params)
    )
    run = _runner(strategy_name, rules, risk, initial_cash, fee_rate, slippage_bps, interval_ms)
    before_validation = train
    before_test = train + validation

    report.grid = _search(run, train, grid)
    eligible = sorted(
        (pm for pm in report.grid if pm[1].trades >= MIN_TRAIN_TRADES),
        key=lambda pm: pm[1].total_return,
        reverse=True,
    )[:SHORTLIST]
    if eligible:
        for params, train_m in eligible:  # the short list is the ONLY thing validation sees
            report.shortlist.append((params, train_m, run(validation, params, before_validation)))
        best_params, best_train, best_val = max(report.shortlist, key=lambda t: t[2].total_return)
        report.chosen_params = dict(best_params)
        report.train_metrics, report.validation_metrics = best_train, best_val
        report.chosen_from_grid = True
        if best_val.trades == 0:
            report.warnings.append("the chosen parameters made no trade in the validation segment")
        if best_val.total_return <= 0:
            report.warnings.append(
                "even the best short-listed parameters did not make money in the validation "
                "segment: nothing here supports using them"
            )
    else:
        report.chosen_params = dict(default_params)
        report.train_metrics = run(train, default_params)
        report.validation_metrics = run(validation, default_params, before_validation)
        report.warnings.append(
            f"no parameter set produced >= {MIN_TRAIN_TRADES} trades on the training data; "
            "falling back to the default parameters"
        )
    report.test_metrics = run(test, report.chosen_params, before_test)  # once, never tuned on
    report.default_test_metrics = run(test, default_params, before_test)

    n_combos = len(report.grid)
    if n_combos > 1:
        report.warnings.append(
            f"{n_combos} parameter combinations were tried on the training data and "
            f"{len(report.shortlist)} on the validation data; the winner is biased upward "
            "(selection effect). Compare train, validation and test."
        )
    if report.test_metrics.trades < MIN_TRADES_FOR_STATISTICS:
        report.warnings.append(
            f"only {report.test_metrics.trades} trades in the test segment: far too few for the "
            "numbers to mean anything statistically"
        )
    if report.train_metrics and report.test_metrics:
        # Compare per-candle returns: the segments have different lengths, so raw totals mislead.
        train_rate = report.train_metrics.total_return / report.train_candles
        test_rate = report.test_metrics.total_return / report.test_candles
        if train_rate > 0 and test_rate < 0.5 * train_rate:
            report.warnings.append(
                "the per-candle test return is less than half of the train return: a typical "
                "sign of over-fitting or of a change in market regime"
            )
    if report.test_metrics.total_return <= report.test_metrics.buy_and_hold_exposure_matched:
        report.warnings.append(
            "on the test segment the strategy did not beat even a buy-and-hold scaled to its own "
            "time in the market: no evidence of skill"
        )
    report.warnings.append(
        "a single instrument, a single split and a single time span: results can be luck and say "
        "nothing about the future. Costs are modelled, but real fills, latency and outages are not."
    )
    return report


# ----------------------------------------------------------------------------- walk-forward


@dataclass(frozen=True, slots=True)
class Fold:
    index: int
    train_range: tuple[int, int]
    test_range: tuple[int, int]
    params: dict[str, Any]
    train: Metrics
    test: Metrics


@dataclass(slots=True)
class WalkForwardReport:
    folds: list[Fold] = field(default_factory=list)
    chained_return: float = 0.0  # compounded out-of-sample returns, fold after fold
    chained_baseline_same_capital: float = 0.0
    chained_baseline_exposure_matched: float = 0.0
    trades: int = 0
    worst_drawdown: float = 0.0
    distinct_choices: int = 0
    warnings: list[str] = field(default_factory=list)


def run_walk_forward(
    candles: Sequence[Candle],
    rules: SymbolRules,
    risk: RiskSettings,
    *,
    strategy_name: str,
    grid: Mapping[str, Sequence[Any]],
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_bps: Decimal,
    interval_ms: int,
    folds: int = 5,
    train_fraction: float = 0.4,
) -> WalkForwardReport:
    """Rolling windows: choose parameters on ``train`` candles, judge them on the next window.

    The windows tile the data after the first training window: with 5 folds and 40% training,
    each test window is 12% of the data and the training window slides forward by the same step.
    """
    if folds < 2:
        raise ConfigError("walk-forward needs at least 2 folds")
    if not 0.2 <= train_fraction <= 0.7:
        raise ConfigError("walk-forward train_fraction must be within [0.2, 0.7]")
    n = len(candles)
    train_len = int(n * train_fraction)
    test_len = (n - train_len) // folds
    if train_len < MIN_TRAIN or test_len < MIN_TEST:
        raise DataError(f"not enough data for {folds} folds: train={train_len} test={test_len}")
    run = _runner(strategy_name, rules, risk, initial_cash, fee_rate, slippage_bps, interval_ms)
    report = WalkForwardReport()
    chained = chained_bh = chained_em = 1.0
    chosen: set[str] = set()
    for k in range(folds):
        a = k * test_len
        train = list(candles[a : a + train_len])
        test = list(candles[a + train_len : a + train_len + test_len])
        scored = [pm for pm in _search(run, train, grid) if pm[1].trades >= MIN_TRAIN_TRADES]
        if not scored:
            report.warnings.append(f"fold {k + 1}: no parameter set traded enough; fold skipped")
            continue
        params, train_m = max(scored, key=lambda pm: pm[1].total_return)
        test_m = run(test, params, train)  # the look-back comes from the training window's end
        report.folds.append(
            Fold(
                k + 1,
                (a, a + train_len),
                (a + train_len, a + train_len + test_len),
                dict(params),
                train_m,
                test_m,
            )
        )
        chained *= 1 + test_m.total_return
        chained_bh *= 1 + test_m.buy_and_hold_same_capital
        chained_em *= 1 + test_m.buy_and_hold_exposure_matched
        report.trades += test_m.trades
        report.worst_drawdown = max(report.worst_drawdown, test_m.max_drawdown)
        chosen.add(repr(sorted(params.items())))
    if not report.folds:
        raise DataError("walk-forward produced no usable fold")
    report.chained_return = chained - 1
    report.chained_baseline_same_capital = chained_bh - 1
    report.chained_baseline_exposure_matched = chained_em - 1
    report.distinct_choices = len(chosen)
    if report.distinct_choices > max(1, len(report.folds) // 2):
        report.warnings.append(
            f"the best parameters changed in {report.distinct_choices} of {len(report.folds)} "
            "folds: the choice is unstable, a sign that it is fitting noise"
        )
    if report.trades < MIN_TRADES_FOR_STATISTICS:
        report.warnings.append(
            f"only {report.trades} out-of-sample trades in total: far too few to conclude anything"
        )
    if report.chained_return <= report.chained_baseline_exposure_matched:
        report.warnings.append(
            "out of sample the strategy did not beat a buy-and-hold scaled to its time in the "
            "market: no evidence of skill"
        )
    report.warnings.append(
        "folds are chained by compounding each fold's return on the starting capital; fills, "
        "latency and outages are not modelled, and the exchange filters are today's."
    )
    return report
