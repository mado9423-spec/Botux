"""Train/test study: choose parameters on the first part of the data, judge them on the rest.

The test segment is contiguous and strictly AFTER the training segment (no shuffling), the
parameter search sees only the training segment, and the chosen parameters are evaluated on the
test segment exactly once. This reduces, but does not remove, the risk of over-fitting: with a
single split and a single instrument, a good test result can still be luck.
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


@dataclass(slots=True)
class StudyReport:
    train_candles: int
    test_candles: int
    grid: list[tuple[dict[str, Any], Metrics]] = field(default_factory=list)
    chosen_params: dict[str, Any] = field(default_factory=dict)
    chosen_from_grid: bool = False
    train_metrics: Metrics | None = None
    test_metrics: Metrics | None = None
    default_params: dict[str, Any] = field(default_factory=dict)
    default_test_metrics: Metrics | None = None
    warnings: list[str] = field(default_factory=list)


def split_train_test(
    candles: Sequence[Candle], train_fraction: float
) -> tuple[list[Candle], list[Candle]]:
    if not 0.3 <= train_fraction <= 0.9:
        raise ConfigError("train_fraction must be within [0.3, 0.9]")
    cut = int(len(candles) * train_fraction)
    return list(candles[:cut]), list(candles[cut:])


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
    train_fraction: float = 0.7,
) -> StudyReport:
    train, test = split_train_test(candles, train_fraction)
    if len(train) < 200 or len(test) < 100:
        raise DataError(
            f"not enough data for a meaningful split: train={len(train)} test={len(test)}"
        )
    report = StudyReport(len(train), len(test), default_params=dict(default_params))

    def run(segment: Sequence[Candle], params: Mapping[str, Any]) -> Metrics:
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
        )
        return compute_metrics(result)

    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values, strict=True))
        try:
            report.grid.append((params, run(train, params)))
        except ConfigError:
            continue  # e.g. fast >= slow
    eligible = [(p, m) for p, m in report.grid if m.trades >= MIN_TRAIN_TRADES]
    if eligible:
        best_params, best_train = max(eligible, key=lambda pm: pm[1].total_return)
        report.chosen_params, report.train_metrics = dict(best_params), best_train
        report.chosen_from_grid = True
    else:
        report.chosen_params = dict(default_params)
        report.train_metrics = run(train, default_params)
        report.warnings.append(
            f"no parameter set produced >= {MIN_TRAIN_TRADES} trades on the training data; "
            "falling back to the default parameters"
        )
    report.test_metrics = run(test, report.chosen_params)
    report.default_test_metrics = run(test, default_params)

    n_combos = len(report.grid)
    if n_combos > 1:
        report.warnings.append(
            f"{n_combos} parameter combinations were tried on the training data; the best one is "
            "biased upward (selection effect). Compare train vs. test."
        )
    if report.test_metrics.trades < MIN_TRADES_FOR_STATISTICS:
        report.warnings.append(
            f"only {report.test_metrics.trades} trades in the test segment: far too few for the "
            "numbers to mean anything statistically"
        )
    if (
        report.train_metrics
        and report.test_metrics.total_return < report.train_metrics.total_return
    ):
        report.warnings.append(
            "test return is below train return: a typical sign of over-fitting or regime change"
        )
    report.warnings.append(
        "a single instrument, a single split and a single time span: results can be luck and say "
        "nothing about the future. Costs are modelled, but real fills, latency and outages are not."
    )
    return report
