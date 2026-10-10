"""Command line interface: ``botux <command>``.

Everything defaults to paper trading. The only command that can ever send real orders is
``botux live`` and it refuses to start unless the whole safety gate (see safety.py) passes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from types import FrameType

from botux import __version__
from botux.app import App, build_live_app, build_paper_app, public_client
from botux.backtest.metrics import format_metrics
from botux.backtest.study import run_study
from botux.clock import SystemClock
from botux.config import (
    SUPPORTED_INTERVALS,
    Settings,
    load_settings,
    settings_fingerprint,
    settings_summary,
)
from botux.data.market import check_candles, fetch_range, load_csv, only_closed, save_csv
from botux.errors import BotuxError
from botux.exchange.client import BinanceSpotClient
from botux.exchange.rules import SymbolRules
from botux.killswitch import KillSwitch
from botux.models import Candle, OrderStatus
from botux.safety import RISK_TEXT, arm_live_interactive, check_live_gate, disarm
from botux.secrets_guard import setup_logging
from botux.storage import Store

log = logging.getLogger("botux")

# Small, deliberately modest search space for the educational strategy.
DEFAULT_GRIDS: dict[str, dict[str, list[int]]] = {
    "sma_cross": {"fast": [10, 20, 30], "slow": [50, 100, 200]},
}


def _settings(args: argparse.Namespace) -> Settings:
    return load_settings(args.config)


def _install_signal_handlers(app: App) -> None:
    def handler(signum: int, _frame: FrameType | None) -> None:
        log.warning("signal %s received: stopping after the current cycle", signum)
        app.loop.request_stop()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


# ---------------------------------------------------------------------------- commands


def cmd_check_config(args: argparse.Namespace) -> int:
    s = _settings(args)
    print(json.dumps(settings_summary(s), indent=2, sort_keys=True))
    print(f"\nsettings fingerprint : {settings_fingerprint(s)}")
    print(f"mode                 : {s.mode}")
    kill = KillSwitch(s.paths.kill_file)
    try:
        check_live_gate(s, kill, SystemClock().now_ms())
        print("live gate            : OPEN (live trading would be allowed)")
    except BotuxError as exc:
        print(f"live gate            : CLOSED ({exc})")
    return 0


def cmd_fetch_data(args: argparse.Namespace) -> int:
    s = _settings(args)
    client = public_client(s)
    clock = SystemClock()
    now = clock.now_ms() + client.sync_time()
    candles = _download(client, args.symbol, args.interval, args.days, now)
    out = (
        Path(args.out)
        if args.out
        else Path(s.paths.data_dir) / f"{args.symbol}_{args.interval}.csv"
    )
    save_csv(out, candles)
    print(f"saved {len(candles)} closed candles to {out}")
    return 0


def _download(
    client: BinanceSpotClient, symbol: str, interval: str, days: int, now_ms: int
) -> list[Candle]:
    start = now_ms - days * 86_400_000
    candles = only_closed(fetch_range(client, symbol, interval, start, now_ms), now_ms)
    report = check_candles(candles, SUPPORTED_INTERVALS[interval])
    print(f"downloaded: {report.describe()}")
    if report.fatal:
        raise BotuxError("downloaded data is inconsistent; refusing to use it")
    return candles


def cmd_backtest(args: argparse.Namespace) -> int:
    s = _settings(args)
    interval = args.interval or s.market.interval
    symbol = args.symbol or s.market.symbols[0]
    client = public_client(s)
    interval_ms = SUPPORTED_INTERVALS[interval]
    if args.data:
        candles = load_csv(Path(args.data))
        if len(candles) > 1:
            data_interval_ms = candles[1].open_time_ms - candles[0].open_time_ms
            if data_interval_ms != interval_ms:
                print(
                    f"note: the CSV has {data_interval_ms // 60_000}-minute candles; using that "
                    f"instead of the configured {interval}"
                )
                interval_ms = data_interval_ms
    else:
        now = SystemClock().now_ms() + client.sync_time()
        candles = _download(client, symbol, interval, args.days, now)
        if args.save:
            save_csv(Path(args.save), candles)
    report_data = check_candles(candles, interval_ms)
    if report_data.fatal:
        raise BotuxError(f"data problems: {report_data.describe()}")
    rules = SymbolRules.from_exchange_info(client.exchange_info(symbol), symbol)

    grid = DEFAULT_GRIDS.get(s.strategy.name, {})
    study = run_study(
        candles,
        rules,
        s.risk,
        strategy_name=s.strategy.name,
        default_params=s.strategy.params,
        grid=grid,
        initial_cash=args.cash,
        fee_rate=s.execution.fee_rate,
        slippage_bps=s.execution.slippage_bps,
        interval_ms=interval_ms,
        train_fraction=args.train_fraction,
    )
    first, last = candles[0].open_time_ms, candles[-1].close_time_ms
    days = (last - first) / 86_400_000
    print(
        f"\n{symbol} {interval}: {len(candles)} candles over {days:.0f} days "
        f"(train {study.train_candles} / test {study.test_candles})"
    )
    print(
        f"costs assumed: fee {s.execution.fee_rate} per fill, "
        f"slippage {s.execution.slippage_bps} bps; "
        f"start cash {args.cash} {s.risk.quote_asset}"
    )
    print(f"\nParameter search on TRAIN data only ({len(study.grid)} combinations):")
    for params, m in sorted(study.grid, key=lambda pm: pm[1].total_return, reverse=True):
        print(
            f"  {params}: return {m.total_return:+.2%}, trades {m.trades}, "
            f"max drawdown {m.max_drawdown:.2%}"
        )
    print(
        f"\nChosen parameters: {study.chosen_params}"
        f"{'' if study.chosen_from_grid else '  (default; grid gave no eligible result)'}"
    )
    if study.train_metrics:
        print("TRAIN result of the chosen parameters (in-sample, optimistic by construction):")
        print(format_metrics(study.train_metrics))
    if study.test_metrics:
        print("TEST result of the chosen parameters (out-of-sample, evaluated once):")
        print(format_metrics(study.test_metrics))
    if study.default_test_metrics:
        print(f"TEST result of the default parameters {study.default_params}:")
        print(format_metrics(study.default_test_metrics))
    print("\nLimitations:")
    for w in study.warnings:
        print(f"  - {w}")
    print("\nThis is an educational simulation, not advice, and not a prediction.")
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    s = _settings(args)
    setup_logging()
    if s.mode != "paper":
        log.warning("config says mode=%r but 'botux paper' always runs the simulator only", s.mode)
    app = build_paper_app(s, act_on_current_candle=args.act_on_current_candle)
    _install_signal_handlers(app)
    try:
        log.info(
            "PAPER trading (simulated fills). Start balance %s %s. Ctrl+C or 'botux kill' to stop.",
            s.paper.starting_balance,
            s.risk.quote_asset,
        )
        app.loop.run(max_cycles=args.max_cycles)
        _print_summary(app, s)
    finally:
        app.close()
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    s = _settings(args)
    setup_logging()
    app = build_live_app(s, os.environ)  # raises LiveTradingNotAllowed unless fully armed
    _install_signal_handlers(app)
    try:
        app.loop.run(max_cycles=args.max_cycles)
    finally:
        app.close()
    return 0


def _print_summary(app: App, s: Settings) -> None:
    curve = app.store.equity_curve()
    if curve:
        first, last = curve[0][1], curve[-1][1]
        print(
            f"equity: start {first:.2f} -> last {last:.2f} {s.risk.quote_asset} "
            f"({len(curve)} snapshots)"
        )
    orders = app.store.list_orders()
    print(f"orders recorded: {len(orders)}")


def cmd_kill(args: argparse.Namespace) -> int:
    s = _settings(args)
    KillSwitch(s.paths.kill_file).trip(args.reason)
    print(f"kill switch SET ({s.paths.kill_file}). A running bot cancels open orders and stops.")
    return 0


def cmd_unkill(args: argparse.Namespace) -> int:
    s = _settings(args)
    KillSwitch(s.paths.kill_file).clear()
    print("kill switch cleared. This was a manual decision; the bot never clears it by itself.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    s = _settings(args)
    kill = KillSwitch(s.paths.kill_file)
    print(
        f"mode: {s.mode} | kill switch: {'SET - ' + kill.reason() if kill.is_set() else 'not set'}"
    )
    db = Path(s.paths.database)
    if db.exists():
        store = Store(db)
        counts: dict[str, int] = {}
        for o in store.list_orders():
            counts[o.status.value] = counts.get(o.status.value, 0) + 1
        print(f"orders by status: {counts or 'none'}")
        uncertain = store.list_orders(
            (OrderStatus.PENDING_SEND, OrderStatus.UNKNOWN, OrderStatus.LOST)
        )
        for o in uncertain:
            print(
                f"  ATTENTION {o.status.value}: {o.symbol} {o.side.value} {o.quantity} "
                f"id={o.client_order_id}"
            )
        store.close()
    return 0


def cmd_arm_live(args: argparse.Namespace) -> int:
    s = _settings(args)
    path = arm_live_interactive(s, SystemClock().now_ms())
    return 0 if path else 1


def cmd_disarm_live(args: argparse.Namespace) -> int:
    disarm(_settings(args))
    print("live trading disarmed.")
    return 0


def cmd_resolve_lost(args: argparse.Namespace) -> int:
    s = _settings(args)
    store = Store(s.paths.database)
    store.kv_set(f"lost:{args.symbol}", "")
    store.log_event(SystemClock().now_ms(), "WARNING", "resolve_lost", {"symbol": args.symbol})
    store.close()
    print(f"{args.symbol}: LOST flag cleared by operator decision.")
    return 0


# ---------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="botux",
        description="Educational Binance Spot bot. Paper trading by default; no profit promised.",
    )
    p.add_argument("--version", action="version", version=f"botux {__version__}")
    p.add_argument("-c", "--config", default=None, help="path to settings TOML (non-secret)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "check-config", help="validate settings and show the live gate state"
    ).set_defaults(func=cmd_check_config)

    f = sub.add_parser("fetch-data", help="download closed candles (public data, no keys)")
    f.add_argument("--symbol", default="BTCUSDT")
    f.add_argument("--interval", default="1h", choices=list(SUPPORTED_INTERVALS))
    f.add_argument("--days", type=int, default=365)
    f.add_argument("--out", default=None)
    f.set_defaults(func=cmd_fetch_data)

    b = sub.add_parser("backtest", help="train/test back-test with fees and slippage")
    b.add_argument("--symbol", default=None)
    b.add_argument("--interval", default=None, choices=list(SUPPORTED_INTERVALS))
    b.add_argument("--days", type=int, default=365)
    b.add_argument("--data", default=None, help="CSV from 'fetch-data' instead of downloading")
    b.add_argument("--save", default=None, help="save the downloaded candles to this CSV")
    b.add_argument("--cash", type=Decimal, default=Decimal(1000))
    b.add_argument("--train-fraction", type=float, default=0.7)
    b.set_defaults(func=cmd_backtest)

    pp = sub.add_parser("paper", help="run the trading loop in PAPER mode (simulated fills)")
    pp.add_argument("--max-cycles", type=int, default=None)
    pp.add_argument(
        "--act-on-current-candle",
        action="store_true",
        help="demo only: act on the latest closed candle at start-up",
    )
    pp.set_defaults(func=cmd_paper)

    lv = sub.add_parser("live", help="REAL TRADING. Refuses unless the full safety gate passes")
    lv.add_argument("--max-cycles", type=int, default=None)
    lv.set_defaults(func=cmd_live)

    k = sub.add_parser("kill", help="emergency stop: create the kill file")
    k.add_argument("--reason", default="manual kill")
    k.set_defaults(func=cmd_kill)
    sub.add_parser("unkill", help="clear the kill switch (manual decision)").set_defaults(
        func=cmd_unkill
    )
    sub.add_parser("status", help="kill switch and order state").set_defaults(func=cmd_status)
    sub.add_parser(
        "arm-live",
        help="interactive confirmation required before live trading",
        description=RISK_TEXT,
    ).set_defaults(func=cmd_arm_live)
    sub.add_parser("disarm-live", help="remove the live arming file").set_defaults(
        func=cmd_disarm_live
    )
    r = sub.add_parser(
        "resolve-lost", help="clear the LOST-order block after checking the exchange"
    )
    r.add_argument("symbol")
    r.set_defaults(func=cmd_resolve_lost)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except BotuxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
