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
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import FrameType

from botux import __version__
from botux.app import App, build_live_app, build_paper_app, public_client
from botux.backtest.metrics import format_metrics
from botux.backtest.study import run_study, run_walk_forward
from botux.clock import SystemClock
from botux.config import (
    SUPPORTED_INTERVALS,
    Settings,
    database_path,
    load_settings,
    settings_fingerprint,
    settings_summary,
)
from botux.data.market import check_candles, fetch_range, load_csv, only_closed, save_csv
from botux.errors import BotuxError
from botux.exchange.client import BinanceSpotClient
from botux.exchange.rules import SymbolRules
from botux.execution.livecheck import run_live_check
from botux.killswitch import KillSwitch
from botux.models import Candle, OrderStatus
from botux.safety import RISK_TEXT, arm_live_interactive, check_live_gate, disarm
from botux.secrets_guard import GLOBAL_REDACTOR, setup_logging
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
    print(f"kill file            : {s.paths.kill_file}")
    print(f"paper database       : {database_path(s, 'paper')}")
    print(f"live database        : {database_path(s, 'live')}")
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


MAX_CASH = 10**12  # anything beyond this is a typo, and float reporting math would overflow


def _positive_decimal(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not value.is_finite() or not 0 < value <= MAX_CASH:
        raise argparse.ArgumentTypeError(
            f"must be a positive number not above {MAX_CASH:,}: {text!r}"
        )
    return value


def cmd_backtest(args: argparse.Namespace) -> int:
    s = _settings(args)
    client = public_client(s)
    try:
        return _backtest(args, s, client)
    finally:
        client.close()


def _backtest(args: argparse.Namespace, s: Settings, client: BinanceSpotClient) -> int:
    interval = args.interval or s.market.interval
    symbol = args.symbol or s.market.symbols[0]
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
    print(f"data check: {report_data.describe()}")
    if report_data.gaps:
        print(
            "WARNING: the data has gaps; a decision that would be filled across a gap is "
            "dropped instead of being filled at a price from after it."
        )
    rules = SymbolRules.from_exchange_info(client.exchange_info(symbol), symbol)

    grid = DEFAULT_GRIDS.get(s.strategy.name, {})
    first, last = candles[0].open_time_ms, candles[-1].close_time_ms
    days = (last - first) / 86_400_000
    utc = "%Y-%m-%d"
    print(
        f"\n{symbol} {interval}: {len(candles)} candles over {days:.0f} days, "
        f"{datetime.fromtimestamp(first / 1000, UTC):{utc}} to "
        f"{datetime.fromtimestamp(last / 1000, UTC):{utc}} (UTC)"
    )
    print("Assumptions (all approximations, none of them guarantees):")
    print(
        f"  - data: public klines of {symbol}, checked for duplicates, gaps and disorder; "
        "orders execute at the NEXT candle's open (no look-ahead)"
    )
    print(
        f"  - costs: fee {s.execution.fee_rate} per fill (check your own tier), slippage "
        f"{s.execution.slippage_bps} bps against you on every fill; no spread or depth model, "
        "complete fills, no latency or outages"
    )
    print(
        f"  - capital: start {args.cash} {s.risk.quote_asset}; each entry is sized by "
        f"risk-per-trade and capped at {s.risk.max_order_notional} per order, so the buy&hold "
        "baselines are scaled to the capital the strategy actually deployed per entry"
    )
    print(
        "  - exchange filters are today's, not historical; the daily-loss halt is reset "
        "automatically each UTC day (the live bot needs a human)"
    )
    if args.walk_forward:
        wf = run_walk_forward(
            candles,
            rules,
            s.risk,
            strategy_name=s.strategy.name,
            grid=grid,
            initial_cash=args.cash,
            fee_rate=s.execution.fee_rate,
            slippage_bps=s.execution.slippage_bps,
            interval_ms=interval_ms,
            folds=args.folds,
        )
        print(f"\nWalk-forward ({len(wf.folds)} folds; parameters chosen on the past only):")
        for fold in wf.folds:
            print(
                f"  fold {fold.index}: chose {fold.params} | train {fold.train.total_return:+.2%} "
                f"-> out-of-sample {fold.test.total_return:+.2%} "
                f"(same-capital buy&hold {fold.test.buy_and_hold_same_capital:+.2%}, "
                f"trades {fold.test.trades}, max drawdown {fold.test.max_drawdown:.2%})"
            )
        print(
            f"Chained out-of-sample return {wf.chained_return:+.2%} | same-capital buy&hold "
            f"{wf.chained_baseline_same_capital:+.2%} | scaled to time in market "
            f"{wf.chained_baseline_exposure_matched:+.2%} | trades {wf.trades} | "
            f"worst fold drawdown {wf.worst_drawdown:.2%} | distinct choices {wf.distinct_choices}"
        )
        print("\nLimitations:")
        for w in wf.warnings:
            print(f"  - {w}")
        print("\nThis is an educational simulation, not advice, and not a prediction.")
        return 0

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
        validation_fraction=args.validation_fraction,
    )
    print(
        f"Split (chronological): train {study.train_candles} / validation "
        f"{study.validation_candles} / test {study.test_candles} candles"
    )
    print(f"\nParameter search on TRAIN data only ({len(study.grid)} combinations):")
    for params, m in sorted(study.grid, key=lambda pm: pm[1].total_return, reverse=True):
        print(
            f"  {params}: return {m.total_return:+.2%}, trades {m.trades}, "
            f"max drawdown {m.max_drawdown:.2%}"
        )
    if study.shortlist:
        print("\nShort list judged on VALIDATION data (the test data is still untouched):")
        for params, train_m, val_m in study.shortlist:
            print(
                f"  {params}: train {train_m.total_return:+.2%} -> validation "
                f"{val_m.total_return:+.2%}, trades {val_m.trades}"
            )
    print(
        f"\nChosen parameters: {study.chosen_params}"
        f"{'' if study.chosen_from_grid else '  (default; grid gave no eligible result)'}"
    )
    if study.train_metrics:
        print("TRAIN result (in-sample, optimistic by construction):")
        print(format_metrics(study.train_metrics))
    if study.validation_metrics:
        print("VALIDATION result (used to pick among the short list):")
        print(format_metrics(study.validation_metrics))
    if study.test_metrics:
        print("TEST result of the chosen parameters (out-of-sample, evaluated once):")
        print(format_metrics(study.test_metrics))
    if study.default_test_metrics:
        print(f"TEST result of the default parameters {study.default_params} (reference only):")
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
    # raises LiveTradingNotAllowed unless fully armed; the settings file is re-read while running
    app = build_live_app(s, os.environ, reload_settings=lambda: _settings(args))
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
    if not KillSwitch(s.paths.kill_file).trip(args.reason):
        raise BotuxError(
            f"the kill file {s.paths.kill_file} could NOT be written, so a running bot will not "
            "see this stop. Stop the process by other means (Ctrl+C / SIGTERM) and fix "
            "paths.kill_file"
        )
    print(
        f"kill switch SET: {s.paths.kill_file}\n"
        "A bot using this same file cancels open orders and stops."
    )
    if not any(Path(database_path(s, m)).exists() for m in ("paper", "live")):
        print(
            "WARNING: no bot database exists at the paths this config resolves to "
            f"({database_path(s, 'paper')} / {database_path(s, 'live')}). If a bot is running "
            "from another directory or with another config, THIS STOP DOES NOT REACH IT: run "
            "the command from the bot's own directory (relative paths in the config are resolved "
            "against the current directory) or put absolute paths in [paths]; otherwise stop the "
            "process itself (Ctrl+C / SIGTERM).",
            file=sys.stderr,
        )
        return 3
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
        f"config mode: {s.mode} | kill switch ({s.paths.kill_file}): "
        f"{'SET - ' + kill.reason() if kill.is_set() else 'not set'}"
    )
    for mode in ("paper", "live"):
        db = Path(database_path(s, mode))
        if not db.exists():
            print(f"[{mode}] no database yet ({db})")
            continue
        store = Store(db)
        counts: dict[str, int] = {}
        for o in store.list_orders():
            counts[o.status.value] = counts.get(o.status.value, 0) + 1
        print(f"[{mode}] orders by status: {counts or 'none'}")
        attention = store.list_orders(
            (OrderStatus.PENDING_SEND, OrderStatus.UNKNOWN, OrderStatus.LOST)
        )
        for o in attention:
            print(
                f"  ATTENTION {o.status.value}: {o.symbol} {o.side.value} {o.quantity} "
                f"id={o.client_order_id}"
            )
        blockers = json.loads(store.kv_get("entry_blockers") or "{}")
        for key, why in sorted(blockers.items()):
            print(f"  NEW ENTRIES BLOCKED [{key}]: {why}")
        store.close()
    return 0


def cmd_arm_live(args: argparse.Namespace) -> int:
    s = _settings(args)
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            "error: arming must be done interactively by a person in a terminal "
            "(stdin/stdout are not a TTY); refusing to read the confirmation from a pipe.",
            file=sys.stderr,
        )
        return 1
    path = arm_live_interactive(s, SystemClock().now_ms())
    return 0 if path else 1


def cmd_disarm_live(args: argparse.Namespace) -> int:
    if not disarm(_settings(args)):
        print(
            "nothing was armed at the arming-file path this config resolves to (relative paths "
            "are resolved against the current directory). A bot running from another directory "
            "or with another config is NOT affected: stop it with 'botux kill' there, or stop "
            "the process.",
            file=sys.stderr,
        )
        return 3
    print(
        "live trading disarmed: a bot that is already running stops NEW entries on its next look "
        "but still manages exits (stop-loss / strategy exit) for positions it holds. To stop "
        "everything use 'botux kill'."
    )
    return 0


def cmd_resolve_lost(args: argparse.Namespace) -> int:
    s = _settings(args)
    db = Path(database_path(s, args.mode))
    if not db.exists():
        raise BotuxError(f"no {args.mode} database at {db}")
    store = Store(db)
    store.resolve_lost(args.symbol)
    store.log_event(SystemClock().now_ms(), "WARNING", "resolve_lost", {"symbol": args.symbol})
    store.close()
    print(f"{args.symbol}: LOST flag cleared in the {args.mode} database by operator decision.")
    return 0


def cmd_ack_balances(args: argparse.Namespace) -> int:
    """Record that a human has looked at an unexplained balance difference.

    Bookkeeping only: nothing is sent to the exchange and nothing is sold or cancelled. The loop
    applies it on its next cycle (a surplus is treated as external; a deficit as a loss of bot
    capital) and then lifts the corresponding block on new entries.
    """
    if not args.confirm_checked:
        raise BotuxError(
            "refusing without --confirm-checked: first compare the account with the bot's records "
            "on the exchange yourself (see 'botux status'), then repeat with --confirm-checked"
        )
    s = _settings(args)
    db = Path(database_path(s, args.mode))
    if not db.exists():
        raise BotuxError(f"no {args.mode} database at {db}")
    store = Store(db)
    store.kv_set("ack:balances", str(SystemClock().now_ms()))
    store.log_event(SystemClock().now_ms(), "WARNING", "ack_balances_requested", {})
    store.close()
    print(
        f"[{args.mode}] acknowledgement recorded; the running (or next) cycle will accept the "
        "current balances as they are. Nothing was sent to the exchange."
    )
    return 0


def cmd_live_check(args: argparse.Namespace) -> int:
    """Read-only rehearsal with the operator's own key; sends no order (see execution/livecheck)."""
    if not args.confirm_contact_binance:
        raise BotuxError(
            "refusing without --confirm-contact-binance: this command contacts Binance with the "
            "API key in your environment/.env (read-only calls plus POST /order/test, which "
            "never reaches the matching engine). It sends no order."
        )
    s = _settings(args)
    setup_logging()
    return run_live_check(s, os.environ)


def cmd_reset_daily_loss(args: argparse.Namespace) -> int:
    """Human decision to lift a tripped daily-loss halt (it does not end at midnight by itself)."""
    if not args.confirm_checked:
        raise BotuxError(
            "refusing without --confirm-checked: first find out WHY the daily loss limit was "
            "reached ('botux status', the log and the exchange), then repeat with "
            "--confirm-checked. Resetting gives the bot a fresh daily allowance."
        )
    s = _settings(args)
    db = Path(database_path(s, args.mode))
    if not db.exists():
        raise BotuxError(f"no {args.mode} database at {db}")
    store = Store(db)
    store.kv_set("ack:daily_loss", str(SystemClock().now_ms()))
    store.log_event(SystemClock().now_ms(), "WARNING", "daily_loss_reset_requested", {})
    store.close()
    print(
        f"[{args.mode}] reset recorded; the running (or next) cycle lifts the halt and starts a "
        "fresh daily reference. Nothing was sent to the exchange."
    )
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
    b.add_argument("--cash", type=_positive_decimal, default=Decimal(1000))
    b.add_argument("--train-fraction", type=float, default=0.5)
    b.add_argument("--validation-fraction", type=float, default=0.2)
    b.add_argument(
        "--walk-forward",
        action="store_true",
        help="rolling choose-on-the-past / judge-on-the-next-window test instead of one split",
    )
    b.add_argument("--folds", type=int, default=5)
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
    r.add_argument("--mode", choices=["paper", "live"], default="live")
    r.set_defaults(func=cmd_resolve_lost)
    a = sub.add_parser(
        "ack-balances",
        help="accept unexplained balance differences once checked (bookkeeping only)",
    )
    a.add_argument("--mode", choices=["paper", "live"], default="live")
    a.add_argument(
        "--confirm-checked",
        action="store_true",
        help="I compared the account with the bot's records and accept the difference",
    )
    a.set_defaults(func=cmd_ack_balances)
    lc = sub.add_parser(
        "live-check",
        help="read-only rehearsal with your own key (order/test only; sends NO order)",
    )
    lc.add_argument(
        "--confirm-contact-binance",
        action="store_true",
        help="I understand this contacts Binance with the API key in my environment",
    )
    lc.set_defaults(func=cmd_live_check)
    d = sub.add_parser(
        "reset-daily-loss",
        help="lift a tripped daily-loss halt after finding out why (human decision)",
    )
    d.add_argument("--mode", choices=["paper", "live"], default="live")
    d.add_argument(
        "--confirm-checked",
        action="store_true",
        help="I know why the limit was reached and accept a fresh daily allowance",
    )
    d.set_defaults(func=cmd_reset_daily_loss)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except BotuxError as exc:
        print(f"error: {GLOBAL_REDACTOR.scrub(str(exc))}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
