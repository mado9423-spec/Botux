"""Composition root: wires settings, connectors, risk, storage and the loop together."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import httpx

from botux.alerts import CompositeNotifier, FileNotifier, LogNotifier
from botux.clock import Clock, SystemClock
from botux.config import Settings, database_path, settings_fingerprint
from botux.errors import BotuxError, ConfigError, ExchangeError, LiveTradingNotAllowed
from botux.exchange.client import BinanceSpotClient
from botux.exchange.rules import SymbolRules
from botux.execution.broker import Broker
from botux.execution.live import build_live_broker
from botux.execution.order_manager import OrderManager
from botux.execution.paper import FaultPlan, PaperBroker
from botux.execution.positions import PositionBook
from botux.killswitch import KillSwitch
from botux.risk.manager import RiskManager
from botux.runner import BinanceMarketFeed, MarketFeed, TradingLoop
from botux.safety import api_key_problems, check_live_gate
from botux.storage import InstanceLock, Store
from botux.strategy.base import create_strategy

log = logging.getLogger(__name__)

# Keys that describe a paper session's in-memory state; cleared when a new paper session starts.
_PAPER_SESSION_PREFIXES = (
    "pos:",
    "lastc:",
    "baseline:",
    "lost:",
    "risk_state",
    "ledger:",
    "ack:",
    "entry_blockers",
)


@dataclass
class App:
    loop: TradingLoop
    store: Store
    client: BinanceSpotClient
    kill: KillSwitch
    paper_broker: PaperBroker | None = None
    lock: InstanceLock | None = None

    def close(self) -> None:
        self.client.close()
        self.store.close()
        if self.lock is not None:
            self.lock.release()


def public_client(
    settings: Settings, *, http: httpx.Client | None = None, clock: Clock | None = None
) -> BinanceSpotClient:
    """A client WITHOUT credentials: it can only reach public market-data endpoints."""
    return BinanceSpotClient(
        market_data_base_url=settings.market.market_data_base_url,
        trading_base_url=settings.market.trading_base_url,
        credentials=None,
        trading_enabled=False,
        http=http,
        clock=clock,
        max_clock_skew_ms=settings.live.max_clock_skew_ms,
    )


def load_rules(client: BinanceSpotClient, settings: Settings) -> dict[str, SymbolRules]:
    return {
        sym: SymbolRules.from_exchange_info(client.exchange_info(sym), sym)
        for sym in settings.market.symbols
    }


def _assemble(
    settings: Settings,
    *,
    client: BinanceSpotClient,
    broker: Broker,
    feed: MarketFeed,
    rules: Mapping[str, SymbolRules],
    store: Store,
    clock: Clock,
    paper_broker: PaperBroker | None,
    act_on_current_candle: bool,
    gate_check: Callable[[], None] | None = None,
    lock: InstanceLock | None = None,
    rules_loader: Callable[[], Mapping[str, SymbolRules]] | None = None,
) -> App:
    kill = KillSwitch(settings.paths.kill_file, clock)
    kill.ensure_writable()  # an emergency stop that cannot be written must stop the start-up
    notifier = CompositeNotifier(LogNotifier(), FileNotifier(settings.paths.alerts_file, clock))
    risk = RiskManager(
        settings.risk,
        fee_rate=settings.execution.fee_rate,
        slippage_bps=settings.execution.slippage_bps,
        clock=clock,
    )
    positions = PositionBook(store, settings.risk.stop_loss_pct)
    strategy = create_strategy(settings.strategy.name, settings.strategy.params)
    order_manager = OrderManager(
        broker,
        store,
        rules,
        notifier,
        clock=clock,
        kill_switch=kill,
        reconcile_attempts=settings.execution.reconcile_attempts,
        reconcile_interval_seconds=settings.execution.reconcile_interval_seconds,
        notional_buffer=settings.risk.min_notional_buffer,
    )
    loop = TradingLoop(
        settings=settings,
        feed=feed,
        broker=broker,
        order_manager=order_manager,
        risk=risk,
        positions=positions,
        store=store,
        kill=kill,
        notifier=notifier,
        clock=clock,
        strategy=strategy,
        rules=rules,
        paper_broker=paper_broker,
        act_on_current_candle=act_on_current_candle,
        gate_check=gate_check,
        rules_loader=rules_loader,
    )
    order_manager.set_fill_callback(loop.on_fill)
    return App(loop, store, client, kill, paper_broker, lock)


def build_paper_app(
    settings: Settings,
    *,
    http: httpx.Client | None = None,
    clock: Clock | None = None,
    store: Store | None = None,
    faults: FaultPlan | None = None,
    feed: MarketFeed | None = None,
    rules: Mapping[str, SymbolRules] | None = None,
    act_on_current_candle: bool = False,
) -> App:
    """Paper trading: public market data in, simulated fills out. No credentials are involved."""
    clock = clock or SystemClock()
    client = public_client(settings, http=http, clock=clock)
    if feed is None:
        client.sync_time()  # measures clock skew using the public /time endpoint
        feed = BinanceMarketFeed(client, settings.market.interval, clock)
    injected_rules = rules is not None
    rules = dict(rules) if rules is not None else load_rules(client, settings)
    lock: InstanceLock | None = None
    owns_store = store is None
    if store is None:
        db = database_path(settings, "paper")
        lock = InstanceLock(db)
        lock.acquire()
        store = Store(db)
    try:
        # BEFORE touching any state: a database that belongs to live mode must never be reset.
        store.claim_mode("paper")
    except ConfigError:
        if owns_store:
            store.close()
        if lock is not None:
            lock.release()
        raise

    # A paper session starts from scratch: the simulator's balances live in memory only.
    store.expire_stale_orders(clock.now_ms(), "stale order from a previous paper session")
    for prefix in _PAPER_SESSION_PREFIXES:
        store.kv_clear_prefix(prefix)

    quote_asset = settings.risk.quote_asset
    paper = PaperBroker(
        rules,
        {quote_asset: settings.paper.starting_balance},
        fee_rate=settings.execution.fee_rate,
        slippage_bps=settings.execution.slippage_bps,
        clock=clock,
        faults=faults,
    )
    return _assemble(
        settings,
        client=client,
        broker=paper,
        feed=feed,
        rules=rules,
        store=store,
        clock=clock,
        paper_broker=paper,
        act_on_current_candle=act_on_current_candle,
        lock=lock,
        # rules supplied by a caller (tests) are fixed; real ones are re-read from the exchange
        rules_loader=None if injected_rules else (lambda: load_rules(client, settings)),
    )


KEY_RECHECK_SECONDS = 3600.0
KEY_RETRY_SECONDS = 60.0
KEY_MAX_LOOKUP_FAILURES = 10


def make_gate_check(
    settings: Settings,
    kill: KillSwitch,
    clock: Clock,
    client: BinanceSpotClient,
    reload_settings: Callable[[], Settings] | None = None,
) -> Callable[[], None]:
    """Re-verify the live gate while running.

    Every call: the settings file on disk, the arming file, the kill switch (cheap, local). The
    API key's permissions are looked up over the network about hourly; a failed lookup is retried
    after a minute, and after too many failures in a row the gate CLOSES (an unverifiable key is
    not a verified one).
    """
    next_key_check = clock.monotonic() + KEY_RECHECK_SECONDS
    failures = 0
    running = settings_fingerprint(settings)

    def check() -> None:
        nonlocal next_key_check, failures
        if reload_settings is not None:
            try:
                on_disk = reload_settings()
            except BotuxError as exc:
                raise LiveTradingNotAllowed(f"settings file unreadable or invalid: {exc}") from None
            if settings_fingerprint(on_disk) != running:
                raise LiveTradingNotAllowed(
                    "settings changed on disk since this process started; restart (and re-arm) "
                    "to apply them"
                )
        check_live_gate(settings, kill, clock.now_ms())
        if clock.monotonic() < next_key_check:
            return
        try:
            info = client.api_restrictions()
        except ExchangeError as exc:
            failures += 1
            next_key_check = clock.monotonic() + KEY_RETRY_SECONDS
            log.warning("could not re-check API key permissions (%d): %s", failures, exc)
            if failures >= KEY_MAX_LOOKUP_FAILURES:
                raise LiveTradingNotAllowed(
                    f"the API key's permissions could not be verified {failures} times in a row"
                ) from None
            return
        failures = 0
        next_key_check = clock.monotonic() + KEY_RECHECK_SECONDS
        problems = api_key_problems(info, settings.live.require_ip_restriction)
        if problems:
            raise LiveTradingNotAllowed("API key permissions changed: " + "; ".join(problems))

    return check


def bind_account(store: Store, uid: str | None) -> None:
    """A live database belongs to ONE Binance account; a different key pointing at another
    account must not inherit this database's baselines, ledger and positions."""
    if not uid:
        raise LiveTradingNotAllowed(
            "the account answer carries no uid, so this database cannot be bound to the account"
        )
    bound = store.kv_get("account_uid")
    if not bound:
        store.kv_set("account_uid", uid)
    elif bound != uid:
        raise LiveTradingNotAllowed(
            "this database was created for a different Binance account; use a separate database "
            "(paths.database) for each account"
        )


def build_live_app(
    settings: Settings,
    env: Mapping[str, str],
    *,
    http: httpx.Client | None = None,
    clock: Clock | None = None,
    store: Store | None = None,
    reload_settings: Callable[[], Settings] | None = None,
) -> App:
    """Live trading. Raises LiveTradingNotAllowed unless every safety gate passes."""
    clock = clock or SystemClock()
    kill = KillSwitch(settings.paths.kill_file, clock)
    broker, client = build_live_broker(settings, env, kill, clock=clock, http=http)
    rules = load_rules(client, settings)
    feed = BinanceMarketFeed(client, settings.market.interval, clock)
    lock: InstanceLock | None = None
    owns_store = store is None
    if store is None:
        db = database_path(settings, "live")
        if db == ":memory:":
            client.close()
            raise ConfigError("live trading needs a persistent database (write-ahead order log)")
        lock = InstanceLock(db)
        try:
            lock.acquire()
        except ConfigError:
            client.close()
            raise
        store = Store(db)
    try:
        store.claim_mode("live")
        bind_account(store, getattr(broker, "account_uid", None))
    except (ConfigError, LiveTradingNotAllowed):
        client.close()
        if owns_store:
            store.close()
        if lock is not None:
            lock.release()
        raise
    return _assemble(
        settings,
        client=client,
        broker=broker,
        feed=feed,
        rules=rules,
        store=store,
        clock=clock,
        paper_broker=None,
        act_on_current_candle=False,
        gate_check=make_gate_check(settings, kill, clock, client, reload_settings),
        lock=lock,
        rules_loader=lambda: load_rules(client, settings),
    )
