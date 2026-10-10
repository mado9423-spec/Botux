"""Composition root: wires settings, connectors, risk, storage and the loop together."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

import httpx

from botux.alerts import CompositeNotifier, FileNotifier, LogNotifier
from botux.clock import Clock, SystemClock
from botux.config import Settings
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
from botux.storage import Store
from botux.strategy.base import create_strategy

# Keys that describe a paper session's in-memory state; cleared when a new paper session starts.
_PAPER_SESSION_PREFIXES = ("pos:", "lastc:", "baseline:", "lost:", "risk_state")


@dataclass
class App:
    loop: TradingLoop
    store: Store
    client: BinanceSpotClient
    kill: KillSwitch
    paper_broker: PaperBroker | None = None

    def close(self) -> None:
        self.client.close()
        self.store.close()


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
) -> App:
    kill = KillSwitch(settings.paths.kill_file, clock)
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
    )
    order_manager.set_fill_callback(loop.on_fill)
    return App(loop, store, client, kill, paper_broker)


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
    rules = dict(rules) if rules is not None else load_rules(client, settings)
    store = store or Store(settings.paths.database)

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
    )


def build_live_app(
    settings: Settings,
    env: Mapping[str, str],
    *,
    http: httpx.Client | None = None,
    clock: Clock | None = None,
    store: Store | None = None,
) -> App:
    """Live trading. Raises LiveTradingNotAllowed unless every safety gate passes."""
    clock = clock or SystemClock()
    kill = KillSwitch(settings.paths.kill_file, clock)
    broker, client = build_live_broker(settings, env, kill, clock=clock, http=http)
    rules = load_rules(client, settings)
    feed = BinanceMarketFeed(client, settings.market.interval, clock)
    store = store or Store(settings.paths.database)
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
    )


def starting_balance(settings: Settings) -> Decimal:
    return settings.paper.starting_balance
