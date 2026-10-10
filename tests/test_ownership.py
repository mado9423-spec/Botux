"""The bot only ever manages money and coins that its OWN records explain.

Everything else on the account (a bigger balance, coins deposited later, withdrawals, orders
placed by the user) must neither be sized from, sold, cancelled, nor silently absorbed.
"""

from __future__ import annotations

import dataclasses
import json
from decimal import Decimal

import pytest

from botux.config import MarketSettings, PaperSettings, Settings, StrategySettings
from botux.errors import ConfigError, NetworkError
from botux.exchange.rules import SymbolRules
from botux.execution.order_manager import OrderIntent
from botux.models import OrderRequest, OrderStatus, OrderType, Side
from botux.risk.manager import RiskManager
from botux.runner import TradingLoop
from botux.storage import Store
from tests.test_runner import Harness, harness

D = Decimal


@pytest.fixture
def cfg(settings: Settings) -> Settings:
    return dataclasses.replace(
        settings,
        strategy=StrategySettings("sma_cross", {"fast": 2, "slow": 3}),
        market=MarketSettings(poll_seconds=1.0),
    )


def buys(h: Harness) -> list[object]:
    return [o for o in h.orders() if o.side is Side.BUY]


def critical_titles(h: Harness) -> list[str]:
    return [t for lvl, t, _ in h.notifier.items if lvl == "CRITICAL"]


class TestBotCapital:
    def test_a_big_account_does_not_enlarge_the_bot(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        s = dataclasses.replace(
            cfg,
            paper=PaperSettings(D(10_000)),
            risk=dataclasses.replace(cfg.risk, bot_capital=D(50)),
        )
        h = harness(s, rules, store)
        h.cycle()
        assert store.equity_curve()[-1][1] == 50  # equity is the BOT's, not the account's
        h.cycle(hours=1)  # BUY signal
        (buy,) = h.orders()
        assert buy.quantity * D("100.01") <= D(50)  # cash-limited by the bot capital (cap is 100)
        assert h.paper.balances()["USDT"].free > D(9_900)  # the rest of the account is untouched

    def test_paper_defaults_to_the_whole_simulated_balance(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        assert store.equity_curve()[-1][1] == cfg.paper.starting_balance

    def test_money_leaving_the_part_of_the_account_the_bot_does_not_use_is_not_its_business(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        s = dataclasses.replace(
            cfg,
            paper=PaperSettings(D(10_000)),
            risk=dataclasses.replace(cfg.risk, bot_capital=D(100), max_daily_loss=D(5)),
        )
        h = harness(s, rules, store)
        h.cycle()
        h.paper._free["USDT"] -= D(60)  # the user spends from the part the bot does not use
        h.cycle()
        assert "balance:USDT" not in h.loop.risk.blockers  # 9,940 still covers the bot's 100
        assert store.equity_curve()[-1][1] == 100 and not h.loop.risk.daily_loss_halted

    def test_live_refuses_to_run_without_an_explicit_capital(self) -> None:
        from botux.config import parse_settings

        s = parse_settings({"mode": "live"})
        assert s.bot_capital is None  # no silent default for real money
        s2 = parse_settings({"mode": "live", "risk": {"bot_capital": "300"}})
        assert s2.bot_capital == 300

    @pytest.mark.parametrize(
        ("risk", "match"),
        [
            ({"bot_capital": "0"}, "must be > 0"),
            ({"bot_capital": "-1"}, "must be > 0"),
            ({"bot_capital": "10", "max_daily_loss": "10"}, "smaller than"),
            ({"bot_capital": "5000"}, "cannot exceed paper.starting_balance"),
        ],
    )
    def test_illogical_capital_is_refused(self, risk: dict[str, str], match: str) -> None:
        from botux.config import parse_settings

        with pytest.raises(ConfigError, match=match):
            parse_settings({"risk": risk})

    def test_the_loop_itself_refuses_to_start_without_a_capital(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        live_like = dataclasses.replace(cfg, mode="live")
        with pytest.raises(ConfigError, match="bot_capital"):
            TradingLoop(
                settings=live_like,
                feed=h.feed,
                broker=h.paper,
                order_manager=h.loop.order_manager,
                risk=h.loop.risk,
                positions=h.loop.positions,
                store=store,
                kill=h.loop._kill,
                notifier=h.notifier,
                clock=h.clock,
                strategy=h.loop._strategy,
                rules={"BTCUSDT": rules},
            )

    def test_a_changed_capital_is_logged_and_rebases_the_daily_reference(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        smaller = dataclasses.replace(cfg, risk=dataclasses.replace(cfg.risk, bot_capital=D(400)))
        risk = RiskManager(smaller.risk, fee_rate=D("0.001"), slippage_bps=D(5))
        TradingLoop(
            settings=smaller,
            feed=h.feed,
            broker=h.paper,
            order_manager=h.loop.order_manager,
            risk=risk,
            positions=h.loop.positions,
            store=store,
            kill=h.loop._kill,
            notifier=h.notifier,
            clock=h.clock,
            strategy=h.loop._strategy,
            rules={"BTCUSDT": rules},
        )
        (event,) = store.events("capital_changed")
        assert json.loads(event["detail"]) == {"from": "1000", "to": "400"}


class TestUnexplainedBalances:
    def test_coins_that_appear_without_a_fill_block_entries_and_are_never_sold(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()  # warm start: baseline recorded (0 BTC)
        h.paper._free["BTC"] = D("0.3")  # a deposit / manual trade the bot knows nothing about
        h.cycle(hours=1)  # the BUY signal candle
        assert buys(h) == []
        assert "balance:BTC" in h.loop.risk.blockers
        assert "Unexpected balance change" in critical_titles(h)
        assert json.loads(store.kv_get("entry_blockers") or "{}").keys() == {"balance:BTC"}
        assert h.loop._position_qty("BTCUSDT", h.paper.balances()) == 0  # not the bot's to sell
        assert h.paper.balances()["BTC"].free == D("0.3")

    def test_acknowledging_lifts_the_block_and_keeps_the_foreign_coins_foreign(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.paper._free["BTC"] = D("0.3")
        h.cycle(hours=1)
        store.kv_set("ack:balances", "1")  # what `botux ack-balances` writes
        h.cycle()
        assert h.loop.risk.blockers == {}
        assert not store.kv_get("ack:balances")  # consumed exactly once
        assert store.kv_get("baseline:BTC") == "0.3"  # the surplus is attributed to the user
        assert store.kv_get("ledger:offset:BTC") == "0"
        assert h.loop._position_qty("BTCUSDT", h.paper.balances()) == 0
        assert [json.loads(e["detail"]) for e in store.events("balances_acknowledged")] == [
            {"offsets": ["BTC: +0.3"]}
        ]

    def test_a_deposit_after_a_bot_trade_is_not_mistaken_for_the_bots_coins(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.cycle(hours=1)  # bot buys
        assert len(buys(h)) == 1
        own = h.loop._position_qty("BTCUSDT", h.paper.balances())
        assert own > 0
        h.paper._free["BTC"] += D("0.5")  # user deposits more of the same coin
        h.cycle()
        assert h.loop._position_qty("BTCUSDT", h.paper.balances()) == own
        assert "balance:BTC" in h.loop.risk.blockers
        for _ in range(4):
            h.cycle(hours=1)  # the strategy sells at candle 8: only the bot's own coins
        sells = [o for o in h.orders() if o.side is Side.SELL]
        assert sells and all(o.quantity <= own for o in sells)
        assert h.paper.balances()["BTC"].free >= D("0.5")

    def test_missing_quote_funds_block_entries_and_count_as_a_loss_once_acknowledged(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.paper._free["USDT"] -= D(500)  # money left the account
        h.cycle()
        assert "balance:USDT" in h.loop.risk.blockers
        store.kv_set("ack:balances", "1")
        h.cycle()
        assert h.loop.risk.blockers.get("balance:USDT") is None
        assert store.kv_get("ledger:offset:USDT") == "-500"
        assert store.equity_curve()[-1][1] == 500
        assert h.loop.risk.daily_loss_halted  # 500 lost > max_daily_loss 20

    def test_nothing_is_flagged_in_normal_trading(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        for hours in (0, 1, 1, 1, 1, 1):
            h.cycle(hours)
        assert len(h.orders()) == 2
        assert h.loop.risk.blockers == {}
        assert critical_titles(h) == []

    def test_unresolved_orders_explain_a_mismatch_so_no_false_alarm_is_raised(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.paper._free["BTC"] = D("0.3")
        now = h.clock.now_ms()
        from botux.storage import OrderRecord

        store.insert_order(
            OrderRecord(
                "bxunknown", "k-u", "BTCUSDT", Side.BUY, OrderType.MARKET, D("0.3"), None,
                OrderStatus.UNKNOWN, now, now,
            )
        )  # fmt: skip
        h.loop.run_cycle()
        assert not any(k.startswith("balance:") for k in h.loop.risk.blockers)
        assert h.loop.order_manager.is_symbol_blocked("BTCUSDT")  # but the symbol is still blocked


class TestForeignOrders:
    def place_foreign(self, h: Harness) -> None:
        h.paper.place_order(
            OrderRequest(
                "someone-elses-1", "BTCUSDT", Side.BUY, OrderType.LIMIT, D("0.1"), D("90"), D("100")
            )
        )

    def test_a_foreign_open_order_blocks_entries_and_is_left_alone(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        self.place_foreign(h)
        h.cycle(hours=1)  # BUY signal
        assert buys(h) == []
        assert "orders:BTCUSDT" in h.loop.risk.blockers
        assert "Foreign open order" in critical_titles(h)
        (still_open,) = h.paper.open_orders("BTCUSDT")
        assert still_open.client_order_id == "someone-elses-1"  # never cancelled by the bot

    def test_the_block_lifts_by_itself_once_the_foreign_order_is_gone(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        self.place_foreign(h)
        h.cycle()
        assert "orders:BTCUSDT" in h.loop.risk.blockers
        h.paper.cancel_order("BTCUSDT", "someone-elses-1")
        h.cycle()
        assert h.loop.risk.blockers == {}

    def test_graceful_shutdown_does_not_cancel_foreign_orders(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        self.place_foreign(h)
        h.loop.request_stop()
        h.loop.run(max_cycles=1)
        assert [o.client_order_id for o in h.paper.open_orders("BTCUSDT")] == ["someone-elses-1"]

    def test_the_bots_own_orders_are_not_foreign(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        result = h.loop.order_manager.submit(
            OrderIntent("mine-1", "BTCUSDT", Side.BUY, OrderType.LIMIT, D("0.1"), D("100"), D("90"))
        )
        assert result.outcome == "accepted" and h.paper.open_orders("BTCUSDT")
        h.cycle()
        assert h.loop.risk.blockers == {}

    def test_when_open_orders_cannot_be_listed_entries_stop_but_the_stop_loss_still_works(
        self, cfg: Settings, rules: SymbolRules, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = harness(cfg, rules, store)
        h.cycle()
        h.cycle(hours=1)  # bot buys
        assert h.loop.positions.get("BTCUSDT") is not None

        def down(_symbol: str) -> list[object]:
            raise NetworkError("exchange unreachable")

        monkeypatch.setattr(h.paper, "open_orders", down)
        h.feed.bid, h.feed.ask = D("97.00"), D("97.01")  # through the stop
        h.loop.run_cycle()
        assert "orders:BTCUSDT" in h.loop.risk.blockers  # fail closed for entries
        stops = [o for o in h.orders() if o.intent_key.startswith("stop:BTCUSDT:")]
        assert stops and stops[0].status is OrderStatus.FILLED  # protection was not starved


class TestTheUsersOwnTradesDoNotHideTheBotsPosition:
    """Review: if the user sold the coins he held before the bot started, the bot's own position
    looked like zero (no stop-loss) and the daily loss was mis-stated."""

    def test_ambiguity_is_reported_loudly_and_an_acknowledgement_restores_protection(
        self, cfg: Settings, rules: SymbolRules, store: Store
    ) -> None:
        h = harness(cfg, rules, store)
        h.paper._free["BTC"] = D(1)  # the user already held 1 BTC when the bot first looked
        h.cycle()
        h.cycle(hours=1)  # the bot buys
        own = h.loop.order_manager.open_orders() or buys(h)
        assert own
        bought = h.paper.balances()["BTC"].free - D(1)
        h.paper._free["BTC"] -= D(1)  # the user sells HIS coin on the exchange's own screen
        h.feed.bid, h.feed.ask = D("90.00"), D("90.01")  # and the market falls through the stop
        h.loop.run_cycle()
        # the bot cannot prove which coins are its own, so it sells nothing and says so
        assert [o for o in h.orders() if o.side is Side.SELL] == []
        messages = [m for _, t, m in h.notifier.items if t == "Unexpected balance change"]
        assert messages and "NOT covered by the stop-loss" in messages[0]
        # a human looks, confirms, and acknowledges: the whole remaining balance is the bot's
        store.kv_set("ack:balances", "1")
        h.clock.advance(61)
        h.loop.run_cycle()
        assert store.kv_get("baseline:BTC") == "0"
        stops = [o for o in h.orders() if o.intent_key.startswith("stop:")]
        assert stops and stops[0].status is OrderStatus.FILLED
        assert stops[0].quantity <= bought
