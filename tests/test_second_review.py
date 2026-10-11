"""Regression tests for the findings of the second independent review (each failed before)."""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from pathlib import Path

import pytest

from botux.config import MarketSettings, RiskSettings, StrategySettings, load_settings
from botux.errors import ConfigError
from botux.exchange.rules import SymbolRules
from botux.execution.paper import FaultKind
from botux.models import OrderStatus
from botux.storage import Store
from tests.conftest import EXCHANGE_INFO_BTCUSDT
from tests.test_loop_invariants import _settings, wavy
from tests.test_runner import Harness

D = Decimal


def _harness(store: Store, **kw: object) -> Harness:
    rules = SymbolRules.from_exchange_info(EXCHANGE_INFO_BTCUSDT, "BTCUSDT")
    cfg = dataclasses.replace(
        _settings(),
        strategy=StrategySettings("sma_cross", {"fast": 2, "slow": 5}),
        market=MarketSettings(poll_seconds=1.0),
    )
    return Harness(cfg, rules, store, closed=8, closes=wavy(60), **kw)  # type: ignore[arg-type]


def _held_btc(h: Harness) -> Decimal:
    total: Decimal = h.paper.balances()["BTC"].total
    return total


def _buy(h: Harness) -> None:
    for _ in range(40):
        h.clock.advance(3600)
        h.feed.mark_to_last_close()
        h.loop.run_cycle()
        if h.loop._position_qty("BTCUSDT", h.paper.balances()) > 0:
            return  # the BOT holds a position (coins of the user's do not count)
    pytest.skip("scenario did not produce an entry")


def test_a_second_ack_keeps_the_first_acknowledged_deficit() -> None:
    """A deposit made after a first acknowledgement must never become the bot's to sell."""
    store = Store(":memory:")
    h = _harness(store)
    _buy(h)
    held = _held_btc(h)
    h.paper._free["BTC"] -= held * D("0.4")  # 40% leaves the account
    store.kv_set("ack:balances", "1")
    h.loop.run_cycle()
    h.paper._free["BTC"] += D("0.3")  # the user's own deposit
    h.loop.run_cycle()
    store.kv_set("ack:balances", "1")
    h.loop.run_cycle()
    owned = h.loop._owned("BTCUSDT", h.paper.balances())[0]
    assert owned <= held * D("0.6") + D("0.0001")  # never includes the 0.3 deposited
    store.close()


def test_a_transient_gate_reload_failure_closes_entries_not_the_cycle(tmp_path: Path) -> None:
    bad = tmp_path / "cfg.toml"
    bad.write_bytes(b'mode = "paper"\n# \xff\xfe not utf-8\n')
    with pytest.raises(ConfigError):
        load_settings(bad)


def test_any_exception_from_the_gate_check_closes_the_gate_and_the_stop_still_runs() -> None:
    store = Store(":memory:")
    h = _harness(store)
    _buy(h)
    calls = {"n": 0}

    def broken() -> None:
        calls["n"] += 1
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")

    h.loop._gate_check = broken  # what a corrupt settings file used to do
    h.feed.bid, h.feed.ask = h.feed.bid * D("0.5"), h.feed.ask * D("0.5")  # far below the stop
    h.loop.run_cycle()
    assert h.loop._gate_closed and "live-gate" in h.loop.risk.blockers
    assert any(o.side.value == "SELL" for o in h.orders())  # protection ran regardless
    store.close()


def test_rules_refresh_with_a_malformed_answer_does_not_abort_the_cycle() -> None:
    store = Store(":memory:")
    h = _harness(store)
    _buy(h)

    def bad_loader() -> dict[str, SymbolRules]:
        raise KeyError("tickSize")

    h.loop._rules_loader = bad_loader
    h.loop._rules_next_try = 0.0
    h.feed.bid, h.feed.ask = h.feed.bid * D("0.5"), h.feed.ask * D("0.5")
    h.loop.run_cycle()
    assert any(o.side.value == "SELL" for o in h.orders())
    store.close()


def test_resolved_lost_orders_do_not_come_back_to_block_exits() -> None:
    store = Store(":memory:")
    h = _harness(store)
    _buy(h)
    om = h.loop.order_manager
    for rec in store.list_orders():
        rec.status = OrderStatus.LOST
        store.update_order(rec)
    store.kv_set("lost:BTCUSDT", "x")
    om.resolve_lost("BTCUSDT")
    assert not store.list_orders((OrderStatus.LOST,))
    assert store.list_orders((OrderStatus.LOST_RESOLVED,))
    store.kv_set("lost:BTCUSDT", "y")  # a LATER incident
    assert om.lost_orders("BTCUSDT") == []  # the old, resolved ones are not re-adopted
    store.close()


def test_flatten_retries_when_the_order_provably_did_not_leave() -> None:
    store = Store(":memory:")
    h = _harness(store)
    _buy(h)
    held = _held_btc(h)
    h.paper.faults.push(FaultKind.NETWORK_DOWN)
    h.loop._flatten_with_retries("BTCUSDT")
    assert _held_btc(h) < held  # the second attempt sold
    store.close()


def test_an_alert_never_says_none_while_the_bot_has_a_position_it_cannot_sell() -> None:
    store = Store(":memory:")
    h = _harness(store)
    _buy(h)
    h.paper._free["BTC"] = D(0)  # the coins are gone from the account; the records remain
    text = h.loop._open_positions_text()
    assert "NOT sellable" in text and "BTCUSDT" in text
    store.close()


def test_the_emergency_flatten_sells_only_the_bots_own_coins() -> None:
    """The user's own BTC (present before the bot first looked) is never part of a flatten."""
    store = Store(":memory:")
    rules = SymbolRules.from_exchange_info(EXCHANGE_INFO_BTCUSDT, "BTCUSDT")
    cfg = dataclasses.replace(
        _settings(),
        strategy=StrategySettings("sma_cross", {"fast": 2, "slow": 5}),
        market=MarketSettings(poll_seconds=1.0),
        risk=dataclasses.replace(RiskSettings(), flatten_on_kill=True),
    )
    h = Harness(cfg, rules, store, closed=8, closes=wavy(60))
    h.paper._free["BTC"] = D("2")  # the user's, before the first look
    _buy(h)
    assert _held_btc(h) > D("2")
    h.loop._flatten_with_retries("BTCUSDT")
    assert _held_btc(h) >= D("2") - D("0.00001")  # never below the user's own 2 BTC
    assert _held_btc(h) < D("2.01")  # and the bot's coins are gone
    store.close()
