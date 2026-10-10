"""The bot's own books: what it owns is what ITS fills explain, never the account's totals."""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from botux.exchange.rules import SymbolRules
from botux.ledger import (
    CAPITAL_KEY,
    EPOCH_KEY,
    Ledger,
    balance_discrepancy,
    build_ledger,
    owned_base,
    tolerance,
)
from botux.models import Fill, OrderStatus, OrderType, Side
from botux.storage import OrderRecord, Store

D = Decimal
FEE = D("0.001")


def view(rules: SymbolRules, rows: list[tuple[str, Side, Fill]], capital: str = "1000"):  # type: ignore[no-untyped-def]
    return build_ledger(D(capital), rows, {"BTCUSDT": rules}, FEE)


class TestFoldingFills:
    def test_empty_ledger_is_just_the_capital(self, rules: SymbolRules) -> None:
        v = view(rules, [])
        assert v.cash == 1000 and v.base_qty("BTCUSDT") == 0

    def test_buy_with_base_fee_spends_quote_and_receives_less_base(
        self, rules: SymbolRules
    ) -> None:
        v = view(rules, [("BTCUSDT", Side.BUY, Fill(D(100), D("2"), D("0.002"), "BTC"))])
        assert v.cash == 1000 - 200
        assert v.base_qty("BTCUSDT") == D("1.998")

    def test_sell_with_quote_fee_receives_less_quote(self, rules: SymbolRules) -> None:
        rows = [
            ("BTCUSDT", Side.BUY, Fill(D(100), D("2"), D("0.002"), "BTC")),
            ("BTCUSDT", Side.SELL, Fill(D(110), D("1.998"), D("0.21978"), "USDT")),
        ]
        v = view(rules, rows)
        assert v.base_qty("BTCUSDT") == 0
        assert v.cash == D("1000") - 200 + D("219.78") - D("0.21978")

    def test_quote_fee_on_a_buy_is_added_to_the_cost(self, rules: SymbolRules) -> None:
        v = view(rules, [("BTCUSDT", Side.BUY, Fill(D(100), D(1), D("0.1"), "USDT"))])
        assert v.cash == D("1000") - 100 - D("0.1") and v.base_qty("BTCUSDT") == 1

    def test_fee_in_a_third_asset_is_flagged_and_not_valued(self, rules: SymbolRules) -> None:
        v = view(rules, [("BTCUSDT", Side.BUY, Fill(D(100), D(1), D("0.0001"), "BNB"))])
        assert v.symbols["BTCUSDT"].third_asset_fees
        assert v.base_qty("BTCUSDT") == 1 and v.cash == 900

    def test_missing_commission_info_falls_back_to_the_configured_fee_and_is_marked(
        self, rules: SymbolRules
    ) -> None:
        v = view(rules, [("BTCUSDT", Side.BUY, Fill(D(100), D(1), D(0), ""))])
        assert v.base_qty("BTCUSDT") == D("0.999")
        assert v.symbols["BTCUSDT"].inexact_qty == 1
        s = view(
            rules,
            [
                ("BTCUSDT", Side.BUY, Fill(D(100), D(1), D(0), "BTC")),
                ("BTCUSDT", Side.SELL, Fill(D(100), D(1), D(0), "")),
            ],
        )
        assert s.cash == D("1000") - 100 + D("100") * (1 - FEE)

    def test_fills_of_unknown_symbols_are_ignored(self, rules: SymbolRules) -> None:
        v = view(rules, [("ETHUSDT", Side.BUY, Fill(D(100), D(1), D(0), "ETH"))])
        assert v.cash == 1000

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])  # rules is immutable
    @given(
        qty=st.decimals(min_value="0.001", max_value="5", places=3),
        buy=st.decimals(min_value="1", max_value="100000", places=2),
        sell=st.decimals(min_value="1", max_value="100000", places=2),
    )
    def test_a_flat_round_trip_never_creates_money_when_fees_are_paid(
        self, rules: SymbolRules, qty: Decimal, buy: Decimal, sell: Decimal
    ) -> None:
        fee_b = qty * FEE
        rows = [
            ("BTCUSDT", Side.BUY, Fill(buy, qty, fee_b, "BTC")),
            ("BTCUSDT", Side.SELL, Fill(sell, qty - fee_b, sell * (qty - fee_b) * FEE, "USDT")),
        ]
        v = view(rules, rows)
        assert v.base_qty("BTCUSDT") == 0
        gross = (sell - buy) * qty
        assert v.cash - 1000 < gross  # fees only ever subtract


class TestOwnershipArithmetic:
    def test_discrepancy_is_what_the_books_do_not_explain(self) -> None:
        assert balance_discrepancy(D("3"), D("1"), D("2"), D(0)) == 0
        assert balance_discrepancy(D("3.5"), D("1"), D("2"), D(0)) == D("0.5")  # a deposit
        assert balance_discrepancy(D("2.5"), D("1"), D("2"), D(0)) == D("-0.5")  # coins missing
        assert balance_discrepancy(D("3.5"), D("1"), D("2"), D("0.5")) == 0  # acknowledged

    def test_owned_is_capped_by_the_ledger_and_by_what_really_exists(self) -> None:
        assert owned_base(D("3"), D("1"), D("2"), D(0)) == 2
        assert (
            owned_base(D("9"), D("1"), D("2"), D("6")) == 2
        )  # an acknowledged surplus is external
        assert owned_base(D("1.5"), D("1"), D("2"), D(0)) == D("0.5")  # fewer coins than recorded
        assert owned_base(D("2.5"), D("1"), D("2"), D("-0.5")) == D("1.5")  # acknowledged deficit
        assert owned_base(D("0.5"), D("1"), D("2"), D(0)) == 0  # never negative

    @given(
        total=st.decimals(min_value="0", max_value="100", places=4),
        baseline=st.decimals(min_value="0", max_value="100", places=4),
        ledger=st.decimals(min_value="0", max_value="100", places=4),
        offset=st.decimals(min_value="-50", max_value="50", places=4),
    )
    def test_owned_never_exceeds_ledger_or_the_account_above_baseline(
        self, total: Decimal, baseline: Decimal, ledger: Decimal, offset: Decimal
    ) -> None:
        got = owned_base(total, baseline, ledger, offset)
        assert got >= 0 and got <= ledger and got <= max(D(0), total - baseline)

    def test_tolerance_is_two_lot_steps_plus_estimated_fees(self, rules: SymbolRules) -> None:
        v = view(rules, [("BTCUSDT", Side.BUY, Fill(D(100), D(10), D(0), ""))])
        assert tolerance(rules, v.symbols["BTCUSDT"], FEE) == rules.step_size * 2 + D(10) * FEE * 2
        exact = view(rules, [("BTCUSDT", Side.BUY, Fill(D(100), D(10), D("0.01"), "BTC"))])
        assert tolerance(rules, exact.symbols["BTCUSDT"], FEE) == rules.step_size * 2


class TestStoreBackedLedger:
    def _buy(self, store: Store, cid: str, qty: str) -> None:
        store.insert_order(
            OrderRecord(
                cid, f"k-{cid}", "BTCUSDT", Side.BUY, OrderType.MARKET, D(qty), None,
                OrderStatus.FILLED, 1, 1, executed_qty=D(qty), cumulative_quote_qty=D(qty) * 100,
            )
        )  # fmt: skip
        store.add_fills(cid, [Fill(D(100), D(qty), D(0), "BTC")], 1)

    def test_fills_before_the_epoch_are_not_this_ledgers(
        self, store: Store, rules: SymbolRules
    ) -> None:
        self._buy(store, "old", "5")  # an earlier (paper) session
        led = Ledger(store, {"BTCUSDT": rules}, capital=D(1000), fee_rate=FEE)
        assert led.view().base_qty("BTCUSDT") == 0 and led.view().cash == 1000
        self._buy(store, "new", "2")
        assert led.view().base_qty("BTCUSDT") == 2 and led.view().cash == 800

    def test_the_epoch_survives_a_restart(self, store: Store, rules: SymbolRules) -> None:
        led = Ledger(store, {"BTCUSDT": rules}, capital=D(1000), fee_rate=FEE)
        self._buy(store, "a", "1")
        again = Ledger(store, {"BTCUSDT": rules}, capital=D(1000), fee_rate=FEE)
        assert again.view().base_qty("BTCUSDT") == 1
        assert store.kv_get(EPOCH_KEY) == "0" and led.capital_changed_from is None

    def test_a_changed_capital_is_noticed(self, store: Store, rules: SymbolRules) -> None:
        Ledger(store, {"BTCUSDT": rules}, capital=D(1000), fee_rate=FEE)
        assert store.kv_get(CAPITAL_KEY) == "1000"
        changed = Ledger(store, {"BTCUSDT": rules}, capital=D(500), fee_rate=FEE)
        assert changed.capital_changed_from == 1000 and store.kv_get(CAPITAL_KEY) == "500"

    def test_offsets_round_trip(self, store: Store, rules: SymbolRules) -> None:
        led = Ledger(store, {"BTCUSDT": rules}, capital=D(1000), fee_rate=FEE)
        assert led.offset("BTC") == 0
        led.set_offset("BTC", D("-0.25"))
        assert led.offset("BTC") == D("-0.25")


@pytest.mark.parametrize("bad", ["0", "-5"])
def test_nonpositive_capital_is_a_config_error(bad: str) -> None:
    from botux.config import parse_settings
    from botux.errors import ConfigError

    with pytest.raises(ConfigError, match="bot_capital"):
        parse_settings({"risk": {"bot_capital": bad}})
