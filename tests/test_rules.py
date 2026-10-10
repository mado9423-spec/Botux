"""Exchange filters: price/quantity precision, venue limits and rounding."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from botux.decimals import D, ceil_to_step, floor_to_step, fmt
from botux.errors import DataError, FilterViolation
from botux.exchange.rules import SymbolRules
from botux.models import OrderType, Side

DEC = Decimal


class TestDecimalHelpers:
    def test_floats_are_rejected(self) -> None:
        with pytest.raises(TypeError):
            D(0.1)  # type: ignore[arg-type]

    def test_invalid_and_non_finite_rejected(self) -> None:
        with pytest.raises(ValueError):
            D("abc")
        with pytest.raises(ValueError):
            D("NaN")

    def test_fmt_plain_notation(self) -> None:
        assert fmt(DEC("0.00100000")) == "0.001"
        assert fmt(DEC("1E+2")) == "100"
        assert fmt(DEC("100.50")) == "100.5"
        assert fmt(DEC("0")) == "0"

    @given(
        st.decimals(min_value=0, max_value=10**6, places=8),
        st.decimals(min_value=DEC("0.00000001"), max_value=1000, places=8),
    )
    def test_floor_to_step_properties(self, value: Decimal, step: Decimal) -> None:
        out = floor_to_step(value, step)
        assert out <= value
        assert value - out < step
        assert out % step == 0
        assert ceil_to_step(value, step) >= value


class TestParsing:
    def test_parses_all_relevant_filters(self, rules: SymbolRules) -> None:
        assert rules.tick_size == DEC("0.01")
        assert rules.step_size == DEC("0.00001")
        assert rules.min_qty == DEC("0.00001")
        assert rules.min_notional == DEC("5")
        assert rules.apply_min_to_market is True
        assert rules.market_max_qty == DEC("107")
        assert rules.percent_up_buy == DEC("5")
        assert rules.max_num_orders == 200
        assert rules.base_asset == "BTC" and rules.quote_asset == "USDT"

    def test_unknown_symbol(self, exchange_info: dict) -> None:  # type: ignore[type-arg]
        with pytest.raises(DataError):
            SymbolRules.from_exchange_info(exchange_info, "ETHUSDT")

    def test_legacy_min_notional_filter(self, exchange_info: dict) -> None:  # type: ignore[type-arg]
        filters = exchange_info["symbols"][0]["filters"]
        filters[:] = [f for f in filters if f["filterType"] != "NOTIONAL"]
        filters.append({"filterType": "MIN_NOTIONAL", "minNotional": "10", "applyToMarket": False})
        r = SymbolRules.from_exchange_info(exchange_info, "BTCUSDT")
        assert r.min_notional == DEC("10") and r.apply_min_to_market is False


class TestBothNotionalFilters:
    def test_min_notional_and_notional_are_evaluated_independently(
        self, exchange_info: dict[str, Any]
    ) -> None:
        filters = exchange_info["symbols"][0]["filters"]
        # NOTIONAL (from the fixture): min 5, applies to market. Add a legacy MIN_NOTIONAL of 10
        # that does NOT apply to market orders, and make NOTIONAL not apply to market either.
        for f in filters:
            if f["filterType"] == "NOTIONAL":
                f["applyMinToMarket"] = False
        filters.append({"filterType": "MIN_NOTIONAL", "minNotional": "10", "applyToMarket": True})
        r = SymbolRules.from_exchange_info(exchange_info, "BTCUSDT")
        assert r.min_notional == DEC("10") and r.extra_min_notionals == ((DEC("5"), False),)
        qty = DEC("0.00009")  # 5.4 USDT at 60000: above 5, below 10
        assert any(
            "NOTIONAL" in p
            for p in r.violations(Side.BUY, OrderType.MARKET, qty, None, DEC("60000"))
        )
        # as a LIMIT order both minimums apply, and the larger one binds
        assert r.violations(Side.BUY, OrderType.LIMIT, qty, DEC("60000.00"), DEC("60000"))
        ok = DEC("0.00020")  # 12 USDT
        assert r.violations(Side.BUY, OrderType.MARKET, ok, None, DEC("60000")) == []


class TestSecondNotionalMinimumBindsOnItsOwn:
    def test_the_nonprimary_filter_can_be_the_one_that_rejects(
        self, exchange_info: dict[str, Any]
    ) -> None:
        filters = exchange_info["symbols"][0]["filters"]
        for f in filters:
            if f["filterType"] == "NOTIONAL":
                f["minNotional"] = "10"  # applies to MARKET orders
        filters.append({"filterType": "MIN_NOTIONAL", "minNotional": "5", "applyToMarket": False})
        r = SymbolRules.from_exchange_info(exchange_info, "BTCUSDT")
        assert r.min_notional == DEC("5") and r.extra_min_notionals == ((DEC("10"), True),)
        qty = DEC("0.00009")  # 5.4 USDT: passes the primary (5) but not the second (10)
        assert any(
            "NOTIONAL" in p
            for p in r.violations(Side.BUY, OrderType.MARKET, qty, None, DEC("60000"))
        )


class TestRounding:
    def test_quantity_is_floored_never_rounded_up(self, rules: SymbolRules) -> None:
        assert rules.round_qty(DEC("0.123456789")) == DEC("0.12345")
        assert rules.round_qty(DEC("0.000019999")) == DEC("0.00001")
        assert rules.round_qty(DEC("0.000009999")) == DEC("0")

    def test_price_is_floored_to_tick(self, rules: SymbolRules) -> None:
        assert rules.round_price(DEC("65000.019")) == DEC("65000.01")

    def test_market_orders_honour_both_lot_filters(self, exchange_info: dict) -> None:  # type: ignore[type-arg]
        # Make MARKET_LOT_SIZE enforce a coarser step than LOT_SIZE.
        for f in exchange_info["symbols"][0]["filters"]:
            if f["filterType"] == "MARKET_LOT_SIZE":
                f["stepSize"] = "0.001"
        r = SymbolRules.from_exchange_info(exchange_info, "BTCUSDT")
        assert r.round_qty(DEC("0.12345"), OrderType.MARKET) == DEC("0.123")
        assert r.round_qty(DEC("0.12345"), OrderType.LIMIT) == DEC("0.12345")
        assert r.violations(Side.BUY, OrderType.MARKET, DEC("0.12345"), None, DEC("60000"))

    @given(st.decimals(min_value=0, max_value=200, places=12))
    def test_rounded_quantity_always_passes_lot_filters(self, qty: Decimal) -> None:
        from tests.conftest import EXCHANGE_INFO_BTCUSDT

        rules = SymbolRules.from_exchange_info(EXCHANGE_INFO_BTCUSDT, "BTCUSDT")
        rounded = rules.round_qty(qty)
        assert rounded <= qty
        assert rounded % rules.step_size == 0
        if rounded >= rules.min_qty:
            problems = [
                p
                for p in rules.violations(Side.SELL, OrderType.MARKET, rounded, None, DEC("100000"))
                if "LOT_SIZE" in p
            ]
            # Only the market max (107) can legitimately fail here.
            assert all("maxQty" in p for p in problems)


class TestValidation:
    def ok(self, rules: SymbolRules, qty: str, price: str | None = None, **kw: object) -> list[str]:
        order_type = OrderType.LIMIT if price else OrderType.MARKET
        return rules.violations(
            kw.get("side", Side.BUY),  # type: ignore[arg-type]
            order_type,
            DEC(qty),
            DEC(price) if price else None,
            DEC(str(kw.get("ref", "60000"))),
        )

    def test_valid_market_and_limit_orders_pass(self, rules: SymbolRules) -> None:
        assert self.ok(rules, "0.001") == []
        assert self.ok(rules, "0.001", "60000.00") == []

    def test_below_min_qty(self, rules: SymbolRules) -> None:
        assert any("LOT_SIZE" in p for p in self.ok(rules, "0.000001"))

    def test_off_step_quantity(self, rules: SymbolRules) -> None:
        assert any("multiple" in p for p in self.ok(rules, "0.001234"))

    def test_below_min_notional(self, rules: SymbolRules) -> None:
        # 0.00005 BTC * 60000 = 3 USDT < 5
        assert any("NOTIONAL" in p for p in self.ok(rules, "0.00005"))

    def test_notional_buffer_demands_headroom(self, rules: SymbolRules) -> None:
        # 0.0000835 BTC is not on the step; use 0.00009 * 60000 = 5.4 passes plain, fails x1.5
        qty = DEC("0.00009")
        assert not rules.violations(Side.BUY, OrderType.MARKET, qty, None, DEC("60000"))
        assert rules.violations(
            Side.BUY, OrderType.MARKET, qty, None, DEC("60000"), notional_buffer=DEC("1.5")
        )

    def test_off_tick_price(self, rules: SymbolRules) -> None:
        assert any("PRICE_FILTER" in p for p in self.ok(rules, "0.001", "60000.005"))

    def test_price_far_from_average_fails_percent_price(self, rules: SymbolRules) -> None:
        assert any("PERCENT_PRICE" in p for p in self.ok(rules, "0.001", "5000.00"))
        assert any("PERCENT_PRICE" in p for p in self.ok(rules, "0.001", "400000.00"))

    def test_halted_symbol_rejected(self, exchange_info: dict) -> None:  # type: ignore[type-arg]
        exchange_info["symbols"][0]["status"] = "HALT"
        r = SymbolRules.from_exchange_info(exchange_info, "BTCUSDT")
        assert any("not TRADING" in p for p in self.ok(r, "0.001"))

    def test_unsupported_order_type(self, rules: SymbolRules) -> None:
        r = SymbolRules.from_exchange_info(
            {"symbols": [{**_entry(rules), "orderTypes": ["LIMIT"]}]}, "BTCUSDT"
        )
        assert any("not supported" in p for p in self.ok(r, "0.001"))

    def test_missing_reference_price_for_market_order(self, rules: SymbolRules) -> None:
        problems = rules.violations(Side.BUY, OrderType.MARKET, DEC("0.001"), None, None)
        assert any("reference price" in p for p in problems)

    def test_validate_raises(self, rules: SymbolRules) -> None:
        with pytest.raises(FilterViolation):
            rules.validate(Side.BUY, OrderType.MARKET, DEC("0.000001"), None, DEC("60000"))

    def test_zero_or_negative_quantity(self, rules: SymbolRules) -> None:
        assert self.ok(rules, "0") and self.ok(rules, "-1")


def _entry(rules: SymbolRules) -> dict:  # type: ignore[type-arg]
    from tests.conftest import EXCHANGE_INFO_BTCUSDT

    return dict(EXCHANGE_INFO_BTCUSDT["symbols"][0])


def _with_filters(
    exchange_info: dict[str, Any], drop: tuple[str, ...] = (), **replacements: dict[str, Any]
) -> SymbolRules:
    """Rules built from the sample exchangeInfo with some filters replaced/added by type."""
    info = {**exchange_info, "symbols": [dict(exchange_info["symbols"][0])]}
    kept = [
        f
        for f in info["symbols"][0]["filters"]
        if f["filterType"] not in replacements and f["filterType"] not in drop
    ]
    info["symbols"][0]["filters"] = kept + list(replacements.values())
    return SymbolRules.from_exchange_info(info, "BTCUSDT")


NOTIONAL = {
    "filterType": "NOTIONAL",
    "minNotional": "5.00000000",
    "applyMinToMarket": True,
    "maxNotional": "100.00000000",
    "applyMaxToMarket": False,
    "avgPriceMins": 5,
}


class TestNotionalAndPercentPriceSemantics:
    """Pin the filter semantics of filters.md so a regression in the local validator is noticed."""

    def test_max_notional_applies_to_limit_orders_always_and_to_market_only_when_asked(
        self, exchange_info: dict[str, Any]
    ) -> None:
        not_to_market = _with_filters(exchange_info, NOTIONAL=NOTIONAL)
        to_market = _with_filters(exchange_info, NOTIONAL={**NOTIONAL, "applyMaxToMarket": True})
        big = DEC("0.002")  # 0.002 x 60000 = 120 USDT > 100
        limit = (Side.BUY, OrderType.LIMIT, big, DEC("60000.00"), DEC("60000"))
        market = (Side.BUY, OrderType.MARKET, big, None, DEC("60000"))
        assert any("NOTIONAL" in p and "> 100" in p for p in not_to_market.violations(*limit))
        assert not any("> 100" in p for p in not_to_market.violations(*market))
        assert any("NOTIONAL" in p and "> 100" in p for p in to_market.violations(*market))
        small = (Side.BUY, OrderType.MARKET, DEC("0.001"), None, DEC("60000"))  # 60 USDT
        assert to_market.violations(*small) == []

    def test_a_market_order_below_min_notional_passes_when_it_is_not_applied_to_market(
        self, exchange_info: dict[str, Any]
    ) -> None:
        r = _with_filters(exchange_info, NOTIONAL={**NOTIONAL, "applyMinToMarket": False})
        tiny = DEC("0.00005")  # 3 USDT < 5
        assert r.violations(Side.BUY, OrderType.MARKET, tiny, None, DEC("60000")) == []
        assert any(
            "NOTIONAL" in p
            for p in r.violations(Side.BUY, OrderType.LIMIT, tiny, DEC("60000.00"), DEC("60000"))
        )  # ... but a LIMIT order always has to meet it

    def test_percent_price_by_side_uses_the_bid_multipliers_for_buys_and_ask_for_sells(
        self, exchange_info: dict[str, Any]
    ) -> None:
        r = _with_filters(
            exchange_info,
            PERCENT_PRICE_BY_SIDE={
                "filterType": "PERCENT_PRICE_BY_SIDE",
                "bidMultiplierUp": "1.1", "bidMultiplierDown": "0.9",
                "askMultiplierUp": "1.5", "askMultiplierDown": "0.5",
                "avgPriceMins": 5,
            },
        )  # fmt: skip
        ref = DEC("60000")
        price = DEC("80000.00")  # +33%: too high for a BUY (limit 1.1), fine for a SELL (1.5)
        buy = r.violations(Side.BUY, OrderType.LIMIT, DEC("0.001"), price, ref)
        sell = r.violations(Side.SELL, OrderType.LIMIT, DEC("0.001"), price, ref)
        assert any("PERCENT_PRICE" in p and "too high" in p for p in buy)
        assert not any("PERCENT_PRICE" in p for p in sell)
        low = DEC("40000.00")  # -33%: too low for a SELL (0.5 floor is fine) vs a BUY (0.9)
        assert any(
            "too low" in p for p in r.violations(Side.BUY, OrderType.LIMIT, DEC("0.001"), low, ref)
        )
        assert not any(
            "PERCENT_PRICE" in p
            for p in r.violations(Side.SELL, OrderType.LIMIT, DEC("0.001"), low, ref)
        )

    def test_the_legacy_percent_price_filter_maps_up_and_down_correctly(
        self, exchange_info: dict[str, Any]
    ) -> None:
        r = _with_filters(
            exchange_info,
            drop=("PERCENT_PRICE_BY_SIDE",),
            PERCENT_PRICE={
                "filterType": "PERCENT_PRICE", "multiplierUp": "1.2", "multiplierDown": "0.8",
                "avgPriceMins": 5,
            },
        )  # fmt: skip
        assert (r.percent_up_buy, r.percent_down_buy) == (DEC("1.2"), DEC("0.8"))
        assert (r.percent_up_sell, r.percent_down_sell) == (DEC("1.2"), DEC("0.8"))
        ref = DEC("1000")

        def problems(side: Side, price: str) -> list[str]:
            return r.violations(side, OrderType.LIMIT, DEC("1"), DEC(price), ref)

        for side in (Side.BUY, Side.SELL):
            assert any("too high" in p for p in problems(side, "1300.00"))
            assert any("too low" in p for p in problems(side, "700.00"))
            assert not any("PERCENT_PRICE" in p for p in problems(side, "1000.00"))

    def test_both_min_notional_and_notional_are_kept_and_evaluated_independently(
        self, exchange_info: dict[str, Any]
    ) -> None:
        r = _with_filters(
            exchange_info,
            MIN_NOTIONAL={
                "filterType": "MIN_NOTIONAL",
                "minNotional": "20.0",
                "applyToMarket": True,
                "avgPriceMins": 5,
            },
            NOTIONAL=NOTIONAL,
        )
        problems = r.violations(Side.BUY, OrderType.MARKET, DEC("0.0002"), None, DEC("60000"))  # 12
        assert any("NOTIONAL" in p for p in problems)  # passes the 5 minimum, fails the 20 one


class TestStatusIsAnEntryRuleNotAnExitRule:
    """A protective SELL must not be refused locally because a stale status says not-TRADING."""

    def test_buys_are_refused_but_sells_are_left_to_the_exchange(
        self, exchange_info: dict[str, Any]
    ) -> None:
        exchange_info["symbols"][0]["status"] = "BREAK"
        r = SymbolRules.from_exchange_info(exchange_info, "BTCUSDT")
        args = (OrderType.MARKET, DEC("0.001"), None, DEC("60000"))
        assert any("not TRADING" in p for p in r.violations(Side.BUY, *args))
        assert not any("not TRADING" in p for p in r.violations(Side.SELL, *args))

    def test_cancel_only_status_is_parsed_and_blocks_entries(
        self, exchange_info: dict[str, Any]
    ) -> None:
        exchange_info["symbols"][0]["status"] = "CANCEL_ONLY"
        r = SymbolRules.from_exchange_info(exchange_info, "BTCUSDT")
        assert r.violations(Side.BUY, OrderType.MARKET, DEC("0.001"), None, DEC("60000"))


class TestMaxPosition:
    def test_it_is_parsed_but_not_enforced_locally(self, exchange_info: dict[str, Any]) -> None:
        r = _with_filters(
            exchange_info,
            MAX_POSITION={"filterType": "MAX_POSITION", "maxPosition": "10.00000000"},
        )
        assert r.max_position == DEC("10")
        assert r.violations(Side.BUY, OrderType.MARKET, DEC("0.001"), None, DEC("60000")) == []
