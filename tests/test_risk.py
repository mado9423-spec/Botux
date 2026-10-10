"""Position sizing and risk limits. Requirement: sizing + risk limits + daily loss."""

from __future__ import annotations

import dataclasses
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from botux.clock import FakeClock
from botux.config import RiskSettings
from botux.exchange.rules import SymbolRules
from botux.models import OrderType, Side
from botux.risk.manager import DAY_MS, PortfolioView, RiskDecision, RiskManager
from tests.conftest import EXCHANGE_INFO_BTCUSDT

D = Decimal
PRICE = D("60000")
FEE, SLIP = D("0.001"), D(5)


def make_rm(**overrides: object) -> RiskManager:
    base = RiskSettings(
        max_order_notional=D(1000), max_position_notional=D(1000), max_total_exposure=D(1000)
    )
    return RiskManager(dataclasses.replace(base, **overrides), fee_rate=FEE, slippage_bps=SLIP)  # type: ignore[arg-type]


def view(
    equity: str = "1000",
    free: str = "1000",
    pos_notional: str = "0",
    pos_qty: str = "0",
    open_orders: int = 0,
    orders_today: int = 0,
) -> PortfolioView:
    return PortfolioView(
        equity=D(equity),
        free_quote=D(free),
        positions_notional={"BTCUSDT": D(pos_notional)},
        position_qty={"BTCUSDT": D(pos_qty)},
        open_orders=open_orders,
        orders_today=orders_today,
    )


def halted(rm: RiskManager) -> bool:
    """Indirection so the type checker does not narrow the property across method calls."""
    return rm.daily_loss_halted


class TestSizing:
    def test_risk_per_trade_is_the_binding_limit(self, rules: SymbolRules) -> None:
        # equity 1000 * 0.5% = 5 USDT at risk. Loss per BTC if the stop is hit = 2% stop distance
        # + slippage (2 x 0.05%) + fees (2 x 0.1%) = 2.3% of 60000 = 1380 -> 0.003623 -> 0.00362
        res = make_rm().size_entry(rules, PRICE, view())
        assert res.binding == "risk_per_trade"
        assert res.quantity == D("0.00362")
        # the worst case INCLUDING trading costs stays inside the risk budget
        assert res.quantity * PRICE * D("0.023") <= D("1000") * D("0.005")

    def test_positions_that_could_not_be_sold_again_are_refused(self, rules: SymbolRules) -> None:
        # 25 USDT equity -> 0.00009 BTC (5.4 USDT, passes the buy-side minimum), but after the
        # 0.1% fee only 0.00008 BTC (4.8 USDT) would be sellable: below MIN_NOTIONAL, so a
        # stop-loss could never be executed. The entry must not be opened.
        res = make_rm().size_entry(rules, PRICE, view(equity="25", free="25"))
        assert res.quantity == 0
        assert any("could not be sold again" in r for r in res.reasons)

    def test_hard_notional_cap_binds_before_risk(self, rules: SymbolRules) -> None:
        res = make_rm(max_order_notional=D(100)).size_entry(rules, PRICE, view())
        assert res.binding == "max_order_notional"
        assert res.quantity * PRICE <= 100

    def test_available_cash_binds_and_reserves_fee_and_slippage(self, rules: SymbolRules) -> None:
        res = make_rm().size_entry(rules, PRICE, view(free="50"))
        assert res.binding == "available_cash"
        assert res.quantity * PRICE * (1 + SLIP / 10_000 + FEE) <= 50

    def test_existing_position_reduces_headroom(self, rules: SymbolRules) -> None:
        res = make_rm(max_position_notional=D(300), max_order_notional=D(300)).size_entry(
            rules, PRICE, view(pos_notional="290", pos_qty="0.0048")
        )
        # only 10 USDT of headroom left under the per-symbol cap -> 0.00016 BTC (9.6 USDT)
        assert res.binding == "max_position_notional"
        assert res.quantity == D("0.00016") and res.quantity * PRICE <= 10

    def test_total_exposure_cap(self, rules: SymbolRules) -> None:
        res = make_rm(max_total_exposure=D(1000)).size_entry(
            rules, PRICE, view(pos_notional="1000")
        )
        assert res.quantity == 0 and res.binding in {"max_position_notional", "max_total_exposure"}

    def test_never_rounds_up_to_reach_the_exchange_minimum(self, rules: SymbolRules) -> None:
        # equity 10: 0.05 USDT at risk -> ~0.0000416 BTC (~2.5 USDT) < min notional 5
        res = make_rm().size_entry(rules, PRICE, view(equity="10", free="10"))
        assert res.quantity == 0
        assert any("NOT be rounded up" in r for r in res.reasons)

    def test_zero_equity_and_bad_price(self, rules: SymbolRules) -> None:
        assert make_rm().size_entry(rules, PRICE, view(equity="0")).quantity == 0
        assert make_rm().size_entry(rules, D(0), view()).quantity == 0

    def test_halted_manager_sizes_nothing(self, rules: SymbolRules) -> None:
        rm = make_rm()
        rm.halt("test")
        res = rm.size_entry(rules, PRICE, view())
        assert res.quantity == 0 and "halted" in res.reasons[0]

    @settings(max_examples=150, deadline=None)
    @given(
        equity=st.decimals(min_value=1, max_value=200_000, places=2),
        free_ratio=st.decimals(min_value=0, max_value=1, places=3),
        price=st.decimals(min_value=100, max_value=200_000, places=2),
        pos=st.decimals(min_value=0, max_value=1500, places=2),
    )
    def test_sizing_invariants(
        self, equity: Decimal, free_ratio: Decimal, price: Decimal, pos: Decimal
    ) -> None:
        rules = SymbolRules.from_exchange_info(EXCHANGE_INFO_BTCUSDT, "BTCUSDT")
        rm = make_rm(
            max_order_notional=D(500), max_position_notional=D(800), max_total_exposure=D(900)
        )
        free = equity * free_ratio
        v = view(equity=str(equity), free=str(free), pos_notional=str(pos))
        res = rm.size_entry(rules, price, v)
        q = res.quantity
        assert q >= 0
        if q == 0:
            return
        notional = q * price
        assert q % rules.step_size == 0 and q >= rules.min_qty
        assert notional <= 500
        assert pos + notional <= 800
        assert pos + notional <= 900
        assert notional * (1 + SLIP / 10_000 + FEE) <= free
        assert notional * D("0.02") <= equity * D("0.005")  # risk at the stop stays within budget


class TestOrderGate:
    def check(
        self,
        rm: RiskManager,
        rules: SymbolRules,
        side: Side,
        qty: str,
        v: PortfolioView,
        order_type: OrderType = OrderType.MARKET,
        price: str | None = None,
    ) -> RiskDecision:
        return rm.check_order(
            rules, side, order_type, D(qty), D(price) if price else None, PRICE, v
        )

    def test_a_normal_buy_is_allowed(self, rules: SymbolRules) -> None:
        assert self.check(make_rm(), rules, Side.BUY, "0.004", view()).allowed

    def test_no_shorting_on_spot(self, rules: SymbolRules) -> None:
        d = self.check(make_rm(), rules, Side.SELL, "0.01", view(pos_qty="0.005"))
        assert not d.allowed and any("no shorting" in r for r in d.reasons)

    def test_selling_what_we_hold_is_allowed(self, rules: SymbolRules) -> None:
        assert self.check(make_rm(), rules, Side.SELL, "0.005", view(pos_qty="0.005")).allowed

    def test_exits_stay_possible_when_entries_are_halted(self, rules: SymbolRules) -> None:
        rm = make_rm()
        rm.halt("kill")
        assert self.check(rm, rules, Side.SELL, "0.005", view(pos_qty="0.005")).allowed
        assert not self.check(rm, rules, Side.BUY, "0.004", view()).allowed

    def test_each_cap_is_enforced(self, rules: SymbolRules) -> None:
        rm = make_rm(
            max_order_notional=D(100), max_position_notional=D(150), max_total_exposure=D(180)
        )
        assert any(
            "max_order_notional" in r
            for r in self.check(rm, rules, Side.BUY, "0.002", view()).reasons
        )  # 120 > 100
        assert any(
            "max_position_notional" in r
            for r in self.check(rm, rules, Side.BUY, "0.001", view(pos_notional="100")).reasons
        )
        rm2 = make_rm(
            max_order_notional=D(100), max_position_notional=D(200), max_total_exposure=D(200)
        )
        v = PortfolioView(D(1000), D(1000), {"BTCUSDT": D(0), "ETHUSDT": D(150)}, {"BTCUSDT": D(0)})
        assert any(
            "max_total_exposure" in r
            for r in rm2.check_order(
                rules, Side.BUY, OrderType.MARKET, D("0.001"), None, PRICE, v
            ).reasons
        )

    def test_insufficient_cash_is_rejected(self, rules: SymbolRules) -> None:
        d = self.check(make_rm(), rules, Side.BUY, "0.004", view(free="100"))
        assert any("insufficient free quote" in r for r in d.reasons)

    def test_order_count_limits(self, rules: SymbolRules) -> None:
        rm = make_rm(max_open_orders=2, max_orders_per_day=5)
        assert any(
            "max_open_orders" in r
            for r in self.check(rm, rules, Side.BUY, "0.001", view(open_orders=2)).reasons
        )
        assert any(
            "max_orders_per_day" in r
            for r in self.check(rm, rules, Side.BUY, "0.001", view(orders_today=5)).reasons
        )
        # exits are not blocked by the per-day entry counter
        assert self.check(
            rm, rules, Side.SELL, "0.005", view(pos_qty="0.005", orders_today=99)
        ).allowed

    def test_limit_price_far_from_reference_is_a_fat_finger(self, rules: SymbolRules) -> None:
        d = self.check(make_rm(), rules, Side.BUY, "0.001", view(), OrderType.LIMIT, "70000.00")
        assert any("deviates" in r for r in d.reasons)
        assert self.check(
            make_rm(), rules, Side.BUY, "0.001", view(), OrderType.LIMIT, "60100.00"
        ).allowed

    def test_all_violations_are_reported_together(self, rules: SymbolRules) -> None:
        d = self.check(make_rm(max_order_notional=D(10)), rules, Side.BUY, "0.004", view(free="1"))
        assert len(d.reasons) >= 2

    def test_exchange_filters_are_part_of_the_gate(self, rules: SymbolRules) -> None:
        d = self.check(make_rm(), rules, Side.BUY, "0.000001", view())
        assert any("LOT_SIZE" in r for r in d.reasons)


class TestDailyLossAndErrors:
    def test_daily_loss_halts_entries_and_only_a_human_lifts_it(self) -> None:
        rm = make_rm(max_daily_loss=D(20))
        day0 = 100 * DAY_MS + 3_600_000
        rm.update_equity(day0, D(1000))
        rm.update_equity(day0 + 1000, D(985))
        assert not halted(rm) and rm.entries_allowed()[0]
        rm.update_equity(day0 + 2000, D(980))  # exactly -20: limit reached
        assert halted(rm) and not rm.entries_allowed()[0]
        assert rm.daily_pnl(D(980)) == D(-20)
        # midnight does NOT lift it (the day's reference moves, the halt stays) ...
        rm.update_equity(101 * DAY_MS + 1, D(980))
        assert halted(rm) and rm.day_start_equity == D(980)
        assert "reset-daily-loss" in rm.entries_allowed()[1]
        rm.update_equity(102 * DAY_MS + 1, D(1100))  # not even a recovery does
        assert halted(rm)
        # ... a human decision does, and starts a fresh reference
        assert rm.reset_daily_loss(D(1100)) is True
        assert not halted(rm) and rm.day_start_equity == D(1100) and rm.entries_allowed()[0]
        assert rm.reset_daily_loss(D(1100)) is False  # nothing left to lift
        rm.update_equity(102 * DAY_MS + 2, D(1080))  # the limit applies afresh
        assert halted(rm)

    def test_rebase_moves_the_reference_without_lifting_a_halt(self) -> None:
        rm = make_rm(max_daily_loss=D(20))
        rm.update_equity(5 * DAY_MS, D(1000))
        rm.rebase_day_start()
        rm.update_equity(5 * DAY_MS + 5, D(400))  # capital was reduced by the operator
        assert rm.day_start_equity == D(400) and not halted(rm)
        rm.update_equity(5 * DAY_MS + 6, D(370))
        assert halted(rm)
        rm.rebase_day_start()
        rm.update_equity(5 * DAY_MS + 7, D(370))
        assert halted(rm)  # a rebase never un-halts

    def test_blockers_stop_entries_and_clear_themselves(self) -> None:
        rm = make_rm()
        assert rm.set_blocker("balance:BTC", "x") is True
        assert rm.set_blocker("balance:BTC", "x") is False  # unchanged: not "new"
        ok, why = rm.entries_allowed()
        assert not ok and "balance:BTC" in why
        assert rm.clear_blocker("balance:BTC") is True and rm.clear_blocker("balance:BTC") is False
        assert rm.entries_allowed()[0]
        assert "blockers" not in rm.export_state()  # recomputed from facts, never persisted

    def test_state_survives_a_restart(self) -> None:
        rm = make_rm(max_daily_loss=D(20))
        rm.update_equity(5 * DAY_MS, D(1000))
        rm.update_equity(5 * DAY_MS + 10, D(900))
        state = rm.export_state()
        fresh = make_rm(max_daily_loss=D(20))
        fresh.restore_state(state)
        assert fresh.daily_loss_halted and fresh.day_start_equity == D(1000)
        fresh.update_equity(5 * DAY_MS + 20, D(1000))  # recovering inside the day must not un-halt
        assert fresh.daily_loss_halted

    def test_consecutive_errors_reach_the_limit_and_reset_on_success(self) -> None:
        rm = make_rm(max_consecutive_errors=3)
        assert [rm.record_error(), rm.record_error()] == [False, False]
        rm.record_success()
        assert [rm.record_error(), rm.record_error(), rm.record_error()] == [False, False, True]

    def test_stop_price(self) -> None:
        assert make_rm(stop_loss_pct=D("0.02")).stop_price(D(1000)) == D(980)

    def test_clock_parameter_is_optional(self) -> None:
        RiskManager(RiskSettings(), fee_rate=FEE, slippage_bps=SLIP, clock=FakeClock())
