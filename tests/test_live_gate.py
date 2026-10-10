"""The live-trading gate. Real orders must be impossible unless EVERY condition holds.

Everything here runs against a fake HTTP transport: no real exchange is ever contacted.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

from botux.app import build_live_app, build_paper_app
from botux.clock import FakeClock
from botux.config import Settings, settings_fingerprint
from botux.errors import (
    ClockSkewError,
    LiveTradingNotAllowed,
    OrderRejected,
    OutcomeUnknownError,
)
from botux.exchange.client import BinanceSpotClient
from botux.exchange.rules import SymbolRules
from botux.execution.live import LiveBroker, parse_order
from botux.execution.order_manager import OrderIntent
from botux.killswitch import KillSwitch
from botux.models import OrderRequest, OrderStatus, OrderType, Side
from botux.safety import (
    ARM_PHRASE,
    ARM_VERSION,
    api_key_problems,
    arm_live_interactive,
    check_live_gate,
    disarm,
)
from botux.secrets_guard import Credentials, Secret
from botux.storage import Store
from tests.conftest import EXCHANGE_INFO_BTCUSDT, FakeExchange

D = Decimal
API_KEY, API_SECRET = "A" * 64, "B" * 64
ENV = {"BINANCE_API_KEY": API_KEY, "BINANCE_API_SECRET": API_SECRET}

GOOD_KEY = {
    "ipRestrict": True,
    "enableReading": True,
    "enableWithdrawals": False,
    "enableInternalTransfer": False,
    "permitsUniversalTransfer": False,
    "enableMargin": False,
    "enableFutures": False,
    "enableVanillaOptions": False,
    "enablePortfolioMarginTrading": False,
    "enableSpotAndMarginTrading": True,
}


def live(settings: Settings) -> Settings:
    return dataclasses.replace(
        settings, mode="live", live=dataclasses.replace(settings.live, enabled=True)
    )


def arm(settings: Settings, clock: FakeClock) -> Path:
    path = arm_live_interactive(
        settings, clock.now_ms(), input_fn=lambda _p: ARM_PHRASE, output_fn=lambda _t: None
    )
    assert path is not None
    return path


def kill_for(settings: Settings, clock: FakeClock) -> KillSwitch:
    return KillSwitch(settings.paths.kill_file, clock)


@pytest.fixture(autouse=True)
def _in_tmp_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # so a stray ./.env in the repo can never leak into a test


def live_exchange(clock: FakeClock, restrictions: dict[str, object] | None = None) -> FakeExchange:
    ex = FakeExchange()
    ex.on(
        "GET", "/api/v3/time", lambda _r: httpx.Response(200, json={"serverTime": clock.now_ms()})
    )
    ex.on(
        "GET",
        "/sapi/v1/account/apiRestrictions",
        dict(GOOD_KEY if restrictions is None else restrictions),
    )
    ex.on("GET", "/api/v3/exchangeInfo", EXCHANGE_INFO_BTCUSDT)

    def new_order(req: httpx.Request) -> httpx.Response:
        body = parse_qs(req.content.decode())
        return httpx.Response(
            200,
            json={
                "symbol": "BTCUSDT",
                "orderId": 28,
                "clientOrderId": body["newClientOrderId"][0],
                "transactTime": clock.now_ms(),
                "price": "0.00000000",
                "origQty": "0.00100000",
                "executedQty": "0.00100000",
                "cummulativeQuoteQty": "60.00000000",
                "status": "FILLED",
                "type": "MARKET",
                "side": "BUY",
                "fills": [
                    {
                        "price": "60000.00",
                        "qty": "0.00100000",
                        "commission": "0.00000100",
                        "commissionAsset": "BTC",
                        "tradeId": 1,
                    }
                ],
            },
        )

    ex.on("POST", "/api/v3/order", new_order)
    return ex


class TestGateRefusals:
    def test_default_settings_refuse(self, settings: Settings, clock: FakeClock) -> None:
        with pytest.raises(LiveTradingNotAllowed, match="mode is not 'live'"):
            check_live_gate(settings, kill_for(settings, clock), clock.now_ms())

    def test_live_mode_alone_is_not_enough(self, settings: Settings, clock: FakeClock) -> None:
        s = dataclasses.replace(settings, mode="live")
        with pytest.raises(LiveTradingNotAllowed, match="enabled is false"):
            check_live_gate(s, kill_for(s, clock), clock.now_ms())

    def test_enabled_but_never_armed(self, settings: Settings, clock: FakeClock) -> None:
        s = live(settings)
        with pytest.raises(LiveTradingNotAllowed, match="not armed"):
            check_live_gate(s, kill_for(s, clock), clock.now_ms())

    def test_every_unmet_condition_is_listed(self, settings: Settings, clock: FakeClock) -> None:
        with pytest.raises(LiveTradingNotAllowed) as e:
            check_live_gate(settings, kill_for(settings, clock), clock.now_ms())
        assert "mode" in str(e.value) and "enabled" in str(e.value) and "not armed" in str(e.value)

    def test_armed_and_enabled_passes(self, settings: Settings, clock: FakeClock) -> None:
        s = live(settings)
        arm(s, clock)
        auth = check_live_gate(s, kill_for(s, clock), clock.now_ms())
        assert auth.fingerprint == settings_fingerprint(s)

    def test_changing_any_setting_after_arming_invalidates_it(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        s = live(settings)
        arm(s, clock)
        looser = dataclasses.replace(s, risk=dataclasses.replace(s.risk, max_order_notional=D(101)))
        with pytest.raises(LiveTradingNotAllowed, match="settings changed"):
            check_live_gate(looser, kill_for(looser, clock), clock.now_ms())

    def test_arming_expires(self, settings: Settings, clock: FakeClock) -> None:
        s = live(settings)
        arm(s, clock)
        clock.advance(25 * 3600)
        with pytest.raises(LiveTradingNotAllowed, match="expired"):
            check_live_gate(s, kill_for(s, clock), clock.now_ms())

    def test_future_dated_arming_is_refused(self, settings: Settings, clock: FakeClock) -> None:
        s = live(settings)
        arm(s, clock)
        clock.advance(-3600)
        with pytest.raises(LiveTradingNotAllowed, match="future"):
            check_live_gate(s, kill_for(s, clock), clock.now_ms())

    def test_kill_switch_closes_the_gate(self, settings: Settings, clock: FakeClock) -> None:
        s = live(settings)
        arm(s, clock)
        kill = kill_for(s, clock)
        kill.trip("stop everything")
        with pytest.raises(LiveTradingNotAllowed, match="kill switch"):
            check_live_gate(s, kill, clock.now_ms())

    @pytest.mark.parametrize(
        "content",
        [
            "not json",
            "[]",
            json.dumps({"version": 99}),
            json.dumps(
                {"version": ARM_VERSION, "fingerprint": "x", "armed_at_ms": 0, "phrase_sha256": "y"}
            ),
        ],
    )
    def test_tampered_or_garbage_arming_files_are_refused(
        self, content: str, settings: Settings, clock: FakeClock
    ) -> None:
        s = live(settings)
        path = Path(s.live.arm_file)
        path.write_text(content)
        with pytest.raises(LiveTradingNotAllowed):
            check_live_gate(s, kill_for(s, clock), clock.now_ms())


class TestArmingCommand:
    def test_wrong_phrase_does_not_arm(self, settings: Settings, clock: FakeClock) -> None:
        s = live(settings)
        out: list[str] = []
        assert arm_live_interactive(s, clock.now_ms(), lambda _p: "yes", out.append) is None
        assert not Path(s.live.arm_file).exists()
        assert any("NOT armed" in line for line in out)

    def test_cannot_arm_while_the_config_is_paper(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        assert (
            arm_live_interactive(settings, clock.now_ms(), lambda _p: ARM_PHRASE, lambda _t: None)
            is None
        )
        assert not Path(settings.live.arm_file).exists()

    def test_the_risks_and_settings_are_shown_before_confirmation(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        s = live(settings)
        out: list[str] = []
        arm_live_interactive(s, clock.now_ms(), lambda _p: "", out.append)
        text = "\n".join(out)
        assert "lose part or all" in text and "max_order_notional" in text and "withdrawals" in text

    def test_disarm_removes_the_file(self, settings: Settings, clock: FakeClock) -> None:
        s = live(settings)
        path = arm(s, clock)
        disarm(s)
        assert not path.exists()


class TestKeysInTheEnvironmentNeverEnableLiveTrading:
    @pytest.fixture(autouse=True)
    def _bomb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(*_a: object, **_k: object) -> None:
            raise AssertionError("credentials were read although the gate is closed")

        monkeypatch.setattr("botux.execution.live.load_credentials", explode)
        monkeypatch.setattr("botux.config.load_credentials", explode)

    @pytest.mark.parametrize("variant", ["paper", "live_not_enabled", "live_enabled_not_armed"])
    def test_gate_closed_means_keys_are_never_even_read_and_no_request_is_made(
        self, variant: str, settings: Settings, clock: FakeClock, store: Store
    ) -> None:
        s = {
            "paper": settings,
            "live_not_enabled": dataclasses.replace(settings, mode="live"),
            "live_enabled_not_armed": live(settings),
        }[variant]
        ex = live_exchange(clock)
        with pytest.raises(LiveTradingNotAllowed):
            build_live_app(s, ENV, http=ex.http(), clock=clock, store=store)
        assert ex.requests == []

    def test_paper_app_ignores_keys_and_a_live_config(
        self, settings: Settings, clock: FakeClock, rules: SymbolRules, store: Store
    ) -> None:
        s = live(settings)  # even a fully 'live' config yields a simulator under build_paper_app
        arm(s, clock)
        ex = FakeExchange()
        app = build_paper_app(
            s,
            http=ex.http(),
            clock=clock,
            store=store,
            feed=object(),  # type: ignore[arg-type]
            rules={"BTCUSDT": rules},
        )
        assert app.paper_broker is not None and not app.client.has_credentials
        assert not app.client.trading_enabled and ex.requests == []


class TestApiKeyPermissions:
    def test_a_minimal_spot_key_is_accepted(self) -> None:
        assert api_key_problems(GOOD_KEY, True) == []

    @pytest.mark.parametrize(
        "flag",
        [
            "enableWithdrawals",
            "enableInternalTransfer",
            "permitsUniversalTransfer",
            "enableMargin",
            "enableFutures",
            "enableVanillaOptions",
            "enablePortfolioMarginTrading",
        ],
    )
    def test_dangerous_permissions_are_refused(self, flag: str) -> None:
        problems = api_key_problems(GOOD_KEY | {flag: True}, True)
        assert len(problems) == 1 and flag in problems[0]

    @pytest.mark.parametrize(
        "flag", ["enableWithdrawals", "enableMargin", "enableFutures", "ipRestrict"]
    )
    def test_missing_fields_fail_closed(self, flag: str) -> None:
        partial = {k: v for k, v in GOOD_KEY.items() if k != flag}
        assert api_key_problems(partial, True)

    def test_ip_restriction_requirement(self) -> None:
        unrestricted = GOOD_KEY | {"ipRestrict": False}
        assert any("IP" in p for p in api_key_problems(unrestricted, True))
        assert api_key_problems(unrestricted, False) == []

    def test_reading_and_spot_trading_must_be_enabled(self) -> None:
        assert (
            len(
                api_key_problems(
                    GOOD_KEY | {"enableReading": False, "enableSpotAndMarginTrading": False}, True
                )
            )
            == 2
        )


class TestMockedLiveBringUp:
    def setup(self, settings: Settings, clock: FakeClock, **kw: object):  # type: ignore[no-untyped-def]
        s = live(settings)
        arm(s, clock)
        ex = live_exchange(clock, **kw)  # type: ignore[arg-type]
        return s, ex

    def test_full_bring_up_and_one_order(
        self, settings: Settings, clock: FakeClock, caplog: pytest.LogCaptureFixture, store: Store
    ) -> None:
        s, ex = self.setup(settings, clock)
        with caplog.at_level(logging.DEBUG):
            app = build_live_app(s, ENV, http=ex.http(), clock=clock, store=store)
            assert app.client.trading_enabled and app.paper_broker is None
            assert (
                ex.count("/sapi/v1/account/apiRestrictions") == 1 and ex.count("/api/v3/order") == 0
            )
            result = app.loop.order_manager.submit(
                OrderIntent("live:1", "BTCUSDT", Side.BUY, OrderType.MARKET, D("0.001"), D("60000"))
            )
        assert result.outcome == "accepted" and result.record is not None
        assert result.record.status is OrderStatus.FILLED
        post = [r for r in ex.requests if r.method == "POST"]
        assert len(post) == 1 and post[0].headers["X-MBX-APIKEY"] == API_KEY
        body = post[0].content.decode()
        assert "signature=" in body and API_SECRET not in body and post[0].url.query == b""
        # nothing sensitive reached logs, the alert file or the database
        dump = "\n".join(store._conn.iterdump())
        alerts = Path(s.paths.alerts_file)
        haystack = caplog.text + dump + (alerts.read_text() if alerts.exists() else "")
        assert API_KEY not in haystack and API_SECRET not in haystack
        app.close()

    def test_a_key_that_can_withdraw_is_refused_before_any_order_endpoint_is_touched(
        self, settings: Settings, clock: FakeClock, store: Store
    ) -> None:
        s, ex = self.setup(settings, clock, restrictions=GOOD_KEY | {"enableWithdrawals": True})
        with pytest.raises(LiveTradingNotAllowed, match="withdrawals"):
            build_live_app(s, ENV, http=ex.http(), clock=clock, store=store)
        assert ex.count("/api/v3/order") == 0 and ex.count("/api/v3/exchangeInfo") == 0

    def test_large_clock_skew_is_refused(
        self, settings: Settings, clock: FakeClock, store: Store
    ) -> None:
        s, ex = self.setup(settings, clock)
        ex.on(
            "GET",
            "/api/v3/time",
            lambda _r: httpx.Response(200, json={"serverTime": clock.now_ms() + 9000}),
        )
        with pytest.raises(ClockSkewError):
            build_live_app(s, ENV, http=ex.http(), clock=clock, store=store)
        assert ex.count("/api/v3/order") == 0


class TestLiveBroker:
    def client(
        self, ex: FakeExchange, clock: FakeClock, *, trading: bool = True
    ) -> BinanceSpotClient:
        return BinanceSpotClient(
            credentials=Credentials(Secret(API_KEY), Secret(API_SECRET)),
            trading_enabled=trading,
            http=ex.http(),
            clock=clock,
        )

    def broker(self, ex: FakeExchange, clock: FakeClock, rules: SymbolRules) -> LiveBroker:
        from botux.safety import LiveAuthorization

        return LiveBroker(self.client(ex, clock), {"BTCUSDT": rules}, LiveAuthorization("f", 0))

    def test_cannot_be_built_without_trading_enabled_or_authorisation(
        self, clock: FakeClock, rules: SymbolRules
    ) -> None:
        from botux.safety import LiveAuthorization

        ex = FakeExchange()
        with pytest.raises(LiveTradingNotAllowed):
            LiveBroker(
                self.client(ex, clock, trading=False), {"BTCUSDT": rules}, LiveAuthorization("f", 0)
            )
        with pytest.raises(LiveTradingNotAllowed):
            LiveBroker(self.client(ex, clock), {"BTCUSDT": rules}, "nope")  # type: ignore[arg-type]

    def req(self, qty: str = "0.001") -> OrderRequest:
        return OrderRequest("c1", "BTCUSDT", Side.BUY, OrderType.MARKET, D(qty), None, D("60000"))

    def test_local_filter_check_happens_before_sending(
        self, clock: FakeClock, rules: SymbolRules
    ) -> None:
        ex = FakeExchange()
        with pytest.raises(OrderRejected):
            self.broker(ex, clock, rules).place_order(self.req("0.000001"))
        assert ex.requests == []

    def test_error_mapping(self, clock: FakeClock, rules: SymbolRules) -> None:
        ex = FakeExchange()
        b = self.broker(ex, clock, rules)
        ex.on(
            "POST",
            "/api/v3/order",
            lambda _r: httpx.Response(400, json={"code": -2010, "msg": "insufficient"}),
        )
        with pytest.raises(OrderRejected) as e:
            b.place_order(self.req())
        assert e.value.code == -2010
        ex.on("POST", "/api/v3/order", lambda _r: httpx.Response(504, text="timeout"))
        with pytest.raises(OutcomeUnknownError):
            b.place_order(self.req())
        ex.on(
            "GET",
            "/api/v3/order",
            lambda _r: httpx.Response(400, json={"code": -2013, "msg": "Order does not exist."}),
        )
        assert b.get_order("BTCUSDT", "c1") is None
        ex.on(
            "DELETE",
            "/api/v3/order",
            lambda _r: httpx.Response(400, json={"code": -2011, "msg": "Unknown order sent."}),
        )
        assert b.cancel_order("BTCUSDT", "c1") is None

    def test_parse_order_statuses_and_unknown_fallback(self) -> None:
        base = {
            "symbol": "BTCUSDT",
            "clientOrderId": "c",
            "side": "BUY",
            "type": "MARKET",
            "origQty": "1",
            "executedQty": "0",
            "cummulativeQuoteQty": "0",
            "orderId": 5,
        }
        assert parse_order(base | {"status": "NEW"}).status is OrderStatus.NEW
        assert parse_order(base | {"status": "EXPIRED_IN_MATCH"}).status is OrderStatus.EXPIRED
        assert parse_order(base | {"status": "SOMETHING_NEW"}).status is OrderStatus.UNKNOWN
        got = parse_order(
            base
            | {
                "status": "FILLED",
                "executedQty": "1",
                "cummulativeQuoteQty": "60000",
                "fills": [
                    {"price": "60000", "qty": "1", "commission": "0.001", "commissionAsset": "BTC"}
                ],
            }
        )
        assert got.avg_price == D(60000) and got.fills[0].commission_asset == "BTC"
