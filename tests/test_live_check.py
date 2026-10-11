"""``live-check``: validation-only rehearsal. Everything here runs against a fake HTTP transport;
nothing in this suite ever contacts Binance, and the real command is left to the operator."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

from botux.cli import main
from botux.clock import FakeClock
from botux.config import Settings
from botux.errors import EndpointNotAllowed, ExchangeError, LiveTradingNotAllowed
from botux.exchange.client import BinanceSpotClient
from botux.execution.livecheck import TEST_ORDER_ID, run_live_check
from botux.secrets_guard import Credentials, Secret
from tests.conftest import FakeExchange
from tests.test_live_gate import API_KEY, API_SECRET, ENV, GOOD_KEY, live, live_exchange

D = Decimal


def rehearsal_exchange(clock: FakeClock, **kw: object) -> FakeExchange:
    ex = live_exchange(clock, **kw)  # type: ignore[arg-type]
    ex.on("GET", "/api/v3/ticker/bookTicker", {"bidPrice": "59999.99", "askPrice": "60000.00"})
    ex.on("GET", "/api/v3/openOrders", [])
    ex.on("POST", "/api/v3/order/test", {})
    return ex


def run(s: Settings, ex: FakeExchange, clock: FakeClock) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = run_live_check(s, ENV, http=ex.http(), clock=clock, out=lines.append)
    return code, lines


class TestRehearsal:
    def test_refuses_unless_the_config_selects_live_and_reads_no_key(
        self, settings: Settings, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*_a: object, **_k: object) -> None:
            raise AssertionError("credentials were read")

        monkeypatch.setattr("botux.execution.livecheck.load_credentials", explode)
        ex = rehearsal_exchange(clock)
        with pytest.raises(LiveTradingNotAllowed, match="mode"):
            run(settings, ex, clock)  # paper
        import dataclasses

        with pytest.raises(LiveTradingNotAllowed):
            run(dataclasses.replace(settings, mode="live"), ex, clock)  # live but not enabled
        assert ex.requests == []

    def test_a_clean_rehearsal_sends_only_reads_and_one_order_test(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        s = live(settings)
        ex = rehearsal_exchange(clock)
        code, lines = run(s, ex, clock)
        assert code == 0, lines
        posts = [r for r in ex.requests if r.method == "POST"]
        assert [p.url.path for p in posts] == ["/api/v3/order/test"]  # never /api/v3/order
        assert not any(r.method == "DELETE" for r in ex.requests)
        body = parse_qs(posts[0].content.decode())
        assert body["side"] == ["BUY"] and body["type"] == ["MARKET"]
        assert body["newClientOrderId"] == [TEST_ORDER_ID] and "signature" in body
        assert D(body["quantity"][0]) * D("60000") >= D(5)  # clears the 5 USDT minimum notional
        text = "\n".join(lines)
        assert "validation only" in text and "does NOT prove" in text
        assert API_KEY not in text and API_SECRET not in text
        assert "withdraw" not in " ".join(r.url.path for r in ex.requests)

    def test_a_key_that_can_withdraw_fails_the_check(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        ex = rehearsal_exchange(clock, restrictions=GOOD_KEY | {"enableWithdrawals": True})
        code, lines = run(live(settings), ex, clock)
        assert code == 1 and any("[FAIL]" in ln and "withdrawals" in ln for ln in lines)

    def test_a_rejected_test_order_fails_the_check_with_the_exchange_reason(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        ex = rehearsal_exchange(clock)
        ex.on(
            "POST",
            "/api/v3/order/test",
            lambda _r: httpx.Response(
                400, json={"code": -2015, "msg": "Invalid API-key, IP, or permissions"}
            ),
        )
        code, lines = run(live(settings), ex, clock)
        assert code == 1 and any("order/test" in ln and "[FAIL]" in ln for ln in lines)

    def test_foreign_open_orders_are_reported_as_a_failure(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        ex = rehearsal_exchange(clock)
        ex.on("GET", "/api/v3/openOrders", [{"symbol": "BTCUSDT", "orderId": 1}])
        code, lines = run(live(settings), ex, clock)
        assert code == 1 and any("open order" in ln and "[FAIL]" in ln for ln in lines)

    def test_an_account_that_cannot_trade_fails(self, settings: Settings, clock: FakeClock) -> None:
        ex = rehearsal_exchange(clock)
        ex.on("GET", "/api/v3/account", {"canTrade": False, "balances": []})
        code, _ = run(live(settings), ex, clock)
        assert code == 1

    def test_an_unreachable_service_is_reported_not_raised(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        ex = rehearsal_exchange(clock)
        ex.on(
            "GET", "/api/v3/account", lambda _r: httpx.Response(451, json={"code": 0, "msg": "x"})
        )
        code, lines = run(live(settings), ex, clock)
        assert code == 1 and any("[FAIL]" in ln for ln in lines)


class TestClientScope:
    def client(
        self, ex: FakeExchange, clock: FakeClock, *, order_test: bool = False
    ) -> BinanceSpotClient:
        return BinanceSpotClient(
            credentials=Credentials(Secret(API_KEY), Secret(API_SECRET)),
            order_test_enabled=order_test,
            http=ex.http(),
            clock=clock,
        )

    def order(self) -> dict[str, object]:
        return {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": D("0.001"),
            "client_order_id": "x1",
        }

    def test_order_test_alone_cannot_place_or_cancel_real_orders(self, clock: FakeClock) -> None:
        ex = rehearsal_exchange(clock)
        c = self.client(ex, clock, order_test=True)
        assert c.test_order(**self.order()) == {}  # type: ignore[arg-type]
        with pytest.raises(EndpointNotAllowed):
            c.new_order(**self.order())  # type: ignore[arg-type]
        with pytest.raises(EndpointNotAllowed):
            c.cancel_order("BTCUSDT", client_order_id="x1")
        assert [r.url.path for r in ex.requests if r.method != "GET"] == ["/api/v3/order/test"]

    def test_it_needs_credentials_and_an_explicit_switch(self, clock: FakeClock) -> None:
        from botux.errors import ConfigError

        with pytest.raises(ConfigError):
            BinanceSpotClient(order_test_enabled=True, clock=clock)
        ex = rehearsal_exchange(clock)
        with pytest.raises(EndpointNotAllowed):
            self.client(ex, clock).test_order(**self.order())  # type: ignore[arg-type]
        assert ex.requests == []

    def test_a_test_order_is_never_retried(self, clock: FakeClock) -> None:
        ex = rehearsal_exchange(clock)
        ex.on("POST", "/api/v3/order/test", lambda _r: httpx.Response(503, text="unavailable"))
        c = self.client(ex, clock, order_test=True)
        with pytest.raises(ExchangeError):
            c.test_order(**self.order())  # type: ignore[arg-type]
        assert ex.count("/api/v3/order/test") == 1


class TestCommand:
    @pytest.fixture
    def cfg_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "botux.toml"
        path.write_text(
            "[paths]\n"
            f'kill_file = "{tmp_path / "KILL"}"\n'
            f'database = "{tmp_path / "db.sqlite3"}"\n'
            f'alerts_file = "{tmp_path / "alerts.jsonl"}"\n'
            "[live]\n"
            f'arm_file = "{tmp_path / "armed.json"}"\n'
        )
        return str(path)

    def test_it_needs_the_explicit_flag_and_never_touches_the_network_without_it(
        self, cfg_path: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_http(*_a: object, **_k: object) -> None:
            raise AssertionError("network used")

        monkeypatch.setattr(httpx.Client, "request", no_http)
        monkeypatch.setenv("BINANCE_API_KEY", "G" * 64)
        monkeypatch.setenv("BINANCE_API_SECRET", "H" * 64)
        assert main(["-c", cfg_path, "live-check"]) == 2
        assert "--confirm-contact-binance" in capsys.readouterr().err
        # with the flag but a paper config it still refuses before reading any key or connecting
        assert main(["-c", cfg_path, "live-check", "--confirm-contact-binance"]) == 2
        assert "mode" in capsys.readouterr().err


class TestTheRehearsalClientCannotTrade:
    def test_the_client_is_built_without_the_right_to_place_or_cancel_orders(
        self, settings: Settings, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        class Spy(BinanceSpotClient):
            def __init__(self, **kw: object) -> None:
                seen.update(kw)
                super().__init__(**kw)  # type: ignore[arg-type]

        monkeypatch.setattr("botux.execution.livecheck.BinanceSpotClient", Spy)
        code, _ = run(live(settings), rehearsal_exchange(clock), clock)
        assert code == 0
        assert seen["trading_enabled"] is False and seen["order_test_enabled"] is True
