"""Paper mode cannot wipe live state; one database per process; stored text is redacted."""

from __future__ import annotations

import dataclasses
import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from botux.app import KEY_RECHECK_SECONDS, App, build_live_app, build_paper_app, make_gate_check
from botux.clock import FakeClock
from botux.config import Settings, database_path
from botux.errors import ConfigError, LiveTradingNotAllowed
from botux.exchange.client import BinanceSpotClient
from botux.exchange.rules import SymbolRules
from botux.killswitch import KillSwitch
from botux.models import OrderStatus, OrderType, Side
from botux.safety import ARM_PHRASE, arm_live_interactive
from botux.secrets_guard import GLOBAL_REDACTOR, Credentials, Secret
from botux.storage import InstanceLock, OrderRecord, Store
from tests.conftest import EXCHANGE_INFO_BTCUSDT, FakeExchange
from tests.test_live_gate import ENV, GOOD_KEY, live, live_exchange

D = Decimal


class FeedStub:
    def now_ms(self) -> int:
        return 0


def halt_reason(app: App) -> str | None:
    """Indirection so the type checker does not narrow the property across method calls."""
    return app.loop.risk.blockers.get("live-gate")


def unresolved(store: Store) -> None:
    store.insert_order(
        OrderRecord(
            "live-order",
            "live-intent",
            "BTCUSDT",
            Side.BUY,
            OrderType.MARKET,
            D("0.1"),
            None,
            OrderStatus.UNKNOWN,
            1,
            1,
        )
    )
    store.kv_set("lost:BTCUSDT", "live-order")
    store.kv_set("pos:BTCUSDT", json.dumps({"entry": "100", "stop": "98"}))
    store.kv_set("baseline:BTC", "0.5")
    store.kv_set("risk_state", json.dumps({"daily_loss_halted": True}))


class TestPaperCannotTouchLiveState:
    def test_a_live_database_is_refused_and_left_untouched(
        self, settings: Settings, clock: FakeClock, rules: SymbolRules, tmp_path: Path
    ) -> None:
        live_db = Store(tmp_path / "shared.sqlite3")
        live_db.claim_mode("live")
        unresolved(live_db)
        with pytest.raises(ConfigError, match="belongs to 'live' mode"):
            build_paper_app(
                settings,
                http=FakeExchange().http(),
                clock=clock,
                store=live_db,
                feed=FeedStub(),  # type: ignore[arg-type]
                rules={"BTCUSDT": rules},
            )
        # nothing was expired, cleared or unblocked
        assert [o.status for o in live_db.list_orders()] == [OrderStatus.UNKNOWN]
        for key in ("lost:BTCUSDT", "pos:BTCUSDT", "baseline:BTC", "risk_state"):
            assert live_db.kv_get(key)
        live_db.close()

    def test_a_paper_database_is_refused_by_live_mode(
        self, settings: Settings, clock: FakeClock, store: Store
    ) -> None:
        store.claim_mode("paper")
        s = live(settings)
        arm_live_interactive(s, clock.now_ms(), lambda _p: ARM_PHRASE, lambda _t: None)
        with pytest.raises(ConfigError, match="belongs to 'paper' mode"):
            build_live_app(s, ENV, http=live_exchange(clock).http(), clock=clock, store=store)

    def test_each_mode_defaults_to_its_own_file(
        self, settings: Settings, clock: FakeClock, rules: SymbolRules
    ) -> None:
        app = build_paper_app(
            settings,
            http=FakeExchange().http(),
            clock=clock,
            feed=FeedStub(),  # type: ignore[arg-type]
            rules={"BTCUSDT": rules},
        )
        assert Path(database_path(settings, "paper")).exists()
        assert not Path(database_path(settings, "live")).exists()
        app.close()

    def test_claim_mode_is_idempotent_for_the_same_mode(self, store: Store) -> None:
        store.claim_mode("paper")
        store.claim_mode("paper")
        with pytest.raises(ConfigError):
            store.claim_mode("live")


class TestSingleInstance:
    def test_a_second_process_on_the_same_database_is_refused(self, tmp_path: Path) -> None:
        first = InstanceLock(tmp_path / "db.sqlite3")
        first.acquire()
        with pytest.raises(ConfigError, match="another Botux instance"):
            InstanceLock(tmp_path / "db.sqlite3").acquire()
        first.release()
        again = InstanceLock(tmp_path / "db.sqlite3")
        again.acquire()  # free again once the first one is gone
        again.release()

    def test_memory_databases_need_no_lock(self) -> None:
        InstanceLock(":memory:").acquire()

    def test_two_paper_apps_cannot_share_a_database(
        self, settings: Settings, clock: FakeClock, rules: SymbolRules
    ) -> None:
        kw = {
            "http": FakeExchange().http(),
            "clock": clock,
            "feed": FeedStub(),
            "rules": {"BTCUSDT": rules},
        }
        first = build_paper_app(settings, **kw)  # type: ignore[arg-type]
        with pytest.raises(ConfigError, match="another Botux instance"):
            build_paper_app(settings, **kw)  # type: ignore[arg-type]
        first.close()
        build_paper_app(settings, **kw).close()  # type: ignore[arg-type]


class TestGateIsCheckedWhileRunning:
    def test_arming_expiry_halts_entries_of_a_running_bot(
        self, settings: Settings, clock: FakeClock, store: Store
    ) -> None:
        s = live(settings)
        arm_live_interactive(s, clock.now_ms(), lambda _p: ARM_PHRASE, lambda _t: None)
        ex = live_exchange(clock)
        app = build_live_app(s, ENV, http=ex.http(), clock=clock, store=store)
        app.loop.run_cycle()  # gate still open (the cycle itself may fail: no klines in this fake)
        assert halt_reason(app) is None
        clock.advance(25 * 3600)  # arming lifetime is 24 h
        app.loop.run_cycle()
        assert "expired" in (halt_reason(app) or "")
        assert app.loop.risk.entries_allowed()[0] is False
        app.close()

    def test_disarming_a_running_bot_halts_entries(
        self, settings: Settings, clock: FakeClock, store: Store
    ) -> None:
        s = live(settings)
        arm_live_interactive(s, clock.now_ms(), lambda _p: ARM_PHRASE, lambda _t: None)
        app = build_live_app(s, ENV, http=live_exchange(clock).http(), clock=clock, store=store)
        Path(s.live.arm_file).unlink()
        app.loop.run_cycle()
        assert "not armed" in (halt_reason(app) or "")
        app.close()

    def key_client(self, ex: FakeExchange, clock: FakeClock) -> BinanceSpotClient:
        return BinanceSpotClient(
            credentials=Credentials(Secret("A" * 64), Secret("B" * 64)), http=ex.http(), clock=clock
        )

    def test_key_permissions_are_rechecked_hourly(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        s = live(settings)
        arm_live_interactive(s, clock.now_ms(), lambda _p: ARM_PHRASE, lambda _t: None)
        ex = FakeExchange()
        state = {"body": dict(GOOD_KEY)}
        ex.on(
            "GET",
            "/sapi/v1/account/apiRestrictions",
            lambda _r: httpx.Response(200, json=state["body"]),
        )
        check = make_gate_check(
            s, KillSwitch(s.paths.kill_file, clock), clock, self.key_client(ex, clock)
        )
        check()
        assert ex.count("/sapi/v1/account/apiRestrictions") == 0  # file check only, not every call
        state["body"] = GOOD_KEY | {"enableWithdrawals": True}
        clock.advance(KEY_RECHECK_SECONDS + 1)
        with pytest.raises(LiveTradingNotAllowed, match="permissions changed"):
            check()

    def test_a_failed_permission_lookup_does_not_close_the_gate(
        self, settings: Settings, clock: FakeClock
    ) -> None:
        s = live(settings)
        arm_live_interactive(s, clock.now_ms(), lambda _p: ARM_PHRASE, lambda _t: None)
        ex = FakeExchange()
        ex.on("GET", "/sapi/v1/account/apiRestrictions", lambda _r: httpx.Response(503))
        check = make_gate_check(
            s, KillSwitch(s.paths.kill_file, clock), clock, self.key_client(ex, clock)
        )
        clock.advance(KEY_RECHECK_SECONDS + 1)
        check()  # a Binance hiccup must not be mistaken for a revoked permission


class TestStoredTextIsRedacted:
    def test_events_and_order_notes_never_hold_a_registered_secret(self, store: Store) -> None:
        secret = "Q" * 64
        GLOBAL_REDACTOR.register(secret)
        store.log_event(1, "ERROR", "x", {"message": f"server echoed {secret}"})
        store.log_event(2, "ERROR", "y", f"plain {secret}")
        rec = OrderRecord(
            "c1",
            "k1",
            "BTCUSDT",
            Side.BUY,
            OrderType.MARKET,
            D(1),
            None,
            OrderStatus.REJECTED,
            1,
            1,
            note=f"rejected: {secret}",
        )
        assert store.insert_order(rec)
        rec.note = f"again {secret}"
        store.update_order(rec)
        dump = "\n".join(store._conn.iterdump())
        assert secret not in dump

    def test_long_unregistered_tokens_are_scrubbed_too(self, store: Store) -> None:
        store.log_event(1, "ERROR", "z", "token " + "k" * 63)  # off-by-one length
        assert "k" * 63 not in "\n".join(store._conn.iterdump())


def test_replace_import_is_used() -> None:
    assert dataclasses.is_dataclass(Settings) and EXCHANGE_INFO_BTCUSDT
