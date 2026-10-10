"""Command line interface smoke and behaviour tests (no real network)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

import botux.cli as cli
from botux.cli import main
from botux.data.market import save_csv
from botux.exchange.client import BinanceSpotClient
from botux.storage import Store
from tests.conftest import EXCHANGE_INFO_BTCUSDT, FakeExchange, candles_from_closes


@pytest.fixture
def cfg_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "botux.toml"
    path.write_text(
        "[paths]\n"
        f'kill_file = "{tmp_path / "KILL"}"\n'
        f'database = "{tmp_path / "db.sqlite3"}"\n'
        f'data_dir = "{tmp_path / "cache"}"\n'
        f'alerts_file = "{tmp_path / "alerts.jsonl"}"\n'
        "[live]\n"
        f'arm_file = "{tmp_path / "armed.json"}"\n'
    )
    return str(path)


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and "botux" in capsys.readouterr().out


def test_check_config_reports_a_closed_live_gate(
    cfg_path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["-c", cfg_path, "check-config"]) == 0
    out = capsys.readouterr().out
    assert "mode                 : paper" in out and "live gate            : CLOSED" in out


def test_invalid_config_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('api_key = "x"\n')
    assert main(["-c", str(bad), "check-config"]) == 2
    assert "error:" in capsys.readouterr().err


def test_kill_status_unkill_cycle(
    cfg_path: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["-c", cfg_path, "kill", "--reason", "test stop"]) == 0
    assert (tmp_path / "KILL").exists()
    assert main(["-c", cfg_path, "status"]) == 0
    assert "SET - test stop" in capsys.readouterr().out
    assert main(["-c", cfg_path, "unkill"]) == 0
    assert not (tmp_path / "KILL").exists()


def test_live_command_refuses_by_default_and_touches_no_network(
    cfg_path: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_http(*_a: object, **_k: object) -> None:
        raise AssertionError("network used")

    monkeypatch.setattr(httpx.Client, "request", no_http)
    monkeypatch.setenv("BINANCE_API_KEY", "G" * 64)
    monkeypatch.setenv("BINANCE_API_SECRET", "H" * 64)
    assert main(["-c", cfg_path, "live"]) == 2
    assert "live trading refused" in capsys.readouterr().err


def test_arm_live_refuses_while_config_is_paper(cfg_path: str, tmp_path: Path) -> None:
    assert main(["-c", cfg_path, "arm-live"]) == 1
    assert not (tmp_path / "armed.json").exists()


def test_disarm_and_resolve_lost(cfg_path: str, tmp_path: Path) -> None:
    assert main(["-c", cfg_path, "disarm-live"]) == 0
    s = Store(tmp_path / "db.sqlite3")
    s.kv_set("lost:BTCUSDT", "abc")
    s.close()
    assert main(["-c", cfg_path, "resolve-lost", "BTCUSDT"]) == 0
    s = Store(tmp_path / "db.sqlite3")
    assert not s.kv_get("lost:BTCUSDT")
    s.close()


def test_backtest_command_end_to_end_from_csv(
    cfg_path: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closes = []
    price = 100.0
    for i in range(700):
        price *= 1 + (0.004 if (i // 40) % 2 == 0 else -0.004)
        closes.append(round(price, 2))
    csv = tmp_path / "data.csv"
    save_csv(csv, candles_from_closes(closes))

    ex = FakeExchange()
    ex.on("GET", "/api/v3/exchangeInfo", EXCHANGE_INFO_BTCUSDT)
    ex.on("GET", "/api/v3/time", {"serverTime": 1})
    monkeypatch.setattr(cli, "public_client", lambda _s, **_k: BinanceSpotClient(http=ex.http()))
    assert main(["-c", cfg_path, "backtest", "--data", str(csv)]) == 0
    out = capsys.readouterr().out
    for needle in (
        "Parameter search on TRAIN data only",
        "TEST result of the chosen parameters",
        "Limitations",
        "not advice",
        "single instrument",
    ):
        assert needle in out
    assert {r.url.path for r in ex.requests} == {"/api/v3/exchangeInfo"}
