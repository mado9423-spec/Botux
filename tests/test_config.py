"""Settings validation: safe defaults, typo-proofing, secret rejection, fingerprint, credentials."""

from __future__ import annotations

import dataclasses
import tomllib
from decimal import Decimal
from pathlib import Path

import pytest

from botux.config import (
    LiveSettings,
    PathSettings,
    Settings,
    load_credentials,
    load_settings,
    parse_settings,
    read_env_file,
    resolve_config_path,
    settings_fingerprint,
)
from botux.errors import ConfigError
from botux.secrets_guard import GLOBAL_REDACTOR

D = Decimal
ROOT = Path(__file__).resolve().parents[1]


def parse(toml: str) -> Settings:
    return parse_settings(tomllib.loads(toml))


class TestSafeDefaults:
    def test_defaults_are_paper_and_live_is_disabled(self) -> None:
        s = Settings()
        assert s.mode == "paper" and s.live.enabled is False
        assert s.risk.flatten_on_kill is False

    def test_empty_config_equals_defaults(self) -> None:
        assert parse("") == Settings()

    def test_shipped_example_config_loads_and_is_paper_only(self) -> None:
        s = load_settings(ROOT / "config" / "botux.example.toml")
        assert s.mode == "paper" and s.live.enabled is False
        assert s.market.symbols == ("BTCUSDT",)
        assert s.execution.order_type == "MARKET"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_settings(tmp_path / "nope.toml")

    def test_invalid_toml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.toml"
        p.write_text("mode = ")
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_settings(p)


class TestValidation:
    @pytest.mark.parametrize(
        ("toml", "message"),
        [
            ('mode = "yolo"', "mode must be"),
            ("nonsense = 1", "unknown top-level"),
            ("[risk]\nmax_daily_los = 5", "unknown key"),
            ("[market]\nsymbols = []", "1 to 5"),
            ('[market]\nsymbols = ["btcusdt"]', "invalid symbol"),
            ('[market]\nsymbols = ["BTCUSDT", "BTCUSDT"]', "duplicates"),
            ('[market]\nsymbols = ["BTCEUR"]', "quote asset"),
            ('[market]\ninterval = "7m"', "unsupported interval"),
            ("[market]\npoll_seconds = 0.1", "poll_seconds"),
            ('[market]\nmarket_data_base_url = "http://data-api.binance.vision"', "https"),
            ('[market]\nmarket_data_base_url = "https://evil.example"', "allow-list"),
            ('[market]\ntrading_base_url = "https://data-api.binance.vision"', "allow-list"),
            ('[market]\ntrading_base_url = "https://api.binance.com/api"', "bare origin"),
            ("[risk]\nstop_loss_pct = 0.02", "quoted string"),
            ('[risk]\nstop_loss_pct = "0.6"', "stop_loss_pct"),
            ('[risk]\nrisk_per_trade_pct = "0.2"', "risk_per_trade_pct"),
            ('[risk]\nmax_order_notional = "500"', "max_order_notional <="),
            ('[risk]\nmax_daily_loss = "0"', "max_daily_loss"),
            ("[risk]\nmax_open_orders = 0", "max_open_orders"),
            ('[risk]\nmin_notional_buffer = "0.9"', "min_notional_buffer"),
            ('[execution]\norder_type = "LIMIT"', "MARKET"),
            ('[execution]\nfee_rate = "0.5"', "fee_rate"),
            ('[execution]\nslippage_bps = "5000"', "slippage_bps"),
            ('[paper]\nstarting_balance = "0"', "starting_balance"),
            ("[live]\narm_max_age_hours = 0", "arm_max_age_hours"),
            ("[live]\nmax_clock_skew_ms = 50", "max_clock_skew_ms"),
            ("[live]\nenabled = 1", "true or false"),
        ],
    )
    def test_bad_values_are_rejected(self, toml: str, message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            parse(toml)

    def test_valid_overrides_are_applied(self) -> None:
        s = parse(
            '[risk]\nmax_daily_loss = "5"\nstop_loss_pct = "0.01"\n[market]\nsymbols = ["ETHUSDT"]'
        )
        assert s.risk.max_daily_loss == D(5) and s.risk.stop_loss_pct == D("0.01")
        assert s.market.symbols == ("ETHUSDT",)


class TestSecretsAreRefused:
    @pytest.mark.parametrize(
        "toml",
        [
            'api_key = "abc"',
            '[market]\napi_secret = "abc"',
            '[live]\nsecret = "x"',
            "[strategy.params]\ntoken = 1",
            '[market]\nsymbols = ["' + "A" * 64 + '"]',
        ],
    )
    def test_secret_like_fields_and_values_are_rejected(self, toml: str) -> None:
        with pytest.raises(ConfigError, match=r"secret|API key"):
            parse(toml)


class TestFingerprint:
    def test_stable_and_sensitive_to_behaviour(self) -> None:
        a = Settings()
        assert settings_fingerprint(a) == settings_fingerprint(Settings())
        risk = dataclasses.replace(a.risk, max_order_notional=D(101))
        assert settings_fingerprint(dataclasses.replace(a, risk=risk)) != settings_fingerprint(a)
        assert settings_fingerprint(dataclasses.replace(a, mode="live")) != settings_fingerprint(a)
        market = dataclasses.replace(a.market, symbols=("ETHUSDT",))
        assert settings_fingerprint(dataclasses.replace(a, market=market)) != settings_fingerprint(
            a
        )

    def test_ignores_local_file_locations(self) -> None:
        a = Settings()
        moved = dataclasses.replace(
            a, paths=PathSettings(kill_file="/x/KILL"), live=LiveSettings(arm_file="/x/armed.json")
        )
        assert settings_fingerprint(moved) == settings_fingerprint(a)


class TestConfigPath:
    def test_precedence(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert resolve_config_path(None, {}) == Path("config/botux.example.toml")
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "botux.toml").write_text("")
        assert resolve_config_path(None, {}) == Path("config/botux.toml")
        assert resolve_config_path(None, {"BOTUX_CONFIG": "env.toml"}) == Path("env.toml")
        assert resolve_config_path("arg.toml", {"BOTUX_CONFIG": "env.toml"}) == Path("arg.toml")


class TestCredentials:
    KEY, SECRET = "k" * 40, "s" * 40

    def test_loaded_from_environment_and_registered_for_redaction(self) -> None:
        creds = load_credentials(
            {"BINANCE_API_KEY": self.KEY, "BINANCE_API_SECRET": self.SECRET}, None
        )
        assert creds.api_key.reveal() == self.KEY
        assert self.KEY not in repr(creds) and self.SECRET not in f"{creds}"
        assert GLOBAL_REDACTOR.scrub(f"oops {self.SECRET}") == "oops ***REDACTED***"

    def test_environment_wins_over_dotenv_file(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(f"BINANCE_API_KEY={'f' * 40}\nBINANCE_API_SECRET={'g' * 40}\n")
        creds = load_credentials({"BINANCE_API_KEY": self.KEY}, env_file)
        assert creds.api_key.reveal() == self.KEY and creds.api_secret.reveal() == "g" * 40

    def test_dotenv_parser(self, tmp_path: Path) -> None:
        p = tmp_path / ".env"
        p.write_text("# comment\nA=1\nB='two'\n\nbad line\nC=\"3\"\n")
        assert read_env_file(p) == {"A": "1", "B": "two", "C": "3"}
        assert read_env_file(tmp_path / "missing") == {}

    @pytest.mark.parametrize(
        "env",
        [
            {},
            {"BINANCE_API_KEY": "your_api_key_here", "BINANCE_API_SECRET": "your_api_secret_here"},
            {"BINANCE_API_KEY": "short", "BINANCE_API_SECRET": "short"},
            {"BINANCE_API_KEY": "k" * 40, "BINANCE_API_SECRET": "bad secret with spaces " * 3},
        ],
    )
    def test_missing_placeholder_or_malformed_credentials_are_refused(
        self, env: dict[str, str]
    ) -> None:
        with pytest.raises(ConfigError):
            load_credentials(env, None)

    def test_env_example_contains_only_placeholders(self) -> None:
        text = (ROOT / ".env.example").read_text()
        values = [
            line.split("=", 1)[1].strip()
            for line in text.splitlines()
            if "=" in line and not line.lstrip().startswith("#")
        ]
        assert values and all(
            v.startswith("your_") for v in values if "KEY" in text or "SECRET" in text
        )
