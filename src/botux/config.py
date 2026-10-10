"""Settings loading and validation.

Design rules:

* Settings are NON-SECRET and live in TOML. Any key or value that looks like a secret is
  rejected, so a key can never be committed by accident through the config file.
* Unknown keys are errors (a typo in a risk limit must not silently fall back to a default).
* Everything defaults to paper trading. Credentials are read only by ``load_credentials`` and
  only the guarded live path calls it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from botux.decimals import D
from botux.errors import ConfigError
from botux.secrets_guard import GLOBAL_REDACTOR, Credentials, Secret

Mode = Literal["paper", "live"]

SUPPORTED_INTERVALS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

# Hosts documented by Binance for the Spot REST API. Signed requests (and therefore the API
# key header) may only ever be sent to a host on TRADING_HOSTS: a typo or a malicious config
# must not be able to point the key at another server.
MARKET_DATA_HOSTS = frozenset(
    {
        "data-api.binance.vision",
        "api.binance.com",
        "api-gcp.binance.com",
        "api1.binance.com",
        "api2.binance.com",
        "api3.binance.com",
        "api4.binance.com",
        "testnet.binance.vision",
    }
)
TRADING_HOSTS = MARKET_DATA_HOSTS - {"data-api.binance.vision"}

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{5,20}$")
_SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|secret|token|passw)")
_LONG_TOKEN_RE = re.compile(r"[A-Za-z0-9]{40,}")
_PLACEHOLDER_RE = re.compile(r"(?i)(your_|changeme|placeholder|example|xxx|<)")


@dataclass(frozen=True, slots=True)
class MarketSettings:
    symbols: tuple[str, ...] = ("BTCUSDT",)
    interval: str = "1h"
    market_data_base_url: str = "https://data-api.binance.vision"
    trading_base_url: str = "https://api.binance.com"
    poll_seconds: float = 30.0
    max_candle_age_factor: float = 3.0


@dataclass(frozen=True, slots=True)
class StrategySettings:
    name: str = "sma_cross"
    params: Mapping[str, Any] = field(default_factory=lambda: {"fast": 20, "slow": 50})


@dataclass(frozen=True, slots=True)
class RiskSettings:
    quote_asset: str = "USDT"
    risk_per_trade_pct: Decimal = Decimal("0.005")
    stop_loss_pct: Decimal = Decimal("0.02")
    max_order_notional: Decimal = Decimal(100)
    max_position_notional: Decimal = Decimal(200)
    max_total_exposure: Decimal = Decimal(300)
    max_daily_loss: Decimal = Decimal(20)
    max_open_orders: int = 3
    max_orders_per_day: int = 20
    max_consecutive_errors: int = 5
    min_notional_buffer: Decimal = Decimal("1.05")
    max_price_deviation_pct: Decimal = Decimal("0.02")
    flatten_on_kill: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    order_type: str = "MARKET"
    fee_rate: Decimal = Decimal("0.001")
    slippage_bps: Decimal = Decimal(5)
    reconcile_attempts: int = 5
    reconcile_interval_seconds: float = 5.0


@dataclass(frozen=True, slots=True)
class PaperSettings:
    starting_balance: Decimal = Decimal(1000)


@dataclass(frozen=True, slots=True)
class LiveSettings:
    enabled: bool = False
    arm_file: str = ".botux/live_armed.json"
    arm_max_age_hours: float = 24.0
    require_ip_restriction: bool = True
    max_clock_skew_ms: int = 1000


@dataclass(frozen=True, slots=True)
class PathSettings:
    kill_file: str = ".botux/KILL"
    database: str = "data/botux.sqlite3"
    data_dir: str = "data/cache"
    alerts_file: str = "data/alerts.jsonl"


@dataclass(frozen=True, slots=True)
class Settings:
    mode: Mode = "paper"
    market: MarketSettings = field(default_factory=MarketSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    paper: PaperSettings = field(default_factory=PaperSettings)
    live: LiveSettings = field(default_factory=LiveSettings)
    paths: PathSettings = field(default_factory=PathSettings)

    @property
    def interval_ms(self) -> int:
        return SUPPORTED_INTERVALS[self.market.interval]


# --------------------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------------------


def _reject_secrets(node: Any, where: str = "") -> None:
    """Refuse secret-looking keys or long token-like values anywhere in the parsed TOML."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            path = f"{where}.{key}" if where else str(key)
            if _SECRET_KEY_RE.search(str(key)):
                raise ConfigError(
                    f"config field '{path}' looks like a secret. Secrets must never be placed in "
                    "config files; use environment variables (see .env.example)."
                )
            _reject_secrets(value, path)
    elif isinstance(node, list | tuple):
        for i, value in enumerate(node):
            _reject_secrets(value, f"{where}[{i}]")
    elif isinstance(node, str) and _LONG_TOKEN_RE.search(node):
        raise ConfigError(f"config value at '{where}' looks like an API key/secret; refusing it.")


def _section(raw: Mapping[str, Any], name: str, allowed: set[str]) -> Mapping[str, Any]:
    table = raw.get(name, {})
    if not isinstance(table, Mapping):
        raise ConfigError(f"[{name}] must be a table")
    unknown = set(table) - allowed
    if unknown:
        raise ConfigError(f"unknown key(s) in [{name}]: {', '.join(sorted(unknown))}")
    return table


def _dec(table: Mapping[str, Any], key: str, default: Decimal) -> Decimal:
    if key not in table:
        return default
    value = table[key]
    if isinstance(value, bool | float):
        raise ConfigError(
            f"'{key}' must be written as a quoted string (e.g. \"0.005\"), not a float"
        )
    try:
        return D(value if isinstance(value, str | int) else str(value))
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"'{key}' is not a valid decimal: {value!r}") from exc


def _int(table: Mapping[str, Any], key: str, default: int) -> int:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{key}' must be an integer")
    return value


def _num(table: Mapping[str, Any], key: str, default: float) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"'{key}' must be a number")
    return float(value)


def _bool(table: Mapping[str, Any], key: str, default: bool) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"'{key}' must be true or false")
    return value


def _str(table: Mapping[str, Any], key: str, default: str) -> str:
    value = table.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"'{key}' must be a string")
    return value


def _check_url(url: str, allowed_hosts: frozenset[str], what: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ConfigError(f"{what} must use https")
    if parsed.hostname not in allowed_hosts:
        raise ConfigError(
            f"{what} host '{parsed.hostname}' is not on the allow-list: {sorted(allowed_hosts)}"
        )
    if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query:
        raise ConfigError(f"{what} must be a bare origin like https://host")


def parse_settings(raw: Mapping[str, Any]) -> Settings:
    """Validate a parsed TOML mapping and return immutable settings."""
    _reject_secrets(raw)
    unknown_top = set(raw) - {
        "mode",
        "market",
        "strategy",
        "risk",
        "execution",
        "paper",
        "live",
        "paths",
    }
    if unknown_top:
        raise ConfigError(f"unknown top-level key(s): {', '.join(sorted(unknown_top))}")

    mode = raw.get("mode", "paper")
    if mode not in ("paper", "live"):
        raise ConfigError('mode must be "paper" or "live"')

    # --- market ---
    m = _section(
        raw,
        "market",
        {
            "symbols",
            "interval",
            "market_data_base_url",
            "trading_base_url",
            "poll_seconds",
            "max_candle_age_factor",
        },
    )
    symbols_raw = m.get("symbols", ["BTCUSDT"])
    if not isinstance(symbols_raw, list) or not symbols_raw or len(symbols_raw) > 5:
        raise ConfigError("market.symbols must be a list of 1 to 5 symbols")
    symbols: list[str] = []
    for s in symbols_raw:
        if not isinstance(s, str) or not _SYMBOL_RE.match(s):
            raise ConfigError(f"invalid symbol: {s!r}")
        symbols.append(s)
    if len(set(symbols)) != len(symbols):
        raise ConfigError("market.symbols contains duplicates")
    interval = _str(m, "interval", "1h")
    if interval not in SUPPORTED_INTERVALS:
        raise ConfigError(
            f"unsupported interval {interval!r}; use one of {list(SUPPORTED_INTERVALS)}"
        )
    dm = MarketSettings()
    market = MarketSettings(
        symbols=tuple(symbols),
        interval=interval,
        market_data_base_url=_str(m, "market_data_base_url", dm.market_data_base_url),
        trading_base_url=_str(m, "trading_base_url", dm.trading_base_url),
        poll_seconds=_num(m, "poll_seconds", 30.0),
        max_candle_age_factor=_num(m, "max_candle_age_factor", 3.0),
    )
    _check_url(market.market_data_base_url, MARKET_DATA_HOSTS, "market.market_data_base_url")
    _check_url(market.trading_base_url, TRADING_HOSTS, "market.trading_base_url")
    if market.poll_seconds < 1:
        raise ConfigError("market.poll_seconds must be >= 1 (be polite to the API)")
    if market.max_candle_age_factor < 1.5:
        raise ConfigError("market.max_candle_age_factor must be >= 1.5")

    # --- strategy ---
    s_tbl = _section(raw, "strategy", {"name", "params"})
    params = s_tbl.get("params", {"fast": 20, "slow": 50})
    if not isinstance(params, Mapping):
        raise ConfigError("strategy.params must be a table")
    strategy = StrategySettings(name=_str(s_tbl, "name", "sma_cross"), params=dict(params))

    # --- risk ---
    r = _section(raw, "risk", {f.name for f in dataclasses.fields(RiskSettings)})
    d = RiskSettings()
    risk = RiskSettings(
        quote_asset=_str(r, "quote_asset", d.quote_asset),
        risk_per_trade_pct=_dec(r, "risk_per_trade_pct", d.risk_per_trade_pct),
        stop_loss_pct=_dec(r, "stop_loss_pct", d.stop_loss_pct),
        max_order_notional=_dec(r, "max_order_notional", d.max_order_notional),
        max_position_notional=_dec(r, "max_position_notional", d.max_position_notional),
        max_total_exposure=_dec(r, "max_total_exposure", d.max_total_exposure),
        max_daily_loss=_dec(r, "max_daily_loss", d.max_daily_loss),
        max_open_orders=_int(r, "max_open_orders", d.max_open_orders),
        max_orders_per_day=_int(r, "max_orders_per_day", d.max_orders_per_day),
        max_consecutive_errors=_int(r, "max_consecutive_errors", d.max_consecutive_errors),
        min_notional_buffer=_dec(r, "min_notional_buffer", d.min_notional_buffer),
        max_price_deviation_pct=_dec(r, "max_price_deviation_pct", d.max_price_deviation_pct),
        flatten_on_kill=_bool(r, "flatten_on_kill", d.flatten_on_kill),
    )
    _validate_risk(risk, market.symbols)

    # --- execution ---
    e = _section(raw, "execution", {f.name for f in dataclasses.fields(ExecutionSettings)})
    de = ExecutionSettings()
    execution = ExecutionSettings(
        order_type=_str(e, "order_type", de.order_type),
        fee_rate=_dec(e, "fee_rate", de.fee_rate),
        slippage_bps=_dec(e, "slippage_bps", de.slippage_bps),
        reconcile_attempts=_int(e, "reconcile_attempts", de.reconcile_attempts),
        reconcile_interval_seconds=_num(e, "reconcile_interval_seconds", 5.0),
    )
    if execution.order_type != "MARKET":
        raise ConfigError(
            'execution.order_type must be "MARKET": brokers also support LIMIT, but the trading '
            "loop does not manage resting orders yet"
        )
    if not Decimal(0) <= execution.fee_rate <= Decimal("0.01"):
        raise ConfigError("execution.fee_rate must be within [0, 0.01]")
    if not Decimal(0) <= execution.slippage_bps <= 1000:
        raise ConfigError("execution.slippage_bps must be within [0, 1000]")
    if execution.reconcile_attempts < 1 or execution.reconcile_interval_seconds < 0:
        raise ConfigError("execution.reconcile_* values are invalid")

    # --- paper ---
    p = _section(raw, "paper", {"starting_balance"})
    paper = PaperSettings(starting_balance=_dec(p, "starting_balance", Decimal(1000)))
    if paper.starting_balance <= 0:
        raise ConfigError("paper.starting_balance must be > 0")

    # --- live ---
    lv = _section(raw, "live", {f.name for f in dataclasses.fields(LiveSettings)})
    dl = LiveSettings()
    live = LiveSettings(
        enabled=_bool(lv, "enabled", dl.enabled),
        arm_file=_str(lv, "arm_file", dl.arm_file),
        arm_max_age_hours=_num(lv, "arm_max_age_hours", dl.arm_max_age_hours),
        require_ip_restriction=_bool(lv, "require_ip_restriction", dl.require_ip_restriction),
        max_clock_skew_ms=_int(lv, "max_clock_skew_ms", dl.max_clock_skew_ms),
    )
    if not 0 < live.arm_max_age_hours <= 168:
        raise ConfigError("live.arm_max_age_hours must be within (0, 168]")
    if not 100 <= live.max_clock_skew_ms <= 2500:
        raise ConfigError("live.max_clock_skew_ms must be within [100, 2500]")

    # --- paths ---
    pt = _section(raw, "paths", {f.name for f in dataclasses.fields(PathSettings)})
    dp = PathSettings()
    paths = PathSettings(
        kill_file=_str(pt, "kill_file", dp.kill_file),
        database=_str(pt, "database", dp.database),
        data_dir=_str(pt, "data_dir", dp.data_dir),
        alerts_file=_str(pt, "alerts_file", dp.alerts_file),
    )

    return Settings(
        mode=mode,
        market=market,
        strategy=strategy,
        risk=risk,
        execution=execution,
        paper=paper,
        live=live,
        paths=paths,
    )


def _validate_risk(risk: RiskSettings, symbols: tuple[str, ...]) -> None:
    for sym in symbols:
        if not sym.endswith(risk.quote_asset) or sym == risk.quote_asset:
            raise ConfigError(f"symbol {sym} does not use quote asset {risk.quote_asset}")
    if not Decimal(0) < risk.risk_per_trade_pct <= Decimal("0.05"):
        raise ConfigError("risk.risk_per_trade_pct must be within (0, 0.05]")
    if not Decimal(0) < risk.stop_loss_pct < Decimal("0.5"):
        raise ConfigError("risk.stop_loss_pct must be within (0, 0.5)")
    for name in (
        "max_order_notional",
        "max_position_notional",
        "max_total_exposure",
        "max_daily_loss",
    ):
        if getattr(risk, name) <= 0:
            raise ConfigError(f"risk.{name} must be > 0")
    if not risk.max_order_notional <= risk.max_position_notional <= risk.max_total_exposure:
        raise ConfigError(
            "risk limits must satisfy max_order_notional <= max_position_notional "
            "<= max_total_exposure"
        )
    for name in ("max_open_orders", "max_orders_per_day", "max_consecutive_errors"):
        if getattr(risk, name) < 1:
            raise ConfigError(f"risk.{name} must be >= 1")
    if risk.min_notional_buffer < 1:
        raise ConfigError("risk.min_notional_buffer must be >= 1")
    if not Decimal(0) < risk.max_price_deviation_pct <= Decimal("0.5"):
        raise ConfigError("risk.max_price_deviation_pct must be within (0, 0.5]")


def resolve_config_path(path: str | Path | None, env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    if path:
        return Path(path)
    if env.get("BOTUX_CONFIG"):
        return Path(env["BOTUX_CONFIG"])
    local = Path("config/botux.toml")
    return local if local.exists() else Path("config/botux.example.toml")


def load_settings(path: str | Path | None = None, env: Mapping[str, str] | None = None) -> Settings:
    cfg_path = resolve_config_path(path, env)
    try:
        with cfg_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {cfg_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {cfg_path}: {exc}") from exc
    return parse_settings(raw)


# --------------------------------------------------------------------------------------
# fingerprint (binds a live "arming" confirmation to the exact settings that were shown)
# --------------------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return format(obj.normalize(), "f")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    return obj


def settings_summary(settings: Settings) -> dict[str, Any]:
    """Everything that affects what live trading would do (excludes local file paths)."""
    data = _jsonable(settings)
    data.pop("paths", None)
    data["live"].pop("arm_file", None)
    data["live"].pop("arm_max_age_hours", None)
    data["paper"] = None  # irrelevant to live behaviour
    return dict(data)


def settings_fingerprint(settings: Settings) -> str:
    blob = json.dumps(settings_summary(settings), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------------------
# credentials (live path only)
# --------------------------------------------------------------------------------------


def read_env_file(path: Path) -> dict[str, str]:
    """Tiny .env parser. Returns values without touching ``os.environ``."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip("'\"")
    return values


def load_credentials(
    env: Mapping[str, str] | None = None, dotenv_path: Path | None = Path(".env")
) -> Credentials:
    """Read the API key pair. Called ONLY from the guarded live path, never in paper mode."""
    env = os.environ if env is None else env
    file_vals = read_env_file(dotenv_path) if dotenv_path else {}
    key = env.get("BINANCE_API_KEY") or file_vals.get("BINANCE_API_KEY") or ""
    secret = env.get("BINANCE_API_SECRET") or file_vals.get("BINANCE_API_SECRET") or ""
    for label, value in (("BINANCE_API_KEY", key), ("BINANCE_API_SECRET", secret)):
        if not value:
            raise ConfigError(f"{label} is not set")
        if _PLACEHOLDER_RE.search(value):
            raise ConfigError(f"{label} still holds a placeholder value")
        if len(value) < 16 or not re.fullmatch(r"[A-Za-z0-9_\-]+", value):
            raise ConfigError(f"{label} does not look like a Binance credential")
    GLOBAL_REDACTOR.register(key)
    GLOBAL_REDACTOR.register(secret)
    return Credentials(api_key=Secret(key), api_secret=Secret(secret))
