"""The gate between paper trading and real orders.

ALL of the following must hold before the live path may even read an API key:

1. ``mode = "live"``                       (settings file)
2. ``[live].enabled = true``               (separate switch)
3. a fresh arming file whose fingerprint equals the current settings (``botux arm-live``,
   interactive: shows the settings and risks and demands an exact confirmation phrase)
4. the kill switch is not set

and, once a key is loaded, the key itself must pass :func:`api_key_problems` (no withdrawal,
transfer, margin, futures or options permission; IP-restricted unless explicitly relaxed).

Honest limits: the arming file is a speed bump against accidents, not authentication. Anyone who
can write files as you can forge it. It exists so that a stray environment variable, a copied
config or a forgotten setting can never start real trading by itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from botux.config import Settings, settings_fingerprint, settings_summary
from botux.errors import LiveTradingNotAllowed
from botux.killswitch import KillSwitch

ARM_PHRASE = "I ACCEPT THE RISK OF REAL TRADING LOSSES"
ARM_VERSION = 1
HOUR_MS = 3_600_000

RISK_TEXT = """\
READ BEFORE ARMING LIVE TRADING
 * You can lose part or all of the money you let this bot trade. Nothing here is a promise of
   profit, and no setting or strategy is safe or guaranteed.
 * Back-tests and paper results do not predict real results: fees, slippage, latency, partial
   fills, outages and regime changes differ, and parameters can be over-fitted.
 * A stop-loss implemented by this bot is only evaluated while the bot is running and connected.
   Gaps, slippage, outages and API errors can produce losses larger than the stop distance.
 * The live connector was not validated against the real exchange by its author (the build
   environment was geo-blocked). Start with an amount you can afford to lose entirely.
 * Use an API key with Spot trading ONLY: no withdrawals, no transfers, no margin/futures,
   and restrict it to this machine's IP address.
"""


@dataclass(frozen=True, slots=True)
class LiveAuthorization:
    """Proof that the live gate passed for exactly these settings."""

    fingerprint: str
    armed_at_ms: int


def _arm_path(settings: Settings) -> Path:
    return Path(settings.live.arm_file)


def _phrase_digest() -> str:
    return hashlib.sha256(ARM_PHRASE.encode()).hexdigest()


def write_arm_file(settings: Settings, now_ms: int) -> Path:
    path = _arm_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": ARM_VERSION,
        "fingerprint": settings_fingerprint(settings),
        "armed_at_ms": now_ms,
        "phrase_sha256": _phrase_digest(),
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".arm-")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)
    return path


def disarm(settings: Settings) -> None:
    _arm_path(settings).unlink(missing_ok=True)


def arm_live_interactive(
    settings: Settings,
    now_ms: int,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> Path | None:
    """Show settings + risks and require the exact phrase. Returns the arm file or None."""
    output_fn(RISK_TEXT)
    output_fn("Settings that will be bound to this confirmation:")
    output_fn(json.dumps(settings_summary(settings), indent=2, sort_keys=True))
    if settings.mode != "live" or not settings.live.enabled:
        output_fn(
            '\nNot armed: set mode = "live" and [live].enabled = true first '
            "(then review the settings above again)."
        )
        return None
    output_fn(f"\nType exactly:  {ARM_PHRASE}")
    if input_fn("> ").strip() != ARM_PHRASE:
        output_fn("Phrase did not match. Live trading NOT armed.")
        return None
    path = write_arm_file(settings, now_ms)
    output_fn(
        f"Armed until the settings change or {settings.live.arm_max_age_hours:g}h pass: {path}"
    )
    return path


def check_live_gate(settings: Settings, kill: KillSwitch, now_ms: int) -> LiveAuthorization:
    """Raise :class:`LiveTradingNotAllowed` listing every unmet condition."""
    problems: list[str] = []
    if settings.mode != "live":
        problems.append("mode is not 'live'")
    if not settings.live.enabled:
        problems.append("[live].enabled is false")
    if kill.is_set():
        problems.append(f"kill switch is set ({kill.reason()})")

    fingerprint = settings_fingerprint(settings)
    armed_at = 0
    path = _arm_path(settings)
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        problems.append(f"not armed: {path} does not exist (run 'botux arm-live')")
    except (OSError, ValueError):
        problems.append(f"arming file {path} is unreadable")
    else:
        if not isinstance(data, dict) or data.get("version") != ARM_VERSION:
            problems.append("arming file has an unknown format")
        else:
            if data.get("phrase_sha256") != _phrase_digest():
                problems.append("arming file does not carry the confirmation phrase")
            if data.get("fingerprint") != fingerprint:
                problems.append("settings changed since arming; run 'botux arm-live' again")
            raw_armed_at = data.get("armed_at_ms")
            if isinstance(raw_armed_at, bool) or not isinstance(raw_armed_at, int):
                problems.append("arming file has an invalid timestamp")
            else:
                armed_at = raw_armed_at
                age_ms = now_ms - armed_at
                if age_ms < -60_000:
                    problems.append("arming file is dated in the future")
                elif age_ms > settings.live.arm_max_age_hours * HOUR_MS:
                    problems.append("arming has expired; run 'botux arm-live' again")
    if problems:
        raise LiveTradingNotAllowed("live trading refused: " + "; ".join(problems))
    return LiveAuthorization(fingerprint=fingerprint, armed_at_ms=armed_at)


def api_key_problems(info: Mapping[str, Any], require_ip_restriction: bool) -> list[str]:
    """Evaluate the answer of GET /sapi/v1/account/apiRestrictions. Fails closed on missing data."""
    problems: list[str] = []
    must_be_false = {
        "enableWithdrawals": "withdrawals",
        "enableInternalTransfer": "internal transfer",
        "permitsUniversalTransfer": "universal transfer",
        "enableMargin": "margin",
        "enableFutures": "futures",
        "enableVanillaOptions": "options",
        "enablePortfolioMarginTrading": "portfolio margin",
    }
    for field_name, label in must_be_false.items():
        if info.get(field_name) is not False:
            problems.append(f"API key must not allow {label} ({field_name} is not false)")
    if info.get("enableReading") is not True:
        problems.append("API key must have reading enabled")
    if info.get("enableSpotAndMarginTrading") is not True:
        problems.append("API key must have Spot trading enabled")
    if require_ip_restriction and info.get("ipRestrict") is not True:
        problems.append("API key must be restricted to trusted IPs (ipRestrict is not true)")
    return problems
