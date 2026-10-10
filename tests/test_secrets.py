"""No secrets in logs, alerts, representations or tracked files."""

from __future__ import annotations

import io
import json
import logging
import re
import subprocess
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from botux.alerts import FileNotifier, LogNotifier
from botux.clock import FakeClock
from botux.exchange.client import BinanceSpotClient
from botux.secrets_guard import (
    REDACTED,
    Credentials,
    RedactingFilter,
    RedactingFormatter,
    Redactor,
    Secret,
    setup_logging,
)
from tests.conftest import FakeExchange

ROOT = Path(__file__).resolve().parents[1]
KEY = "Q" * 64
SECRET = "Z" * 64


class TestRedactor:
    def test_registered_literals_are_removed_everywhere(self) -> None:
        r = Redactor([SECRET])
        assert SECRET not in r.scrub(f"a {SECRET} b {SECRET}")
        assert r.scrub("short") == "short"  # nothing to hide

    def test_signature_query_and_header_patterns(self) -> None:
        r = Redactor()
        sig = "ab" * 32
        out = r.scrub(f"GET /api/v3/account?timestamp=1&signature={sig} X-MBX-APIKEY: abcdef123456")
        assert sig not in out and "abcdef123456" not in out and REDACTED in out

    def test_generic_secret_assignments(self) -> None:
        r = Redactor()
        for text in (
            "api_key=abcd1234",
            "apiSecret: 'zzzz9999'",
            "password = hunter2",
            "token=t0k3n",
        ):
            assert REDACTED in r.scrub(text), text

    def test_any_64_char_token_looks_like_a_binance_key(self) -> None:
        text = f"leaked {'A1' * 32} here"
        assert "A1" * 32 not in Redactor().scrub(text)

    def test_ordinary_text_is_untouched(self) -> None:
        text = "BTCUSDT BUY 0.001 @ 60000.00 status=FILLED"
        assert Redactor().scrub(text) == text


class TestLogging:
    def make_logger(self, redactor: Redactor) -> tuple[logging.Logger, io.StringIO]:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(RedactingFormatter("%(levelname)s %(message)s", redactor))
        handler.addFilter(RedactingFilter(redactor))
        logger = logging.getLogger(f"test.secrets.{id(stream)}")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        return logger, stream

    def test_message_arguments_and_tracebacks_are_scrubbed(self) -> None:
        r = Redactor([SECRET])
        logger, stream = self.make_logger(r)
        logger.info("plain %s", SECRET)
        logger.info(f"fstring {SECRET}")
        logger.info("dict %s", {"secret": SECRET})
        try:
            raise RuntimeError(f"boom {SECRET}")
        except RuntimeError:
            logger.exception("failed")
        out = stream.getvalue()
        assert SECRET not in out and out.count(REDACTED) >= 4

    def test_setup_logging_quiets_http_libraries_that_print_urls(self) -> None:
        setup_logging()
        assert logging.getLogger("httpx").level >= logging.WARNING
        assert logging.getLogger("httpcore").level >= logging.WARNING
        for h in logging.getLogger().handlers:
            assert any(isinstance(f, RedactingFilter) for f in h.filters)
            assert isinstance(h.formatter, RedactingFormatter)

    def test_connector_never_logs_keys_signatures_or_parameters(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ex = FakeExchange()
        ex.on("GET", "/api/v3/account", {"balances": []})
        ex.routes[("POST", "/api/v3/order")] = lambda _r: httpx.Response(500, text="oops")
        client = BinanceSpotClient(
            credentials=Credentials(Secret(KEY), Secret(SECRET)),
            trading_enabled=True,
            http=ex.http(),
            clock=FakeClock(),
        )
        with caplog.at_level(logging.DEBUG):
            client.account()
            with pytest.raises(Exception):  # noqa: B017 - any failure is fine here
                client.new_order(
                    symbol="BTCUSDT",
                    side="BUY",
                    order_type="MARKET",
                    quantity=Decimal("0.001"),
                    client_order_id="idsecret",
                )
        log_text = caplog.text
        assert KEY not in log_text and SECRET not in log_text
        assert "signature" not in log_text and "idsecret" not in log_text


class TestRepresentations:
    def test_secret_and_credentials_never_show_values(self) -> None:
        s, c = Secret(KEY), Credentials(Secret(KEY), Secret(SECRET))
        for obj in (s, c):
            for text in (repr(obj), str(obj), f"{obj}", f"{obj!r}"):
                assert KEY not in text and SECRET not in text
        assert s.reveal() == KEY


class TestAlerts:
    def test_file_alerts_are_redacted(self, tmp_path: Path) -> None:
        r = Redactor([SECRET])
        FileNotifier(tmp_path / "alerts.jsonl", FakeClock(), r).notify(
            "CRITICAL", f"title {SECRET}", f"message {SECRET} signature={'cd' * 32}"
        )
        text = (tmp_path / "alerts.jsonl").read_text()
        assert SECRET not in text and "cd" * 32 not in text
        assert json.loads(text)["level"] == "CRITICAL"

    def test_log_alerts_are_redacted(self, caplog: pytest.LogCaptureFixture) -> None:
        r = Redactor([SECRET])
        with caplog.at_level(logging.INFO):
            LogNotifier(r).notify("WARNING", "t", f"leak {SECRET}")
        assert SECRET not in caplog.text


# ---------------------------------------------------------------- repository hygiene


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="not a git checkout")
class TestRepositoryHygiene:
    @pytest.mark.parametrize(
        "path",
        [
            ".env",
            ".env.local",
            "config/botux.toml",
            ".botux/live_armed.json",
            ".botux/KILL",
            "data/botux.sqlite3",
            "key.pem",
            "logs/x.log",
            "secrets/api.txt",
        ],
    )
    def test_sensitive_paths_are_git_ignored(self, path: str) -> None:
        assert _git("check-ignore", "-q", path).returncode == 0, f"{path} is not ignored"

    @pytest.mark.parametrize(
        "path", [".env.example", "config/botux.example.toml", "src/botux/config.py"]
    )
    def test_templates_and_sources_are_not_ignored(self, path: str) -> None:
        assert _git("check-ignore", "-q", path).returncode == 1

    def test_no_secret_shaped_content_in_any_non_ignored_file(self) -> None:
        listing = _git("ls-files", "--cached", "--others", "--exclude-standard").stdout.split("\n")
        token = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{64}(?![A-Za-z0-9])")
        private_key = re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")
        offenders: list[str] = []
        for name in filter(None, listing):
            if name == "uv.lock" or not (ROOT / name).is_file():
                continue  # the lock file legitimately contains sha256 digests
            try:
                text = (ROOT / name).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if token.search(text) or private_key.search(text):
                offenders.append(name)
        assert offenders == []
