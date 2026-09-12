"""The balance probe: free, cached, and never optimistic about what it does not know.

No network — `httpx.get` is replaced, and every test asserts on what the module does
with the answer rather than on the provider.

The property that matters most here is the last section's: a probe that fails must
report UNKNOWN. A monitoring feature that fails quiet is worse than none, because it
converts "we don't know" into "we're fine" — which is the exact sentence the 2026-08-01
incident wrote into the logs for six hours.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from healthee.core.config import get_settings
from healthee.insights import credits


@pytest.fixture
def keyed(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A configured deployment: a key, and the model ids Settings demands beside it."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-a-secret")
    monkeypatch.setenv("DEFAULT_MODEL", "vendor/cheap")
    monkeypatch.setenv("COACH_MODEL", "vendor/strong")
    monkeypatch.setenv("POSTGRES_PASSWORD", "unit-test-pw")
    get_settings.cache_clear()
    credits.reset_cache()
    yield
    get_settings.cache_clear()
    credits.reset_cache()


def _respond(monkeypatch: pytest.MonkeyPatch, response: httpx.Response) -> list[dict]:
    """Point the probe at a canned response; return the list of calls it makes."""
    calls: list[dict] = []

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        calls.append({"url": url, **kwargs})
        return response

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


def _balance(total: float, used: float) -> httpx.Response:
    return httpx.Response(200, json={"data": {"total_credits": total, "total_usage": used}})


# ── the happy path ────────────────────────────────────────────────────────────


def test_the_reading_is_the_providers_own_arithmetic(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:  # noqa: ARG001
    _respond(monkeypatch, _balance(200.0, 187.5))
    reading = credits.read_balance()
    assert reading.status == credits.OK
    assert reading.remaining_usd == pytest.approx(12.5)
    assert reading.total_credits_usd == 200.0
    assert reading.total_usage_usd == 187.5


def test_the_key_is_sent_as_a_bearer_header_and_appears_nowhere_else(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:  # noqa: ARG001
    calls = _respond(monkeypatch, _balance(200.0, 0.0))
    credits.read_balance()
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key-not-a-secret"
    assert "test-key-not-a-secret" not in calls[0]["url"]


def test_an_unconfigured_deployment_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No key ⇒ no AI layer by choice, which must not read as a broken one."""
    # `chdir` first: `Settings` reads `env_file=".env"` resolved against the CWD, so
    # `delenv` alone does NOT unset the field — the real `apps/server/.env` supplies it,
    # and this passed in CI and in worktrees (neither has one) while failing on a
    # developer box. Same trap `conftest._hermetic_settings_env` documents.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "unit-test-pw")
    get_settings.cache_clear()
    calls = _respond(monkeypatch, _balance(1.0, 0.0))
    assert credits.read_balance().status == credits.UNCONFIGURED
    assert calls == [], "an unconfigured box still called the provider"
    get_settings.cache_clear()


def test_a_non_openrouter_gateway_has_no_balance_to_read(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:  # noqa: ARG001
    """Ollama Cloud (or any non-OpenRouter gateway) has no `/credits` endpoint.

    The probe must report UNCONFIGURED without touching the network — the balance is
    not "fine", it is simply not a thing this provider reports, and the transport-health
    signal is the real spend/outage signal there.
    """
    monkeypatch.setenv("LLM_BASE_URL", "https://ollama.com/v1")
    get_settings.cache_clear()
    calls = _respond(monkeypatch, _balance(200.0, 0.0))
    assert credits.read_balance().status == credits.UNCONFIGURED
    assert calls == [], "a non-OpenRouter gateway still called the credits endpoint"
    get_settings.cache_clear()


# ── the state machine an alert is built on ────────────────────────────────────


@pytest.mark.parametrize(
    ("total", "used", "state"),
    [
        (200.0, 10.0, credits.BALANCE_OK),
        (200.0, 195.0, credits.BALANCE_LOW),  # $5 left, threshold $20
        (200.0, 200.0, credits.BALANCE_EXHAUSTED),
        (200.0, 260.0, credits.BALANCE_EXHAUSTED),  # overdrawn is not "low"
    ],
)
def test_balance_state_separates_top_up_soon_from_already_402ing(
    monkeypatch: pytest.MonkeyPatch, keyed: None, total: float, used: float, state: str
) -> None:  # noqa: ARG001
    _respond(monkeypatch, _balance(total, used))
    assert credits.balance_state(credits.read_balance(), low_threshold_usd=20.0) == state


def test_a_zero_threshold_disables_the_warning_but_not_the_exhausted_alert(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:  # noqa: ARG001
    _respond(monkeypatch, _balance(200.0, 199.99))
    reading = credits.read_balance()
    assert credits.balance_state(reading, low_threshold_usd=0.0) == credits.BALANCE_OK


def test_the_threshold_defaults_to_the_configured_one(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:  # noqa: ARG001
    monkeypatch.setenv("LLM_LOW_BALANCE_USD", "100")
    get_settings.cache_clear()
    _respond(monkeypatch, _balance(200.0, 150.0))  # $50 left — fine at 20, low at 100
    assert credits.balance_state(credits.read_balance()) == credits.BALANCE_LOW


# ── failure is UNKNOWN, never a plausible-looking number ──────────────────────


def test_a_non_2xx_is_an_error_reading_carrying_only_the_status_code(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:  # noqa: ARG001
    """A revoked key 401s here, and that is the point — the probe validates the key too.

    The body is provider-controlled free text, so only the code is kept.
    """
    _respond(monkeypatch, httpx.Response(401, json={"error": {"message": "vendor-x/secret-model"}}))
    reading = credits.read_balance()
    assert reading.status == credits.ERROR
    assert reading.remaining_usd is None
    assert "401" in (reading.error or "")
    assert "secret-model" not in (reading.error or "")
    assert credits.balance_state(reading) == credits.BALANCE_UNKNOWN


def test_a_network_failure_is_an_error_reading_not_a_zero_balance(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:  # noqa: ARG001
    def boom(url: str, **_kwargs: Any) -> httpx.Response:  # noqa: ARG001
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "get", boom)
    reading = credits.read_balance()
    assert reading.status == credits.ERROR
    assert credits.balance_state(reading) == credits.BALANCE_UNKNOWN


def test_an_unexpected_payload_shape_is_unknown_not_a_balance_of_zero(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:  # noqa: ARG001
    """A silent 0 here would page hourly forever; a silent large number would hide a fire."""
    _respond(monkeypatch, httpx.Response(200, json={"data": {"credits": 5}}))
    reading = credits.read_balance()
    assert reading.status == credits.ERROR
    assert reading.remaining_usd is None


def test_a_failed_probe_is_logged_and_not_only_returned(
    monkeypatch: pytest.MonkeyPatch, keyed: None, caplog: pytest.LogCaptureFixture
) -> None:  # noqa: ARG001
    _respond(monkeypatch, httpx.Response(500))
    with caplog.at_level("WARNING"):
        credits.read_balance()
    assert any("UNKNOWN" in r.getMessage() for r in caplog.records)


# ── the cache bounds what an unauthenticated /readyz can amplify into ─────────


def test_a_second_read_inside_the_ttl_makes_no_second_call(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:  # noqa: ARG001
    calls = _respond(monkeypatch, _balance(200.0, 1.0))
    for _ in range(20):
        credits.read_balance()
    assert len(calls) == 1, "an unauthenticated probe amplified into outbound calls"


def test_force_bypasses_the_cache_for_the_watcher_that_decides_to_page(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:  # noqa: ARG001
    calls = _respond(monkeypatch, _balance(200.0, 1.0))
    credits.read_balance()
    credits.read_balance(force=True)
    assert len(calls) == 2


def test_an_error_is_cached_too_so_a_broken_endpoint_is_not_hammered(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:  # noqa: ARG001
    """A cached error still says "we do not know", which is the honest answer."""
    calls = _respond(monkeypatch, httpx.Response(503))
    for _ in range(10):
        assert credits.read_balance().status == credits.ERROR
    assert len(calls) == 1
