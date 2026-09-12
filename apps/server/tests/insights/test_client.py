"""Transport unit tests — the JSON output-format seam, and the model id stays unlogged.

No network: the OpenAI SDK object is replaced by a fake that records the kwargs
handed to ``chat.completions.create`` so we can assert ``response_format`` is (and
is NOT) forwarded, without changing the prose path.

The second concern here is a standing owner constraint: **the model we run must not be
discoverable**. Keeping the ids out of git (they resolve from ``DEFAULT_MODEL`` /
``COACH_MODEL``) is undone by a runtime log line that prints them, which is what
``llm completion: model=…`` did on every single call. The tests below capture the log
records and assert the id is absent and the TIER is present, so an operator keeps the
diagnostic and nobody reading logs learns the id.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from healthee.core.config import get_settings
from healthee.insights import client as client_module
from healthee.insights import transport_health
from healthee.insights.client import OpenRouterClient, tier_of


class _FakeCompletions:
    def __init__(self, finish_reason: str | None = None, usage: Any = None) -> None:
        self.kwargs: dict[str, Any] | None = None
        self._finish_reason = finish_reason
        self._usage = usage

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        message = SimpleNamespace(content="ok", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason=self._finish_reason)
        return SimpleNamespace(choices=[choice], usage=self._usage)


class _FakeSDK:
    def __init__(self, finish_reason: str | None = None, usage: Any = None) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(finish_reason, usage))


def _client_with_fake(
    finish_reason: str | None = None, usage: Any = None
) -> tuple[OpenRouterClient, _FakeSDK]:
    client = OpenRouterClient()
    fake = _FakeSDK(finish_reason, usage)
    client._client = lambda: fake  # type: ignore[method-assign]  # inject the fake SDK
    return client, fake


# A value that could not plausibly be anything but the id we passed, so "the id is
# absent" is a real assertion rather than a coincidence of short strings.
_SECRET_MODEL = "vendor-x/never-log-me-9000"


@pytest.fixture
def configured_models(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Point both tiers at known ids so ``tier_of`` has something to resolve against."""
    monkeypatch.setenv("COACH_MODEL", _SECRET_MODEL)
    monkeypatch.setenv("DEFAULT_MODEL", "vendor-x/cheap-tier-1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_response_format_is_forwarded_to_the_sdk() -> None:
    client, fake = _client_with_fake()
    client.complete([{"role": "user", "content": "x"}], response_format={"type": "json_object"})
    assert fake.chat.completions.kwargs is not None
    assert fake.chat.completions.kwargs["response_format"] == {"type": "json_object"}


def test_response_format_absent_by_default_keeps_prose_path_unchanged() -> None:
    client, fake = _client_with_fake()
    client.complete([{"role": "user", "content": "x"}])
    assert fake.chat.completions.kwargs is not None
    assert "response_format" not in fake.chat.completions.kwargs


# ── coach-tier provider routing ──────────────────────────────────────────────


@pytest.fixture
def provider_order(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Set the provider order this client sees, without touching the settings cache.

    `get_settings.cache_clear()` is NOT usable here — `tests/conftest.py` builds the
    settings once at session start inside its own environment, so clearing the cache
    surfaces `POSTGRES_PASSWORD must be set` in whatever test runs next.
    """

    def _set(order: str) -> None:
        real = client_module.get_settings()
        monkeypatch.setattr(
            client_module,
            "get_settings",
            lambda: SimpleNamespace(
                default_model=real.default_model,
                coach_model=real.coach_model,
                llm_provider_order=order,
                llm_base_url=real.llm_base_url,
            ),
        )

    return _set


def test_no_provider_block_is_sent_when_no_order_is_configured(
    provider_order: Any, configured_models: None
) -> None:  # noqa: ARG001 — the fixture is the environment
    """Empty means "say nothing", so OpenRouter's own routing is untouched."""
    provider_order("")
    client, fake = _client_with_fake()
    client.complete([{"role": "user", "content": "x"}], model=_SECRET_MODEL)
    assert fake.chat.completions.kwargs is not None
    assert "extra_body" not in fake.chat.completions.kwargs


def test_the_coach_tier_gets_the_order_and_keeps_fallbacks(
    provider_order: Any, configured_models: None
) -> None:  # noqa: ARG001 — the fixture is the environment
    """The owner asked for these providers "then fallback if none of them works"."""
    provider_order("baidu/fp8, wafer/fast ,reka/fp4,fireworks")
    client, fake = _client_with_fake()
    client.complete([{"role": "user", "content": "x"}], model=_SECRET_MODEL)
    assert fake.chat.completions.kwargs is not None
    assert fake.chat.completions.kwargs["extra_body"] == {
        "provider": {
            "order": ["baidu/fp8", "wafer/fast", "reka/fp4", "fireworks"],
            "allow_fallbacks": True,
        }
    }


def test_the_DEFAULT_tier_is_never_routed(provider_order: Any, configured_models: None) -> None:  # noqa: ARG001, N802 — the fixture is the environment
    """THE regression, and it is an outage rather than a degradation.

    The configured tags serve the COACH model. The default tier is a different model
    that most of them do not host, so a blanket order sends recs, briefings and notable
    cards to providers that cannot answer them — the nightly chain going dark, wearing a
    routing preference as a disguise.
    """
    provider_order("baidu/fp8,wafer/fast,reka/fp4,fireworks")
    client, fake = _client_with_fake()
    client.complete([{"role": "user", "content": "x"}], model="vendor-x/cheap-tier-1")
    assert fake.chat.completions.kwargs is not None
    assert "extra_body" not in fake.chat.completions.kwargs


# ── non-OpenRouter gateways get no OpenRouter-only extras ───────────────────


@pytest.fixture
def ollama_base_url(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Point the client at a non-OpenRouter gateway (Ollama Cloud)."""

    def _set() -> None:
        real = client_module.get_settings()
        monkeypatch.setattr(
            client_module,
            "get_settings",
            lambda: SimpleNamespace(
                default_model=real.default_model,
                coach_model=real.coach_model,
                llm_provider_order="baidu/fp8,wafer/fast",
                llm_base_url="https://ollama.com/v1",
            ),
        )

    return _set


def test_a_non_openrouter_gateway_gets_no_x_title_header(
    ollama_base_url: Any, configured_models: None
) -> None:  # noqa: ARG001 — the fixture is the environment
    """`X-Title` is an OpenRouter concept; Ollama Cloud must not receive it."""
    ollama_base_url()
    client, fake = _client_with_fake()
    client.complete([{"role": "user", "content": "x"}], model=_SECRET_MODEL)
    assert fake.chat.completions.kwargs is not None
    assert "extra_headers" not in fake.chat.completions.kwargs


def test_a_non_openrouter_gateway_gets_no_provider_routing(
    ollama_base_url: Any, configured_models: None
) -> None:  # noqa: ARG001 — the fixture is the environment
    """Provider routing is OpenRouter-only; a gateway that ignores it must not get it."""
    ollama_base_url()
    client, fake = _client_with_fake()
    client.complete([{"role": "user", "content": "x"}], model=_SECRET_MODEL)
    assert fake.chat.completions.kwargs is not None
    assert "extra_body" not in fake.chat.completions.kwargs


# ── the model id must not be discoverable from the logs ──────────────────────


def test_the_completion_log_names_the_tier_not_the_model_id(
    caplog: pytest.LogCaptureFixture, configured_models: None
) -> None:  # noqa: ARG001 — the fixture is the environment
    client, _ = _client_with_fake()
    with caplog.at_level(logging.INFO):
        client.complete([{"role": "user", "content": "x"}], model=_SECRET_MODEL)
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "tier=coach" in logged
    assert _SECRET_MODEL not in logged


def test_the_truncation_warning_names_the_tier_not_the_model_id(
    caplog: pytest.LogCaptureFixture, configured_models: None
) -> None:  # noqa: ARG001 — the fixture is the environment
    """The same leak was in the warning added when the reasoning-token trap was found."""
    client, _ = _client_with_fake(finish_reason="length")
    with caplog.at_level(logging.WARNING):
        client.complete([{"role": "user", "content": "x"}], model=_SECRET_MODEL)
    warnings = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "TRUNCATED" in warnings
    assert "tier=coach" in warnings
    assert _SECRET_MODEL not in warnings


def test_tier_of_distinguishes_the_two_configured_tiers(configured_models: None) -> None:  # noqa: ARG001
    """Operators need "which surface's model", and both tiers must be tellable apart."""
    assert tier_of(_SECRET_MODEL) == "coach"
    assert tier_of("vendor-x/cheap-tier-1") == "default"


# ── provider-counted usage reaches the caller (the eval harness reads it) ────


def test_the_providers_token_counts_are_carried_on_the_response() -> None:
    """Including reasoning tokens — the field that made the max_tokens trap unreadable."""
    usage = SimpleNamespace(
        prompt_tokens=33_446,
        completion_tokens=1068,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=892),
    )
    client, _ = _client_with_fake(usage=usage)
    response = client.complete([{"role": "user", "content": "x"}])
    assert response.usage is not None
    assert response.usage.prompt_tokens == 33_446
    assert response.usage.completion_tokens == 1068
    assert response.usage.reasoning_tokens == 892


def test_a_provider_that_reports_no_usage_is_unknown_not_zero() -> None:
    """ "We don't know" and "it cost nothing" are different states (standards §Errors)."""
    client, _ = _client_with_fake(usage=None)
    assert client.complete([{"role": "user", "content": "x"}]).usage is None


def test_a_usage_block_without_a_reasoning_breakdown_still_reports_what_it_has() -> None:
    """Not every provider breaks reasoning out; a missing detail is 0, never a crash."""
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
    client, _ = _client_with_fake(usage=usage)
    response = client.complete([{"role": "user", "content": "x"}])
    assert response.usage is not None
    assert (response.usage.prompt_tokens, response.usage.reasoning_tokens) == (100, 0)


def test_an_id_matching_neither_setting_is_reported_as_unconfigured(
    configured_models: None,
) -> None:  # noqa: ARG001
    """A caller passing an explicit model, or an env changed under a running process."""
    assert tier_of("vendor-x/some-other-model") == "unconfigured"
    assert tier_of("") == "unset"


# ── every attempt reaches the transport health record (the 2026-08-01 incident) ──


class _RefusingCompletions:
    """A transport that always fails the way a spent account does."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def create(self, **_kwargs: Any) -> Any:
        raise _ProviderError(self.status_code)


class _ProviderError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


def _refusing_client(status_code: int) -> OpenRouterClient:
    client = OpenRouterClient()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=_RefusingCompletions(status_code)))
    client._client = lambda: fake  # type: ignore[method-assign]
    return client


def test_a_successful_completion_records_the_transport_as_healthy() -> None:
    client, _ = _client_with_fake()
    client.complete([{"role": "user", "content": "x"}])
    assert transport_health.snapshot().status == transport_health.OK


def test_a_402_is_recorded_and_still_raises_out_of_the_transport() -> None:
    """Both halves: the record learns, and the caller is NOT told a lie about it.

    Recording must not become swallowing — the chain supervisor still has to see the
    exception and report the step as failed (standards §Errors).
    """
    client = _refusing_client(402)
    with pytest.raises(_ProviderError):
        client.complete([{"role": "user", "content": "x"}])
    snapshot = transport_health.snapshot()
    assert snapshot.consecutive_failures == 1
    assert snapshot.last_error_kind == transport_health.CREDIT
    assert snapshot.last_status_code == 402


def test_repeated_402s_drive_the_record_to_down() -> None:
    """The live incident, reproduced through the real transport rather than the record."""
    client = _refusing_client(402)
    for _ in range(transport_health.OUTAGE_THRESHOLD):
        with pytest.raises(_ProviderError):
            client.complete([{"role": "user", "content": "x"}])
    assert transport_health.snapshot().status == transport_health.DOWN


def test_an_unset_key_is_a_config_fact_and_never_a_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Running without the AI layer is a supported configuration, not an outage.

    `_client()` raises before any request exists, so counting it would report the
    transport as broken on a box that was deliberately never given a key.
    """
    # `chdir` first: `Settings` reads `env_file=".env"` resolved against the CWD, so
    # `delenv` alone does NOT unset the field — the real `apps/server/.env` supplies it,
    # and this passed in CI and in worktrees (neither has one) while failing on a
    # developer box. Same trap `conftest._hermetic_settings_env` documents.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "unit-test-pw")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError):
        OpenRouterClient().complete([{"role": "user", "content": "x"}])
    assert transport_health.snapshot().status == transport_health.UNKNOWN
    get_settings.cache_clear()
