"""The LLM transport is time-bounded, and a timeout is REPORTED — never an empty card.

Two separate properties, because the bug had two halves:

1. **The limits reach the SDK.** `insights/client.py` passed no `timeout`/`max_retries`,
   so it silently inherited the SDK's `Timeout(read=600)` + `max_retries=2` — 3 × 600 s
   = 30 minutes of hang for ONE stuck call. Since 6.4c the scheduler is a
   single-threaded tick loop over `active_users()`, so that call blocks every later
   owner's chain for half an hour while the tick just waits. Asserting the numbers
   *in settings* would prove nothing; the test asserts what the SDK constructor is
   actually handed.
2. **A timeout surfaces.** The failure mode this codebase must never have is a broken
   transport that reads as "no data" — so the timeout is followed all the way out: it
   is not caught by the choke point, and it lands on the chain's supervisor as a
   reported `failed`.

No network: the SDK constructor is faked and the client stub raises.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from uuid import UUID

import httpx
import openai
import pytest

from healthee.core.config import get_settings
from healthee.insights import grounded, pipeline
from healthee.insights.client import ChatResponse, OpenRouterClient
from healthee.jobs import chain

_OWNER = UUID("55555555-5555-5555-5555-555555555555")


class _FakeSDK:
    """Captures the kwargs `OpenRouterClient` hands the `openai` constructor."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        _FakeSDK.last_kwargs = kwargs


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the SDK's local `from openai import OpenAI` at a capturing fake.

    The limit vars are cleared so the defaults test sees the CODE's defaults rather
    than whatever the ambient shell exported (the `env` fixture's rule, same reason).

    The model ids are set alongside the key because `Settings` refuses the
    key-set-with-a-blank-id state outright (#56) — a client configured that way could
    never exist in a real process, so a fixture must not manufacture one.
    """
    _FakeSDK.last_kwargs = {}
    monkeypatch.setattr(openai, "OpenAI", _FakeSDK)
    for var in ("LLM_TIMEOUT_S", "LLM_MAX_RETRIES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-a-secret")
    monkeypatch.setenv("DEFAULT_MODEL", "vendor/cheap-test-model")
    monkeypatch.setenv("COACH_MODEL", "vendor/strong-test-model")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_the_configured_timeout_and_retries_reach_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
    fake_sdk: None,  # noqa: ARG001
) -> None:
    monkeypatch.setenv("LLM_TIMEOUT_S", "12.5")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    get_settings.cache_clear()

    OpenRouterClient()._client()  # noqa: SLF001 — the construction IS the subject

    assert _FakeSDK.last_kwargs["timeout"] == 12.5
    assert _FakeSDK.last_kwargs["max_retries"] == 0


def test_the_default_limits_are_sane_not_the_sdks_half_hour(
    fake_sdk: None,  # noqa: ARG001
) -> None:
    """Unset env must still bound the call — the hang was a *default*, so is the fix.

    The upper bounds are the point, not the exact values: whatever is chosen, one call
    must not be able to occupy the single-threaded sweep for the SDK's 30 minutes.
    """
    OpenRouterClient()._client()  # noqa: SLF001

    timeout = _FakeSDK.last_kwargs["timeout"]
    retries = _FakeSDK.last_kwargs["max_retries"]
    assert 0 < timeout <= 120, "a Flash-class call answers in seconds; this is a hang bound"
    assert 0 <= retries <= 1, "retries multiply the timeout — that is what makes a hang an outage"
    assert (retries + 1) * timeout <= 180, "worst case for one call, vs the SDK's 1800 s"


def _timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx.Request("POST", "https://openrouter.test/v1"))


class _TimingOutLLM:
    """A transport that always times out — what a wedged provider looks like."""

    def complete(self, messages: list[dict], **_kw: Any) -> ChatResponse:  # noqa: ARG002
        raise _timeout_error()


def test_a_timeout_propagates_out_of_the_choke_point_and_is_never_an_empty_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`grounded_ask` must RAISE, not return a blank/fallback `GroundedResult`.

    An honest fallback means "the model could not ground this"; a timeout means "we
    never got an answer". Collapsing the second into the first is exactly the swallow
    that leaves a dead surface looking merely quiet.
    """
    # The context/retrieval stages now live on the shared pipeline (both surfaces run
    # them), so the DB-backed builders are stubbed there rather than on `grounded`.
    monkeypatch.setattr(pipeline, "user_context", lambda *a, **kw: "# CONTEXT")  # noqa: ARG005
    monkeypatch.setattr(pipeline, "evidence", lambda *a, **kw: ("# EVIDENCE", []))  # noqa: ARG005

    with pytest.raises(openai.APITimeoutError):
        grounded.grounded_ask("How is my recovery trending?", _OWNER, "UTC", client=_TimingOutLLM())


def test_a_timeout_is_reported_to_the_job_health_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    """The supervisor turns the raised timeout into a logged + Telegrammed `failed`."""
    sent: list[str] = []
    monkeypatch.setattr(chain, "send_matrix", lambda text, **_: sent.append(text) or True)

    def raiser() -> dict:
        raise _timeout_error()

    outcome = chain._run_supervised("warm", raiser)  # noqa: SLF001 — the supervisor IS the subject

    assert outcome.status == "failed"
    assert outcome.detail is None  # not "no data" — a failure, distinguishable by the caller
    assert any("chain step 'warm' failed" in text for text in sent)
