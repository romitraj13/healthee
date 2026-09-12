"""LLM chat client (OpenAI-compatible API) — the ONE LLM transport.

The provider is any OpenAI-compatible gateway: OpenRouter by default, or a
self-hosted / alternative endpoint (e.g. Ollama Cloud) via ``LLM_BASE_URL``. We
point the ``openai`` SDK at that base URL. Every LLM surface reaches this through
the grounded-ask choke point (``grounded.py``), never directly (standards §2:
"LLM access only via the grounded-ask choke point").

The API key is read once from ``core.config`` and never logged (standards
§Errors: no secret ever reaches a log line). Neither is the MODEL ID: it is kept out of
git on purpose (ids resolve from ``DEFAULT_MODEL`` / ``COACH_MODEL``) and a log line that
prints it undoes that at runtime, so every diagnostic here names the TIER instead
(:func:`tier_of`). Tests inject a stub implementing the ``LLMClient`` protocol, so no
network call happens under pytest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from healthee.core.config import get_settings
from healthee.core.logging import get_logger
from healthee.insights import transport_health

log = get_logger(__name__)

# Model ids live in settings (env: DEFAULT_MODEL / COACH_MODEL), NOT hardcoded here —
# the source never reveals which models we run; they're resolved per call. Per-surface
# tiers: default_model = the cheap, high-volume tier (recs/insights/notable/daily-
# action); coach_model = a stronger tier for the interactive coach (low volume, high
# engagement, where answer quality is most felt). Low temperature: health data wants
# factual, reproducible output, not creative tails. See docs/PRICING.md §6 / task #23.
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9
# A CEILING, not a spend: output is billed on tokens actually produced, so raising this
# costs nothing until an answer genuinely needs the room.
#
# 2000 was measured to be too small, and the old comment ("headroom so verbose/reasoning
# models aren't truncated") was wrong for the tier we actually run. A REASONING model
# spends its thinking out of this SAME budget: measured on the coach tier, one ordinary
# answer used 1068 completion tokens of which **892 were reasoning** — leaving ~180 for
# the visible reply. Longer answers then stopped mid-citation, the validator's truncation
# guard correctly refused them, and the coach shipped the honest fallback to ordinary
# questions. The failure read as "can't ground that" when the real cause was "ran out of
# room to finish the sentence".
DEFAULT_MAX_TOKENS = 10_000


def default_model() -> str:
    """The cheap, high-volume model tier (batch surfaces: recs/insights/notable/action)."""
    return get_settings().default_model


def coach_model() -> str:
    """The stronger model tier for the interactive coach (low volume, quality-first)."""
    return get_settings().coach_model


def tier_of(model: str) -> str:
    """Which configured TIER an id is — the diagnostic value a log line may carry.

    Standing owner constraint: the model we run must not be discoverable. It is kept out
    of git by design (ids resolve from ``DEFAULT_MODEL`` / ``COACH_MODEL``), and then the
    runtime logged it on every completion — the same category of leak as a bot token in a
    request URL, differing only in blast radius. Operators need to know *which surface's
    model* answered, not which model: "the coach tier is slow" and "the default tier is
    slow" are the two different diagnoses, and both are legible without the id.

    ``unconfigured`` is deliberately distinguishable from the two known tiers — an id
    that matches neither setting is a real operational fact (a caller passing an explicit
    ``model=``, or an env that changed under a running process) and hiding it would trade
    one silence for another.
    """
    settings = get_settings()
    if not model:
        return "unset"
    if model == settings.coach_model:
        return "coach"
    if model == settings.default_model:
        return "default"
    return "unconfigured"


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Usage:
    """Provider-COUNTED tokens for one completion — the measured number, not an estimate.

    ``reasoning_tokens`` is carried separately because it is the field that made a live
    bug unreadable: a reasoning model spends its thinking out of the same ``max_tokens``
    budget as its visible answer, so 892 of 1068 completion tokens were invisible and the
    coach's answers stopped mid-citation (VERIFICATION_2026_08_01 §6). That was diagnosed
    by instrumenting this call by hand; keeping the number costs nothing and makes the
    diagnosis repeatable — including by the grounding eval harness, which reads it
    through an injected client to report what an answer actually cost.

    ``cached_prompt_tokens`` is the same kind of field for the other half of the bill.
    The coach's system turn — persona, the owner's context and the evidence block — is
    built ONCE per request and re-sent unchanged on every round, ~40,000 tokens of it.
    Whether the provider serves that from its cache is the difference between a round
    that re-reads the corpus and one that does not, and it decides which lever makes
    the coach faster. Nothing recorded it, so "is the prompt the problem?" could only
    be answered by guessing.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0


@dataclass(frozen=True)
class ChatResponse:
    """A single assistant turn: its text plus any tool calls it requested.

    ``tool_calls`` is the raw OpenAI-shaped list (or None); the coach tool-loop
    (WP5b) consumes it. Insight surfaces use only ``text``.

    ``usage`` is None when the provider (or a test stub) reported none — "we don't know"
    and "zero tokens" are different states and stay distinguishable.
    """

    text: str
    tool_calls: list[Any] | None = None
    usage: Usage | None = None


class LLMClient(Protocol):
    """The one method the choke point needs. A stub implements this in tests."""

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        response_format: dict | None = None,
    ) -> ChatResponse: ...


class OpenAICompatClient:
    """Real transport: the ``openai`` SDK pointed at the configured base URL.

    Constructed lazily so importing this module never requires a key or touches
    the network. The endpoint is ``settings.llm_base_url`` (OpenRouter by default,
    any OpenAI-compatible gateway otherwise) and the key is
    ``settings.effective_llm_api_key()``.
    """

    def __init__(self) -> None:
        self._sdk: Any | None = None

    def _client(self) -> Any:
        """The SDK client, built once — with OUR limits, never the SDK's defaults.

        `timeout`/`max_retries` are passed explicitly because omitting them inherits
        `Timeout(read=600)` and `max_retries=2` — 30 minutes of hang for one stuck
        call, which the single-threaded scheduler tick pays for EVERY later owner
        (see `core.config.llm_timeout_s` for the numbers and why).
        """
        if self._sdk is None:
            settings = get_settings()
            key = settings.effective_llm_api_key()
            if not key:
                raise RuntimeError(
                    "LLM_API_KEY/OPENROUTER_API_KEY is unset — LLM features are unavailable"
                )
            from openai import OpenAI  # local import: heavy SDK, only when a call happens

            self._sdk = OpenAI(
                api_key=key,
                base_url=settings.llm_base_url,
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            )
        return self._sdk

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        response_format: dict | None = None,
    ) -> ChatResponse:
        """One completion. Returns the assistant text + any tool calls.

        ``response_format`` (e.g. ``{"type": "json_object"}``) is forwarded to the
        SDK when supplied — the grounded-ask choke point sets it for JSON surfaces
        (recs) so the model returns a parseable object, not fenced prose. Default
        ``None`` leaves the request unchanged (prose path is byte-identical).

        Errors propagate (the endpoint layer degrades to an honest error body) —
        never swallowed. A timeout is one of them: the SDK raises
        ``openai.APITimeoutError`` once ``llm_timeout_s`` is exceeded and it travels
        out through the choke point untouched, so it lands on the chain's supervisor
        (logged + Telegram-notified) instead of quietly becoming an empty answer.
        A blank card and a broken transport are different states and must stay so
        (standards §Errors). The key is passed to the SDK, never logged.

        Every attempt — success or failure — is recorded in ``insights.transport_health``,
        which is what makes a dead AI layer VISIBLE without any probe ever paying for a
        completion. The recording brackets the SDK call only: `_client()` raising for an
        unset key is a *configuration* fact, not a transport one, and counting it as a
        failed call would report the AI layer as broken on a box deliberately running
        without it.
        """
        model = model or get_settings().default_model  # resolve the env-configured default
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        if tools:
            kwargs["tools"] = tools
        if response_format is not None:
            kwargs["response_format"] = response_format
        # OpenRouter-only extras: the X-Title header and provider routing are
        # OpenRouter concepts. A non-OpenRouter gateway (Ollama Cloud, a local
        # vLLM, …) rejects or ignores them, so they are sent only when the base
        # URL is actually OpenRouter's.
        if get_settings().llm_base_url == _OPENROUTER_BASE_URL:
            kwargs["extra_headers"] = {"X-Title": "healthee"}
            provider = _provider_routing(model)
            if provider is not None:
                kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "provider": provider}
        sdk = self._client()
        started = time.monotonic()
        try:
            raw = sdk.chat.completions.create(**kwargs)
        except Exception as exc:  # recorded on the health surface, then re-raised untouched
            transport_health.record_failure(exc)
            raise
        elapsed = time.monotonic() - started
        transport_health.record_success()
        choice = raw.choices[0]
        message = choice.message
        _warn_if_truncated(choice, model)
        usage = _usage(raw)
        # tier, never the id: the model we run must not be discoverable, and a log line
        # is a place it reaches operators, log shippers and anyone with read access.
        #
        # ⛔ **The duration and the token split are the point.** Without them "the coach
        # is slow" could only be answered by subtracting timestamps of consecutive log
        # lines and guessing which half was to blame — which is how a 46,000-token prompt
        # got named as the cause of a ~100 s round before anybody had checked whether the
        # time was going into INPUT at all. This tier is a reasoning model and its
        # thinking is DECODE, generated a token at a time; prompt size and reasoning
        # length are different problems with different fixes, and one line here tells
        # them apart. Counts and seconds only — never a prompt, never an answer.
        log.info(
            "llm completion: tier=%s tools=%d in %.1fs "
            "(prompt=%d cached=%d completion=%d reasoning=%d)",
            tier_of(model),
            len(tools or []),
            elapsed,
            usage.prompt_tokens if usage else -1,
            usage.cached_prompt_tokens if usage else -1,
            usage.completion_tokens if usage else -1,
            usage.reasoning_tokens if usage else -1,
        )
        return ChatResponse(
            text=message.content or "",
            tool_calls=getattr(message, "tool_calls", None),
            usage=usage,
        )


def _provider_routing(model: str) -> dict[str, Any] | None:
    """The ordered provider preference for the COACH tier, or None to say nothing.

    Scoped by tier rather than applied globally, and that is the whole safety of it: the
    configured tags serve the coach model at a chosen quantization, while the default
    tier is a different model most of them do not host. Sent on every call, a preference
    meant for the coach would route the nightly chain to providers that cannot answer it.

    ``allow_fallbacks`` is True and is not configurable. The owner asked for these
    providers "then fallback if none of them works", which is exactly this flag: the
    order is walked first, and anything else that serves the model answers rather than
    the request failing. A hard restriction is a different feature and would need its own
    argument — a coach that 503s because four named providers were busy is a worse
    outcome than an answer from a fifth.
    """
    if tier_of(model) != "coach":
        return None
    raw = get_settings().llm_provider_order
    order = [tag.strip() for tag in raw.split(",") if tag.strip()]
    return {"order": order, "allow_fallbacks": True} if order else None


def _warn_if_truncated(choice: Any, model: str) -> None:
    """Say so when the API stopped because it ran out of room.

    We used to drop the whole response object and keep only ``.message``, so the ONLY
    signal left was the validator noticing an unclosed '[' downstream — a string
    heuristic standing in for a fact the transport already knew. A truncated answer is a
    transport problem wearing a grounding problem's clothes, and it cost a live debugging
    session to tell them apart. Log it loudly; the validator still refuses the text, but
    now the cause is in the logs at the point it happened rather than inferred three
    layers up.

    ``getattr`` with a default, not ``choice.finish_reason``: not every provider (or test
    stub) sets it, and a missing stop-reason must not turn a good answer into an
    AttributeError. Absent ⇒ we simply don't know, which is not "truncated".
    """
    if getattr(choice, "finish_reason", None) != "length":
        return
    log.warning(
        "llm answer TRUNCATED by max_tokens=%d (tier=%s) — the answer stopped "
        "mid-sentence; raise DEFAULT_MAX_TOKENS rather than reading this as a "
        "grounding failure",
        DEFAULT_MAX_TOKENS,
        tier_of(model),
    )


def _usage(raw: Any) -> Usage | None:
    """The provider's own token counts, or None when it reported none.

    Read defensively with ``getattr``: not every provider behind OpenRouter returns a
    ``usage`` block, and none of them is required to break down reasoning tokens. A
    missing count must never turn a good answer into an AttributeError — a cost figure
    is diagnostic, an answer is the product.
    """
    usage = getattr(raw, "usage", None)
    if usage is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return Usage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
        # Read as defensively as the rest: providers spell this differently and some
        # omit it. A zero here means "not reported", which is NOT the same as "no cache
        # hit" — the log line says the number, and reading it as a claim either way
        # would be exactly the guess this field exists to replace.
        cached_prompt_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
    )


@lru_cache(maxsize=1)
def get_client() -> OpenAICompatClient:
    """Process-wide real client. ``grounded_ask`` defaults to this; tests pass a stub."""
    return OpenAICompatClient()


# Backward-compatible alias: callers and tests that imported the old name keep
# working. The class is provider-agnostic now; the alias is a name, not a behaviour.
OpenRouterClient = OpenAICompatClient
