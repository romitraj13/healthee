"""The OpenRouter balance — the cheap version of never running out again.

``GET /api/v1/credits`` returns the account's ``total_credits`` and ``total_usage`` in
dollars. It is a plain read on a normal key: **it spends no tokens and costs nothing**,
which is what makes it safe to poll on a timer where a completion call would not be.
(``GET /api/v1/activity`` would give per-day spend, but it requires a *management* key
and 403s on a normal one — deliberately not built on.)

Two jobs, and both matter:

* **Warn before zero.** ``transport_health`` can only tell us the layer is dead once it
  is; this tells us it is nearly dead while there is still time to top up. That is the
  whole difference between an alert and a post-mortem.
* **Prove the key still works.** The probe authenticates with the same key every
  completion uses, so a revoked or mistyped key fails here too — and this runs in the
  scheduler, on a timer, regardless of which process the failing completions were in.
  ``transport_health`` is per-process memory; this reading is a property of the ACCOUNT
  and is therefore the same fact everywhere.

## The reading is CACHED, and an error is cached too

``/readyz`` is unauthenticated, so an uncached read here would let anyone drive outbound
requests at OpenRouter one-for-one with requests at us. The TTL bounds it to one call
per :data:`_TTL_S` per process no matter how hard the endpoint is hit.

Failures are cached on the same terms, which looks wrong for about a second and is not:
a cached *error* still reports "we do not know the balance", which is the honest answer
and the one the caller must act on. Retrying the broken endpoint on every request would
turn a provider outage into a request-rate amplifier without learning anything sooner.

## "Unknown" is never "fine"

A probe that failed returns :data:`ERROR` with the reason. It does not return a
plausible-looking zero, and it does not return the last good reading as though it were
current (the stale-as-current bug class this repo has now hit three times). Standards
§Errors: "no data" and "operation failed" are different states and must be
distinguishable by the caller — here they are ``unconfigured`` and ``error``.

## What is NOT carried

The error path records the HTTP status code and the exception TYPE, never the response
body and never the request headers. The key is a secret and the body is free text we do
not control; a status code is the whole diagnosis and cannot leak either.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from healthee.core.config import get_settings
from healthee.core.logging import get_logger

log = get_logger(__name__)

_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
# The OpenRouter base URL — the provider whose credits endpoint this module reads.
# Kept in sync with `insights.client._OPENROUTER_BASE_URL`; a non-OpenRouter
# `LLM_BASE_URL` means the balance probe has nothing to read and reports unconfigured.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_TIMEOUT_S = 8.0

# How long one reading stands. Five minutes: long enough that a hammered `/readyz`
# makes one outbound call per five, short enough that an operator who just topped the
# account up sees it clear without restarting anything.
_TTL_S = 300.0

# Reading states. `UNCONFIGURED` is not a failure — a deployment with no
# `OPENROUTER_API_KEY` is running without the AI layer on purpose, which is a supported
# configuration (`core.config`), and reporting it as broken would be a false alarm.
OK = "ok"
ERROR = "error"
UNCONFIGURED = "unconfigured"

# Balance states, derived from a reading plus the configured threshold.
BALANCE_OK = "ok"
BALANCE_LOW = "low"
BALANCE_EXHAUSTED = "exhausted"
BALANCE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class CreditsReading:
    """One answer from the balance endpoint — or one honest failure to get it."""

    status: str
    checked_at: datetime
    remaining_usd: float | None = None
    total_credits_usd: float | None = None
    total_usage_usd: float | None = None
    error: str | None = None


def balance_state(reading: CreditsReading, low_threshold_usd: float | None = None) -> str:
    """Classify a reading: ``ok`` / ``low`` / ``exhausted`` / ``unknown``.

    Separate from :class:`CreditsReading` because the threshold is deployment config,
    not a property of the number — the same balance is comfortable on one box and an
    emergency on another, and a dataclass that reached into settings to decide would
    make the reading untestable without an environment.

    ``exhausted`` is ``<= 0`` and not ``< low_threshold``: at or below zero every call
    402s, which is a different event from "top this up soon" and gets different words.
    """
    if reading.status != OK or reading.remaining_usd is None:
        return BALANCE_UNKNOWN
    if reading.remaining_usd <= 0:
        return BALANCE_EXHAUSTED
    threshold = (
        low_threshold_usd if low_threshold_usd is not None else get_settings().llm_low_balance_usd
    )
    return BALANCE_LOW if reading.remaining_usd < threshold else BALANCE_OK


_lock = threading.Lock()
_cached: CreditsReading | None = None
_cached_at_monotonic: float | None = None


def read_balance(*, force: bool = False) -> CreditsReading:
    """The account balance, from cache when it is fresh enough.

    ``force`` skips the cache — used by the scheduler's watcher, which has its own,
    slower cadence and wants a reading it knows is current when it decides whether to
    page somebody.

    Freshness is measured on ``time.monotonic``, not the wall clock: an NTP step or a
    DST-adjusted clock must not be able to make a cached reading look older or younger
    than it is (the calendar-date-vs-instant lesson, one layer down).
    """
    global _cached, _cached_at_monotonic
    if not force:
        with _lock:
            fresh = (
                _cached is not None
                and _cached_at_monotonic is not None
                and time.monotonic() - _cached_at_monotonic < _TTL_S
            )
            if fresh and _cached is not None:
                return _cached
    reading = _fetch()
    with _lock:
        _cached, _cached_at_monotonic = reading, time.monotonic()
    return reading


def _fetch() -> CreditsReading:
    """One real call to the credits endpoint. Never raises; every failure is a reading.

    The credits endpoint is an OpenRouter concept — a non-OpenRouter gateway (Ollama
    Cloud, a local vLLM, …) has no ``/api/v1/credits`` to read, so the probe returns
    ``UNCONFIGURED`` without touching the network. That is the honest state: the
    balance is not \"fine\", it is simply not a thing this provider reports, and the
    transport-health signal (dead key / repeated failures) is the real spend/outage
    signal on such a provider.
    """
    settings = get_settings()
    if settings.llm_base_url != _OPENROUTER_BASE_URL:
        return CreditsReading(status=UNCONFIGURED, checked_at=datetime.now(tz=UTC))
    key = settings.openrouter_api_key.strip()
    if not key:
        return CreditsReading(status=UNCONFIGURED, checked_at=datetime.now(tz=UTC))
    try:
        response = httpx.get(
            _CREDITS_URL,
            headers={"Authorization": f"Bearer {key}"},  # the key goes to the wire, never a log
            timeout=_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        return _error(f"{type(exc).__name__} reaching the credits endpoint")
    if not response.is_success:
        # The status code only — the body is provider-controlled free text.
        return _error(f"HTTP {response.status_code} from the credits endpoint")
    return _parse(response)


def _parse(response: httpx.Response) -> CreditsReading:
    """Turn a 2xx body into a reading, or into a named failure if it is not the shape."""
    try:
        data = response.json()["data"]
        total = float(data["total_credits"])
        used = float(data["total_usage"])
    except (ValueError, KeyError, TypeError) as exc:
        # A shape change must read as "we do not know", never as a balance of zero —
        # which would page every hour forever, or (worse, with the sign flipped) look fine.
        return _error(f"unexpected credits payload ({type(exc).__name__})")
    return CreditsReading(
        status=OK,
        checked_at=datetime.now(tz=UTC),
        remaining_usd=total - used,
        total_credits_usd=total,
        total_usage_usd=used,
    )


def _error(detail: str) -> CreditsReading:
    """A failed probe, logged once here so it is never only in a response body."""
    log.warning("openrouter balance check failed: %s — the balance is UNKNOWN", detail)
    return CreditsReading(status=ERROR, checked_at=datetime.now(tz=UTC), error=detail)


def reset_cache() -> None:
    """Drop the cached reading — for tests, and for a caller that just changed the key."""
    global _cached, _cached_at_monotonic
    with _lock:
        _cached, _cached_at_monotonic = None, None
