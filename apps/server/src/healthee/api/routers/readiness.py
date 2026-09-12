"""Dependency readiness — GET /readyz (no auth). Separate from /healthz on purpose.

## Why not just extend /healthz

``/healthz`` is what the container healthcheck and nginx act on. Its 503 means *restart
me / take me out of rotation*, and that is the only thing it may ever mean. Teaching it
about OpenRouter would make a provider outage restart our API in a loop — trading a dead
AI layer for a flapping read API, which is strictly worse. ``core.config``'s
``_require_model_ids_when_ai_key_is_set`` already records this argument in its own
docstring; this endpoint is the other side of it.

So the two probes are split by *who acts on them*:

* ``/healthz`` — liveness + DB. An ORCHESTRATOR reads it. Contract unchanged.
* ``/readyz`` — every dependency this deployment needs to be fully itself, the AI layer
  included. A HUMAN or a monitor reads it, and nothing restarts because of it.

## It costs nothing per probe

The rule that shaped this endpoint: a health check that makes a paid LLM call per probe
is a worse bug than the one it detects. Nothing here calls a model.

* the transport signal is a passive read of ``insights.transport_health``, which is
  filled in by traffic that was going to happen anyway;
* the balance comes from ``insights.credits``, whose endpoint is free and whose reading
  is TTL-cached, so hammering ``/readyz`` cannot amplify into outbound calls.

## What it reports, and what it deliberately does not

No dollar figures. This endpoint is unauthenticated (nginx proxies ``/`` wholesale, so
anything mounted here is public) and the account's economics are nobody else's business;
the amounts go to the operator's Telegram channel and the logs instead. The body carries
the STATE — ``ok`` / ``low`` / ``exhausted`` / ``unknown`` — which is what a monitor
needs to act, and no model id or key can appear in any of it.

## 503 only for the two states that are really broken

The DB being unreachable, or the AI layer being demonstrably dead (transport ``down``,
or a balance measured at zero). Never for ``degraded`` (a blip), never for ``unknown``
(no evidence either way), and never for ``low`` (a warning, and the layer still works).
A probe that 503s on uncertainty is a probe that gets ignored.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from healthee.api.routers.health import db_ok
from healthee.core.config import get_settings
from healthee.core.logging import get_logger
from healthee.insights import credits, transport_health

log = get_logger(__name__)

router = APIRouter(tags=["health"])


class LlmReadiness(BaseModel):
    """What this process knows about the LLM layer, and how it came to know it."""

    configured: bool
    transport: str
    consecutive_failures: int
    last_error_kind: str | None
    last_status_code: int | None
    balance: str
    balance_checked: bool
    balance_error: str | None


class Readiness(BaseModel):
    """Readiness body: overall verdict plus one entry per dependency."""

    status: str
    db: str
    llm: LlmReadiness


def _llm_readiness() -> LlmReadiness:
    """Read both LLM signals. Neither call spends a token; neither can raise."""
    configured = bool(get_settings().effective_llm_api_key().strip())
    health = transport_health.snapshot()
    reading = credits.read_balance()
    return LlmReadiness(
        configured=configured,
        transport=health.status,
        consecutive_failures=health.consecutive_failures,
        last_error_kind=health.last_error_kind,
        last_status_code=health.last_status_code,
        balance=credits.balance_state(reading),
        # Whether the number is a measurement at all — `unknown` alone cannot say
        # whether we asked and failed or never asked (standards §Errors).
        balance_checked=reading.status == credits.OK,
        balance_error=reading.error,
    )


def _ai_layer_is_dead(llm: LlmReadiness) -> bool:
    """True only on evidence, never on the absence of it.

    A deployment with no key configured is not broken — running without the AI layer is
    a supported configuration — so it is excluded before anything else is considered.
    """
    if not llm.configured:
        return False
    return llm.transport == transport_health.DOWN or llm.balance == credits.BALANCE_EXHAUSTED


@router.get("/readyz", response_model=Readiness)
def readyz(response: Response) -> Readiness:
    """Return 200 when every dependency is usable, 503 when the DB or the AI layer is not."""
    database = db_ok()
    llm = _llm_readiness()
    if database and not _ai_layer_is_dead(llm):
        return Readiness(status="ok", db="ok", llm=llm)
    log.warning(
        "readyz degraded: db=%s llm_transport=%s balance=%s",
        "ok" if database else "fail",
        llm.transport,
        llm.balance,
    )
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return Readiness(status="degraded", db="ok" if database else "fail", llm=llm)
