"""The ``briefing`` chain step — the morning "daily-insight" Telegram message.

A short grounded summary of where the user stands today, generated THROUGH the
choke point (v2 context, cited, blocking-validated) and sent via ``core.notify``
(which never raises and no-ops cleanly when Telegram is unconfigured). Honest by
construction: a flat or below-par day is reported as such, and if the model can't ground
a claim the choke point returns its honest fallback — which we still send, plainly,
rather than inventing an upbeat line.

## This step no longer generates on the normal path (#95)

The briefing task asked for "today's single most useful action" and the daily-action
prompt asked for that same line and nothing else — two full-corpus calls for one answer.
Since #95 the ``warm`` step's merged morning call (``insights.morning``) writes the
briefing BODY into the per-day cache along with the action, so this step usually spends
**zero** LLM calls and simply date-stamps and sends what was warmed.

It still generates when there is nothing warmed to send, and that is the whole coupling
story: a merged call that could not ship, or a ``correlate`` failure that skipped ``warm``
entirely (``jobs/chain.py``), leaves this step doing exactly what it did before #95. The
cost saving is conditional on the merged call succeeding; the briefing arriving is not.
"""

from __future__ import annotations

from datetime import date
from uuid import UUID

from healthee.core.logging import get_logger
from healthee.core.notify import send_matrix
from healthee.core.tenancy import user_today
from healthee.insights import coaching, morning
from healthee.insights.client import LLMClient

log = get_logger(__name__)


def send_briefing(
    user_id: UUID, tz: str, day: date | None = None, *, client: LLMClient | None = None
) -> dict:
    """Send ``user_id``'s grounded briefing over Telegram. Returns a status dict.

    ``day`` defaults to the OWNER's local today (from their ``tz``), not a global one.
    ``client`` is injectable for tests. Errors from generation propagate to the
    supervised chain runner; the Telegram send itself never raises (``core.notify``).

    ``source`` names which path produced the text — ``warm`` (the merged morning call,
    no model run here) or ``standalone`` (this step generated). It is on the status dict
    because "the briefing went out" and "the briefing cost a call" are two different
    facts, and the job health surface is where the second one is legible.

    ## Why a past ``day`` is refused rather than honoured

    ``day`` reaches exactly one thing: the ``healthee briefing — {day}`` header
    ``_compose`` prints. Every input is unconditionally today's — ``cached_line`` is
    keyed on the owner's own today and takes no day, and ``morning.generate_briefing``
    anchors on ``user_today(tz)`` throughout. So a call with a past ``day`` sent
    **today's judgement under an older date**, over the one channel where the reader
    has no other date to check it against.

    That is the stale-as-current lie ``docs/AS_OF_DAY.md`` section 3 names, and
    ``jobs/recs.py`` already refuses its own version of it in thirty lines of argument.
    This is the same refusal for the same reason: authoring a past day's judgement now
    is a new claim, not a record (AS_OF_DAY section 6), so making ``day`` bind the inputs
    is not the repair either. It is loud rather than a silent clamp — the only caller
    that can reach it is a hand-run ``run_chain(..., force=True)``, and a back-fill that
    would have mislabelled a message must fail where somebody can see it.

    ``recs`` raises first on the same forced chain, so nothing that used to work stops
    working; what changes is that the briefing can no longer go out mis-dated behind it.
    """
    today = user_today(tz)
    if day is not None and day != today:
        raise ValueError(
            f"send_briefing cannot stamp a briefing {day}: every input it has is "
            f"{today}'s (cached_line and generate_briefing take no reference day), so "
            "the message would carry a date its own content never answered for "
            "(docs/AS_OF_DAY.md section 6)."
        )
    day = today
    warmed = coaching.cached_line(user_id, tz, coaching.MORNING_BRIEFING_KEY)
    body, status = _body(user_id, tz, warmed, client=client)
    sent = send_matrix(_compose(day, body))
    log.info(
        "briefing[%s] %s: sent=%s source=%s validated=%s refused=%s",
        user_id,
        day,
        sent,
        status["source"],
        status["validated"],
        status["refused"],
    )
    return {"ok": True, "day": day.isoformat(), "sent": sent, **status}


def _body(
    user_id: UUID, tz: str, warmed: str | None, *, client: LLMClient | None
) -> tuple[str, dict]:
    """The message body plus what it cost — the warmed text, or a generation of our own.

    A warmed body is only ever cached after it validated (``coaching._cache``), so this
    path is validated-by-construction; the standalone path reports what the choke point
    concluded, including an honest fallback, which is still sent.
    """
    if warmed is not None:
        return warmed, {"source": "warm", "validated": True, "refused": False}
    result = morning.generate_briefing(user_id, tz, client=client)
    return result.text, {
        "source": "standalone",
        "validated": result.validated,
        "refused": result.refused,
    }


def _compose(day: date, body: str) -> str:
    """Date-stamped header + the grounded body (never presented as undated)."""
    return f"healthee briefing — {day.isoformat()}\n\n{body}"
