"""The alerting half: turn a dead AI layer into a message somebody actually receives.

## The incident this is the answer to

2026-08-01: the OpenRouter account hit its $200 ceiling. Every LLM call 402'd — coach,
insight cards, the whole nightly chain — for hours, while ``/healthz`` returned
``{"status":"ok","db":"ok"}``. It surfaced because an eval run happened to die. A pull
surface alone (``/readyz``) would not have fixed that: nobody was looking. Something had
to *push*.

The Telegram channel that already carries chain-step failures is that something, and it
is reused rather than duplicated — a second notifier would be a second thing to
configure, a second thing to leave unconfigured, and a second place for the
"notifications are broken" failure to hide. This module owns the POLICY (when is it
worth a message?) and ``core.notify`` stays the only sink.

## Two signals, because neither alone is enough

* ``insights.transport_health`` — what real traffic is doing, in THIS process. It is
  free and immediate, but it is per-process memory: an outage that so far has only hit
  the api container is invisible here.
* ``insights.credits`` — the account balance. It costs nothing (a free endpoint, no
  tokens) and it is a property of the ACCOUNT, so it sees an exhausted balance no matter
  which process the failing calls were in. It also authenticates with the same key, so a
  revoked key fails the probe and is reported.

Together: the balance probe usually fires FIRST (before zero, by design), and the
transport signal is the backstop for everything a balance cannot express — a dead key, a
provider outage, a network partition.

## Why a scheduler tick and not the request path

``send_matrix`` is a synchronous HTTP POST with a 10 s timeout. On a request path that
is latency somebody pays for, on an error path it is latency somebody pays for while
already having a bad time. The scheduler is a loop that already wakes on a timer and
already owns the Telegram health surface, so the alert costs nothing anybody is waiting
for. The api's contribution is ``/readyz``, which is a pull.

## Edge-triggered, with recovery — never level-triggered

An alert repeated every tick is an alert that gets muted, and a muted channel is worse
than no channel because it also swallows the chain-failure messages it shares. So each
condition alerts ONCE on the way in and ONCE on the way out. The state is in memory,
like ``scheduler._ATTEMPT_BUDGET``, and for the same reason: a container restart is a
legitimate reason to re-announce a still-broken world.

``degraded`` and ``unknown`` deliberately neither alert nor clear an alert. Alerting on
``degraded`` is alerting on the blip this design exists to ignore; *clearing* on it
would re-arm the alarm mid-outage and page again on the next failure.
"""

from __future__ import annotations

import time

from healthee.core.config import get_settings
from healthee.core.logging import get_logger
from healthee.core.notify import send_matrix
from healthee.insights import credits, transport_health

log = get_logger(__name__)

# How often the balance is re-read. Hourly: the endpoint is free, but the number moves
# slowly and an alert about it is not more useful for being one tick fresher. It also
# bounds the worst case for the signal the transport record cannot see — an account
# emptied entirely by api-container traffic is reported within the hour.
BALANCE_INTERVAL_S = 3600.0

_DOWN_ALERT = (
    "⛔ LLM transport DOWN — {failures} consecutive failures (last: {kind}{status}).\n"
    "Every AI surface is failing: the coach, the insight cards and the nightly chain. "
    "{advice}"
)
_WILL_NOT_HEAL = "This kind does not fix itself — check the OpenRouter balance and key."
_MAY_HEAL = "This kind is often transient; if it persists, check the provider's status."
_RECOVERED_ALERT = "✅ LLM transport recovered — a completion succeeded again."

_EXHAUSTED_ALERT = (
    "⛔ OpenRouter credits EXHAUSTED — ${remaining:.2f} left of ${total:.2f}.\n"
    "Every LLM call will 402 until the account is topped up, and /healthz will stay "
    "green throughout."
)
_LOW_ALERT = (
    "⚠️ OpenRouter credits LOW — ${remaining:.2f} left of ${total:.2f} "
    "(warning threshold ${threshold:.2f}). Top up before it reaches zero."
)
_UNKNOWN_ALERT = (
    "⚠️ OpenRouter balance check FAILED — {error}.\n"
    "The balance is UNKNOWN, which is not the same as fine: if the key is dead this is "
    "how it says so."
)
_BALANCE_OK_ALERT = "✅ OpenRouter balance is healthy again — ${remaining:.2f} remaining."


class LlmWatch:
    """Watches the two LLM health signals and pushes a message when one changes state.

    A class rather than module globals so the scheduler's state is injectable and a test
    can drive ``check`` at arbitrary instants (the same reasoning as ``Sweeper``).
    """

    def __init__(self, balance_interval_s: float = BALANCE_INTERVAL_S) -> None:
        self._balance_interval_s = balance_interval_s
        self._transport_alerted = False
        self._balance_alerted: str | None = None
        self._next_balance_probe: float | None = None

    def check(self, now_monotonic: float | None = None) -> None:
        """One pass over both signals. Called from the scheduler's tick; never raises.

        ``send_matrix`` is already no-raise by contract, and both readers below are
        pure/no-raise, so there is nothing here to swallow — the module deliberately has
        no ``try`` of its own, because a bare guard around a health check is how a health
        check starts failing quietly.
        """
        self._check_transport()
        self._check_balance(now_monotonic if now_monotonic is not None else time.monotonic())

    # ── the transport signal ──────────────────────────────────────────────────

    def _check_transport(self) -> None:
        health = transport_health.snapshot()
        if health.status == transport_health.DOWN and not self._transport_alerted:
            send_matrix(_down_text(health))
            self._transport_alerted = True
            return
        if health.status == transport_health.OK and self._transport_alerted:
            send_matrix(_RECOVERED_ALERT)
            self._transport_alerted = False

    # ── the balance signal ────────────────────────────────────────────────────

    def _check_balance(self, now: float) -> None:
        """Re-read the balance when due, and alert on a CHANGE of state.

        ``force=True`` bypasses the module's TTL cache: that cache exists to protect the
        provider from ``/readyz`` traffic, and a watcher on its own hourly cadence
        deciding whether to wake somebody up should be reading a number it knows is
        current rather than one up to five minutes old.
        """
        if self._next_balance_probe is not None and now < self._next_balance_probe:
            return
        self._next_balance_probe = now + self._balance_interval_s
        reading = credits.read_balance(force=True)
        if reading.status == credits.UNCONFIGURED:
            # No key ⇒ this deployment runs without the AI layer on purpose. Watching a
            # balance that does not exist would page an operator about a choice they made.
            return
        state = credits.balance_state(reading)
        if state == self._balance_alerted:
            return  # already said this; edge-triggered, not level-triggered
        if state == credits.BALANCE_OK:
            if self._balance_alerted is not None:
                send_matrix(_BALANCE_OK_ALERT.format(remaining=reading.remaining_usd or 0.0))
            self._balance_alerted = None
            return
        send_matrix(_balance_text(state, reading))
        self._balance_alerted = state


def _down_text(health: transport_health.TransportHealth) -> str:
    """The outage message — kind and status code only, never the provider's own words.

    A provider error body is free text we do not control, and a 400 for a bad model id
    contains that id verbatim. The model we run must not be discoverable
    (``insights.client.tier_of``), so the alert carries the diagnosis and not the quote.
    """
    status = f", HTTP {health.last_status_code}" if health.last_status_code else ""
    return _DOWN_ALERT.format(
        failures=health.consecutive_failures,
        kind=health.last_error_kind,
        status=status,
        advice=_WILL_NOT_HEAL if health.will_not_self_heal else _MAY_HEAL,
    )


def _balance_text(state: str, reading: credits.CreditsReading) -> str:
    """The balance message for a non-ok state. Dollar figures go to Telegram, not to HTTP.

    ``/readyz`` reports the STATE and no numbers: it is unauthenticated, and the
    account's economics are nobody else's business. This channel is the operator's own,
    so it carries the numbers that make the message actionable.
    """
    if state == credits.BALANCE_UNKNOWN:
        return _UNKNOWN_ALERT.format(error=reading.error or "no reason reported")
    template = _EXHAUSTED_ALERT if state == credits.BALANCE_EXHAUSTED else _LOW_ALERT
    return template.format(
        remaining=reading.remaining_usd or 0.0,
        total=reading.total_credits_usd or 0.0,
        threshold=get_settings().llm_low_balance_usd,
    )
