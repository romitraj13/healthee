"""In-process scheduler — ticks, and runs each owner's chain at THEIR local fire time.

Replaces the legacy ``scheduler.py`` cron loop with the structural change the rebuild
mandates: it does NOT ``subprocess.call(["healthee", …])``. Each owner's chain runs
in-process through ``chain.run_chain`` — supervised, so a failure is logged and
Telegram-notified rather than lost (standards §2: "no ``subprocess``/``Popen`` for
in-process work — jobs run as supervised functions").

## The model: tick + the per-owner per-day marker (Phase 6.4c, MULTI_USER.md §6)

The loop does not compute "the next fire instant" and sleep to it. It **ticks** on a
bounded interval and, on every tick, asks of each active owner: *is it past their
local fire time, and has their chain not already run for their local day?* If so, run
it. Idempotence is not the loop's job — it belongs to ``run_chain``'s per-owner
per-day dedup marker (the folded ``kv`` PK), which is what makes this safe by
construction: a missed tick, a container restart, a slow run, or an owner who signed
up after their own fire time all resolve correctly on the next tick.

**Why not a next-fire timer** (the trap this design exists to avoid): a
"compute the next matching instant, sleep, fire, repeat" loop returns *tomorrow* once
an instant has passed. Extended per-(owner) it fires owner A at their 10:30, then
recomputes and sees owner B's identical 10:30 has *just passed* — so B's next fire
becomes tomorrow, and B is silently skipped every single day. Any owner sharing a
fire instant with an earlier-sorted owner never runs. A tick asks a *question about
the present* instead of scheduling a future instant, so it cannot express that bug.

## One chain per owner per local day (the 10:30/10:45/11:00 stagger is gone)

The three staggered timers are collapsed into a single ``run_chain`` per owner per
local day: ``run_chain`` already sequences correlate → recs → warm → briefing *with* the
dependency (a correlate failure skips recs), so the stagger only re-implemented that
ordering with sleep(). It was not spreading LLM load — the steps were fired in the
same fixed order at fixed offsets, and the sweep runs all owners back-to-back within
one timer anyway, so N owners already produced N chains in a burst.

## The clock says EARLIEST; the data says GO (the arrival gate)

``DAILY_FIRE`` (10:30) **in each owner's own timezone** is the earliest the chain may
start, never the instant it does. Reaching it is one of two conditions; the other is
``data_gate.day_data_arrived`` — **the night that woke today** is actually in the
database. There is no global fire zone (the old ``TZ``): an owner's
``app_user.timezone`` is the only zone in play for them.

The gate asks for the night rather than for any of the day's data, and the difference
is not academic: a local day starts at midnight, so an owner awake at 00:54 who opens
the app syncs an hour of post-midnight heart rate stamped with today's date. A gate
keyed on "any data" opens on that, and 10:30 then computes a day whose sleep has not
happened yet — the very defect below, wearing the fix as a hat. At ``SLEEP_FALLBACK``
(14:00) the gate widens to accept a day with samples and no session, so a strap-off
night costs a few hours' latency rather than the whole day's chain.

Until this gate existed, this docstring claimed 10:30 was chosen as "late morning,
AFTER a typical late wake + app push, so recs/briefing don't compute on last night's
not-yet-synced sleep". Nothing checked. The app's auto-sync fires only on a foreground
transition, so an owner who had not opened the app by 10:30 had handed us nothing since
yesterday — and the chain ran anyway, writing recommendations and a briefing about a
night no row described, then setting the per-day marker so the real sleep arriving at
noon got NO chain at all. A comment about a clock was standing in for a check, which is
the failure class this codebase keeps finding in itself; ``jobs/data_gate.py`` opens
with the evidence.

**The gate can stay shut all day, and that is an outcome, not a hang.** A day nothing
arrived for gets no chain and no briefing, because "not enough data" beats an
optimistic guess and a briefing about an unrecorded night is the guess. What it must
never be is *silent*: past ``QUIET_DAY_NOTICE`` the owner-day is reported to the
Telegram health surface once — once, not once per tick — so a strap that stopped
syncing surfaces as itself instead of as a product that quietly stopped speaking.

Two consequences worth naming rather than discovering:

* a sync landing at 23:55 local still runs that day's chain on the next tick — the loop
  asks its question every tick, so late is late, not lost;
* a sync landing after local midnight does NOT retro-run yesterday. The marker and the
  question are both per local day. Yesterday's briefing delivered tomorrow is not a
  briefing, and authoring a past day's analysis now is forbidden anyway
  (``docs/AS_OF_DAY.md`` section 6, and ``recs.generate_recs`` refuses the call).

Run it as its own container CMD::

    python -m healthee.jobs.scheduler
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID
from zoneinfo import ZoneInfo

from healthee.core.entitlement import warn_if_self_host_unlocked
from healthee.core.logging import configure_logging, get_logger
from healthee.core.notify import send_matrix
from healthee.core.tenancy import Tenant, active_users
from healthee.jobs import chain
from healthee.jobs.data_gate import day_data_arrived
from healthee.jobs.llm_watch import LlmWatch

log = get_logger(__name__)

_UTC = ZoneInfo("UTC")

# How often the loop wakes to ask "is anyone due?". Also the bound on how long a
# container stop signal can be blocked (the old loop chunked its sleep for exactly
# this reason; the tick interval subsumes that). It is the granularity of a fire, not
# a deadline: an owner's chain starts within one tick of their local fire time.
_TICK_INTERVAL_S = 300

# How many times ONE owner's chain may be attempted for ONE of their local days.
#
# The dedup marker is only set when NO step failed (deliberately — see
# `chain._nothing_failed`), so a *failing* chain stays unmarked and would otherwise be
# retried on every tick for the rest of that owner's day: ~150 identical Telegram
# alerts, plus a re-sent briefing each time (briefing runs even when a step fails).
# That is alert fatigue, which silently costs the health surface its meaning. This
# budget keeps the retry — a transient DB blip at 10:30 must not cost an owner their
# day — while bounding it. It is NOT idempotence (the marker is); it is a retry
# budget, and it deliberately lives in memory: a container restart is itself a good
# reason to try again.
_ATTEMPT_BUDGET = 3

# Statuses the per-tick sweep does NOT log. All three mean "nothing happened, and that
# is correct": the owner's fire time has not arrived, their day is already done, or the
# gate is still shut. Each recurs on every one of a day's ~150 ticks, and a log line per
# tick would bury the statuses that do mean something. "no data yet" is not silent
# overall — `_report_quiet_day` raises it to the health surface once, at
# `QUIET_DAY_NOTICE`, which is the difference between quiet and hidden.
_QUIET_STATUSES = frozenset({"waiting", "deduped", "no data yet"})


@dataclass(frozen=True)
class FireTime:
    """The local wall-clock time an owner's daily chain runs at."""

    hour: int
    minute: int


DAILY_FIRE = FireTime(hour=10, minute=30)

# When the gate widens from "the night has arrived" to "anything from the day has".
#
# Before this hour the chain waits for the night that woke today, because that is what
# the morning's analysis is about. After it, a day with samples but no session is
# allowed through — otherwise an owner who took the strap off for one night would get
# no chain at all, and "you recorded no sleep last night" is a true and useful thing
# for a day to say.
#
# The hour is a JUDGEMENT and is written here rather than in `data_gate` so that it
# sits beside the other fire times and so that module stays free of clocks: by early
# afternoon, a night that was going to be handed over has been. Its cost is that a
# strap-off day's chain lands at 14:00 instead of 10:30 — a few hours' latency on the
# days with no night, in exchange for not fabricating on the days that have one.
SLEEP_FALLBACK = FireTime(hour=14, minute=0)

# When a day on which NOTHING has arrived is reported to the health surface. Late
# enough that an owner who syncs in the evening has long since fired and been marked
# done, so reaching this hour with the gate still shut means the day really is empty
# rather than merely late. It is a NOTICE, not a deadline: it fires no chain, and the
# gate stays open for the rest of the local day.
QUIET_DAY_NOTICE = FireTime(hour=22, minute=0)


def local_now(tenant: Tenant, now_utc: datetime) -> datetime:
    """``now_utc`` as wall-clock time in the owner's own zone."""
    return now_utc.astimezone(ZoneInfo(tenant.tz))


def is_due(local: datetime, fire: FireTime = DAILY_FIRE) -> bool:
    """True once the owner's local wall clock has reached their fire time.

    At/after — never "equals". An equality test would need the tick to land on the
    exact minute, so a tick landing at 10:31 would skip the owner for the whole day;
    "past it" plus the dedup marker is what makes a late tick self-heal instead.
    """
    return (local.hour, local.minute) >= (fire.hour, fire.minute)


class Sweeper:
    """The per-tick sweep over active owners, holding the in-memory retry budget.

    A class rather than module globals so the loop's state is injectable and a test
    can drive ``tick`` at arbitrary instants without patching a global (standards §2:
    no import-time side effects, everything testable).
    """

    def __init__(
        self,
        fire: FireTime = DAILY_FIRE,
        budget: int = _ATTEMPT_BUDGET,
        *,
        gate: Callable[[UUID, date, str, bool], bool] | None = None,
        sleep_fallback: FireTime = SLEEP_FALLBACK,
        quiet_notice: FireTime = QUIET_DAY_NOTICE,
    ) -> None:
        self._fire = fire
        self._budget = budget
        # Injected for the same reason the budget is: the loop's contract is "fires
        # only once the day's data is here", and a test that had to seed a hypertable
        # to assert it would be testing the gate's SQL instead of the loop's rule.
        # `tests/jobs/test_data_gate.py` tests the SQL, against a real database.
        #
        # A `None` sentinel resolved HERE rather than a default argument of
        # `day_data_arrived`, matching `generate_recs(client=None)`: a default argument
        # is bound once at def time, so it would capture the real gate permanently and
        # every one of the loop's existing tests would open a database connection to
        # assert something about a clock.
        self._gate = gate if gate is not None else day_data_arrived
        self._sleep_fallback = sleep_fallback
        self._quiet_notice = quiet_notice
        self._attempts: dict[tuple[UUID, date], int] = {}
        self._quiet_reported: set[tuple[UUID, date]] = set()

    def tick(self, now_utc: datetime | None = None) -> None:
        """Run the chain for every owner whose local fire time has passed today.

        The ``active_users()`` lookup is deliberately OUTSIDE the per-owner boundary:
        if the owner list itself can't be read there is no sweep to continue, so that
        error propagates rather than being swallowed into a silently sweep-less night.
        """
        now_utc = now_utc or datetime.now(tz=_UTC)
        tenants = active_users()
        seen: set[tuple[UUID, date]] = set()
        for tenant in tenants:
            status, day = self._run_for(tenant, now_utc)
            if day is not None:
                seen.add((tenant.id, day))
            if status not in _QUIET_STATUSES:
                log.info("chain for owner %s → %s", tenant.id, status)
        self._prune(seen)

    def _run_for(self, tenant: Tenant, now_utc: datetime) -> tuple[str, date | None]:
        """One owner's tick, isolating THIS owner's failure from the rest of the sweep.

        ``run_chain`` supervises each step's body, but not everything can fail inside
        that boundary: resolving the owner's local day from ``tenant.tz`` happens
        first, so one bad ``app_user.timezone`` row would otherwise abort the sweep for
        every later owner — one user's bad data silently costing everyone else their
        nightly chain. The failure is logged with the owner's id and reported to the
        Telegram health surface, never swallowed (standards §1); it just doesn't stop
        the sweep.
        """
        day: date | None = None
        try:
            local = local_now(tenant, now_utc)
            day = local.date()
            if not is_due(local, self._fire):
                return "waiting", day
            # Before `SLEEP_FALLBACK` only the night opens the gate; after it, any of
            # the day's data does. See that constant for why the widening is timed.
            without_night = is_due(local, self._sleep_fallback)
            if not self._gate(tenant.id, day, tenant.tz, without_night):
                self._report_quiet_day(tenant, local, day)
                return "no data yet", day
            return self._attempt(tenant, day), day
        except Exception as exc:  # supervised: report, then continue the sweep
            log.exception("chain failed for owner %s", tenant.id)
            send_matrix(f"chain failed for owner {tenant.id}: {exc}")
            if day is not None:
                self._spend(tenant, day)
            return "failed", day

    def _attempt(self, tenant: Tenant, day: date) -> str:
        """Run this owner's chain for ``day``, unless already done or out of budget."""
        if self._attempts.get((tenant.id, day), 0) >= self._budget:
            return "exhausted"
        result = chain.run_chain(tenant.id, tenant.tz, day)
        if result.deduped:
            return "deduped"  # the marker says this day already ran — costs no budget
        self._spend(tenant, day)
        failed = [step.name for step in result.steps if step.status == "failed"]
        return f"ran (failed steps: {', '.join(failed)})" if failed else "ran"

    def _spend(self, tenant: Tenant, day: date) -> None:
        """Charge one attempt against this owner-day; announce the last one."""
        spent = self._attempts.get((tenant.id, day), 0) + 1
        self._attempts[(tenant.id, day)] = spent
        if spent >= self._budget:
            log.warning(
                "chain budget spent for owner %s on %s — no more retries today", tenant.id, day
            )
            send_matrix(
                f"chain for owner {tenant.id} exhausted its {self._budget} attempts "
                f"for {day} — no further retries until tomorrow"
            )

    def _report_quiet_day(self, tenant: Tenant, local: datetime, day: date) -> None:
        """Once past ``QUIET_DAY_NOTICE`` with nothing arrived, say so — exactly once.

        A gate that can stay shut is only honest if the shut state is visible. Without
        this the difference between "the owner's strap has not synced in three days"
        and "everything is fine" is that one of them produces no Telegram traffic —
        and so does the other.

        The once-ness is the point and it is why this keeps a set rather than
        re-deriving the condition: the sweep asks this question on every tick after
        22:00, which is ~24 times, and 24 identical alerts is the alert fatigue
        `_ATTEMPT_BUDGET` exists to prevent, arriving through a different door.
        """
        if not is_due(local, self._quiet_notice):
            return
        key = (tenant.id, day)
        if key in self._quiet_reported:
            return
        self._quiet_reported.add(key)
        log.warning("no data has arrived for owner %s on %s — no chain run", tenant.id, day)
        send_matrix(
            f"no data has arrived for owner {tenant.id} on {day} — their daily chain "
            f"has not run. Nothing measured on that day has reached the server."
        )

    def _prune(self, seen: set[tuple[UUID, date]]) -> None:
        """Drop budget and notice entries for days/owners this tick no longer sees.

        Without it both collections grow one entry per owner per day for the process's
        life. They are pruned TOGETHER against the same `seen` set deliberately: an
        owner-day that survived in one and not the other would be a day that could be
        re-alerted after its budget was forgotten, or budgeted after its alert was.
        """
        self._attempts = {key: n for key, n in self._attempts.items() if key in seen}
        self._quiet_reported = {key for key in self._quiet_reported if key in seen}


def main() -> None:
    """The tick loop. Blocks forever."""
    configure_logging()
    warn_if_self_host_unlocked()
    log.info(
        "healthee scheduler starting (pid=%d, earliest fire=%02d:%02d in each owner's own "
        "timezone AND ONLY once their day's data has arrived, quiet-day notice=%02d:%02d, "
        "tick=%ds)",
        os.getpid(),
        DAILY_FIRE.hour,
        DAILY_FIRE.minute,
        QUIET_DAY_NOTICE.hour,
        QUIET_DAY_NOTICE.minute,
        _TICK_INTERVAL_S,
    )
    sweeper = Sweeper()
    # The LLM health watch rides this loop rather than owning a timer of its own: it is
    # cheap, it needs no tenant, and the loop is already the process that owns the
    # Telegram health surface. It runs AFTER the sweep so that the tick's own chain
    # failures are already in the transport record when it reads it — checking first
    # would report every outage one full tick late.
    watch = LlmWatch()
    while True:
        sweeper.tick()
        watch.check()
        time.sleep(_TICK_INTERVAL_S)


if __name__ == "__main__":
    main()
