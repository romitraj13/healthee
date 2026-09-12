"""The alert actually fires — and does not fire on a blip, and does not fire twice.

On 2026-08-01 every LLM call 402'd for hours behind a green `/healthz` and the only
reason anyone found out was that an unrelated eval run died. An alert nobody has proved
fires is not an alert, so this file drives the watcher over the real record and asserts
on the messages it pushes.

Four properties, each of which the incident or the fix depends on:

1. **N consecutive failures page; one does not.** A channel that fires on every blip
   gets muted, and it is shared with the chain-failure messages.
2. **Once per outage, plus a recovery.** Level-triggered alerting at a 300 s tick would
   be ~12 identical messages an hour, which is the same muting by a slower route.
3. **The balance warns BEFORE zero**, on a probe that spends no tokens.
4. **A failed balance probe is reported as unknown**, never as fine.

No network: `send_matrix` is replaced and the credits probe is stubbed.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from healthee.core.config import get_settings
from healthee.insights import credits
from healthee.insights import transport_health as th
from healthee.jobs import llm_watch


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Capture what would have gone to Telegram, and give each test a clean record."""
    messages: list[str] = []
    monkeypatch.setattr(llm_watch, "send_matrix", lambda text, **_kw: bool(messages.append(text)))
    monkeypatch.setenv("POSTGRES_PASSWORD", "unit-test-pw")
    monkeypatch.delenv("LLM_LOW_BALANCE_USD", raising=False)
    get_settings.cache_clear()
    th.reset()
    yield messages
    th.reset()
    get_settings.cache_clear()


def _reading(remaining: float, total: float = 200.0) -> credits.CreditsReading:
    return credits.CreditsReading(
        status=credits.OK,
        checked_at=datetime.now(tz=UTC),
        remaining_usd=remaining,
        total_credits_usd=total,
        total_usage_usd=total - remaining,
    )


def _balance(monkeypatch: pytest.MonkeyPatch, reading: credits.CreditsReading) -> list[bool]:
    """Stub the probe; return the list of `force` flags it was called with."""
    probes: list[bool] = []

    def fake(*, force: bool = False) -> credits.CreditsReading:
        probes.append(force)
        return reading

    monkeypatch.setattr(llm_watch.credits, "read_balance", fake)
    return probes


@pytest.fixture
def healthy_balance(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """The balance is fine, so a test about the transport is only about the transport."""
    return _balance(monkeypatch, _reading(remaining=150.0))


# ── the transport outage ──────────────────────────────────────────────────────


def test_one_402_does_not_page(sent: list[str], healthy_balance: list[bool]) -> None:  # noqa: ARG001
    th.record_failure(_StatusError(402))
    llm_watch.LlmWatch().check(now_monotonic=0.0)
    assert sent == []


def test_repeated_402s_page_once_with_the_diagnosis(
    sent: list[str],
    healthy_balance: list[bool],  # noqa: ARG001
) -> None:
    watch = llm_watch.LlmWatch()
    for _ in range(th.OUTAGE_THRESHOLD):
        th.record_failure(_StatusError(402))
    watch.check(now_monotonic=0.0)
    assert len(sent) == 1
    assert "DOWN" in sent[0]
    assert "credit" in sent[0] and "402" in sent[0]
    # A 402 does not fix itself, and the message has to say which kind of wait this is.
    assert "balance" in sent[0].lower()


def test_a_standing_outage_is_not_repeated_every_tick(
    sent: list[str],
    healthy_balance: list[bool],  # noqa: ARG001
) -> None:
    """At a 300 s tick, level-triggered alerting is ~12 identical messages an hour."""
    watch = llm_watch.LlmWatch()
    for _ in range(th.OUTAGE_THRESHOLD):
        th.record_failure(_StatusError(402))
    for tick in range(10):
        watch.check(now_monotonic=float(tick))
    assert len(sent) == 1


def test_recovery_is_announced_and_re_arms_the_alarm(
    sent: list[str],
    healthy_balance: list[bool],  # noqa: ARG001
) -> None:
    watch = llm_watch.LlmWatch()
    for _ in range(th.OUTAGE_THRESHOLD):
        th.record_failure(_StatusError(402))
    watch.check(now_monotonic=0.0)
    th.record_success()
    watch.check(now_monotonic=1.0)
    assert len(sent) == 2
    assert "recovered" in sent[1]
    # …and a second outage after the recovery pages again.
    for _ in range(th.OUTAGE_THRESHOLD):
        th.record_failure(_StatusError(402))
    watch.check(now_monotonic=2.0)
    assert len(sent) == 3


def test_a_degraded_transport_neither_pages_nor_clears_a_standing_alert(
    sent: list[str],
    healthy_balance: list[bool],  # noqa: ARG001
) -> None:
    """Clearing on `degraded` would re-arm mid-outage and page again on the next failure."""
    watch = llm_watch.LlmWatch()
    for _ in range(th.OUTAGE_THRESHOLD):
        th.record_failure(_StatusError(402))
    watch.check(now_monotonic=0.0)
    th.record_success()  # one call got through…
    th.record_failure(_StatusError(402))  # …and the next did not: degraded
    watch.check(now_monotonic=1.0)
    assert len(sent) == 1, "a mid-outage blip re-announced the outage"


def test_a_transient_kind_is_described_differently_from_a_dead_account(
    sent: list[str],
    healthy_balance: list[bool],  # noqa: ARG001
) -> None:
    for _ in range(th.OUTAGE_THRESHOLD):
        th.record_failure(_StatusError(429))
    llm_watch.LlmWatch().check(now_monotonic=0.0)
    assert "transient" in sent[0]


def test_the_outage_message_carries_no_provider_text(
    sent: list[str],
    healthy_balance: list[bool],  # noqa: ARG001
) -> None:
    """The model id must not be discoverable, and a 400 body contains it verbatim."""

    class _BadModelError(Exception):
        def __init__(self) -> None:
            super().__init__("vendor-x/never-log-me-9000 is not a valid model ID")
            self.status_code = 400

    for _ in range(th.OUTAGE_THRESHOLD):
        th.record_failure(_BadModelError())
    llm_watch.LlmWatch().check(now_monotonic=0.0)
    assert "never-log-me" not in sent[0]


# ── the balance ───────────────────────────────────────────────────────────────


def test_a_low_balance_warns_before_zero(monkeypatch: pytest.MonkeyPatch, sent: list[str]) -> None:
    """The cheap version of never having this happen again."""
    _balance(monkeypatch, _reading(remaining=5.0))
    llm_watch.LlmWatch().check(now_monotonic=0.0)
    assert len(sent) == 1
    assert "LOW" in sent[0] and "5.00" in sent[0]


def test_an_exhausted_balance_is_a_different_message_from_a_low_one(
    monkeypatch: pytest.MonkeyPatch, sent: list[str]
) -> None:
    _balance(monkeypatch, _reading(remaining=0.0))
    llm_watch.LlmWatch().check(now_monotonic=0.0)
    assert "EXHAUSTED" in sent[0]
    assert "402" in sent[0]


def test_a_failed_balance_probe_says_unknown_rather_than_nothing(
    monkeypatch: pytest.MonkeyPatch, sent: list[str]
) -> None:
    """A monitoring feature that fails quiet converts "we don't know" into "we're fine"."""
    _balance(
        monkeypatch,
        credits.CreditsReading(
            status=credits.ERROR, checked_at=datetime.now(tz=UTC), error="HTTP 401"
        ),
    )
    llm_watch.LlmWatch().check(now_monotonic=0.0)
    assert len(sent) == 1
    assert "UNKNOWN" in sent[0] and "401" in sent[0]


def test_an_unconfigured_deployment_is_never_paged_about_a_balance(
    monkeypatch: pytest.MonkeyPatch, sent: list[str]
) -> None:
    """Running without the AI layer is a choice, not an incident."""
    _balance(
        monkeypatch,
        credits.CreditsReading(status=credits.UNCONFIGURED, checked_at=datetime.now(tz=UTC)),
    )
    llm_watch.LlmWatch().check(now_monotonic=0.0)
    assert sent == []


def test_the_balance_alert_is_edge_triggered_with_a_recovery(
    monkeypatch: pytest.MonkeyPatch, sent: list[str]
) -> None:
    low = _reading(remaining=5.0)
    healthy = _reading(remaining=150.0)
    current = {"reading": low}
    monkeypatch.setattr(llm_watch.credits, "read_balance", lambda **_kw: current["reading"])
    watch = llm_watch.LlmWatch(balance_interval_s=1.0)
    watch.check(now_monotonic=0.0)
    watch.check(now_monotonic=10.0)
    assert len(sent) == 1, "the low-balance warning repeated every probe"
    current["reading"] = healthy
    watch.check(now_monotonic=20.0)
    assert len(sent) == 2
    assert "healthy again" in sent[1]


def test_low_escalating_to_exhausted_is_announced_as_a_new_state(
    monkeypatch: pytest.MonkeyPatch, sent: list[str]
) -> None:
    """Getting worse is news; staying the same is not."""
    current = {"reading": _reading(remaining=5.0)}
    monkeypatch.setattr(llm_watch.credits, "read_balance", lambda **_kw: current["reading"])
    watch = llm_watch.LlmWatch(balance_interval_s=1.0)
    watch.check(now_monotonic=0.0)
    current["reading"] = _reading(remaining=0.0)
    watch.check(now_monotonic=10.0)
    assert len(sent) == 2
    assert "EXHAUSTED" in sent[1]


# ── the probe cadence ─────────────────────────────────────────────────────────


def test_the_balance_is_probed_on_the_first_check_then_on_its_own_cadence(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[str],  # noqa: ARG001
) -> None:
    """Hourly by default: the endpoint is free, but the number moves slowly."""
    probes = _balance(monkeypatch, _reading(remaining=150.0))
    watch = llm_watch.LlmWatch(balance_interval_s=3600.0)
    watch.check(now_monotonic=0.0)
    assert len(probes) == 1, "the first check never asked"
    for tick in range(1, 12):  # 11 more ticks inside the hour
        watch.check(now_monotonic=tick * 300.0)
    assert len(probes) == 1, "the watcher probed on every tick"
    watch.check(now_monotonic=3601.0)
    assert len(probes) == 2


def test_the_watcher_forces_a_fresh_reading_before_deciding_to_page(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[str],  # noqa: ARG001
) -> None:
    """The TTL cache exists to protect the provider from /readyz, not to age this decision."""
    probes = _balance(monkeypatch, _reading(remaining=150.0))
    llm_watch.LlmWatch().check(now_monotonic=0.0)
    assert probes == [True]
