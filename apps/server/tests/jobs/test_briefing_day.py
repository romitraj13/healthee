"""C2 · `send_briefing` may only stamp a message with the day its content answered for.

``day`` reached exactly one thing — the ``healthee briefing — {day}`` header
``_compose`` prints — while every input is unconditionally today's: ``cached_line`` is
keyed on the owner's own today and takes no day, and ``morning.generate_briefing``
anchors on ``user_today(tz)`` throughout.

So ``run_chain(user_id, tz, day=<past>, force=True)`` sent **today's judgement under an
older date**, over the one channel where the reader has no other date beside it to check
against. That is the stale-as-current lie ``docs/AS_OF_DAY.md`` section 3 names, and
``jobs/recs.py`` has refused its own version of it since B2 —
``tests/jobs/test_recs_day.py`` is this file's sibling and its argument applies verbatim.

No database, no Telegram and no model: the refusal is the first thing ``send_briefing``
does, which is itself the property worth pinning. A check that runs after the message
has gone out is a different check.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from healthee.core.tenancy import SENTINEL_TZ, SENTINEL_USER_ID, user_today
from healthee.jobs import briefing


class _NeverSent:
    """Fails if anything is generated or sent — the refusal must precede every cost."""

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201, ARG002
        raise AssertionError("a refused day must not reach the model")


def test_a_past_day_is_refused_before_anything_is_generated_or_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE finding. Today's judgement must not be sent under an older date."""
    monkeypatch.setattr(
        briefing, "send_matrix", lambda *_a, **_k: pytest.fail("a refused day was sent")
    )
    yesterday = user_today(SENTINEL_TZ) - timedelta(days=1)

    with pytest.raises(ValueError) as raised:
        briefing.send_briefing(SENTINEL_USER_ID, SENTINEL_TZ, yesterday, client=_NeverSent())

    message = str(raised.value)
    assert yesterday.isoformat() in message
    assert "AS_OF_DAY" in message, "the refusal names the rule it is enforcing"


def test_a_future_day_is_refused_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mirror. A briefing dated tomorrow is a claim about a day that has not been."""
    monkeypatch.setattr(
        briefing, "send_matrix", lambda *_a, **_k: pytest.fail("a refused day was sent")
    )
    with pytest.raises(ValueError):
        briefing.send_briefing(
            SENTINEL_USER_ID,
            SENTINEL_TZ,
            user_today(SENTINEL_TZ) + timedelta(days=1),
            client=_NeverSent(),
        )


def test_the_owners_own_today_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must not be a blanket refusal: the nightly chain's call still runs.

    Stubbed at the first thing past the comparison — the warmed-line lookup — so this
    asserts the day check passed and nothing else.
    """
    reached: list[str] = []

    def _cached(user_id, tz, key):  # noqa: ANN001, ANN202, ARG001
        reached.append("cached_line")
        raise RuntimeError("stop here")

    monkeypatch.setattr(briefing.coaching, "cached_line", _cached)

    with pytest.raises(RuntimeError, match="stop here"):
        briefing.send_briefing(SENTINEL_USER_ID, SENTINEL_TZ, user_today(SENTINEL_TZ))
    assert reached == ["cached_line"]


def test_no_day_at_all_is_still_the_owners_today(monkeypatch: pytest.MonkeyPatch) -> None:
    """The chain's ordinary call passes no day, and must keep working unchanged."""
    reached: list[str] = []
    monkeypatch.setattr(
        briefing.coaching,
        "cached_line",
        lambda *_a: reached.append("cached_line") or "warmed body",
    )
    sent: list[str] = []
    monkeypatch.setattr(briefing, "send_matrix", lambda text: sent.append(text) or True)

    result = briefing.send_briefing(SENTINEL_USER_ID, SENTINEL_TZ)

    today = user_today(SENTINEL_TZ)
    assert result["day"] == today.isoformat()
    assert sent and sent[0].startswith(f"healthee briefing — {today.isoformat()}")
