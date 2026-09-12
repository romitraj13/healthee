"""Fixtures shared by the two scheduler suites.

A ``conftest.py`` rather than importing the fixtures from ``_sweeper_bed``: pytest
discovers fixtures by name from here, whereas importing one puts a second binding of
the same name in the importing module — which is a redefinition to the linter and a
fixture that shadows itself to a reader. The non-fixture bed (the owners, the fake
chain, the fake gate) stays in ``_sweeper_bed`` where it can be imported explicitly.

Both are opt-in: a test in this directory gets them only by naming them.
"""

from __future__ import annotations

import pytest
from tests.jobs._sweeper_bed import FakeChain

from healthee.jobs import chain, scheduler


@pytest.fixture
def fake_chain(monkeypatch: pytest.MonkeyPatch) -> FakeChain:
    """``chain.run_chain`` replaced by the counting stub, dedup marker and all."""
    fake = FakeChain()
    monkeypatch.setattr(chain, "run_chain", fake.run_chain)
    return fake


@pytest.fixture
def notices(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every Telegram the sweep sends, in order — the health surface as a list."""
    sent: list[str] = []
    monkeypatch.setattr(scheduler, "send_matrix", lambda text, **_: sent.append(text) or True)
    return sent
