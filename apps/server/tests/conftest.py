"""Shared test fixtures.

Unit tests set a minimal valid environment via the `env` fixture. Integration
tests depend on `db` (or `db_env`), which auto-skips when no TimescaleDB is
reachable — so `pytest` is green on a laptop with no database and exercises the
real DB in CI, where the service container sets POSTGRES_*.

## Every run gets its OWN database AND its own role (#119)

`app_role_pool` creates both, named after a per-process id. A private database alone
is **not** enough — the role is a cluster-level object, so two suites against one
PostgreSQL instance meet it even from different databases, and they corrupt each
other's run in ways that surface as failures in unrelated code. `tests/_isolation.py`
carries the full account; read it before changing anything session-scoped here.

## The whole suite runs as the LEAST-PRIVILEGE app role (6.5b-2)

`app_role_pool` below is autouse and session-scoped: it provisions the real
least-privilege role and points the **app pool** at it for the entire run, while
migrations / `TRUNCATE` / re-keying keep using the admin (`admin_connection`).

This is not tidiness — it is the acceptance bar for RLS. `POSTGRES_APP_*` is unset
in dev and CI, so without this fixture the app pool falls back to the admin
**superuser**, which `rolbypassrls` — every `0008` policy would be inert, every RLS
test would pass while protecting nothing, and any tenant query left on plain
`transaction()` would silently keep working here and return zero rows in prod. That
is the exact illusion 6.5b-1 exists to prevent, and it has bitten this project
before. `tests/db/test_rls.py::test_the_app_pool_is_never_privileged` asserts the
premise so a future env change cannot quietly undo it.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import psycopg
import pytest
from psycopg import sql
from tests._isolation import TEST_APP_ROLE, TEST_DATABASE, create_database, drop_database

from healthee.core import db as db_module
from healthee.core import logging as logging_module
from healthee.core.config import get_settings
from healthee.core.tenancy import SENTINEL_USER_ID
from healthee.db import migrate, provision_app_role
from healthee.insights import credits, transport_health

# How long a seeded entitlement runs for. Absurdly long on purpose: `is_premium`
# requires `now < current_period_end`, so a short term would turn the suite into a time
# bomb that starts 402-ing on some future afternoon.
_ENTITLEMENT_YEARS = 50


def entitle(user_id: UUID, *, premium: bool = True) -> None:
    """Give (or take) an owner's premium entitlement — the ONE way a test does it (6.6a).

    Written on the ADMIN connection, and that is the fixture proving the design rather
    than working around it: `provision_app_role` REVOKEs every write privilege on
    `subscription` from the role the app connects as, precisely so a request path cannot
    mint entitlement (MULTI_USER.md §12.7). A seeder that could INSERT here on the app
    pool would mean the revoke had not happened.

    Every bed that exercises a gated surface calls this EXPLICITLY. It is deliberately
    not autouse: `subscription` survives the truncate lists of the other seeds, so an
    implicit grant would leave one test file passing only because an earlier one had
    run — which is exactly what happened while 6.6a was being built, and is invisible
    until someone runs a single file.
    """
    ends = datetime.now(tz=UTC) + timedelta(days=365 * _ENTITLEMENT_YEARS)
    status, period_end = ("active", ends) if premium else ("canceled", datetime.now(tz=UTC))
    with db_module.admin_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO subscription (user_id, status, plan, current_period_end, granted_by) "
            "VALUES (%s, %s, 'test', %s, 'tests.conftest') "
            "ON CONFLICT (user_id) DO UPDATE SET status = EXCLUDED.status, "
            "  plan = EXCLUDED.plan, current_period_end = EXCLUDED.current_period_end",
            (user_id, status, period_end),
        )


# Vars with defaults that the defaults tests assert on — cleared so the unit
# environment is hermetic (a dev shell that exports e.g. POSTGRES_PORT must not
# leak into a test asserting the default value).
_DEFAULTED_ENV_VARS = (
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_APP_USER",
    "POSTGRES_APP_PASSWORD",
    "LOG_LEVEL",
    "API_HOST",
    "API_PORT",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "OPENROUTER_API_KEY",
    "DEFAULT_MODEL",
    "COACH_MODEL",
    "LLM_TIMEOUT_S",
    "LLM_MAX_RETRIES",
    "MATRIX_HOMESERVER",
    "MATRIX_ACCESS_TOKEN",
    "MATRIX_ROOM_ID",
    "SUPABASE_JWT_SECRET",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_PROJECT_REF",
    "SUPABASE_JWT_AUD",
    "SIGNUPS_OPEN",
    "SIGNUP_ALLOWLIST",
    "SELF_HOST_UNLOCKED",
    "UPGRADE_URL",
    "LLM_LOW_BALANCE_USD",
)


@pytest.fixture(scope="session", autouse=True)
def _allow_the_bootstrap_to_build_settings() -> Iterator[None]:
    """Let the suite's own bootstrap construct `Settings` before it has an app role.

    `core.config` refuses to boot with blank `POSTGRES_APP_*` unless
    `ALLOW_ADMIN_DB_FALLBACK=true` asks for the transitional state (the B1 fix). The
    suite's bootstrap is exactly that state and only that state: `_db_reachable()` and
    `app_role_pool` both build `Settings` to reach `admin_db_url` BEFORE the throwaway
    role exists, and `app_role_pool` then sets `POSTGRES_APP_*` to the real role for
    every test that follows.

    ⚠ Without this the failure is silent and mis-reads as a code break:
    `_db_reachable()` catches the refusal, returns False, and the whole integration
    suite auto-skips — the exact "looks exactly like a code break and isn't" shape
    CONTRIBUTING.md warns about for the `POSTGRES_*` vars themselves.

    Session-scoped and autouse because `os.environ` is process-wide. It does NOT weaken
    what is under test: `tests/db/test_rls.py::test_the_app_pool_is_never_privileged`
    still asserts from `pg_roles` that the pool is on the least-privilege role, and
    `tests/test_config.py` constructs `Settings` without this var to exercise the
    refusal itself.
    """
    previous = os.environ.get("ALLOW_ADMIN_DB_FALLBACK")
    os.environ["ALLOW_ADMIN_DB_FALLBACK"] = "true"
    get_settings.cache_clear()
    yield
    if previous is None:
        os.environ.pop("ALLOW_ADMIN_DB_FALLBACK", None)
    else:
        os.environ["ALLOW_ADMIN_DB_FALLBACK"] = previous
    get_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def _geodata_caches_are_hermetic(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point the SRTM and basemap caches at a throwaway directory for the run.

    Both default to `/var/cache/healthee/...` — the container's volume, which on a
    developer's machine is not writable and in CI would be a directory the suite
    silently populates and never cleans. Neither cache holds anything a test
    asserts on: `tests/derive/_gps_seed.py` uses ocean coordinates precisely so
    the DEM lookup misses, and the tile tests stub the fetch. Session-scoped
    because `os.environ` is process-wide and this only has to be true once.
    """
    root = tmp_path_factory.mktemp("geocache")
    previous = {name: os.environ.get(name) for name in ("SRTM_CACHE_DIR", "MAP_TILE_CACHE_DIR")}
    os.environ["SRTM_CACHE_DIR"] = str(root / "srtm")
    os.environ["MAP_TILE_CACHE_DIR"] = str(root / "tiles")
    get_settings.cache_clear()
    yield
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clean_llm_health() -> Iterator[None]:
    """Give every test a transport record and a balance cache with no history.

    Both are process-wide singletons by design — the whole point of
    `insights.transport_health` is that one record accumulates across every call the
    process makes. Under pytest that means one test's stubbed 402 would still be in the
    streak when the next test reads it, and a test asserting "unknown, nothing has
    happened yet" would pass or fail depending on alphabetical file order. Autouse,
    because the tests most likely to be polluted are the ones that never mention either
    module.
    """
    transport_health.reset()
    credits.reset_cache()
    yield
    transport_health.reset()
    credits.reset_cache()


@pytest.fixture(autouse=True)
def _keep_caplog_capturing() -> Iterator[None]:
    """Stop `configure_logging()` from silently deleting pytest's capture handler (#119).

    `core.logging.configure_logging` does `root.handlers.clear()` on its FIRST call in a
    process. pytest's `caplog` works by adding a handler to that same root logger, so any
    test that reaches an ops `main()`, `create_app()` or `scheduler.main()` before reading
    `caplog` had its capture removed mid-test — and only sometimes, because the second and
    later calls are no-ops. That makes a caplog assertion pass or fail on **test order**,
    and an empty `caplog.text` makes a NEGATIVE assertion ("the secret is not in the log")
    pass while proving nothing at all. A vacuous assertion is worse than no assertion: it
    reports a safety it does not provide.

    Fixed once, here, rather than test by test: the flag is pinned so `configure_logging`
    is a no-op for the duration of every test, and restored afterwards. What that gives up
    is nothing — the handler/level/httpx-pin wiring is what `tests/test_logging.py` exists
    for, and it clears the flag itself to exercise the real thing.

    `tests/db/test_stale_derived.py::test_a_plain_run_says_so_in_the_log` is this
    fixture's canary: it drives the operator's real `main()` and asserts on the log, so
    removing the pin below turns it red.
    """
    was_configured = logging_module._configured
    logging_module._configured = True
    yield
    logging_module._configured = was_configured


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Minimal, hermetic environment for constructing Settings in a unit test.

    Only the two effectively-required vars are set; every var that has a default
    is cleared so a test of the defaults sees the code's defaults, not whatever
    the ambient shell exported.

    Clearing the env vars is NOT sufficient on its own: `Settings.model_config`
    sets `env_file=".env"`, resolved against the CWD, so a developer running the
    suite from `apps/server/` (where a real `.env` lives) had its values injected
    into a "defaults" test no matter what `delenv` did — and the file wins for any
    var the fixture cleared. That failed only on a machine with a `.env`, never in
    CI and never in a worktree (`.env` is gitignored, so it isn't copied), which is
    exactly the shape of a phantom: it cost an agent a chase and could not be
    reproduced. Chdir'ing to an empty tmp dir makes the fixture hermetic against
    the file too, which is what its name already promised.
    """
    monkeypatch.chdir(tmp_path)
    for var in _DEFAULTED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "unit-test-pw")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _db_reachable() -> bool:
    """True if the configured DB accepts a trivial query within a short timeout."""
    try:
        settings = get_settings()
    except Exception:  # Settings can't even be built (e.g. no password) → skip
        return False
    try:
        with psycopg.connect(settings.admin_db_url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
    except Exception:
        return False
    return True


def drop_test_role(role: str) -> None:
    """Remove the role and every privilege granted to it, so the DB is left as found.

    `DROP OWNED BY` is what revokes the grants and the `ALTER DEFAULT PRIVILEGES`
    entries; a plain `DROP ROLE` would fail while they exist.
    """
    with db_module.admin_connection() as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
        cur.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


@pytest.fixture(scope="session", autouse=True)
def app_role_pool() -> Iterator[str | None]:
    """Build this run's private database + least-privilege role; point the pool at both.

    Yields the role name, or None when no DB is reachable (unit-only runs stay green
    on a laptop with no database — the integration tests skip themselves anyway).

    **Both halves are per-run and both are load-bearing.** The database is private
    because the seed fixtures `TRUNCATE` shared tables; the ROLE is private because a
    role is cluster-scoped, so a private database does not cover it — two suites
    against one PostgreSQL instance would still re-key and then drop each other's
    login. `tests/_isolation.py` has the measured account of that failure.

    Migrations run against the NEW database, and before `provision_app_role`, because
    provision grants table-by-table from an explicit list: on an empty database those
    GRANTs have nothing to grant on.

    A fresh random password per run — a fixed one in git is a credential in git even
    on a throwaway database. Torn down with `DROP OWNED BY` + `DROP ROLE` and then
    `DROP DATABASE`, so neither the local DB nor CI's accumulates either across runs.
    """
    if not _db_reachable():
        yield None
        return
    admin_url = get_settings().admin_db_url  # the CONFIGURED database — our bootstrap
    monkeypatch = pytest.MonkeyPatch()
    create_database(admin_url)
    monkeypatch.setenv("POSTGRES_DB", TEST_DATABASE)
    monkeypatch.setenv("POSTGRES_APP_USER", TEST_APP_ROLE)
    monkeypatch.setenv("POSTGRES_APP_PASSWORD", secrets.token_urlsafe(24))
    get_settings.cache_clear()
    db_module.close_pool()  # so the next get_pool() connects as the app role
    migrate.apply_migrations()
    provision_app_role.provision()
    yield TEST_APP_ROLE
    leaked = owner_ids() - {SENTINEL_USER_ID}
    db_module.close_pool()
    drop_test_role(TEST_APP_ROLE)
    monkeypatch.undo()
    get_settings.cache_clear()
    drop_database(admin_url)
    _refuse_leaked_owners(leaked)


def _refuse_leaked_owners(leaked: set[UUID]) -> None:
    """Fail the session when it leaves `app_user` rows behind (#119).

    The private database means a leak no longer poisons the NEXT run, so nothing would
    ever complain — and the leak itself is still a bug: `--user`-less ops tooling
    (`db/rederive.py`) walks every active owner, so a suite that invents owners and does
    not remove them makes those runs slower and less deterministic the longer a
    developer's box has been in use. It reached ~1,288 rows before anyone noticed,
    because nothing was watching.

    Raised at session teardown rather than asserted in a test, because "the suite left
    the database as it found it" is a property of the whole run and no single test can
    see it. A run that fails for other reasons may trip this too — the message names the
    ids, which is the fastest route to the fixture that forgot its teardown.
    """
    if leaked:
        raise RuntimeError(
            f"the suite leaked {len(leaked)} app_user row(s): {sorted(map(str, leaked))}. "
            "Every test that provisions an owner must remove it — request the "
            "`owner_sweep` fixture, or DELETE the row in the fixture's own teardown."
        )


@pytest.fixture(scope="session")
def db_env(app_role_pool: str | None) -> None:
    """Skip the test unless a TimescaleDB is reachable from the current env."""
    if app_role_pool is None:
        pytest.skip("no reachable TimescaleDB — integration test skipped")


@pytest.fixture
def db(db_env: None) -> Iterator[None]:  # noqa: ARG001 — gates on reachability
    """Fresh DB pool for an integration test; closes it afterwards so the next
    test rebuilds against current config."""
    db_module.close_pool()
    yield
    db_module.close_pool()


def owner_ids() -> set[UUID]:
    """Every `app_user` id, on the ADMIN connection — the question spans owners."""
    with db_module.admin_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM app_user")
        return {row[0] for row in cur.fetchall()}


@pytest.fixture
def owner_sweep(db_env: None) -> Iterator[None]:  # noqa: ARG001 — gates on reachability
    """Delete every `app_user` row the test provisions. The teardown that was missing (#119).

    By DIFFERENCE rather than by a list of ids the test remembers to name, because a list
    is exactly what was being forgotten — and because two of the three ways an owner
    appears are invisible at the call site: a JIT provision on the first authenticated
    request (`core.supabase_auth`), and `tests/contracts/seed_owner_b`. Only the plain
    `INSERT INTO app_user` announces itself.

    Requested via `pytest.mark.usefixtures` at the top of the leaking modules, which also
    fixes the ordering: `usefixtures` marks enter the fixture closure ahead of the test's
    own arguments, so the snapshot is taken before a bed fixture seeds its second owner.

    A run whose sentinel row is missing is left ALONE. `tests/db/test_claim_sentinel.py`
    re-keys that row to another id, so a sweep during a failed claim would read the
    re-keyed sentinel as a new owner and delete it — cascading the whole tenant tree away
    to tidy up.
    """
    before = owner_ids()
    yield
    after = owner_ids()
    if SENTINEL_USER_ID not in after:
        return  # mid-re-key (claim_sentinel) — not ours to tidy
    leaked = after - before
    if not leaked:
        return
    with db_module.admin_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM app_user WHERE id = ANY(%s)", (list(leaked),))
