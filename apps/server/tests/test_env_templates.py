"""The env templates and the compose file must stay a complete inventory of `Settings`.

This is the cheap check that would have prevented #56. Every field on
`core.config.Settings` defaults to something constructible, so a var missing from the
file a deployer copies produces an API that **starts, reports `/healthz` green, and
runs a dead subsystem**. Nothing in the deploy can see that: the templates are prose,
compose is YAML, and `Settings` is Python, so the three drift apart silently and the
only detector was someone noticing a 400 in a log.

Three invariants, each catching a different half of the drift:

1. **`infra/.env.example` declares every field, uncommented.** It is the file an
   operator copies to `infra/.env`; a commented line there is a var they will not set.
2. **`apps/server/.env.example` declares every field**, commented allowed — it is the
   local-dev inventory, and a var like `API_HOST` is documented rather than required.
3. **The compose `api` service passes every field through.** `LLM_TIMEOUT_S` and
   `LLM_MAX_RETRIES` were in `infra/.env.example` and *not* in compose, so setting
   them in prod's `.env` did nothing at all — a template promising something the
   container never received, which is #56's exact shape one layer down.

Plus the reverse direction: a key in a template that `Settings` does not read is
either a typo (`DEFAULT_MODLE=`) or a var nobody consumes. Both are caught by pinning
the deploy-only names explicitly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from healthee.core.config import Settings

# tests/test_env_templates.py → apps/server → apps → repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_INFRA_TEMPLATE = _REPO_ROOT / "infra" / ".env.example"
_DEV_TEMPLATE = _REPO_ROOT / "apps" / "server" / ".env.example"
_COMPOSE = _REPO_ROOT / "infra" / "docker" / "docker-compose.prod.yml"

# Vars in `infra/.env.example` that `Settings` deliberately does NOT read: they are
# consumed by the shell (deploy.sh, backup/pg_dump_backup.sh), never by the app. Named
# one by one rather than pattern-matched, so a typo'd app var cannot hide among them.
_DEPLOY_ONLY_VARS = frozenset(
    {
        "DEPLOY_BRANCH",  # infra/deploy.sh — the branch the VPS resets to
        "BACKUP_DIR",  # infra/backup/pg_dump_backup.sh — where dumps are written
        "BACKUP_RETENTION_DAYS",  # …and how long they are kept
        "OFFBOX_CMD",  # …and the optional off-box copy command
        "GOTRUE_DB_USER",  # compose only — the gotrue service's own Postgres user
        "GOTRUE_DB_PASSWORD",  # compose only — the gotrue service's own Postgres password
        "GOTRUE_DB_NAME",  # compose only — the gotrue service's own Postgres database
    }
)

# Vars the DEV template may declare and the infra one may not: they belong to tooling that
# runs on a developer's machine and must never reach the VPS (`infra/DEPLOY.md` §E2).
_DEV_ONLY_VARS = frozenset(
    {
        "EVAL_OPENROUTER_API_KEY",  # tests/grounding_eval — a paid arm's own small-limit key
    }
)

_ASSIGNMENT = re.compile(r"^(?P<commented>#\s*)?(?P<name>[A-Z][A-Z0-9_]*)\s*=")


def _settings_vars() -> set[str]:
    """Every env var name `Settings` reads, as the templates spell it."""
    return {name.upper() for name in Settings.model_fields}


def _declared(template: Path, *, include_commented: bool) -> set[str]:
    """Var names assigned in `template`, optionally counting commented-out lines."""
    found: set[str] = set()
    for line in template.read_text().splitlines():
        match = _ASSIGNMENT.match(line)
        if match and (include_commented or not match["commented"]):
            found.add(match["name"])
    return found


def _compose_env(service: str) -> set[str]:
    """The env var names a compose service passes into its container."""
    services = yaml.safe_load(_COMPOSE.read_text())["services"]
    return set(services[service]["environment"])


def test_the_infra_template_declares_every_setting_uncommented() -> None:
    """`infra/.env.example` is the file a deployer copies — nothing may be missing.

    Uncommented specifically: a commented var is documentation, and an operator who
    copies this file will not set it. That is how `SUPABASE_JWT_SECRET` could be
    absent on a box whose API was green.
    """
    missing = _settings_vars() - _declared(_INFRA_TEMPLATE, include_commented=False)
    assert not missing, (
        f"infra/.env.example does not declare {sorted(missing)} — a deployer copying "
        f"it gets an app that starts with those silently blank"
    )


def test_the_dev_template_declares_every_setting() -> None:
    """`apps/server/.env.example` is the local-dev inventory; commented counts here.

    A dev never needs `API_HOST`, but the template still has to *name* it, or the next
    person adding a setting has no file that lists what exists.
    """
    missing = _settings_vars() - _declared(_DEV_TEMPLATE, include_commented=True)
    assert not missing, f"apps/server/.env.example does not mention {sorted(missing)}"


def test_the_templates_declare_nothing_the_app_does_not_read() -> None:
    """The reverse direction: an unknown key is a typo or a var nobody consumes."""
    base = _settings_vars() | _DEPLOY_ONLY_VARS
    for template, allowed in ((_INFRA_TEMPLATE, base), (_DEV_TEMPLATE, base | _DEV_ONLY_VARS)):
        unknown = _declared(template, include_commented=True) - allowed
        assert not unknown, (
            f"{template.name} declares {sorted(unknown)}, which `Settings` does not "
            f"read and is not a known deploy-only var — a typo'd name is silently inert"
        )


def test_the_eval_key_is_offered_to_developers_and_withheld_from_the_vps() -> None:
    """`EVAL_OPENROUTER_API_KEY` belongs on a dev machine and nowhere near production.

    Both halves matter. Naming it in the dev template is how anyone learns the variable
    exists at all; keeping it out of `infra/.env.example` is `DEPLOY.md` §E2's actual
    rule — an eval key that reaches the VPS is just the production key again, and #102
    was two keys' worth of traffic on one balance.
    """
    assert "EVAL_OPENROUTER_API_KEY" in _declared(_DEV_TEMPLATE, include_commented=True), (
        "apps/server/.env.example never mentions EVAL_OPENROUTER_API_KEY, so the "
        "separate-key setup is documented only where nobody copies from"
    )
    assert "EVAL_OPENROUTER_API_KEY" not in _declared(_INFRA_TEMPLATE, include_commented=True), (
        "infra/.env.example is copied onto the VPS — an eval key must never be deployed"
    )


def test_the_api_container_receives_every_setting() -> None:
    """Compose must forward every var, or the template is promising the operator air.

    `LLM_TIMEOUT_S`/`LLM_MAX_RETRIES` lived in `infra/.env.example` while compose never
    passed them, so a bound the file documented in detail was simply not in effect.
    """
    missing = _settings_vars() - _compose_env("api")
    assert not missing, (
        f"docker-compose.prod.yml's api service does not pass {sorted(missing)} — "
        f"setting them in infra/.env would have no effect on the container"
    )


@pytest.mark.parametrize("var", ["OPENROUTER_API_KEY", "DEFAULT_MODEL", "COACH_MODEL"])
def test_the_scheduler_receives_the_llm_config(var: str) -> None:
    """The scheduler IS the nightly chain, so its LLM config is not optional.

    It is not checked for the full set (it verifies no JWTs and binds no port), but a
    missing model id or bound here breaks the surface that runs unattended.
    """
    assert var in _compose_env("scheduler"), f"the scheduler container never receives {var}"


def test_the_llm_bounds_reach_the_scheduler() -> None:
    """The single-threaded tick loop is where an unbounded call costs the most."""
    scheduler = _compose_env("scheduler")
    assert {"LLM_TIMEOUT_S", "LLM_MAX_RETRIES"} <= scheduler, (
        "the scheduler runs the nightly chain on one thread — without these it "
        "inherits the SDK's 600 s × 3, and one stuck call blocks every later owner"
    )


def test_the_scheduler_receives_the_self_host_unlock() -> None:
    """The nightly chain decides entitlement too (§12.3), so the flag must reach it.

    Without it a SELF-HOSTED install would serve its owner the AI layer over HTTP while
    the scheduler — a separate container, reading its own environment — treated them as
    free and silently generated nothing: recs, briefing and the daily action all absent,
    with a green API and no error anywhere. That is a whole-feature outage that looks
    like "the AI just isn't very good".
    """
    assert "SELF_HOST_UNLOCKED" in _compose_env("scheduler"), (
        "the scheduler container never receives SELF_HOST_UNLOCKED — a self-hosted "
        "install's nightly chain would skip every AI step for every owner"
    )
