"""Unit tests for core.config — parsing, defaults, and the singleton accessor."""

from __future__ import annotations

import pytest

from healthee.core.config import Settings, get_settings


def test_defaults_apply_when_only_required_var_set(
    env: None,  # noqa: ARG001 — sets the required password
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Clear any ambient POSTGRES_* (a developer shell or CI with POSTGRES_PORT set
    # would otherwise fail this defaults test) so it exercises the code defaults.
    # The optional-integration keys need the same treatment for the same reason: a
    # shell that sourced `.env` for the Postgres creds also carries OPENROUTER_API_KEY,
    # which turned the assertions below into a failure that reproduces on a developer
    # machine and never in CI — a phantom that cost an agent real time chasing it.
    for var in (
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "OPENROUTER_API_KEY",
        "TELEGRAM_BOT_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.postgres_host == "localhost"
    assert settings.postgres_port == 5432
    assert settings.postgres_db == "healthee"
    assert settings.api_port == 8765
    assert settings.log_level == "INFO"
    # Optional integrations default to empty (they no-op when blank).
    assert settings.openrouter_api_key == ""
    assert settings.telegram_bot_token == ""


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "5544")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.postgres_host == "db.internal"
    assert settings.postgres_port == 5544
    assert settings.log_level == "DEBUG"
    get_settings.cache_clear()


def test_admin_db_url_is_libpq_conninfo(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear ambient host/port/user so the conninfo reflects the code defaults,
    # not whatever a dev shell exported.
    for var in ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret")
    monkeypatch.setenv("POSTGRES_DB", "healthee")
    get_settings.cache_clear()
    url = get_settings().admin_db_url
    assert "dbname=healthee" in url
    assert "password=s3cret" in url
    assert "host=localhost" in url
    get_settings.cache_clear()


# ── the app-role credential split (6.5b-1, MULTI_USER.md §3.3) ────────────────


def test_app_db_url_falls_back_to_the_admin_creds(env: None) -> None:  # noqa: ARG001
    """Unset app creds ⇒ exactly the pre-split behaviour, so this is safe to deploy
    before the role is provisioned. `core/db` warns that RLS cannot apply."""
    settings = get_settings()
    assert settings.app_role_configured is False
    assert settings.app_db_url == settings.admin_db_url


def test_app_db_url_uses_the_app_creds_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "admin-pw")
    monkeypatch.setenv("POSTGRES_USER", "healthee")
    monkeypatch.setenv("POSTGRES_APP_USER", "healthee_app")
    monkeypatch.setenv("POSTGRES_APP_PASSWORD", "app-pw")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.app_role_configured is True
    assert "user=healthee_app " in settings.app_db_url
    assert "password=app-pw" in settings.app_db_url
    # The admin identity is unchanged — the two never bleed into each other.
    assert "user=healthee " in settings.admin_db_url
    assert "password=admin-pw" in settings.admin_db_url
    get_settings.cache_clear()


def test_half_set_app_creds_are_a_loud_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse the ambiguity: silently connecting as the ADMIN while the operator
    believes the least-privilege role is in force is the exact security theatre the
    split exists to end."""
    monkeypatch.setenv("POSTGRES_PASSWORD", "admin-pw")
    monkeypatch.setenv("POSTGRES_APP_USER", "healthee_app")
    monkeypatch.delenv("POSTGRES_APP_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="must be set together"):
        Settings()
    monkeypatch.delenv("POSTGRES_APP_USER")
    monkeypatch.setenv("POSTGRES_APP_PASSWORD", "app-pw")
    with pytest.raises(ValueError, match="must be set together"):
        Settings()


# ── the "configured for AI, cannot do AI" state (#56) ─────────────────────────
#
# Every one of these uses the `env` fixture, which chdirs to a tmp dir. That is load-
# bearing, not decorative: `Settings.model_config` sets `env_file=".env"` resolved
# against the CWD, so run from `apps/server/` with a real `.env` these tests would read
# the developer's actual OPENROUTER_API_KEY and DEFAULT_MODEL and assert nothing. That
# exact trap produced a phantom failure here days ago.


@pytest.mark.parametrize(
    ("blank_var", "other_var"),
    [("DEFAULT_MODEL", "COACH_MODEL"), ("COACH_MODEL", "DEFAULT_MODEL")],
)
def test_an_ai_key_with_a_blank_model_id_is_refused(
    env: None,  # noqa: ARG001 — hermetic env + tmp CWD, see the note above
    monkeypatch: pytest.MonkeyPatch,
    blank_var: str,
    other_var: str,
) -> None:
    """Key set + either id blank ⇒ construction fails, so the process never starts.

    This is the state prod ran in: the blank id goes to OpenRouter verbatim, every
    call 400s, and `/healthz` stays green because it is a liveness+DB probe. Refusing
    at construction moves the discovery to the deploy, in front of the operator.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv(other_var, "vendor/some-model")
    monkeypatch.delenv(blank_var, raising=False)
    with pytest.raises(ValueError, match=blank_var):
        Settings()


def test_a_whitespace_only_model_id_is_refused_too(
    env: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`DEFAULT_MODEL=" "` is exactly as dead as blank — and looks set in a diff."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("DEFAULT_MODEL", "   ")
    monkeypatch.setenv("COACH_MODEL", "vendor/strong")
    with pytest.raises(ValueError, match="DEFAULT_MODEL"):
        Settings()


def test_no_ai_key_and_no_model_ids_is_a_valid_configuration(
    env: None,  # noqa: ARG001
) -> None:
    """Running WITHOUT the AI layer must stay bootable — the check is opt-in.

    Without this the fail-fast would be a landmine for every self-hoster who wants the
    tracking without the LLM, which the product explicitly supports.
    """
    settings = Settings()
    assert settings.openrouter_api_key == ""
    assert settings.default_model == ""
    assert settings.coach_model == ""


def test_an_ai_key_with_both_model_ids_is_accepted(
    env: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mutation guard: the validator must not reject the correct configuration."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("DEFAULT_MODEL", "vendor/cheap")
    monkeypatch.setenv("COACH_MODEL", "vendor/strong")
    settings = Settings()
    assert settings.default_model == "vendor/cheap"
    assert settings.coach_model == "vendor/strong"


def test_model_ids_without_a_key_are_not_a_config_error(
    env: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the key drives the requirement — ids left set while the key is pulled is
    how someone temporarily disables the AI layer, and it is a working state."""
    monkeypatch.setenv("DEFAULT_MODEL", "vendor/cheap")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert Settings().default_model == "vendor/cheap"


def test_missing_password_is_a_loud_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with pytest.raises(ValueError):  # pydantic ValidationError is a ValueError
        Settings()


def test_get_settings_is_cached(env: None) -> None:  # noqa: ARG001
    assert get_settings() is get_settings()


# ── provider-agnostic LLM key + base URL (self-host / Ollama Cloud) ─────────


def test_llm_base_url_defaults_to_openrouter(env: None) -> None:  # noqa: ARG001
    assert Settings().llm_base_url == "https://openrouter.ai/api/v1"


def test_llm_api_key_wins_over_openrouter_key(
    env: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "ollama-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    monkeypatch.setenv("DEFAULT_MODEL", "cheap")
    monkeypatch.setenv("COACH_MODEL", "strong")
    assert Settings().effective_llm_api_key() == "ollama-key"


def test_effective_key_falls_back_to_openrouter_key(
    env: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    monkeypatch.setenv("DEFAULT_MODEL", "cheap")
    monkeypatch.setenv("COACH_MODEL", "strong")
    assert Settings().effective_llm_api_key() == "sk-or-x"


def test_effective_key_is_blank_without_any_key(env: None) -> None:  # noqa: ARG001
    assert Settings().effective_llm_api_key() == ""


def test_llm_api_key_alone_drives_the_model_id_guard(
    env: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider-agnostic key must trigger the same fail-fast as the legacy one."""
    monkeypatch.setenv("LLM_API_KEY", "ollama-key")
    monkeypatch.setenv("DEFAULT_MODEL", "some-model")
    monkeypatch.delenv("COACH_MODEL", raising=False)
    with pytest.raises(ValueError, match="COACH_MODEL"):
        Settings()
