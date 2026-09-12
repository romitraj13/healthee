"""Which configurations this server REFUSES to boot on, and why each one.

Split out of `core/config.py` when the auth audit's two new refusals (B1, C3)
pushed that file past the 400-line gate. The seam is a real one rather than a place
to put the overflow: `config.py` answers *what the environment holds* — the fields,
their defaults and the two conninfo helpers — and this answers *what combinations
are not a deployment*. They change for different reasons, and the arguments below
are long precisely because each one is a decision somebody will want to reverse.

Every function here raises `ValueError` and returns nothing useful, so the
validators in `config.py` stay four lines each and read as a list of the ambiguities
this codebase refuses. Plain functions over primitives rather than a mixin: a mixin
would have to re-declare the fields it reads, and two declarations of a settings
field is the shape of problem this file exists to refuse.

The house rule they all follow: **an ambiguity is refused, never interpreted.** A
setting with two possible readings and no way for the operator to see which one was
taken is how a dead subsystem, or an inert isolation guarantee, survives for weeks
behind a green /healthz.
"""

from __future__ import annotations


def require_password(value: str) -> str:
    if not value:
        raise ValueError("POSTGRES_PASSWORD must be set")
    return value


def require_app_creds_together(user: str, password: str) -> None:
    """App user and password are both-or-neither — never one alone.

    Half-set creds are the dangerous case: the pool would either try a
    passwordless login or connect as the ADMIN while the operator believes the
    least-privilege role is in force. Both failure modes are silent, and the
    second one is the exact security theatre this split exists to end. Refuse
    the ambiguity instead of picking an interpretation.
    """
    if bool(user) != bool(password):
        raise ValueError(
            "POSTGRES_APP_USER and POSTGRES_APP_PASSWORD must be set together "
            "(set both to use the least-privilege app role, or neither to fall "
            "back to the admin creds)"
        )


def refuse_an_unasked_for_rls_bypass(app_role_configured: bool, allow_fallback: bool) -> None:
    """Refuse to boot on the admin fallback unless the operator ASKED for it.

    A blank `POSTGRES_APP_USER` means the request/job pool connects as the
    owner/admin role, and a superuser ignores every policy `0008` creates —
    `FORCE ROW LEVEL SECURITY` does not bind it either, and `0008` deliberately does
    not FORCE. So in that state RLS, the backstop underneath every explicit
    `AND user_id = %s`, is decoration. This product is MULTI-TENANT, so that
    backstop is the isolation guarantee, not a nicety.

    Until now it was announced by one WARNING at pool open, and **both shipped env
    templates defaulted to the unsafe side** — an operator who copied a template and
    filled in what was obviously required got the RLS-inert configuration and a log
    line. Whether the live deployment was in that state was, at audit time, an open
    question nobody could answer from the repository.

    ### Why refuse-to-boot rather than "make the fallback impossible where tenants
    exist"

    Both were on the table. The tenant-count check has to run against the database
    at pool open, which means it can only fire AFTER the deployment is already
    serving — and worse, the dangerous moment is when a SECOND owner signs up, on a
    process that booted safely and is still running. A check that cannot see the
    moment it exists for is not the guarantee it looks like. This one fires at boot,
    in front of the operator, before a single request; it needs no database; and it
    is the shape `config.py` already uses three times over
    (`_require_app_creds_together`, `_require_model_ids_when_ai_key_is_set`,
    `_require_an_ordered_zoom_range`) — an ambiguity refused rather than
    interpreted.

    ### And why there is an opt-out at all

    The bootstrap really is two deploys (`infra/DEPLOY.md` B2): the role cannot be
    provisioned before the deploy that provisions it, so step 1 MUST be bootable on
    the admin creds. Refusing outright would make the documented procedure
    impossible. `ALLOW_ADMIN_DB_FALLBACK=true` keeps step 1 working while turning
    the transitional state into something written down in the deployment's own env
    file, where the next person can see it — instead of the absence of a value,
    which looks identical to never having been considered.
    """
    if app_role_configured or allow_fallback:
        return
    raise ValueError(
        "POSTGRES_APP_USER / POSTGRES_APP_PASSWORD are blank, so the pool would "
        "connect as the admin role — which BYPASSES Row-Level Security and leaves "
        "tenant isolation resting on the explicit user_id filters alone. Provision "
        "the least-privilege role (python -m healthee.db.provision_app_role) and set "
        "both vars. During the two-deploy bootstrap (infra/DEPLOY.md B2 step 1), set "
        "ALLOW_ADMIN_DB_FALLBACK=true to ask for the transitional state explicitly."
    )


def require_model_ids_when_ai_key_is_set(
    api_key: str, default_model: str, coach_model: str
) -> None:
    """Refuse "configured for AI, but cannot do AI" — it is never a valid state.

    A blank model id is not a default, it is a dead subsystem: the id is forwarded
    to OpenRouter verbatim and comes back **400 on every call** — insight cards,
    the coach, and the nightly recs/briefing chain. Nothing in the deploy can see
    it. `/healthz` is a liveness+DB probe, so it goes green; the LLM failures land
    in Telegram and the scheduler log. Prod ran in exactly this state.

    **Fail fast, not warn.** The admin-creds fallback used to be the contrasting
    precedent here — a documented, deliberately-transitional step of a two-deploy
    bootstrap (`infra/DEPLOY.md` B2) that had to stay bootable. The auth audit
    settled that differently: `_refuse_an_unasked_for_rls_bypass` above now refuses
    it too, and the bootstrap stays bootable by ASKING for the transitional state
    (`ALLOW_ADMIN_DB_FALLBACK=true`) rather than by leaving a variable blank.

    There is no deploy order, no bootstrap and no
    migration in which "key set, model id blank" is correct, which makes it the
    same shape as `_require_app_creds_together` above: an ambiguity to refuse, not
    an interpretation to pick. Refusing turns it into a *deploy-time* failure, in
    front of the operator, instead of a silence discovered weeks later.

    Not running the AI layer stays a first-class, bootable configuration: leave
    `OPENROUTER_API_KEY` blank and this never fires. The check only triggers on a
    state the operator explicitly asked for and then half-configured.

    Deliberately NOT mirrored into `/healthz`: with this validator the state cannot
    exist in a live process, and a probe that 503s on a config problem would let
    Docker's healthcheck restart the container into the same config forever —
    trading a dead AI layer for a flapping read API, which is worse than the
    disease. Config correctness belongs at boot; `/healthz` stays a signal an
    orchestrator can act on.
    """
    if not api_key:
        return
    blank = [
        name
        for name, value in (
            ("DEFAULT_MODEL", default_model),
            ("COACH_MODEL", coach_model),
        )
        if not value.strip()
    ]
    if blank:
        raise ValueError(
            f"an LLM API key is set but {' and '.join(blank)} is blank — a blank "
            "model id reaches the provider verbatim and every LLM call returns 400, "
            "with nothing failing in /healthz. Set the model id(s), or unset the "
            "LLM key (LLM_API_KEY / OPENROUTER_API_KEY) to run without the AI layer."
        )


def require_a_usable_tile_template(value: str) -> str:
    """Refuse a template this server cannot safely expand.

    Two failures, both silent without this. A template missing a placeholder
    expands to the SAME url for every tile, so the cache fills with one image
    and the whole basemap is one square of the world repeated — which looks
    like a rendering bug in the app. And a non-http scheme reaches
    `urllib.request.urlopen` verbatim: `file:///etc/passwd` would make the
    tile route a file-read primitive with the z/x/y ignored.
    """
    if not value.startswith(("http://", "https://")):
        raise ValueError(f"MAP_TILE_URL must be an http(s) url, not {value!r}")
    missing = [token for token in ("{z}", "{x}", "{y}") if token not in value]
    if missing:
        raise ValueError(
            f"MAP_TILE_URL is missing {' and '.join(missing)} — a template without "
            f"them expands to one tile for every request"
        )
    return value


def require_an_ordered_zoom_range(min_zoom: int, max_zoom: int) -> None:
    """An inverted range refuses every tile, which reads as "the map is broken"."""
    if min_zoom > max_zoom:
        raise ValueError(
            f"MAP_TILE_MIN_ZOOM ({min_zoom}) is above MAP_TILE_MAX_ZOOM "
            f"({max_zoom}) — no zoom would ever be servable"
        )
