"""Outbound notifications — Matrix, for job status and failures.

One sink: `send_matrix(text)`. It is fire-and-forget by contract and **must
never raise into its caller** — a failed notification cannot be allowed to break
the pipeline it reports on. Every failure is logged with context (standards §1:
failures are logged through the one logging path, never silently swallowed) and
returns `False`; success returns `True`.

Unconfigured (no homeserver / access token / room id) is not a failure — it logs
at debug and no-ops, so callers don't have to gate on config.

The transport is the Matrix client-server API: a `PUT` of an `m.room.message`
event with `msgtype: m.text` to the room, authenticated with a Bearer access
token. The access token lives in the `Authorization` header (never the URL), so
it cannot leak into an httpx request-line log the way a Telegram bot token did
in the URL path.
"""

from __future__ import annotations

import uuid

import httpx

from healthee.core.config import get_settings
from healthee.core.logging import get_logger

log = get_logger(__name__)

_TIMEOUT_S = 10.0


def send_matrix(text: str) -> bool:
    """POST one `m.text` message to the configured Matrix room. True on success.

    Returns False (logging why) on any failure or when notifications aren't
    configured. Never raises.
    """
    settings = get_settings()
    homeserver = settings.matrix_homeserver.strip().rstrip("/")
    token = settings.matrix_access_token.strip()
    room_id = settings.matrix_room_id.strip()
    if not homeserver or not token or not room_id:
        log.debug("matrix not configured — skipping notification")
        return False

    # A fresh txn id per send: the client-server API deduplicates on it, and a
    # fixed id would collapse repeated identical alerts into one event.
    url = f"{homeserver}/_matrix/client/v3/rooms/{room_id}/send/m.room.message/{uuid.uuid4()}"
    payload = {
        "msgtype": "m.text",
        "body": text,
    }
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = httpx.put(url, json=payload, headers=headers, timeout=_TIMEOUT_S)
    except httpx.HTTPError as exc:
        log.warning("matrix send failed (network): %s", exc)
        return False
    except Exception as exc:  # last-resort guard — notify must never raise
        log.warning("matrix send failed (unexpected): %s", exc)
        return False

    if response.is_success:
        return True
    log.warning("matrix HTTP %d: %s", response.status_code, response.text[:200])
    return False
