"""Logging setup — one configured root for the whole app.

Every module logs through the stdlib `logging` tree (`get_logger(__name__)`);
this is the single place the root handler and level are configured. Level comes
from `Settings.log_level` (the one config path — no direct env reads). Format is
concise and carries the logger name so failures are traceable to their module
(standards §1: "errors logged with context through the one logging path").
"""

from __future__ import annotations

import logging

from healthee.core.config import get_settings

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured = False


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once. Idempotent — repeat calls are no-ops.

    `level` overrides `Settings.log_level` when given (tests, one-off scripts).
    Called at app startup; anything logging before this still works via stdlib
    defaults, it just isn't formatted yet.
    """
    global _configured
    if _configured:
        return
    resolved = (level or get_settings().log_level).upper()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)
    # httpx logs every request line at INFO as `HTTP Request: POST <full url> ...`,
    # and a secret can live in the URL path (a Telegram bot token did:
    # api.telegram.org/bot<TOKEN>/…). Matrix keeps its access token in the
    # Authorization header instead, but the pin stays as defence in depth — a
    # request line is never something we want at INFO.
    # Pin its logger to WARNING regardless of the root level: we never want the
    # request line, and nothing we rely on lives below WARNING there.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _configured = True


class _PathFilter(logging.Filter):
    """Drops access-log records whose line mentions [prefix]."""

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self._prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        return self._prefix not in record.getMessage()


def silence_access_log_for(prefix: str) -> None:
    """Stop uvicorn's access log printing request lines under [prefix].

    The basemap tile route is the caller, and the reason is the same one that
    pins `httpx` to WARNING above: the SECRET IS IN THE URL. There it is a
    Telegram bot token; here it is z/x/y, which is a street corner and a
    timestamp — a location history assembled one access line at a time, which
    would undo the entire point of proxying the tiles in the first place.

    Applied to the `uvicorn.access` logger rather than the root, because it is
    uvicorn that owns that handler, and applied at startup rather than in
    [configure_logging] because uvicorn configures its loggers after ours.
    Idempotent: a second call for the same prefix adds nothing.
    """
    access = logging.getLogger("uvicorn.access")
    for existing in access.filters:
        if isinstance(existing, _PathFilter) and existing._prefix == prefix:
            return
    access.addFilter(_PathFilter(prefix))


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Use `__name__` at each call site so log lines
    identify their originating module."""
    return logging.getLogger(name)
