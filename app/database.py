"""Database engine and session handling."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from .config import Settings

logger = logging.getLogger(__name__)

_engine = None


def get_engine(settings: Settings):
    """Create the engine once and reuse it."""
    global _engine
    if _engine is not None:
        return _engine

    url = settings.database_url
    connect_args = {}
    if url.startswith("sqlite"):
        # SQLite refuses cross-thread use by default, and FastAPI runs
        # synchronous handlers in a worker thread pool.
        connect_args["check_same_thread"] = False
        path = url.split("///")[-1]
        if path not in (":memory:", ""):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(url, connect_args=connect_args, echo=False)
    return _engine


def init_db(settings: Settings) -> None:
    """Create tables. Called at startup so the first request does not pay
    for it, and so a bad database URL fails loudly at boot."""
    SQLModel.metadata.create_all(get_engine(settings))
    logger.info("database ready", extra={"url": _redact(settings.database_url)})


def session_scope(settings: Settings) -> Iterator[Session]:
    with Session(get_engine(settings)) as session:
        yield session


def reset_engine() -> None:
    """Drop the cached engine. Used by tests to swap databases."""
    global _engine
    _engine = None


def _redact(url: str) -> str:
    """Strip credentials before a URL reaches the logs."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    return f"{scheme}://***@{rest.rpartition('@')[2]}"
