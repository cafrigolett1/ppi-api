"""Database tables.

SQLModel, so one class is both the ORM table and a Pydantic model - there is
no second set of schemas to keep in sync with these columns.

One row per document, not per HTTP request. A batch of twenty writes twenty
rows sharing a ``request_id``, so latency, tokens and errors are attributable
to the document that produced them.

Deduplication lives here rather than in a side cache, because this table is
also what a dashboard reads. Two stores would mean the dashboard could not
tell a real call from a replayed one.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


def make_dedupe_key(doc_id: str, engine: str, prompt_id: str | None) -> str:
    """Identity of a piece of work: this document, this engine, this prompt.

    Text is deliberately not part of the key - a doc_id is assumed stable. The
    text hash is stored alongside so a dashboard can still detect the case
    where a document changed under a reused id, which would otherwise replay a
    stale answer silently.
    """
    digest = hashlib.sha256()
    for part in (doc_id, engine, prompt_id or ""):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:32]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class RequestLog(SQLModel, table=True):
    __tablename__ = "request_logs"

    id: int | None = Field(default=None, primary_key=True)

    # -- correlation ------------------------------------------------------
    request_id: str = Field(
        index=True, description="Shared by every document in one HTTP request."
    )
    doc_id: str | None = Field(
        default=None,
        index=True,
        description="Caller-supplied identifier. Required for deduplication.",
    )
    endpoint: str = Field(index=True)
    created_at: datetime = Field(default_factory=utcnow, index=True)

    # -- what was asked ---------------------------------------------------
    engine: str = Field(index=True)
    prompt_id: str | None = Field(
        default=None,
        index=True,
        description="Fingerprint of the prompt used. Null for engines that "
        "have no prompt. Changing a prompt changes this, so previous results "
        "are not reused.",
    )
    input_tokens: int = Field(description="Estimated, not exact - see app/tokens.py.")
    input_chars: int
    text_hash: str | None = Field(
        default=None,
        index=True,
        description="Fingerprint of the input. Not part of the dedupe key, "
        "but stored so a document silently changing under a reused doc_id is "
        "detectable.",
    )

    # -- deduplication ----------------------------------------------------
    dedupe_key: str | None = Field(
        default=None,
        index=True,
        description="doc_id + engine + prompt_id. Null when no doc_id was "
        "supplied, which disables deduplication for that call.",
    )
    served_from_log_id: int | None = Field(
        default=None,
        index=True,
        description="Set when this request was answered from an earlier row "
        "instead of running inference. Null means the work was actually done, "
        "so a dashboard counts unique inferences with "
        "`WHERE served_from_log_id IS NULL`.",
    )
    served_from_timestamp: datetime | None = Field(
        default=None,
        description="created_at of the row this was served from. Denormalised "
        "so a dashboard can show result age without a join.",
    )

    # -- payloads ---------------------------------------------------------
    # Nullable because storing the input is configurable: it is the sensitive
    # data this service exists to remove.
    input_text: str | None = None
    output_text: str | None = None
    entities_json: str | None = Field(
        default=None,
        description="Detected spans as JSON. Stored so a replay returns the "
        "same spans as the original call, not just the redacted string.",
    )

    # -- what happened ----------------------------------------------------
    status: str = Field(index=True, description="'ok' or 'error'.")
    entity_count: int = 0
    counts_by_label: str | None = Field(
        default=None, description='JSON object, e.g. {"PERSON": 2}.'
    )
    latency_ms: float = Field(
        default=0.0,
        description="Detection time. On a replayed row this is the latency of "
        "the replay itself, near zero - the original is on the source row.",
    )

    error_type: str | None = Field(default=None, index=True)
    error_message: str | None = None