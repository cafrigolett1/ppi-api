"""Request and response models.

Separate from ``models.py``: the wire format and the database schema change
for different reasons, and coupling them turns a column rename into a breaking
API change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import EngineName
from .entities import Label

# A cheap first guard, checked by the schema before any tokenising happens.
MAX_TEXT_CHARS = 100_000


class DetectedEntity(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    start: int
    end: int
    label: Label
    text: str
    score: float


class AnonymizeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Marie Dubois works at Ordina. IBAN BE68 5390 0754 7034.",
                "engine": "presidio-be",
            }
        }
    )

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    doc_id: str | None = Field(
        default=None,
        max_length=200,
        description="Stable identifier for this document. Supplying one "
        "enables deduplication: a repeat of the same doc_id with the same "
        "engine and prompt is answered from the log instead of re-running "
        "inference. Omit it to force a fresh call.",
    )
    engine: EngineName | None = Field(
        default=None, description="Defaults to the configured engine."
    )
    include_entities: bool = Field(
        default=False,
        description="Return the detected spans. Off by default: they reveal "
        "exactly what was removed and where.",
    )


class AnonymizeResponse(BaseModel):
    request_id: str
    redacted_text: str
    entity_count: int
    counts_by_label: dict[str, int]
    entities: list[DetectedEntity] | None = None
    engine: str
    input_tokens: int
    latency_ms: float
    cached: bool = Field(
        default=False,
        description="True when this was answered from a previous log row "
        "rather than by running inference.",
    )
    served_from_log_id: int | None = Field(
        default=None, description="The log row this was served from."
    )
    served_from_timestamp: datetime | None = Field(
        default=None, description="When that original result was produced."
    )


class BatchItem(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class BatchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {"id": "doc-1", "text": "Marie Dubois works at Ordina."},
                    {"id": "doc-2", "text": "Call 015 29 58 58 for details."},
                ]
            }
        }
    )

    items: list[BatchItem] = Field(min_length=1)
    engine: EngineName | None = None
    include_entities: bool = False


class BatchItemResult(BaseModel):
    """One item's outcome.

    Status is per item: a batch returns 200 with a mix of successes and
    failures, so one malformed document does not discard the rest.
    """

    id: str
    status: Literal["ok", "error"]
    redacted_text: str | None = None
    entity_count: int | None = None
    counts_by_label: dict[str, int] | None = None
    entities: list[DetectedEntity] | None = None
    input_tokens: int | None = None
    latency_ms: float | None = None
    cached: bool = False
    error: str | None = None


class BatchResponse(BaseModel):
    request_id: str
    results: list[BatchItemResult] = Field(description="In request order.")
    succeeded: int
    failed: int
    engine: str
    total_input_tokens: int
    latency_ms: float


class LogEntry(BaseModel):
    """A stored request, as returned by /v1/logs."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    request_id: str
    doc_id: str | None
    endpoint: str
    created_at: datetime
    engine: str
    input_tokens: int
    input_chars: int
    status: str
    entity_count: int
    latency_ms: float
    prompt_id: str | None
    served_from_log_id: int | None
    served_from_timestamp: datetime | None
    error_type: str | None
    error_message: str | None


class StatsResponse(BaseModel):
    total_requests: int
    inferences_run: int = Field(
        default=0, description="Rows where work was actually done."
    )
    served_from_cache: int = Field(
        default=0, description="Rows answered from an earlier result."
    )
    succeeded: int
    failed: int
    total_input_tokens: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    by_engine: dict[str, int]


class EngineInfo(BaseModel):
    name: str
    description: str
    available: bool
    prompt_id: str | None = Field(
        default=None,
        description="Fingerprint of the prompt. Part of the deduplication "
        "key, so a prompt change makes earlier results ineligible for reuse.",
    )
    prompt: str | None = Field(
        default=None,
        description="System prompt for LLM engines, so a benchmark can record "
        "exactly what was run. Null for the Presidio engines.",
    )


class HealthResponse(BaseModel):
    status: str
    engines: list[str]
    database: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
