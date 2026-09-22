"""Anonymisation service: engines, detection, and persistence of every call.

Synchronous and sequential throughout. Detection is CPU-bound (spaCy) or
blocking I/O (the LLM client), and FastAPI runs synchronous handlers in a
worker thread, so the event loop is never blocked. A batch of N documents is
N calls in a loop. If throughput ever becomes the bottleneck, more replicas is
a simpler answer than making a single request concurrent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from sqlmodel import Session, select

from .config import Settings
from .entities import Detector, Entity, anonymize, resolve_overlaps
from .models import RequestLog, hash_text, make_dedupe_key
from .tokens import HeuristicTokenCounter

logger = logging.getLogger(__name__)

ENGINE_DESCRIPTIONS: dict[str, str] = {
    # Stock Presidio is not exposed: it is strictly worse than this on Belgian
    # data - no national-number recogniser at all, and its IBAN pattern drops
    # valid matches - so serving both would only add a slower, less accurate
    # option. app/detectors/belgian.py documents what the recognisers fix.
    "presidio-be": "Presidio plus checksum-validated Belgian IBAN and national number.",
    "llm-zero-shot": "LLM extraction from instructions alone.",
    "llm-few-shot": "LLM extraction with worked examples.",
}
LLM_ENGINES = frozenset({"llm-zero-shot", "llm-few-shot"})


class InputTooLarge(Exception):
    def __init__(self, tokens: int, limit: int) -> None:
        super().__init__(f"input is approximately {tokens} tokens, limit is {limit}")
        self.tokens = tokens
        self.limit = limit


class EngineUnavailable(Exception):
    """A known engine that this deployment cannot serve, e.g. no API key."""


@dataclass(frozen=True)
class Result:
    redacted_text: str
    entities: list[Entity]
    input_tokens: int
    latency_ms: float
    served_from_log_id: int | None = None
    served_from_timestamp: datetime | None = None

    @property
    def counts_by_label(self) -> dict[str, int]:
        return dict(Counter(e.label for e in self.entities))

    @property
    def cached(self) -> bool:
        return self.served_from_log_id is not None


class AnonymizationService:
    """Holds the loaded engines and runs detection.

    Engines are built once at construction, not per request: loading
    ``en_core_web_lg`` takes several seconds, and doing it lazily means the
    first caller after every deploy pays for it.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tokens = HeuristicTokenCounter(settings.chars_per_token)
        self.engines: dict[str, Detector] = {}
        self._load_engines()

    # -- setup -------------------------------------------------------------

    def _load_engines(self) -> None:
        from .detectors.presidio import PresidioDetector

        for name in ("presidio", "presidio-be"):
            started = time.perf_counter()
            self.engines[name] = PresidioDetector(
                score_threshold=self.settings.score_threshold,
                belgian=(name == "presidio-be"),
            )
            logger.info(
                "engine loaded",
                extra={
                    "engine": name,
                    "seconds": round(time.perf_counter() - started, 2),
                },
            )

        if not self.settings.llm_available:
            logger.info("LLM engines disabled: no API key configured")
            return

        from .detectors.llm import CohereChatClient, FewShotExample, LLMDetector

        client = CohereChatClient(
            api_key=self.settings.cohere_api_key.get_secret_value(),
            model=self.settings.llm_model,
        )
        self.engines["llm-zero-shot"] = LLMDetector(client, name="llm-zero-shot")

        shots = [
            FewShotExample.model_validate(e)
            for e in json.loads(self.settings.few_shot_path.read_text(encoding="utf-8"))
        ]
        self.engines["llm-few-shot"] = LLMDetector(
            client, examples=shots, name="llm-few-shot"
        )
        logger.info("LLM engines loaded", extra={"shots": len(shots)})

    def prompt_id(self, engine_name: str) -> str | None:
        """Fingerprint of an engine's prompt, or None if it has none.

        Part of the dedupe key, so editing a prompt makes previous results
        ineligible for reuse without anything being cleared by hand.
        """
        prompt = getattr(self.engines.get(engine_name), "prompt", None)
        if not prompt:
            return None
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]

    def describe_engines(self) -> list[dict]:
        return [
            {
                "name": n,
                "description": d,
                "available": n in self.engines,
                # Only the LLM engines have one; Presidio is configuration,
                # not a prompt.
                "prompt": getattr(self.engines.get(n), "prompt", None),
                "prompt_id": self.prompt_id(n),
            }
            for n, d in ENGINE_DESCRIPTIONS.items()
        ]

    def get_engine(self, name: str) -> Detector:
        engine = self.engines.get(name)
        if engine is None:
            raise EngineUnavailable(
                f"engine {name!r} is not available in this deployment"
                + (" (LLM engines require COHERE_API_KEY)" if name in LLM_ENGINES else "")
            )
        return engine

    # -- detection ---------------------------------------------------------

    def check_tokens(self, text: str, limit: int | None = None) -> int:
        count = self.tokens.count(text)
        ceiling = limit or self.settings.max_input_tokens
        if count > ceiling:
            raise InputTooLarge(count, ceiling)
        return count

    def anonymize(self, text: str, engine_name: str) -> Result:
        engine = self.get_engine(engine_name)
        tokens = self.check_tokens(text)

        started = time.perf_counter()
        try:
            found = engine.detect(text)
        except TypeError:
            found = engine.detect(text, doc_id="")

        resolved = resolve_overlaps(found)
        return Result(
            redacted_text=anonymize(text, resolved),
            entities=resolved,
            input_tokens=tokens,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def find_reusable(session: Session, dedupe_key: str | None) -> RequestLog | None:
    """The most recent successful row for this exact piece of work.

    Rows that were themselves replays are excluded, so every replay points at
    the row that actually ran inference rather than forming a chain.
    """
    if not dedupe_key:
        return None

    statement = (
        select(RequestLog)
        .where(RequestLog.dedupe_key == dedupe_key)
        .where(RequestLog.status == "ok")
        .where(RequestLog.served_from_log_id.is_(None))
        .order_by(RequestLog.id.desc())
        .limit(1)
    )
    return session.exec(statement).first()


def replay(row: RequestLog, text: str) -> Result | None:
    """Rebuild a Result from a stored row, or None if the row cannot serve one.

    A row logged with output storage disabled has no redacted text to return,
    so it is not reusable however well its key matches.
    """
    if row.output_text is None:
        return None

    entities: list[Entity] = []
    if row.entities_json:
        entities = [Entity.model_validate(e) for e in json.loads(row.entities_json)]

    if row.text_hash and row.text_hash != hash_text(text):
        # Same doc_id, different content. The key ignores text by design, so
        # this is the one case it cannot catch - refuse the replay rather than
        # return an answer computed from different input.
        logger.warning(
            "doc_id reused with different text; not replaying",
            extra={"doc_id": row.doc_id, "log_id": row.id},
        )
        return None

    return Result(
        redacted_text=row.output_text,
        entities=entities,
        input_tokens=row.input_tokens,
        latency_ms=0.0,
        served_from_log_id=row.id,
        served_from_timestamp=row.created_at,
    )


def record(
    session: Session,
    settings: Settings,
    *,
    request_id: str,
    endpoint: str,
    engine: str,
    text: str,
    input_tokens: int,
    doc_id: str | None = None,
    prompt_id: str | None = None,
    result: Result | None = None,
    error: Exception | None = None,
) -> None:
    """Write one row per document.

    A replayed request still writes a row, carrying ``served_from_log_id`` and
    the original timestamp. That keeps the table a complete record of traffic
    while letting a dashboard separate work from reuse:

        WHERE served_from_log_id IS NULL     -- inferences actually run
        WHERE served_from_log_id IS NOT NULL -- requests served from cache

    Persistence must never break the response: if the database is unreachable
    the caller still gets their redacted text, and the failure is logged. An
    anonymisation service that returns 500 because its audit log is down has
    traded its primary job for its secondary one.
    """
    entry = RequestLog(
        request_id=request_id,
        doc_id=doc_id,
        endpoint=endpoint,
        engine=engine,
        prompt_id=prompt_id,
        input_tokens=input_tokens,
        input_chars=len(text),
        text_hash=hash_text(text),
        dedupe_key=make_dedupe_key(doc_id, engine, prompt_id) if doc_id else None,
        served_from_log_id=result.served_from_log_id if result else None,
        served_from_timestamp=result.served_from_timestamp if result else None,
        input_text=text if settings.store_input_text else None,
        output_text=(
            result.redacted_text
            if result is not None and settings.store_output_text
            else None
        ),
        entities_json=(
            json.dumps([e.model_dump(mode="json") for e in result.entities])
            if result is not None and settings.store_output_text
            else None
        ),
        status="ok" if error is None else "error",
        entity_count=len(result.entities) if result else 0,
        counts_by_label=json.dumps(result.counts_by_label) if result else None,
        latency_ms=result.latency_ms if result else 0.0,
        error_type=type(error).__name__ if error else None,
        error_message=str(error) if error else None,
    )

    try:
        session.add(entry)
        session.commit()
    except Exception:
        session.rollback()
        logger.exception(
            "failed to persist request log", extra={"request_id": request_id}
        )
