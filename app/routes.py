"""HTTP routes.

Handlers are deliberately thin: validate, call the service, record the call,
shape the response. Anything longer than that belongs in ``service.py``.

They are ``def`` rather than ``async def`` on purpose. Detection blocks, and
FastAPI runs synchronous handlers in a worker thread pool, so the event loop
stays free without any asyncio in our own code.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlmodel import Session, func, select

from .config import Settings
from .database import session_scope
from .models import RequestLog, make_dedupe_key
from .schemas import (
    AnonymizeRequest,
    AnonymizeResponse,
    BatchItemResult,
    BatchRequest,
    BatchResponse,
    DetectedEntity,
    EngineInfo,
    ErrorResponse,
    HealthResponse,
    LogEntry,
    StatsResponse,
)
from .service import (
    AnonymizationService,
    EngineUnavailable,
    InputTooLarge,
    Result,
    find_reusable,
    new_request_id,
    record,
    replay,
)

logger = logging.getLogger(__name__)

router = APIRouter()

ERRORS: dict[int | str, dict] = {
    413: {"model": ErrorResponse, "description": "Input exceeds the token limit"},
    502: {"model": ErrorResponse, "description": "Detection failed"},
    503: {"model": ErrorResponse, "description": "Engine unavailable"},
}


# -- dependencies ----------------------------------------------------------


def get_config(request: Request) -> Settings:
    """Read settings from application state, NOT from ``get_settings()``.

    ``get_settings()`` is lru_cached and reads the environment, so a Settings
    object passed into ``create_app`` would be ignored here - silently. That
    matters most for ``store_input_text``: a deployment that turns off input
    persistence would still have written every raw request to the database.
    Application state is the single source of truth for what this instance is
    actually configured to do.
    """
    return request.app.state.settings


def get_service(request: Request) -> AnonymizationService:
    return request.app.state.service


SettingsDep = Annotated[Settings, Depends(get_config)]


def get_session(settings: SettingsDep):
    yield from session_scope(settings)


ServiceDep = Annotated[AnonymizationService, Depends(get_service)]
SessionDep = Annotated[Session, Depends(get_session)]


def _entities(result: Result, include: bool) -> list[DetectedEntity] | None:
    if not include:
        return None
    return [DetectedEntity.model_validate(e) for e in result.entities]


def _response(
    request_id: str, result: Result, engine: str, include_entities: bool
) -> AnonymizeResponse:
    return AnonymizeResponse(
        request_id=request_id,
        redacted_text=result.redacted_text,
        entity_count=len(result.entities),
        counts_by_label=result.counts_by_label,
        entities=_entities(result, include_entities),
        engine=engine,
        input_tokens=result.input_tokens,
        latency_ms=result.latency_ms,
        cached=result.cached,
        served_from_log_id=result.served_from_log_id,
        served_from_timestamp=result.served_from_timestamp,
    )


@router.post(
    "/v1/anonymize",
    response_model=AnonymizeResponse,
    responses=ERRORS,
    tags=["anonymisation"],
)
def anonymize_one(
    payload: AnonymizeRequest,
    service: ServiceDep,
    session: SessionDep,
    settings: SettingsDep,
) -> AnonymizeResponse:
    """Redact one document.

    Supplying ``doc_id`` enables deduplication: a repeat of the same document
    with the same engine and prompt is answered from the log rather than by
    running inference again.
    """
    engine = payload.engine or settings.default_engine
    request_id = new_request_id()
    prompt_id = service.prompt_id(engine)

    if payload.doc_id:
        previous = find_reusable(
            session, make_dedupe_key(payload.doc_id, engine, prompt_id)
        )
        reused = replay(previous, payload.text) if previous is not None else None
        if reused is not None:
            # The replay is still recorded, pointing at the row it came from,
            # so the table stays a complete record of traffic.
            record(
                session,
                settings,
                request_id=request_id,
                endpoint="/v1/anonymize",
                engine=engine,
                text=payload.text,
                input_tokens=reused.input_tokens,
                doc_id=payload.doc_id,
                prompt_id=prompt_id,
                result=reused,
            )
            return _response(request_id, reused, engine, payload.include_entities)

    try:
        result = service.anonymize(payload.text, engine)
    except EngineUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except InputTooLarge as exc:
        # Recorded even though nothing was processed: a caller repeatedly
        # hitting the ceiling is exactly what the log is for.
        record(
            session,
            settings,
            request_id=request_id,
            endpoint="/v1/anonymize",
            engine=engine,
            text=payload.text,
            input_tokens=exc.tokens,
            doc_id=payload.doc_id,
            prompt_id=prompt_id,
            error=exc,
        )
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    except Exception as exc:
        record(
            session,
            settings,
            request_id=request_id,
            endpoint="/v1/anonymize",
            engine=engine,
            text=payload.text,
            input_tokens=0,
            doc_id=payload.doc_id,
            prompt_id=prompt_id,
            error=exc,
        )
        logger.exception("detection failed", extra={"request_id": request_id})
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "detection failed") from exc

    record(
        session,
        settings,
        request_id=request_id,
        endpoint="/v1/anonymize",
        engine=engine,
        text=payload.text,
        input_tokens=result.input_tokens,
        doc_id=payload.doc_id,
        prompt_id=prompt_id,
        result=result,
    )

    return _response(request_id, result, engine, payload.include_entities)


@router.post(
    "/v1/anonymize/batch",
    response_model=BatchResponse,
    responses=ERRORS,
    tags=["anonymisation"],
)
def anonymize_batch(
    payload: BatchRequest,
    service: ServiceDep,
    session: SessionDep,
    settings: SettingsDep,
) -> BatchResponse:
    """Redact several documents, one model call each, processed in order.

    Each item carries its own status, so one failure returns an error for that
    document while the rest still come back.
    """
    engine = payload.engine or settings.default_engine
    request_id = new_request_id()

    if len(payload.items) > settings.max_batch_size:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"batch of {len(payload.items)} exceeds the limit of "
            f"{settings.max_batch_size}",
        )

    # Whole-batch budget, checked before any work so an over-budget request
    # costs nothing rather than billing for the half it got through.
    total_tokens = sum(service.tokens.count(item.text) for item in payload.items)
    if total_tokens > settings.max_batch_tokens:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"batch is approximately {total_tokens} tokens, limit is "
            f"{settings.max_batch_tokens}",
        )

    try:
        service.get_engine(engine)
    except EngineUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    started = time.perf_counter()
    prompt_id = service.prompt_id(engine)
    results: list[BatchItemResult] = []

    for item in payload.items:
        previous = find_reusable(session, make_dedupe_key(item.id, engine, prompt_id))
        reused = replay(previous, item.text) if previous is not None else None
        if reused is not None:
            record(
                session,
                settings,
                request_id=request_id,
                endpoint="/v1/anonymize/batch",
                engine=engine,
                text=item.text,
                input_tokens=reused.input_tokens,
                doc_id=item.id,
                prompt_id=prompt_id,
                result=reused,
            )
            results.append(
                BatchItemResult(
                    id=item.id,
                    status="ok",
                    redacted_text=reused.redacted_text,
                    entity_count=len(reused.entities),
                    counts_by_label=reused.counts_by_label,
                    entities=_entities(reused, payload.include_entities),
                    input_tokens=reused.input_tokens,
                    latency_ms=reused.latency_ms,
                    cached=True,
                )
            )
            continue

        try:
            result = service.anonymize(item.text, engine)
        except Exception as exc:
            logger.warning(
                "batch item failed",
                extra={
                    "request_id": request_id,
                    "doc_id": item.id,
                    "error_type": type(exc).__name__,
                },
            )
            record(
                session,
                settings,
                request_id=request_id,
                endpoint="/v1/anonymize/batch",
                engine=engine,
                text=item.text,
                input_tokens=getattr(exc, "tokens", 0),
                doc_id=item.id,
                error=exc,
            )
            results.append(BatchItemResult(id=item.id, status="error", error=str(exc)))
            continue

        record(
            session,
            settings,
            request_id=request_id,
            endpoint="/v1/anonymize/batch",
            engine=engine,
            text=item.text,
            input_tokens=result.input_tokens,
            doc_id=item.id,
            result=result,
        )
        results.append(
            BatchItemResult(
                id=item.id,
                status="ok",
                redacted_text=result.redacted_text,
                entity_count=len(result.entities),
                counts_by_label=result.counts_by_label,
                entities=_entities(result, payload.include_entities),
                input_tokens=result.input_tokens,
                latency_ms=result.latency_ms,
            )
        )

    succeeded = sum(1 for r in results if r.status == "ok")
    return BatchResponse(
        request_id=request_id,
        results=results,
        succeeded=succeeded,
        failed=len(results) - succeeded,
        engine=engine,
        total_input_tokens=total_tokens,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )


# -- observability ---------------------------------------------------------


@router.get("/v1/logs", response_model=list[LogEntry], tags=["observability"])
def list_logs(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=500),
    status_filter: str | None = Query(None, alias="status"),
) -> list[RequestLog]:
    """Recent calls, newest first. Payload columns are omitted deliberately."""
    statement = select(RequestLog).order_by(RequestLog.id.desc()).limit(limit)
    if status_filter:
        statement = statement.where(RequestLog.status == status_filter)
    return list(session.exec(statement))


@router.get("/v1/stats", response_model=StatsResponse, tags=["observability"])
def stats(session: SessionDep) -> StatsResponse:
    """Totals, token usage and latency across everything recorded."""
    total = session.exec(select(func.count(RequestLog.id))).one() or 0
    ok = (
        session.exec(
            select(func.count(RequestLog.id)).where(RequestLog.status == "ok")
        ).one()
        or 0
    )
    run = (
        session.exec(
            select(func.count(RequestLog.id)).where(
                RequestLog.served_from_log_id.is_(None)
            )
        ).one()
        or 0
    )
    tokens = session.exec(select(func.sum(RequestLog.input_tokens))).one() or 0
    avg = session.exec(
        select(func.avg(RequestLog.latency_ms)).where(RequestLog.status == "ok")
    ).one()

    # p95 computed in Python: SQLite has no percentile function, and the log
    # is small enough that sorting it is cheaper than a window query.
    latencies = sorted(
        session.exec(select(RequestLog.latency_ms).where(RequestLog.status == "ok"))
    )
    p95 = (
        latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
        if latencies
        else None
    )

    by_engine = dict(
        session.exec(
            select(RequestLog.engine, func.count(RequestLog.id)).group_by(
                RequestLog.engine
            )
        ).all()
    )

    return StatsResponse(
        total_requests=total,
        inferences_run=run,
        served_from_cache=total - run,
        succeeded=ok,
        failed=total - ok,
        total_input_tokens=int(tokens),
        avg_latency_ms=round(avg, 2) if avg else None,
        p95_latency_ms=round(p95, 2) if p95 else None,
        by_engine=by_engine,
    )


@router.get("/v1/engines", response_model=list[EngineInfo], tags=["anonymisation"])
def list_engines(service: ServiceDep) -> list[dict]:
    return service.describe_engines()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(service: ServiceDep, session: SessionDep) -> HealthResponse:
    try:
        session.exec(select(func.count(RequestLog.id))).one()
        database = "ok"
    except Exception:
        database = "unavailable"
    return HealthResponse(status="ok", engines=sorted(service.engines), database=database)