"""Benchmark runner.

Drives the **running API over HTTP** rather than importing the engines
directly, so the numbers describe what the service actually serves - overlap
resolution, token limits and all - instead of a parallel configuration that
drifts from production.

Every call carries the document's ``doc_id``, which means the service answers
repeats from its own request log instead of re-running inference. There is no
separate benchmark cache: a second store would mean the dashboard could not
tell a real call from a replayed one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import requests

from ..entities import Entity
from .metrics import Evaluation, MatchMode, score_document

logger = logging.getLogger(__name__)


def load_dataset(path: Path, source: str | None = None) -> list[dict]:
    """Read a JSONL dataset of ``{id, source, text, entities}`` records.

    ``source`` filters to one subset. The consolidated file mixes three sets
    with very different label distributions, and a single blended score hides
    which kind of input fails.
    """
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if source is None or record.get("source") == source:
                records.append(record)
    if source is not None and not records:
        raise ValueError(f"no records with source={source!r} in {path}")
    return records


def sources(path: Path) -> list[str]:
    """Distinct sources present in a dataset, in first-seen order."""
    seen: list[str] = []
    for record in load_dataset(path):
        s = record.get("source", "unknown")
        if s not in seen:
            seen.append(s)
    return seen


def gold_entities(record: dict) -> list[Entity]:
    return [
        Entity(
            start=e["start"],
            end=e["end"],
            label=e["label"],
            text=record["text"][e["start"] : e["end"]],
            source="gold",
        )
        for e in record["entities"]
    ]


def available_engines(api_url: str = "http://localhost:8000") -> list[str]:
    response = requests.get(f"{api_url}/v1/engines", timeout=30)
    response.raise_for_status()
    return [e["name"] for e in response.json() if e["available"]]


def engine_prompts(api_url: str = "http://localhost:8000") -> dict[str, str | None]:
    """Prompts as reported by the running service.

    Read from the API rather than imported from the source, so what gets
    logged is what actually ran - not a copy that is assumed to match.
    """
    response = requests.get(f"{api_url}/v1/engines", timeout=30)
    response.raise_for_status()
    return {e["name"]: e.get("prompt") for e in response.json()}


def run_engine(
    dataset: list[dict],
    engine: str,
    dataset_name: str,
    api_url: str = "http://localhost:8000",
    mode: MatchMode = "strict",
    timeout: int = 180,
) -> Evaluation:
    """Send every document to the API and score the responses.

    One document per call, mirroring how the service is actually used. A
    failed document is recorded as a total miss rather than skipped: an engine
    that errors on hard inputs should not score better than one that answers
    them badly.
    """
    evaluation = Evaluation(engine=engine, dataset=dataset_name, mode=mode)
    replayed = 0

    for record in dataset:
        gold = gold_entities(record)

        try:
            response = requests.post(
                f"{api_url}/v1/anonymize",
                json={
                    "text": record["text"],
                    # Lets the service deduplicate against its request log.
                    "doc_id": record["id"],
                    "engine": engine,
                    "include_entities": True,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            logger.warning(
                "document failed",
                extra={"doc_id": record["id"], "error": str(exc)},
            )
            score_document(evaluation, record["id"], record["text"], [], gold)
            continue

        predicted = [
            Entity(
                start=e["start"],
                end=e["end"],
                label=e["label"],
                text=e["text"],
                score=e["score"],
                source=engine,
            )
            for e in (body.get("entities") or [])
        ]

        score_document(evaluation, record["id"], record["text"], predicted, gold)
        evaluation.input_tokens.append(body["input_tokens"])

        if body.get("cached"):
            replayed += 1
        else:
            # A replayed row reports ~0 ms, which would drag the latency
            # distribution toward zero and describe nothing real. Only
            # measured calls count.
            evaluation.latencies_ms.append(body["latency_ms"])

    if replayed:
        logger.info(
            "%s: %d/%d documents served from the request log",
            engine,
            replayed,
            len(dataset),
        )
    evaluation.replayed = replayed
    return evaluation


def run_benchmark(
    dataset_path: Path,
    engines: list[str] | None = None,
    api_url: str = "http://localhost:8000",
    mode: MatchMode = "strict",
    experiment: str = "pii-anonymisation",
    tracking_uri: str | None = None,
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
    log_errors: bool = True,
    source: str | None = None,
) -> dict[str, Evaluation]:
    """Run every engine over the dataset. One MLflow run per engine.

    Returns ``{engine: Evaluation}``.

    Deduplication happens in the service, keyed on doc_id + engine + prompt,
    so a re-run does not repeat inference. Editing a prompt invalidates only
    that engine's results; deleting the request log clears everything.
    """
    from . import tracking

    dataset_path = Path(dataset_path)
    dataset = load_dataset(dataset_path, source=source)
    engines = engines or available_engines(api_url)
    prompts = engine_prompts(api_url)

    tracking.setup(experiment=experiment, tracking_uri=tracking_uri)
    dataset_name = source or dataset_path.stem

    results: dict[str, Evaluation] = {}
    failures: dict[str, Exception] = {}

    with tracking.benchmark_run(
        run_name or f"{dataset_name}-{mode}",
        tags={"dataset": dataset_name, "match_mode": mode},
    ):
        for engine in engines:
            logger.info("running %s", engine)
            # One engine failing - a bad response shape, a scoring bug on one
            # document, an MLflow write error - must not discard runs that
            # already completed and logged successfully. Each engine gets its
            # own try, and the loop always continues to the next one.
            try:
                evaluation = run_engine(
                    dataset, engine, dataset_name, api_url=api_url, mode=mode
                )
                tracking.log_evaluation(
                    evaluation,
                    params={"documents": len(dataset), **(params or {})},
                    prompt=prompts.get(engine),
                    log_errors=log_errors,
                    run_name=engine,
                )
                results[engine] = evaluation
            except Exception as exc:
                logger.exception("engine %s failed, continuing", engine)
                failures[engine] = exc

    if failures:
        print(f"\n{len(failures)} engine(s) failed and were skipped:")
        for engine, exc in failures.items():
            print(f"   {engine}: {type(exc).__name__}: {exc}")
        print(f"completed: {list(results)}")

    return results