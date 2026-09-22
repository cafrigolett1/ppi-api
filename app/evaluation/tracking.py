"""MLflow tracking for anonymisation benchmarks.

Each engine-on-a-dataset becomes one run, nested under a parent run for the
whole benchmark, so MLflow's compare view lines the engines up side by side.

Error artefacts include the **source document** alongside each miss, because a
label and an offset alone rarely explain why something failed - you need to
see the sentence it sat in.

That means the artefact contains the evaluation text in full. For the
synthetic evaluation data here that is fine. Pointed at real documents it
would put a copy of the material in MLflow's artefact store, so
``redact_error_text=True`` drops both the entity text and the document while
keeping labels and offsets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .metrics import Evaluation, per_label_table, summary_metrics

logger = logging.getLogger(__name__)


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def setup(
    experiment: str = "pii-anonymisation",
    tracking_uri: str | None = None,
    artifact_location: str | Path | None = None,
) -> None:
    """Point MLflow at a tracking store and select the experiment.

    Defaults to a local SQLite file rather than the classic ``./mlruns``
    directory: as of MLflow 3.x the filesystem backend is in maintenance mode
    and raises unless ``MLFLOW_ALLOW_FILE_STORE=true`` is set. SQLite needs no
    server and supports the current feature set.

    View it with::

        mlflow ui --backend-store-uri sqlite:///mlflow.db

    Pass an ``http://`` URI for a shared tracking server.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)

    # Pin the artifact location explicitly. It otherwise defaults to ./mlruns
    # relative to the *working directory of the logging process* - so a
    # notebook whose kernel starts in notebooks/ writes its artifacts to
    # notebooks/mlruns while the tracking database sits at the project root.
    # Two stores, one of which you will not think to delete.
    if artifact_location is None and tracking_uri and tracking_uri.startswith("sqlite:"):
        db_path = Path(tracking_uri.replace("sqlite:///", "").replace("sqlite://", ""))
        if db_path.name:
            artifact_location = db_path.resolve().parent / "mlartifacts"

    existing = mlflow.get_experiment_by_name(experiment)
    if existing is None and artifact_location is not None:
        location = Path(artifact_location).resolve()
        location.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(experiment, artifact_location=location.as_uri())

    mlflow.set_experiment(experiment)


@contextmanager
def benchmark_run(name: str, tags: dict[str, Any] | None = None):
    """Parent run grouping one benchmark's engine runs."""
    import mlflow

    with mlflow.start_run(run_name=name) as run:
        base_tags = {"run_type": "benchmark"}
        if sha := _git_sha():
            base_tags["git_sha"] = sha
        mlflow.set_tags({**base_tags, **(tags or {})})
        yield run


def log_evaluation(
    evaluation: Evaluation,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    nested: bool = True,
    log_errors: bool = True,
    redact_error_text: bool = False,
    prompt: str | None = None,
    run_name: str | None = None,
) -> str:
    """Log one engine's evaluation as an MLflow run. Returns the run id.

    ``params`` should carry everything needed to reproduce the run - model
    version, thresholds, dataset. Anything that changes the numbers and is not
    recorded makes the run unreproducible.

    ``prompt`` is logged as an artefact and fingerprinted into a ``prompt_hash``
    param. MLflow truncates params at 500 characters, so the prompt itself
    cannot be one - but the hash makes runs comparable at a glance, and the
    artefact holds the text.
    """
    import mlflow

    name = run_name or f"{evaluation.engine}-{evaluation.dataset}"

    with mlflow.start_run(run_name=name, nested=nested) as run:
        mlflow.log_params(
            {
                "engine": evaluation.engine,
                "dataset": evaluation.dataset,
                "match_mode": evaluation.mode,
                **(params or {}),
            }
        )

        run_tags = {"engine": evaluation.engine, "dataset": evaluation.dataset}
        if sha := _git_sha():
            run_tags["git_sha"] = sha
        mlflow.set_tags({**run_tags, **(tags or {})})

        if prompt:
            mlflow.log_param(
                "prompt_hash", hashlib.sha256(prompt.encode()).hexdigest()[:12]
            )
            mlflow.log_param("prompt_chars", len(prompt))
            mlflow.log_text(prompt, "prompt.txt")

        mlflow.log_metrics(summary_metrics(evaluation))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            (tmp_path / "per_label.json").write_text(
                json.dumps(per_label_table(evaluation), indent=2), encoding="utf-8"
            )
            mlflow.log_artifact(str(tmp_path / "per_label.json"))

            if log_errors:
                errors = []
                for d in evaluation.documents:
                    if not (d.missed or d.spurious):
                        continue
                    entry: dict[str, Any] = {
                        "doc_id": d.doc_id,
                        # The sentence the failure happened in. A label and an
                        # offset on their own rarely explain the miss.
                        "input": None if redact_error_text else d.text,
                        "missed": [
                            {
                                "label": e.label,
                                "start": e.start,
                                "end": e.end,
                                "text": None if redact_error_text else e.text,
                            }
                            for e in d.missed
                        ],
                        "spurious": [
                            {
                                "label": e.label,
                                "start": e.start,
                                "end": e.end,
                                "text": None if redact_error_text else e.text,
                            }
                            for e in d.spurious
                        ],
                    }
                    errors.append(entry)

                (tmp_path / "errors.json").write_text(
                    json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                mlflow.log_artifact(str(tmp_path / "errors.json"))

        return run.info.run_id
