"""Entity recognition performance dashboard.

Joins logged predictions against ground truth to answer the question an
anonymisation service actually needs answered: which model finds which
entities correctly, and where does each one fail.

The join key is `doc_id`. A logged request only has a ground-truth match when
its `doc_id` corresponds to a record in the evaluation dataset - which is
exactly what happens when the benchmark notebook or the seeded traffic below
runs requests through the API with `doc_id` set to the record's `id`. Traffic
with no matching gold entry (a real caller's `doc_id`, or none at all) is
excluded from accuracy and shown separately as "unscored" so the two are never
silently mixed.

Scoring reuses `app.evaluation.metrics` rather than reimplementing it, so the
numbers here agree with what the benchmark notebook reports for the same data.

    uv run streamlit run dashboard.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.entities import Entity  # noqa: E402
from app.evaluation.metrics import (  # noqa: E402
    Evaluation,
    score_document,
    summary_metrics,
)

DB_PATH = Path("data/requests.db")
GOLD_PATH = Path("data/evaluation.jsonl")

st.set_page_config(
    page_title="Entity Recognition Performance", page_icon="🎯", layout="wide"
)


# -- data loading -------------------------------------------------------


@st.cache_data(ttl=30)
def load_gold(path: str) -> dict[str, dict]:
    """doc_id -> {text, source, entities} from the evaluation dataset."""
    if not Path(path).exists():
        return {}
    gold = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            gold[record["id"]] = record
    return gold


@st.cache_data(ttl=30)
def load_logs(db_path: str) -> pd.DataFrame:
    """Successful, non-replayed requests with a doc_id.

    Replays are excluded: they carry the original prediction, which is
    already counted once on the row that actually ran inference, and their
    latency (irrelevant here, but still true of the row) is not a fresh
    measurement either.
    """
    if not Path(db_path).exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """
            SELECT doc_id, engine, prompt_id, created_at, entities_json,
                   entity_count, status, latency_ms
            FROM request_logs
            WHERE status = 'ok'
              AND served_from_log_id IS NULL
              AND doc_id IS NOT NULL
            """,
            conn,
        )
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df["prompt_short"] = df["prompt_id"].apply(
            lambda p: p[:8] if isinstance(p, str) else "—"
        )
    return df


def score_engine(rows: pd.DataFrame, gold: dict[str, dict], mode: str) -> Evaluation:
    """Score one engine's predictions against gold for whichever of its rows
    have a matching doc_id. Documents scored more than once (re-run after a
    prompt edit) use only the most recent row, so a stale prediction from
    before a fix does not sit in the average alongside the corrected one."""
    latest = rows.sort_values("created_at").groupby("doc_id").tail(1)
    engine_name = rows["engine"].iloc[0] if len(rows) else "unknown"
    evaluation = Evaluation(engine=engine_name, dataset="request_logs", mode=mode)

    for _, row in latest.iterrows():
        record = gold.get(row["doc_id"])
        if record is None:
            continue
        gold_entities = [
            Entity(start=e["start"], end=e["end"], label=e["label"], source="gold")
            for e in record["entities"]
        ]
        predicted = [
            Entity(
                start=e["start"],
                end=e["end"],
                label=e["label"],
                text=e.get("text", ""),
                score=e.get("score", 1.0),
                source=engine_name,
            )
            for e in json.loads(row["entities_json"] or "[]")
        ]
        score_document(
            evaluation, row["doc_id"], record["text"], predicted, gold_entities
        )

    return evaluation


# -- app --------------------------------------------------------------------

st.title("🎯 Entity Recognition Performance")

logs = load_logs(str(DB_PATH))
gold = load_gold(str(GOLD_PATH))

if logs.empty:
    st.warning(
        f"No logged requests at `{DB_PATH}`. Run some documents through the "
        "API first - the benchmark notebook is the fastest way to populate "
        "this."
    )
    st.stop()

if not gold:
    st.warning(
        f"No ground truth at `{GOLD_PATH}`. Accuracy scoring needs it; build "
        "it with `uv run scripts/build_dataset.py`."
    )
    st.stop()

logs["has_gold"] = logs["doc_id"].isin(gold)
scored_logs = logs[logs["has_gold"]]
unscored_count = len(logs) - len(scored_logs)

if scored_logs.empty:
    st.warning(
        "None of the logged requests have a doc_id matching the evaluation "
        "dataset, so nothing can be scored. This happens when requests were "
        "made without `doc_id`, or with ids that don't correspond to "
        "`data/evaluation.jsonl` records."
    )
    st.stop()

st.caption(
    f"{len(scored_logs)} scored requests across "
    f"{scored_logs['doc_id'].nunique()} documents"
    + (
        f" · {unscored_count} unscored (no matching doc_id) excluded"
        if unscored_count
        else ""
    )
)

# -- sidebar --------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    engines = sorted(scored_logs["engine"].unique())
    selected_engines = st.multiselect("Engines", engines, default=engines)

    sources = sorted({gold[d]["source"] for d in scored_logs["doc_id"] if d in gold})
    selected_sources = st.multiselect(
        "Dataset subset",
        sources,
        default=sources,
        help="The evaluation set mixes subsets with different label mixes "
        "(provided / belgian / edge). A pooled score across them is "
        "misleading, so filter to one to read it cleanly.",
    )

    mode = st.radio(
        "Match mode",
        ["strict", "partial"],
        help="Strict: exact boundaries. Partial: any overlap with the "
        "correct label counts. The gap between them separates boundary "
        "errors from real misses.",
    )

filtered_doc_ids = {
    d
    for d in scored_logs["doc_id"].unique()
    if d in gold and gold[d]["source"] in selected_sources
}
working = scored_logs[
    scored_logs["engine"].isin(selected_engines)
    & scored_logs["doc_id"].isin(filtered_doc_ids)
]

if working.empty:
    st.info("No scored requests match the current filters.")
    st.stop()

# -- score every engine -----------------------------------------------------

evaluations: dict[str, Evaluation] = {}
for engine in selected_engines:
    rows = working[working["engine"] == engine]
    if not rows.empty:
        evaluations[engine] = score_engine(rows, gold, mode)

tab_compare, tab_labels, tab_errors, tab_ops = st.tabs(
    ["Model comparison", "Per-label breakdown", "Missed & spurious", "Volume & latency"]
)

# -- model comparison -------------------------------------------------------

with tab_compare:
    st.subheader(f"Headline metrics ({mode} matching)")
    st.caption(
        "Document leakage rate is the fraction of documents still containing "
        "at least one unredacted entity - the number that decides whether "
        "output is safe to release, and not derivable from span recall alone."
    )

    rows = []
    for engine, ev in evaluations.items():
        m = summary_metrics(ev)
        rows.append(
            {
                "engine": engine,
                "documents": int(m["documents"]),
                "doc leakage": m["document_leakage_rate"],
                "direct-ID recall": m["direct_identifier_recall"],
                "precision": m["micro_precision"],
                "recall": m["micro_recall"],
                "F2": m["micro_f2"],
                "over-redaction/doc": m["over_redaction_per_doc"],
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("doc leakage")
    st.dataframe(
        summary_df.style.format(
            {c: "{:.2f}" for c in summary_df.columns if c not in ("engine", "documents")}
        ),
        width="stretch",
        hide_index=True,
    )

    if len(summary_df) > 1:
        metric_choice = st.selectbox(
            "Chart metric",
            ["doc leakage", "direct-ID recall", "precision", "recall", "F2"],
        )
        fig = px.bar(summary_df, x="engine", y=metric_choice, title=metric_choice)
        st.plotly_chart(fig, width="stretch")

# -- per-label breakdown ----------------------------------------------------

with tab_labels:
    st.subheader("Recall by entity label")
    st.caption(
        "Where the real differences live. `ORG`, `JOB`, `UNIVERSITY` and "
        "`AMOUNT` are outside Presidio's default entity set and score 0.00 "
        "there regardless of the Belgian checksum recognisers - this is "
        "where the LLM engines are expected to win."
    )

    recall_table = pd.DataFrame(
        {
            engine: {label: s.recall for label, s in ev.per_label.items() if s.support}
            for engine, ev in evaluations.items()
        }
    ).sort_index()
    if not recall_table.empty:
        st.dataframe(
            recall_table.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1).format(
                "{:.2f}"
            ),
            width="stretch",
        )

    st.subheader("Precision by entity label")
    st.caption("Catches an engine buying recall with over-redaction.")
    precision_table = pd.DataFrame(
        {
            engine: {label: s.precision for label, s in ev.per_label.items() if s.support}
            for engine, ev in evaluations.items()
        }
    ).sort_index()
    if not precision_table.empty:
        st.dataframe(
            precision_table.style.background_gradient(
                cmap="RdYlGn", vmin=0, vmax=1
            ).format("{:.2f}"),
            width="stretch",
        )

    st.subheader("Support (gold spans per label)")
    support = {
        label: s.support
        for ev in evaluations.values()
        for label, s in ev.per_label.items()
        if s.support
    }
    if support:
        support_df = pd.DataFrame(
            sorted(support.items(), key=lambda x: -x[1]), columns=["label", "gold spans"]
        )
        fig = px.bar(
            support_df, x="label", y="gold spans", title="Ground-truth spans per label"
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "A label with very few gold spans gives an unstable recall "
            "figure - one miss can swing it by a large margin."
        )

# -- missed & spurious -------------------------------------------------

with tab_errors:
    st.subheader("What each engine gets wrong")
    engine_choice = st.selectbox("Engine", list(evaluations), key="error_engine")
    ev = evaluations[engine_choice]

    leaked = [d for d in ev.documents if d.leaked]
    over_redacted = [d for d in ev.documents if d.spurious]

    c1, c2 = st.columns(2)
    c1.metric("Documents with a miss", len(leaked))
    c2.metric("Documents with a spurious span", len(over_redacted))

    st.markdown("**Missed entities** (false negatives)")
    missed_rows = [
        {"doc_id": d.doc_id, "label": e.label, "text": e.text, "context": d.text[:80]}
        for d in ev.documents
        for e in d.missed
    ]
    if missed_rows:
        missed_df = pd.DataFrame(missed_rows)
        label_filter = st.multiselect(
            "Filter by label",
            sorted(missed_df["label"].unique()),
            key="missed_label_filter",
        )
        if label_filter:
            missed_df = missed_df[missed_df["label"].isin(label_filter)]
        st.dataframe(missed_df, width="stretch", hide_index=True)
    else:
        st.success("No misses for this engine under the current filters.")

    st.markdown("**Spurious entities** (false positives)")
    spurious_rows = [
        {"doc_id": d.doc_id, "label": e.label, "text": e.text, "context": d.text[:80]}
        for d in ev.documents
        for e in d.spurious
    ]
    if spurious_rows:
        st.dataframe(pd.DataFrame(spurious_rows), width="stretch", hide_index=True)
    else:
        st.success("No spurious detections for this engine under the current filters.")

    if unscored_count:
        st.divider()
        st.caption(
            f"{unscored_count} logged request(s) have no matching doc_id in "
            "the evaluation dataset and are excluded from every number on "
            "this page."
        )

# -- volume & latency (secondary) --------------------------------------

with tab_ops:
    st.subheader("Request volume")
    st.caption(
        "Operational context, not the focus of this dashboard - kept here "
        "for reference. Only scored (doc_id-matched) requests are shown."
    )

    hourly = (
        working.set_index("created_at")
        .groupby("engine")
        .resample("h")
        .size()
        .reset_index(name="requests")
    )
    fig = px.line(hourly, x="created_at", y="requests", color="engine", markers=True)
    fig.update_layout(xaxis_title="Hour (UTC)", yaxis_title="Requests")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Latency by engine")
    fig2 = px.box(working, x="engine", y="latency_ms", points="outliers")
    st.plotly_chart(fig2, width="stretch")

    label_counts: Counter = Counter()
    for ev in evaluations.values():
        for label, s in ev.per_label.items():
            label_counts[label] += s.true_positives
    if label_counts:
        st.subheader("Correctly detected entities, by label")
        det_df = pd.DataFrame(
            sorted(label_counts.items(), key=lambda x: -x[1]), columns=["label", "count"]
        )
        st.plotly_chart(px.bar(det_df, x="label", y="count"), width="stretch")