"""Streamlit demo for the anonymisation API.

Deliberately a thin client: it holds no detection logic and imports nothing
from ``app/``. It talks to the running service over HTTP exactly as any other
caller would, so what you see in the demo is what the API actually returns.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT = 120

EXAMPLE = (
    "Marie Dubois joined Ordina Belgium as a Data Scientist in 2019. "
    "Her email is marie.dubois@ordina.be and her phone is 015 29 58 58. "
    "Her IBAN is BE68 5390 0754 7034 and her national number is 82.05.30-025.56."
)

st.set_page_config(page_title="PII Anonymisation", page_icon="🔒", layout="wide")
st.title("PII Anonymisation")
st.caption(f"Calling `{API_URL}`")


def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return None

    if r.status_code >= 400:
        detail = r.json().get("detail", r.text)
        st.error(f"{r.status_code}: {detail}")
        return None
    return r.json()


# -- sidebar ---------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    engines_info = api_get("/v1/engines") or []
    available = [e["name"] for e in engines_info if e["available"]]

    if not available:
        st.warning("No engines available. Is the API running?")
        engine = None
    else:
        engine = st.selectbox("Engine", available)
        description = next(
            (e["description"] for e in engines_info if e["name"] == engine), ""
        )
        st.caption(description)

    show_entities = st.checkbox("Show detected spans", value=True)

    unavailable = [e["name"] for e in engines_info if not e["available"]]
    if unavailable:
        st.caption(f"Unavailable: {', '.join(unavailable)}")

    health = api_get("/health")
    if health:
        st.success(f"API ok · db {health['database']}")


tab_single, tab_batch, tab_activity = st.tabs(["Single", "Batch", "Activity"])


# -- single ----------------------------------------------------------------

with tab_single:
    text = st.text_area("Text to anonymise", value=EXAMPLE, height=160)

    if st.button("Anonymise", type="primary", disabled=not engine):
        result = api_post(
            "/v1/anonymize",
            {"text": text, "engine": engine, "include_entities": show_entities},
        )
        if result:
            st.subheader("Redacted")
            st.code(result["redacted_text"], language=None)

            a, b, c = st.columns(3)
            a.metric("Entities", result["entity_count"])
            b.metric("Input tokens", result["input_tokens"])
            c.metric("Latency", f"{result['latency_ms']:.0f} ms")

            if result["counts_by_label"]:
                st.bar_chart(result["counts_by_label"])

            if result.get("entities"):
                st.subheader("Detected spans")
                st.dataframe(result["entities"], use_container_width=True)


# -- batch -----------------------------------------------------------------

with tab_batch:
    st.caption("One document per line. Each line is a separate model call.")
    lines = st.text_area(
        "Documents",
        value="Marie Dubois works at Ordina.\n"
        "Call 015 29 58 58 for details.\n"
        "Nothing sensitive in this sentence.",
        height=160,
    )

    if st.button("Anonymise batch", type="primary", disabled=not engine):
        items = [
            {"id": f"line-{i + 1}", "text": line.strip()}
            for i, line in enumerate(lines.splitlines())
            if line.strip()
        ]
        if not items:
            st.warning("Nothing to send.")
        else:
            result = api_post("/v1/anonymize/batch", {"items": items, "engine": engine})
            if result:
                a, b, c = st.columns(3)
                a.metric("Succeeded", result["succeeded"])
                b.metric("Failed", result["failed"])
                c.metric("Total tokens", result["total_input_tokens"])

                st.dataframe(
                    [
                        {
                            "id": r["id"],
                            "status": r["status"],
                            "output": r["redacted_text"] or r["error"],
                            "entities": r["entity_count"],
                            "ms": r["latency_ms"],
                        }
                        for r in result["results"]
                    ],
                    use_container_width=True,
                )


# -- activity --------------------------------------------------------------

with tab_activity:
    if st.button("Refresh"):
        st.rerun()

    stats = api_get("/v1/stats")
    if stats:
        a, b, c, d = st.columns(4)
        a.metric("Calls", stats["total_requests"])
        b.metric("Failed", stats["failed"])
        c.metric("Tokens in", stats["total_input_tokens"])
        d.metric(
            "p95 latency",
            f"{stats['p95_latency_ms']:.0f} ms" if stats["p95_latency_ms"] else "—",
        )

    logs = api_get("/v1/logs?limit=50")
    if logs:
        st.dataframe(logs, use_container_width=True)
    elif logs == []:
        st.info("No calls recorded yet.")
