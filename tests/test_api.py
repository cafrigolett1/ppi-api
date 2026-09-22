"""HTTP behaviour and persistence."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlmodel import select

from app.models import RequestLog


def test_anonymize_single(client: TestClient):
    r = client.post("/v1/anonymize", json={"text": "Marie works at Ordina."})
    assert r.status_code == 200

    body = r.json()
    assert body["redacted_text"] == "<PERSON> works at <ORG>."
    assert body["counts_by_label"] == {"PERSON": 1, "ORG": 1}
    assert body["input_tokens"] > 0
    assert body["request_id"]
    assert body["entities"] is None


def test_entities_are_opt_in(client: TestClient):
    body = client.post(
        "/v1/anonymize", json={"text": "Marie here.", "include_entities": True}
    ).json()
    assert body["entities"] == [
        {"start": 0, "end": 5, "label": "PERSON", "text": "Marie", "score": 1.0}
    ]


def test_every_call_is_recorded(client: TestClient, session):
    client.post("/v1/anonymize", json={"text": "Marie works at Ordina."})

    row = session.exec(select(RequestLog)).one()
    assert row.endpoint == "/v1/anonymize"
    assert row.engine == "presidio-be"
    assert row.status == "ok"
    assert row.entity_count == 2
    assert row.input_tokens > 0
    assert row.input_chars == len("Marie works at Ordina.")
    assert row.latency_ms >= 0
    assert row.created_at is not None
    assert row.input_text == "Marie works at Ordina."
    assert row.output_text == "<PERSON> works at <ORG>."
    assert row.error_type is None


def test_input_text_is_not_stored_when_disabled(settings, detector, tmp_path):
    """The stored input is the sensitive data; it must be possible to opt out."""
    from fastapi.testclient import TestClient as TC
    from sqlmodel import Session, SQLModel, create_engine
    from sqlmodel.pool import StaticPool

    from app import database
    from app.main import create_app
    from tests.conftest import FakeService

    database.reset_engine()
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    database._engine = engine
    SQLModel.metadata.create_all(engine)

    private = settings.model_copy(update={"store_input_text": False})
    app = create_app(settings=private, service=FakeService(private, detector))

    with TC(app) as c:
        c.post("/v1/anonymize", json={"text": "Marie works at Ordina."})

    with Session(engine) as s:
        row = s.exec(select(RequestLog)).one()
        assert row.input_text is None
        assert row.output_text == "<PERSON> works at <ORG>."
        assert row.input_chars == 22  # length still recorded
    database.reset_engine()


def test_failure_is_recorded_with_the_error(client: TestClient, detector, session):
    detector.fail_on = {"POISON"}
    r = client.post("/v1/anonymize", json={"text": "POISON here."})
    assert r.status_code == 502

    row = session.exec(select(RequestLog)).one()
    assert row.status == "error"
    assert row.error_type == "RuntimeError"
    assert "exploded" in row.error_message


def test_oversized_input_is_413_and_recorded(client: TestClient, session):
    r = client.post("/v1/anonymize", json={"text": "word " * 500})
    assert r.status_code == 413
    assert "tokens" in r.json()["detail"]

    row = session.exec(select(RequestLog)).one()
    assert row.status == "error"
    assert row.error_type == "InputTooLarge"


def test_unavailable_engine_is_503(client: TestClient):
    r = client.post("/v1/anonymize", json={"text": "Marie", "engine": "llm-zero-shot"})
    assert r.status_code == 503


def test_empty_text_rejected(client: TestClient):
    assert client.post("/v1/anonymize", json={"text": ""}).status_code == 422


# -- batch ----------------------------------------------------------------


def test_batch_returns_results_in_order(client: TestClient):
    items = [{"id": f"d{i}", "text": f"Doc {i} names Marie."} for i in range(4)]
    body = client.post("/v1/anonymize/batch", json={"items": items}).json()

    assert [r["id"] for r in body["results"]] == ["d0", "d1", "d2", "d3"]
    assert body["succeeded"] == 4
    assert body["failed"] == 0
    assert body["total_input_tokens"] > 0


def test_batch_partial_failure_is_200(client: TestClient, detector):
    detector.fail_on = {"POISON"}
    body = client.post(
        "/v1/anonymize/batch",
        json={
            "items": [
                {"id": "good", "text": "Marie is here."},
                {"id": "bad", "text": "POISON."},
            ]
        },
    ).json()

    by_id = {r["id"]: r for r in body["results"]}
    assert by_id["good"]["status"] == "ok"
    assert by_id["bad"]["status"] == "error"
    assert body["succeeded"] == 1 and body["failed"] == 1


def test_batch_writes_one_row_per_document(client: TestClient, session):
    items = [{"id": f"d{i}", "text": "Marie"} for i in range(3)]
    client.post("/v1/anonymize/batch", json={"items": items})

    rows = list(session.exec(select(RequestLog)))
    assert len(rows) == 3
    assert {r.doc_id for r in rows} == {"d0", "d1", "d2"}
    assert len({r.request_id for r in rows}) == 1  # shared correlation id


def test_batch_size_limit(client: TestClient):
    items = [{"id": str(i), "text": "Marie"} for i in range(6)]
    assert client.post("/v1/anonymize/batch", json={"items": items}).status_code == 413


def test_batch_token_budget_checked_before_any_work(client: TestClient, detector):
    items = [{"id": str(i), "text": "word " * 90} for i in range(5)]
    r = client.post("/v1/anonymize/batch", json={"items": items})
    assert r.status_code == 413
    assert detector.calls == 0


# -- observability --------------------------------------------------------


def test_logs_endpoint(client: TestClient):
    client.post("/v1/anonymize", json={"text": "Marie"})
    client.post("/v1/anonymize", json={"text": "Ordina"})

    rows = client.get("/v1/logs").json()
    assert len(rows) == 2
    assert rows[0]["id"] > rows[1]["id"]  # newest first
    assert "input_text" not in rows[0]  # payloads not exposed here


def test_logs_can_filter_by_status(client: TestClient, detector):
    detector.fail_on = {"POISON"}
    client.post("/v1/anonymize", json={"text": "Marie"})
    client.post("/v1/anonymize", json={"text": "POISON"})

    assert len(client.get("/v1/logs?status=error").json()) == 1
    assert len(client.get("/v1/logs?status=ok").json()) == 1


def test_stats(client: TestClient, detector):
    client.post("/v1/anonymize", json={"text": "Marie"})
    detector.fail_on = {"POISON"}
    client.post("/v1/anonymize", json={"text": "POISON"})

    stats = client.get("/v1/stats").json()
    assert stats["total_requests"] == 2
    assert stats["succeeded"] == 1
    assert stats["failed"] == 1
    assert stats["total_input_tokens"] > 0
    assert stats["by_engine"]["presidio-be"] == 2


def test_stats_on_empty_database(client: TestClient):
    stats = client.get("/v1/stats").json()
    assert stats["total_requests"] == 0
    assert stats["avg_latency_ms"] is None


def test_health(client: TestClient):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert "presidio-be" in body["engines"]


def test_engines_endpoint(client: TestClient):
    engines = {e["name"]: e for e in client.get("/v1/engines").json()}
    assert engines["presidio-be"]["available"] is True
    assert engines["llm-zero-shot"]["available"] is False


# -- deduplication --------------------------------------------------------


def test_repeat_with_same_doc_id_is_served_from_the_log(client: TestClient, detector):
    first = client.post(
        "/v1/anonymize", json={"text": "Marie works at Ordina.", "doc_id": "d1"}
    ).json()
    assert first["cached"] is False
    assert detector.calls == 1

    second = client.post(
        "/v1/anonymize", json={"text": "Marie works at Ordina.", "doc_id": "d1"}
    ).json()
    assert second["cached"] is True
    assert second["redacted_text"] == first["redacted_text"]
    assert second["served_from_log_id"] is not None
    assert second["served_from_timestamp"] is not None
    assert detector.calls == 1  # inference not repeated


def test_replay_returns_the_same_entities(client: TestClient):
    body = {"text": "Marie works at Ordina.", "doc_id": "d1", "include_entities": True}
    first = client.post("/v1/anonymize", json=body).json()
    second = client.post("/v1/anonymize", json=body).json()
    assert second["cached"] is True
    assert second["entities"] == first["entities"]


def test_without_doc_id_nothing_is_deduplicated(client: TestClient, detector):
    for _ in range(3):
        body = client.post("/v1/anonymize", json={"text": "Marie here."}).json()
        assert body["cached"] is False
    assert detector.calls == 3


def test_replay_writes_a_row_pointing_at_the_original(client: TestClient, session):
    client.post("/v1/anonymize", json={"text": "Marie here.", "doc_id": "d1"})
    client.post("/v1/anonymize", json={"text": "Marie here.", "doc_id": "d1"})

    rows = list(session.exec(select(RequestLog).order_by(RequestLog.id)))
    assert len(rows) == 2  # the replay is recorded, not silently dropped
    assert rows[0].served_from_log_id is None
    assert rows[1].served_from_log_id == rows[0].id
    assert rows[1].served_from_timestamp == rows[0].created_at


def test_different_doc_id_is_not_deduplicated(client: TestClient, detector):
    client.post("/v1/anonymize", json={"text": "Marie here.", "doc_id": "d1"})
    body = client.post(
        "/v1/anonymize", json={"text": "Marie here.", "doc_id": "d2"}
    ).json()
    assert body["cached"] is False
    assert detector.calls == 2


def test_changed_text_under_the_same_doc_id_is_not_replayed(client: TestClient, detector):
    """The dedupe key ignores text by design, so the stored hash is the only
    guard against a document changing under a reused id."""
    client.post("/v1/anonymize", json={"text": "Marie here.", "doc_id": "d1"})
    body = client.post(
        "/v1/anonymize", json={"text": "Ordina here instead.", "doc_id": "d1"}
    ).json()

    assert body["cached"] is False
    assert body["redacted_text"] == "<ORG> here instead."
    assert detector.calls == 2


def test_failures_are_not_replayed(client: TestClient, detector, session):
    detector.fail_on = {"POISON"}
    assert (
        client.post("/v1/anonymize", json={"text": "POISON", "doc_id": "d1"}).status_code
        == 502
    )

    detector.fail_on = set()
    body = client.post("/v1/anonymize", json={"text": "POISON", "doc_id": "d1"}).json()
    assert body["cached"] is False  # an error row is never reused


def test_batch_deduplicates_per_item(client: TestClient, detector):
    items = [{"id": "d1", "text": "Marie here."}, {"id": "d2", "text": "Ordina here."}]
    client.post("/v1/anonymize/batch", json={"items": items})
    assert detector.calls == 2

    body = client.post("/v1/anonymize/batch", json={"items": items}).json()
    assert all(r["cached"] for r in body["results"])
    assert detector.calls == 2


def test_stats_separates_work_from_reuse(client: TestClient):
    client.post("/v1/anonymize", json={"text": "Marie here.", "doc_id": "d1"})
    client.post("/v1/anonymize", json={"text": "Marie here.", "doc_id": "d1"})
    client.post("/v1/anonymize", json={"text": "Ordina here.", "doc_id": "d2"})

    stats = client.get("/v1/stats").json()
    assert stats["total_requests"] == 3
    assert stats["inferences_run"] == 2
    assert stats["served_from_cache"] == 1
