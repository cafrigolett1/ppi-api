"""Fixtures. No test loads spaCy or calls an API."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import database  # noqa: E402
from app.config import Settings  # noqa: E402
from app.entities import Entity  # noqa: E402
from app.main import create_app  # noqa: E402
from app.service import AnonymizationService  # noqa: E402


class FakeDetector:
    """Finds fixed surface forms. Can be told to fail."""

    name = "fake"

    def __init__(self, matches: dict[str, str] | None = None) -> None:
        self.matches = matches or {"Marie": "PERSON", "Ordina": "ORG"}
        self.fail_on: set[str] = set()
        self.calls = 0

    def detect(self, text: str) -> list[Entity]:
        self.calls += 1
        if any(token in text for token in self.fail_on):
            raise RuntimeError("detector exploded")
        found = []
        for surface, label in self.matches.items():
            start = text.find(surface)
            if start != -1:
                found.append(
                    Entity(
                        start=start,
                        end=start + len(surface),
                        label=label,
                        text=surface,
                        source=self.name,
                    )
                )
        return found


class FakeService(AnonymizationService):
    """Service with engines injected instead of loaded."""

    def __init__(self, settings: Settings, detector: FakeDetector) -> None:
        self.settings = settings
        from app.tokens import HeuristicTokenCounter

        self.tokens = HeuristicTokenCounter(settings.chars_per_token)
        self.engines = {"presidio-be": detector}

    def _load_engines(self) -> None:  # never loads a real model
        return


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="dev",
        default_engine="presidio-be",
        database_url="sqlite://",  # in-memory
        data_dir=tmp_path,
        max_input_tokens=100,
        max_batch_size=5,
        max_batch_tokens=300,
    )


@pytest.fixture
def detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def client(settings: Settings, detector: FakeDetector):
    """Client backed by a shared in-memory SQLite database.

    StaticPool keeps every connection pointed at the same in-memory database;
    without it each connection gets its own and the tables vanish between
    calls.
    """
    database.reset_engine()
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    database._engine = engine
    SQLModel.metadata.create_all(engine)

    app = create_app(settings=settings, service=FakeService(settings, detector))
    with TestClient(app) as running:
        yield running

    database.reset_engine()


@pytest.fixture
def session(client) -> Session:
    """A session on the same database the client writes to."""
    with Session(database._engine) as s:
        yield s
