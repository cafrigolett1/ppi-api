"""Entity representation, overlap resolution and masking.

Spans are the ground truth everywhere in this package. Masked text is only
ever a *rendering* of spans, never the source: masked strings cannot be scored
per entity, cannot express overlaps, and cannot be reversed.
"""

from __future__ import annotations

from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

Label = Literal[
    "PERSON",
    "ORG",
    "JOB",
    "EMAIL_ADDRESS",
    "LOCATION",
    "AMOUNT",
    "DATE_TIME",
    "UNIVERSITY",
    "PHONE_NUMBER",
    "URL",
    "IBAN",
    "SSN",
]

REQUIRED_LABELS: tuple[str, ...] = (
    "PERSON",
    "ORG",
    "JOB",
    "EMAIL_ADDRESS",
    "LOCATION",
    "AMOUNT",
    "DATE_TIME",
    "UNIVERSITY",
    "PHONE_NUMBER",
    "URL",
    "IBAN",
    "SSN",
)

# Labels whose surface form is structural and checksum-verifiable. These win
# ties during resolution: arithmetic beats inference.
DETERMINISTIC_LABELS: frozenset[str] = frozenset({"IBAN", "SSN"})

# Preferred label when detectors claim overlapping spans. More specific labels
# outrank the general ones they are a subtype of.
LABEL_PRIORITY: dict[str, int] = {
    "IBAN": 100,
    "SSN": 100,
    "EMAIL_ADDRESS": 95,
    "PHONE_NUMBER": 90,
    "URL": 85,
    "AMOUNT": 80,
    "UNIVERSITY": 75,  # more specific than ORG
    "DATE_TIME": 70,
    "JOB": 65,
    "PERSON": 60,
    "ORG": 55,
    "LOCATION": 50,
}


class Entity(BaseModel):
    """A labelled character span.

    Frozen so that a detector cannot mutate another detector's output during
    resolution, and so entities can be compared by value in the scorer.
    """

    model_config = ConfigDict(frozen=True)

    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]
    label: Label
    text: str = ""
    score: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    source: str = ""
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_span(self) -> Entity:
        if self.end <= self.start:
            raise ValueError(f"empty or inverted span: [{self.start}, {self.end})")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Entity) -> bool:
        return self.start < other.end and other.start < self.end

    def same_span(self, other: Entity) -> bool:
        return self.start == other.start and self.end == other.end


class GoldEntity(BaseModel):
    """A span as stored in a dataset file."""

    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]
    label: Label


class GoldDocument(BaseModel):
    """One annotated document. Validates that every span is inside the text."""

    id: str
    text: str = Field(min_length=1)
    entities: list[GoldEntity] = Field(default_factory=list)

    @model_validator(mode="after")
    def _spans_within_text(self) -> GoldDocument:
        length = len(self.text)
        for entity in self.entities:
            if entity.end > length:
                raise ValueError(
                    f"{self.id}: span [{entity.start}, {entity.end}) "
                    f"exceeds text length {length}"
                )
            if entity.end <= entity.start:
                raise ValueError(f"{self.id}: inverted span {entity}")
        return self

    def to_entities(self) -> list[Entity]:
        return [
            Entity(
                start=e.start,
                end=e.end,
                label=e.label,
                text=self.text[e.start : e.end],
                source="gold",
            )
            for e in self.entities
        ]


def resolve_overlaps(entities: list[Entity]) -> list[Entity]:
    """Keep one entity per overlapping cluster.

    Ordering, applied in turn:
      1. Checksum-verifiable labels beat inferred ones.
      2. Higher label priority wins (UNIVERSITY over ORG).
      3. Longer span wins - "New York City" over "New York".
      4. Higher detector score wins.
    """

    def rank(entity: Entity) -> tuple:
        return (
            entity.label in DETERMINISTIC_LABELS,
            LABEL_PRIORITY.get(entity.label, 0),
            entity.length,
            entity.score,
        )

    kept: list[Entity] = []
    for entity in sorted(entities, key=rank, reverse=True):
        if not any(entity.overlaps(k) for k in kept):
            kept.append(entity)

    return sorted(kept, key=lambda e: e.start)


def anonymize(text: str, entities: list[Entity]) -> str:
    """Render spans as ``<LABEL>`` placeholders."""
    out: list[str] = []
    cursor = 0
    for entity in sorted(entities, key=lambda e: e.start):
        if entity.start < cursor:
            continue  # defensive: resolve_overlaps should have removed this
        out.append(text[cursor : entity.start])
        out.append(f"<{entity.label}>")
        cursor = entity.end
    out.append(text[cursor:])
    return "".join(out)


class Detector(Protocol):
    """Anything that turns text into entities.

    A Protocol rather than a base class: detectors are built from unrelated
    libraries and should not have to inherit from us to be usable.
    """

    name: str

    def detect(self, text: str) -> list[Entity]: ...
