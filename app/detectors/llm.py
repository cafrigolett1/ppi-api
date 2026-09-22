"""LLM entity extraction, zero-shot and few-shot.

The prompting is the easy half. The hard half is that an LLM returns entity
*strings*, while evaluation and anonymisation both need character offsets.
Three failure modes fall out of that gap, and all three are measured rather
than assumed away:

  repeats       The model lists "Paris" once when the document contains it
                three times. Mapping only the first occurrence leaves two
                unredacted - a silent leak. Every occurrence is mapped.

  hallucination The returned string does not appear in the source at all.
                It cannot be mapped to a span, so it is counted separately
                as an unmappable extraction and reported.

  normalisation The model helpfully rewrites what it found ("BE68539007547034"
                for "BE68 5390 0754 7034", or title-casing a name). Exact
                match is tried first, then a whitespace-insensitive and
                case-insensitive pass.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from ..entities import Entity, Label

LABEL_GUIDE = """\
PERSON         individual people, full or partial names
ORG            companies, agencies, institutions that are not universities
JOB            job titles, roles, professions, and professional titles
               ("software engineer", "CEO", "Dr.", "Professor")
EMAIL_ADDRESS  email addresses
LOCATION       countries, cities, states, provinces, regions, street
               addresses, postal codes, area codes, and nationality words
               used of a place (e.g. "Belgian", "US")
AMOUNT         monetary values with a currency
DATE_TIME      dates, years, times, timestamps
UNIVERSITY     universities, colleges, schools, academic institutions
PHONE_NUMBER   telephone numbers in any national format
URL            web addresses, with or without a scheme
IBAN           international bank account numbers
SSN            national identification numbers (e.g. Belgian rijksregisternummer)\
"""

ZERO_SHOT_SYSTEM = f"""\
You extract sensitive entities from text for anonymisation.

Return ONLY a JSON array. Each element must be an object with exactly two keys:
  "text"  - the exact substring as it appears in the input, copied character
            for character, with no reformatting
  "label" - one of the labels below

Labels:
{LABEL_GUIDE}

Rules:
- Copy "text" verbatim from the input. Do not normalise spacing, casing or
  punctuation.
- If the same entity appears more than once, list it once; every occurrence
  will be handled downstream.
- Prefer the most specific label: UNIVERSITY over ORG, IBAN over SSN.
- If there are no entities, return [].
- No markdown fences, no commentary, no trailing text.

LOCATION granularity - this is the most common source of error:
- Keep one place reference as ONE entity, including its components.
  "Seattle, Washington" is a single LOCATION.
  "Kerkstraat 14, 2300 Turnhout" is a single LOCATION.
  A full street address is one identifying unit - a doorstep, not three
  independent facts - so never split it.
- Two references to DIFFERENT places are separate entities. In "moved from
  Ghent to Namur" that is two LOCATIONs.
- Include a leading article when it is part of the name:
  "the United States", "the Walloon Region".
- A nationality or demonym describing a place is a LOCATION: the "US" in
  "a naturalized US citizen", the "Belgian" in "a Belgian citizen".
- A bare area code or postal code is a LOCATION.

Boundaries:
- Do not include a trailing sentence period in an entity. "Apple Inc." at the
  end of a sentence is the entity "Apple Inc".
- A professional title is its own JOB entity, and the PERSON span excludes
  it. "Dr. Lucas Mertens" is two entities: JOB "Dr." and PERSON
  "Lucas Mertens". "Professor Anne Verstraeten" is JOB "Professor" and
  PERSON "Anne Verstraeten". Never fold the title into the name.
- A date range is two separate DATE_TIME entities, one per date:
  "from 2009 to 2017" gives "2009" and "2017", not "2009 to 2017".

Do not over-extract:
- AMOUNT is monetary only. "8 million people" is a population, not an amount.
- A company VAT or registration number is not an SSN. SSN is a personal
  national identification number.
- A hospital or clinic is an ORG, even when attached to a university.\
"""

FEW_SHOT_SYSTEM = (
    ZERO_SHOT_SYSTEM
    + """

Worked examples follow. Match their granularity exactly, especially for
LOCATION: one place reference stays whole including its components, while
references to different places are separate entities."""
)


class ChatClient(Protocol):
    """Minimal provider surface. Anything satisfying this can be benchmarked."""

    def complete(self, system_prompt: str, messages: list[dict]) -> str: ...


class ExtractedEntity(BaseModel):
    """One item as returned by the model.

    Pydantic does the schema enforcement here rather than hand-written dict
    checks: an unknown label, a missing key or a non-string value is rejected
    with a precise message instead of silently becoming a bad span. Extraction
    output is untrusted input, and this is the validation boundary.
    """

    text: str = Field(min_length=1)
    label: Label


class FewShotExample(BaseModel):
    text: str
    entities: list[ExtractedEntity] = Field(default_factory=list)


class CohereChatClient:
    """Adapter for Cohere's v2 chat endpoint."""

    def __init__(
        self, api_key: str, model: str = "command-a-03-2025", max_tokens: int = 2000
    ) -> None:
        import cohere

        self._client = cohere.ClientV2(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system_prompt: str, messages: list[dict]) -> str:
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "system", "content": system_prompt}, *messages],
            max_tokens=self._max_tokens,
            temperature=0.0,  # extraction is not a creative task
        )
        return response.message.content[0].text


class LLMDetector:
    """Zero-shot or few-shot extraction via a chat model."""

    def __init__(
        self,
        client: ChatClient,
        examples: list[FewShotExample] | None = None,
        name: str | None = None,
    ) -> None:
        self._client = client
        self._examples = examples or []
        self.shot_count = len(self._examples)
        self.name = name or ("llm-few-shot" if self._examples else "llm-zero-shot")
        self.unmappable: list[dict] = []

    # -- prompting ---------------------------------------------------------

    def _system_prompt(self) -> str:
        return FEW_SHOT_SYSTEM if self._examples else ZERO_SHOT_SYSTEM

    @property
    def prompt(self) -> str:
        """The full system prompt, including few-shot turns when present.

        Exposed so a benchmark logs the prompt the service actually ran,
        rather than one reconstructed from the same source and assumed to
        match.
        """
        parts = [self._system_prompt()]
        for example in self._examples:
            parts.append(f"\n[example input]\n{example.text}")
            parts.append(
                "[example output]\n"
                + json.dumps(
                    [e.model_dump() for e in example.entities], ensure_ascii=False
                )
            )
        return "\n".join(parts)

    def _messages(self, text: str) -> list[dict]:
        messages: list[dict] = []
        for example in self._examples:
            messages.append({"role": "user", "content": example.text})
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        [e.model_dump() for e in example.entities],
                        ensure_ascii=False,
                    ),
                }
            )
        messages.append({"role": "user", "content": text})
        return messages

    # -- parsing -----------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> list:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Recover the first array in the response rather than losing the
            # whole document to one stray sentence of preamble.
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if not match:
                raise ValueError(
                    f"no JSON array in model output: {raw[:200]!r}"
                ) from None
            parsed = json.loads(match.group())

        if not isinstance(parsed, list):
            raise ValueError(f"expected a JSON array, got {type(parsed).__name__}")
        return parsed

    # -- span mapping ------------------------------------------------------

    def _validate(self, items: list, doc_id: str) -> list[ExtractedEntity]:
        """Coerce raw JSON into validated items, recording what was rejected."""
        valid: list[ExtractedEntity] = []
        for item in items:
            if isinstance(item, dict):
                # Models sometimes lower-case the label despite instructions.
                label = item.get("label")
                if isinstance(label, str):
                    item = {**item, "label": label.strip().upper()}
            try:
                valid.append(ExtractedEntity.model_validate(item))
            except ValidationError as exc:
                self.unmappable.append(
                    {
                        "doc_id": doc_id,
                        "reason": "schema_violation",
                        "item": item,
                        "detail": exc.errors()[0]["msg"] if exc.errors() else "",
                    }
                )
        return valid

    def _map_to_spans(
        self, text: str, items: list[ExtractedEntity], doc_id: str
    ) -> list[Entity]:
        found: list[Entity] = []

        for item in items:
            surface = item.text.strip()
            label = item.label
            offsets = self._find_all(text, surface)
            if not offsets:
                # The model returned something that is not in the document.
                self.unmappable.append(
                    {
                        "doc_id": doc_id,
                        "reason": "not_in_source",
                        "text": surface,
                        "label": label,
                    }
                )
                continue

            for start, end in offsets:
                found.append(
                    Entity(
                        start=start,
                        end=end,
                        label=label,
                        text=text[start:end],
                        score=0.8,
                        source=self.name,
                        metadata={"returned_as": surface},
                    )
                )

        # Same span returned twice under one label is one entity.
        unique: dict[tuple[int, int, str], Entity] = {}
        for entity in found:
            unique.setdefault((entity.start, entity.end, entity.label), entity)
        return sorted(unique.values(), key=lambda e: (e.start, e.end))

    @staticmethod
    def _find_all(text: str, surface: str) -> list[tuple[int, int]]:
        """All occurrences, trying progressively looser matching."""
        spans = [(m.start(), m.end()) for m in re.finditer(re.escape(surface), text)]
        if spans:
            return spans

        # Case-insensitive.
        spans = [
            (m.start(), m.end())
            for m in re.finditer(re.escape(surface), text, re.IGNORECASE)
        ]
        if spans:
            return spans

        # Whitespace-insensitive: the model stripped or added spaces, which is
        # common for IBANs and phone numbers.
        loose = r"\s*".join(re.escape(ch) for ch in re.sub(r"\s+", "", surface))
        return [(m.start(), m.end()) for m in re.finditer(loose, text, re.IGNORECASE)]

    # -- detector interface ------------------------------------------------

    def detect(self, text: str, doc_id: str = "") -> list[Entity]:
        raw = self._client.complete(self._system_prompt(), self._messages(text))
        items = self._validate(self._parse_json(raw), doc_id)
        return self._map_to_spans(text, items, doc_id)
