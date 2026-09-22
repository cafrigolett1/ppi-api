"""Presidio detector, in stock and Belgian-enhanced configurations.

``build_nlp_engine`` exists because an ``AnalyzerEngine`` constructed with no
arguments loads its own copy of the spaCy pipeline. Creating two analysers
that way loads ``en_core_web_lg`` twice - roughly 1 GB of resident memory and
double the startup time, for two identical copies. The engine is built once
and shared.

All data here is Belgian, so ``US_SSN`` is not requested: against a label set
where SSN means rijksregisternummer it would only add false positives.
"""

from __future__ import annotations

import logging

from ..entities import Entity

logger = logging.getLogger(__name__)

PRESIDIO_LABEL_MAP: dict[str, str] = {
    "PERSON": "PERSON",
    "ORGANIZATION": "ORG",
    "NRP": "LOCATION",
    "LOCATION": "LOCATION",
    "DATE_TIME": "DATE_TIME",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "URL": "URL",
    "IBAN_CODE": "IBAN",
    "BE_NATIONAL_NUMBER": "SSN",
}

BASE_ENTITIES = [
    "PERSON",
    "LOCATION",
    "NRP",
    "DATE_TIME",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "URL",
    "IBAN_CODE",
]


# spaCy's NER emits the full OntoNotes label set. Presidio has no mapping for
# most of these and logs a warning per unmapped label, then keeps them anyway.
# They are filtered out downstream by PRESIDIO_LABEL_MAP, so the only effect is
# noise - but declaring them here stops the warning and skips the work.
#
# Do NOT add labels Presidio does map: PERSON, ORG, GPE, LOC, DATE, TIME and
# NORP all feed entity types this service uses. FAC is handled below.
SPACY_LABELS_TO_IGNORE = [
    "CARDINAL",
    "ORDINAL",
    "PERCENT",
    "QUANTITY",
    "MONEY",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "EVENT",
    "PRODUCT",
]


# spaCy's FAC label covers buildings, airports and landmarks - "The Eiffel
# Tower". Presidio has no default mapping for it, so those spans are dropped
# with a warning. Mapping it to LOCATION keeps them, alongside Presidio's own
# defaults for the other labels.
SPACY_ENTITY_MAPPING = {
    "PERSON": "PERSON",
    "NORP": "NRP",
    "FAC": "LOCATION",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "ORG": "ORGANIZATION",
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
}


def build_nlp_engine(spacy_model: str = "en_core_web_lg"):
    """Load the spaCy pipeline once, for sharing across analysers."""
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}],
            "ner_model_configuration": {
                "labels_to_ignore": SPACY_LABELS_TO_IGNORE,
                "model_to_presidio_entity_mapping": SPACY_ENTITY_MAPPING,
            },
        }
    )
    return provider.create_engine()


class PresidioDetector:
    def __init__(
        self,
        score_threshold: float = 0.4,
        belgian: bool = False,
        nlp_engine=None,
        spacy_model: str = "en_core_web_lg",
    ) -> None:
        from presidio_analyzer import AnalyzerEngine

        self.belgian = belgian
        self.name = "presidio-be" if belgian else "presidio"
        self._threshold = score_threshold
        self._entities = list(BASE_ENTITIES)

        self._analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine or build_nlp_engine(spacy_model)
        )

        if belgian:
            from .belgian import (
                BelgianIbanRecognizer,
                BelgianNationalNumberRecognizer,
            )

            self._analyzer.registry.add_recognizer(BelgianIbanRecognizer())
            self._analyzer.registry.add_recognizer(BelgianNationalNumberRecognizer())
            self._entities.append("BE_NATIONAL_NUMBER")

    def detect(self, text: str) -> list[Entity]:
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=self._entities,
            score_threshold=self._threshold,
        )

        found: list[Entity] = []
        for r in results:
            label = PRESIDIO_LABEL_MAP.get(r.entity_type)
            if label is None:
                continue
            found.append(
                Entity(
                    start=r.start,
                    end=r.end,
                    label=label,
                    text=text[r.start : r.end],
                    score=float(r.score),
                    source=self.name,
                )
            )
        return found
