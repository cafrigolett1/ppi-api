"""Evaluation metrics for anonymisation.

Standard NER scoring answers "how many spans did you get right". For
anonymisation the question is "is this document safe to release", and the two
come apart badly. A model at 0.95 span recall sounds strong, but across
documents averaging five entities each roughly 22% of documents still contain
at least one unredacted entity - and one missed national number makes the
whole document unsafe. The unit of risk is the document, not the span.

So the metrics here are grouped:

  per-label    precision / recall / F1, because aggregate numbers hide that
               some labels score zero
  safety       document leakage rate and recall on direct identifiers, which
               are what a data protection officer would ask about
  utility      over-redaction, because redacting everything scores perfect
               recall and produces useless text

Costs are asymmetric: a missed IBAN is a breach, a false positive is an
over-redacted word. F1 weights them equally, which is wrong here, so F2
(recall weighted four times precision) is reported alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..entities import Entity

MatchMode = Literal["strict", "partial"]

# Direct identifiers: these alone identify a person, so their recall is
# tracked separately from labels like DATE_TIME that only contribute context.
DIRECT_IDENTIFIERS: frozenset[str] = frozenset(
    {"IBAN", "SSN", "EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"}
)


@dataclass
class LabelScore:
    label: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    support: int = 0

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        return self.fbeta(1.0)

    @property
    def f2(self) -> float:
        return self.fbeta(2.0)

    def fbeta(self, beta: float) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r)


@dataclass
class DocumentOutcome:
    doc_id: str
    text: str = ""
    """The source document, kept so an error report can show the sentence a
    miss happened in rather than just a label and an offset."""
    missed: list[Entity] = field(default_factory=list)
    spurious: list[Entity] = field(default_factory=list)
    gold_count: int = 0

    @property
    def leaked(self) -> bool:
        """True if anything that should have been redacted was not."""
        return bool(self.missed)

    @property
    def leaked_direct_identifier(self) -> bool:
        return any(e.label in DIRECT_IDENTIFIERS for e in self.missed)


@dataclass
class Evaluation:
    engine: str
    dataset: str
    mode: MatchMode
    per_label: dict[str, LabelScore] = field(default_factory=dict)
    documents: list[DocumentOutcome] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    input_tokens: list[int] = field(default_factory=list)
    redacted_chars: int = 0
    total_chars: int = 0
    unmappable: int = 0
    replayed: int = 0
    """Documents answered from the request log rather than fresh inference.
    Their latency is excluded, so latency metrics describe measured calls."""

    # -- aggregate span scores -------------------------------------------

    @property
    def micro(self) -> LabelScore:
        total = LabelScore(label="micro")
        for s in self.per_label.values():
            total.true_positives += s.true_positives
            total.false_positives += s.false_positives
            total.false_negatives += s.false_negatives
            total.support += s.support
        return total

    @property
    def macro_f1(self) -> float:
        scored = [s for s in self.per_label.values() if s.support]
        return sum(s.f1 for s in scored) / len(scored) if scored else 0.0

    @property
    def macro_recall(self) -> float:
        scored = [s for s in self.per_label.values() if s.support]
        return sum(s.recall for s in scored) / len(scored) if scored else 0.0

    # -- safety -----------------------------------------------------------

    @property
    def document_leakage_rate(self) -> float:
        """Fraction of documents with at least one unredacted entity.

        The headline safety number, and not derivable from span recall.
        """
        if not self.documents:
            return 0.0
        return sum(d.leaked for d in self.documents) / len(self.documents)

    @property
    def direct_identifier_leakage_rate(self) -> float:
        if not self.documents:
            return 0.0
        return sum(d.leaked_direct_identifier for d in self.documents) / len(
            self.documents
        )

    @property
    def direct_identifier_recall(self) -> float:
        scored = [
            s for k, s in self.per_label.items() if k in DIRECT_IDENTIFIERS and s.support
        ]
        if not scored:
            return 0.0
        tp = sum(s.true_positives for s in scored)
        fn = sum(s.false_negatives for s in scored)
        return tp / (tp + fn) if (tp + fn) else 0.0

    @property
    def clean_document_rate(self) -> float:
        """Fraction of documents fully redacted with no misses."""
        return 1.0 - self.document_leakage_rate

    # -- utility ----------------------------------------------------------

    @property
    def over_redaction_rate(self) -> float:
        """Spurious redactions per document.

        Redacting everything scores perfect recall and destroys the data, so
        this is the counterweight that stops recall being gamed.
        """
        if not self.documents:
            return 0.0
        return sum(len(d.spurious) for d in self.documents) / len(self.documents)

    @property
    def redacted_char_fraction(self) -> float:
        return self.redacted_chars / self.total_chars if self.total_chars else 0.0

    # -- performance -------------------------------------------------------

    def latency_percentile(self, p: float) -> float | None:
        if not self.latencies_ms:
            return None
        ordered = sorted(self.latencies_ms)
        index = min(int(len(ordered) * p), len(ordered) - 1)
        return ordered[index]

    @property
    def mean_latency_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def total_input_tokens(self) -> int:
        return sum(self.input_tokens)

    @property
    def documents_per_second(self) -> float | None:
        total_seconds = sum(self.latencies_ms) / 1000
        if not total_seconds:
            return None
        return len(self.latencies_ms) / total_seconds


def _matches(pred: Entity, gold: Entity, mode: MatchMode) -> bool:
    if pred.label != gold.label:
        return False
    if mode == "strict":
        return pred.start == gold.start and pred.end == gold.end
    return pred.start < gold.end and gold.start < pred.end


def score_document(
    evaluation: Evaluation,
    doc_id: str,
    text: str,
    predicted: list[Entity],
    gold: list[Entity],
) -> DocumentOutcome:
    """Score one document and fold the result into ``evaluation``."""
    for entity in gold:
        evaluation.per_label.setdefault(
            entity.label, LabelScore(label=entity.label)
        ).support += 1

    unmatched_gold = list(gold)
    matched_pred: list[Entity] = []

    for pred in predicted:
        hit = next(
            (g for g in unmatched_gold if _matches(pred, g, evaluation.mode)), None
        )
        if hit is not None:
            unmatched_gold.remove(hit)
            matched_pred.append(pred)
            evaluation.per_label.setdefault(
                pred.label, LabelScore(label=pred.label)
            ).true_positives += 1

    spurious = [p for p in predicted if p not in matched_pred]
    for pred in spurious:
        evaluation.per_label.setdefault(
            pred.label, LabelScore(label=pred.label)
        ).false_positives += 1

    for miss in unmatched_gold:
        evaluation.per_label[miss.label].false_negatives += 1

    outcome = DocumentOutcome(
        doc_id=doc_id,
        text=text,
        missed=list(unmatched_gold),
        spurious=spurious,
        gold_count=len(gold),
    )
    evaluation.documents.append(outcome)

    evaluation.total_chars += len(text)
    evaluation.redacted_chars += sum(e.length for e in predicted)
    return outcome


def summary_metrics(evaluation: Evaluation) -> dict[str, float]:
    """The metrics worth tracking, and only those.

    Deliberately short. A dashboard with forty numbers is one nobody reads,
    and per-label figures belong in the artefact table rather than as forty
    separate MLflow metrics.

    Safety, then quality, then cost:

      document_leakage_rate     fraction of documents still containing an
                                unredacted entity - the number that decides
                                whether output can be released, and not
                                derivable from span recall
      direct_identifier_recall  recall over IBAN, SSN, email, phone, person
      micro_recall / precision  span-level, aggregated
      micro_f2                  recall weighted 4x precision, because a missed
                                IBAN is a breach and a false positive is an
                                over-redacted word
      over_redaction_per_doc    the counterweight - redacting everything
                                scores perfect recall and destroys the data
      latency_*                 p50 and p95; means hide the tail
    """
    micro = evaluation.micro
    metrics: dict[str, float] = {
        "document_leakage_rate": evaluation.document_leakage_rate,
        "direct_identifier_recall": evaluation.direct_identifier_recall,
        "micro_precision": micro.precision,
        "micro_recall": micro.recall,
        "micro_f2": micro.f2,
        "over_redaction_per_doc": evaluation.over_redaction_rate,
        "documents": float(len(evaluation.documents)),
        "gold_spans": float(micro.support),
    }

    metrics["replayed_documents"] = float(evaluation.replayed)

    if evaluation.mean_latency_ms is not None:
        metrics["latency_measured_documents"] = float(len(evaluation.latencies_ms))
        metrics["latency_p50_ms"] = evaluation.latency_percentile(0.50) or 0.0
        metrics["latency_p95_ms"] = evaluation.latency_percentile(0.95) or 0.0
        metrics["latency_total_s"] = sum(evaluation.latencies_ms) / 1000
    if evaluation.input_tokens:
        metrics["total_input_tokens"] = float(evaluation.total_input_tokens)

    return metrics


def per_label_table(evaluation: Evaluation) -> dict[str, dict[str, float]]:
    """Per-label breakdown, logged as an artefact rather than as metrics."""
    return {
        label: {
            "precision": round(s.precision, 4),
            "recall": round(s.recall, 4),
            "f1": round(s.f1, 4),
            "support": s.support,
            "tp": s.true_positives,
            "fp": s.false_positives,
            "fn": s.false_negatives,
        }
        for label, s in sorted(evaluation.per_label.items())
        if s.support or s.false_positives
    }