"""
The Experience contract: Observation, Interpretation, Judgment, Episode.

Four separate record types — deliberately. Collapsing them loses the
supervision signal that makes the learning loop work.

Erasure rule: observation text is PII-tokenised at ingestion. The reversible
map lives in a separate deletable store. Dropping map entries makes a person
unlinkable without touching the corpus.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


class Actor(str, Enum):
    USER = "user"
    AGENT = "agent"
    OPERATOR = "operator"
    SYSTEM = "system"


class Origin(str, Enum):
    PRODUCTION = "production"
    IMPORT = "import"
    SYNTHETIC = "synthetic"
    MANUAL = "manual"


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)
    observation_id: str = Field(default_factory=lambda: _id("obs"))
    instance_id: str
    # Required at creation, nullable after erasure: `Store.forget` clears
    # it in place so a subject's messages cannot be re-grouped by session.
    session_id: str | None
    episode_id: str | None = None
    actor: Actor = Actor.USER
    channel: str = "unknown"
    text: str
    payload: dict[str, Any] = Field(default_factory=dict)
    origin: Origin = Origin.PRODUCTION
    subject_key: str | None = None
    content_hash: str = ""
    locale: str | None = None
    occurred_at: datetime = Field(default_factory=_now)
    ingested_at: datetime = Field(default_factory=_now)
    source: str = "unknown"
    source_ref: str | None = None

    @property
    def is_evaluable(self) -> bool:
        return self.origin is not Origin.SYNTHETIC


@dataclass(frozen=True)
class SubjectMapEntry:
    """
    One link between a tokenised placeholder and a subject fingerprint,
    scoped to the observation it was extracted from.

    Observation-scoped, deliberately. The same placeholder — `{PHONE_1}` —
    stands for a different person in a different observation, so a link is
    only meaningful alongside the `observation_id` it came from. This is
    what `Store.append_subject_map` records and what `forget` drops.

    `fingerprint` is the subject key, never a raw identifier: the phone
    number was turned into it at ingestion and does not exist in the store.
    """

    observation_id: str
    placeholder: str
    fingerprint: str


class EntityGuess(BaseModel):
    model_config = ConfigDict(frozen=True)
    field: str
    value: str
    normalized: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    char_start: int | None = None
    char_end: int | None = None

    @property
    def effective(self) -> str:
        return self.normalized or self.value


class Interpretation(BaseModel):
    model_config = ConfigDict(frozen=True)
    interpretation_id: str = Field(default_factory=lambda: _id("int"))
    observation_id: str
    instance_id: str
    labeler: str
    labeler_kind: Literal["model", "llm", "external", "human", "rule"] = "model"
    intent: str | None = None
    intent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    alternatives: list[tuple[str, float]] = Field(default_factory=list)
    entities: list[EntityGuess] = Field(default_factory=list)
    taxonomy_fingerprint: str = ""
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=_now)

    def entity(self, field: str) -> EntityGuess | None:
        return next((e for e in self.entities if e.field == field), None)

    def entity_map(self) -> dict[str, str]:
        return {e.field: e.effective for e in self.entities}


class JudgmentSource(str, Enum):
    HUMAN = "human"
    CORRECTION = "correction"
    OUTCOME = "outcome"
    CONSENSUS = "consensus"


SOURCE_TRUST: dict[JudgmentSource, float] = {
    JudgmentSource.HUMAN: 1.0,
    JudgmentSource.CORRECTION: 0.8,
    JudgmentSource.OUTCOME: 0.6,
    JudgmentSource.CONSENSUS: 0.4,
}


class Judgment(BaseModel):
    model_config = ConfigDict(frozen=True)
    judgment_id: str = Field(default_factory=lambda: _id("jdg"))
    observation_id: str
    instance_id: str
    source: JudgmentSource
    judge: str
    gold_intent: str | None = None
    intent_judged: bool = False
    gold_entities: dict[str, str | None] = Field(default_factory=dict)
    taxonomy_fingerprint: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = ""
    supersedes: str | None = None
    created_at: datetime = Field(default_factory=_now)

    @property
    def trust(self) -> float:
        return SOURCE_TRUST[self.source] * self.confidence

    @property
    def is_gold(self) -> bool:
        return self.source is JudgmentSource.HUMAN


class Outcome(str, Enum):
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ESCALATED = "escalated"
    FAILED = "failed"
    ONGOING = "ongoing"
    UNKNOWN = "unknown"


class Episode(BaseModel):
    episode_id: str = Field(default_factory=lambda: _id("epi"))
    instance_id: str
    session_id: str
    observation_ids: list[str] = Field(default_factory=list)
    outcome: Outcome = Outcome.ONGOING
    outcome_detail: str = ""
    capability: str | None = None
    intent: str | None = None
    turn_count: int = 0
    clarify_count: int = 0
    correction_count: int = 0
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None

    @property
    def weight(self) -> float:
        if self.outcome is Outcome.COMPLETED and self.correction_count == 0:
            return 2.0
        if self.outcome is Outcome.COMPLETED:
            return 1.2
        if self.outcome in {Outcome.ABANDONED, Outcome.ESCALATED, Outcome.FAILED}:
            return 0.5
        return 1.0


class FieldVerdict(str, Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    MISSED = "missed"
    HALLUCINATED = "hallucinated"
    UNJUDGED = "unjudged"


class InterpretationScore(BaseModel):
    interpretation_id: str
    judgment_id: str
    intent_verdict: FieldVerdict
    entity_verdicts: dict[str, FieldVerdict] = Field(default_factory=dict)

    @property
    def fully_correct(self) -> bool:
        judged = [v for v in self.entity_verdicts.values() if v is not FieldVerdict.UNJUDGED]
        return self.intent_verdict in {FieldVerdict.CORRECT, FieldVerdict.UNJUDGED} and all(
            v is FieldVerdict.CORRECT for v in judged
        )

    @property
    def failures(self) -> dict[str, FieldVerdict]:
        bad = {FieldVerdict.WRONG, FieldVerdict.MISSED, FieldVerdict.HALLUCINATED}
        out = {f: v for f, v in self.entity_verdicts.items() if v in bad}
        if self.intent_verdict in bad:
            out["__intent__"] = self.intent_verdict
        return out


def score(interp: Interpretation, judgment: Judgment) -> InterpretationScore:
    intent_verdict = FieldVerdict.UNJUDGED
    if judgment.intent_judged:
        intent_verdict = (
            FieldVerdict.CORRECT if interp.intent == judgment.gold_intent else FieldVerdict.WRONG
        )
    verdicts: dict[str, FieldVerdict] = {}
    guessed = {e.field: e.effective.strip().lower() for e in interp.entities}
    for field, gold_value in judgment.gold_entities.items():
        got = guessed.get(field)
        if gold_value is None:
            verdicts[field] = FieldVerdict.HALLUCINATED if got is not None else FieldVerdict.CORRECT
        elif got is None:
            verdicts[field] = FieldVerdict.MISSED
        elif got == gold_value.strip().lower():
            verdicts[field] = FieldVerdict.CORRECT
        else:
            verdicts[field] = FieldVerdict.WRONG
    for field in guessed:
        verdicts.setdefault(field, FieldVerdict.UNJUDGED)
    return InterpretationScore(
        interpretation_id=interp.interpretation_id,
        judgment_id=judgment.judgment_id,
        intent_verdict=intent_verdict,
        entity_verdicts=verdicts,
    )


class Disagreement(BaseModel):
    observation_id: str
    left: str
    right: str
    intent_differs: bool
    entity_fields_differing: list[str] = Field(default_factory=list)
    left_confidence: float = 0.0
    right_confidence: float = 0.0

    @property
    def review_priority(self) -> float:
        base = min(self.left_confidence, self.right_confidence)
        return base * (
            1.0 + 0.5 * len(self.entity_fields_differing) + (1.0 if self.intent_differs else 0.0)
        )


def disagree(left: Interpretation, right: Interpretation) -> Disagreement | None:
    if left.taxonomy_fingerprint != right.taxonomy_fingerprint:
        raise ValueError(
            f"cannot compare interpretations across taxonomy versions "
            f"({left.taxonomy_fingerprint} vs {right.taxonomy_fingerprint})"
        )
    lm, rm = left.entity_map(), right.entity_map()
    fields = sorted(
        f
        for f in set(lm) | set(rm)
        if lm.get(f, "").strip().lower() != rm.get(f, "").strip().lower()
    )
    intent_differs = left.intent != right.intent
    if not intent_differs and not fields:
        return None
    return Disagreement(
        observation_id=left.observation_id,
        left=left.labeler,
        right=right.labeler,
        intent_differs=intent_differs,
        entity_fields_differing=fields,
        left_confidence=left.intent_confidence,
        right_confidence=right.intent_confidence,
    )
