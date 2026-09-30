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
    # The observation this one answers. An agent's reply and the outcome
    # that followed both carry the `observation_id` of the user event
    # that triggered them, which is what makes a single interaction
    # queryable as a whole — see `CaptureEvent`. None on the first event
    # of an interaction, and on every record written by the batch ingest
    # path, which groups by session instead.
    correlation_id: str | None = None
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


# --------------------------------------------------------------------------
# The capture contract
# --------------------------------------------------------------------------
#
# What an application posts to `POST /v1/capture`, in any language, with
# no Syntri-side knowledge of what the application does.
#
# An interaction is three events, not one. The user said something, the
# agent answered, and something happened as a result — and the third is
# the only one that says whether the first two went well. A contract
# that captured just the inbound message would collect utterances and no
# supervision signal, which is the difference between a corpus and a
# training set.
#
#     input    ──> event_id: obs_a1b2…
#                      │
#     output   ──> correlation_id: obs_a1b2…
#     outcome  ──> correlation_id: obs_a1b2…
#
# The link is the input event's own id, handed back to the caller in the
# response so it can be threaded onto whatever follows. The caller keeps
# nothing else and needs no second round trip.


class CaptureEventType(str, Enum):
    """
    What kind of event this is, and therefore what links to what.

    INPUT    something arrived from a user. Starts an interaction; its
             `event_id` is the handle for everything that follows.
    OUTPUT   what the application answered. Links to an input.
    OUTCOME  what actually happened — the payment settled, the user gave
             up, a human took over. Links to the same input.
    """

    INPUT = "input"
    OUTPUT = "output"
    OUTCOME = "outcome"


class CaptureEvent(BaseModel):
    """
    One event, from any application, in any shape.

    `payload` is opaque on purpose. Syntri stores it, tokenises the
    identifiers it finds in it and learns from it, but requires no
    particular keys: the customer decides what an event of theirs
    contains, and a contract that demanded a schema would make every new
    application a change to this file.

    PII is *not* the caller's problem. The server tokenises phone
    numbers, emails and account numbers out of `payload` at ingestion,
    before anything is written. An SDK that tokenised client-side would
    put the erasure guarantee in someone else's deployment and make each
    caller's fingerprints incompatible with every other caller's.
    """

    instance_id: str
    event_type: CaptureEventType
    payload: dict[str, Any] = Field(default_factory=dict)
    #: Optional. Groups events into a conversation. Output and outcome
    #: events inherit the input's when they name none.
    session_id: str | None = None
    #: Required for `output` and `outcome`, and must be None on `input` —
    #: an input starts an interaction, so there is nothing for it to
    #: point at. Carries the `event_id` the server returned for the
    #: input event.
    correlation_id: str | None = None
    #: Caller-supplied context. `metadata.channel` becomes the
    #: Observation's channel when present.
    metadata: dict[str, Any] | None = None
    #: When it happened, not when it arrived. The server fills this in
    #: if it is absent, but a caller that batches or retries should send
    #: its own — arrival time is not event time.
    timestamp: datetime | None = None


class CaptureResponse(BaseModel):
    """
    The outcome of a capture, always at HTTP 200.

    The caller is a production application whose request handler is not
    a place where an exception is survivable, so the verdict is in the
    body and never in the status line. `status` is one of:

        captured        written; `event_id` names the row
        unprocessable   the event could not be linked or read
        write_failed    the store would not take it

    Auth failures are the exception and stay 401/403: a bad key is a
    configuration error that no amount of quiet retrying fixes.

    `event_id` is None on every failure path, which is why it is
    nullable. On an `input` event it is returned twice — once as itself
    and once as `correlation_id` — so the caller can thread the output
    and outcome onto it without inventing an id of its own.
    """

    event_id: str | None = None
    status: str
    correlation_id: str | None = None
    #: Why, when `status` is not "captured". For the caller's logs; never
    #: load-bearing, and never a reason to fail a request.
    detail: str | None = None


# --------------------------------------------------------------------------
# Quality criteria
# --------------------------------------------------------------------------
#
# What separates an interaction worth learning from as a positive example
# from one that merely happened. Evaluated server-side against an outcome
# event's payload when it is captured, so an SDK or an instance config can
# declare "good" once and the loop applies it everywhere.
#
# The criteria live here, in the contract, rather than in each caller: a
# customer who changes what "good" means changes one config, not every
# place that reads an outcome.


class QualityOperator(str, Enum):
    """How a `QualitySignal` compares the resolved field to its value.

    `exists`/`not_exists` test presence alone and ignore `value`; the
    ordered comparisons coerce to float and fail closed on anything that
    is not numeric; `contains` tests membership (substring or element).
    """

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    CONTAINS = "contains"


class QualitySignal(BaseModel):
    """One condition on a dot-notation path into an outcome payload.

    `field` is a dotted path resolved against the payload dict, e.g.
    `payload.result.score`. `value` is unused for `exists`/`not_exists`.
    """

    field: str
    operator: QualityOperator
    value: Any | None = None


class QualityConfig(BaseModel):
    """The criteria that decide whether an outcome counts as good.

    Two independent gates, both optional. When both are set both must
    pass; when neither is set everything passes — no criteria means no
    opinion, not "nothing is good".
    """

    good_signal: QualitySignal | None = None
    min_outcome_score: float | None = None
    require_all_three: bool = True
    score_field: str | None = None

    def evaluate(self, outcome_payload: dict) -> bool:
        """Return True if `outcome_payload` passes the configured criteria.

        Called server-side when an outcome event is captured.
        """
        if self.require_all_three and not outcome_payload:
            return False

        checks: list[bool] = []
        if self.good_signal is not None:
            checks.append(self._eval_signal(self.good_signal, outcome_payload))
        if self.min_outcome_score is not None:
            checks.append(self._eval_score(outcome_payload))

        if not checks:
            return True
        return all(checks)

    @staticmethod
    def _resolve(path: str, payload: dict) -> tuple[bool, Any]:
        """Walk a dotted path; return (found, value)."""
        current: Any = payload
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return False, None
            current = current[part]
        return True, current

    def _eval_signal(self, signal: QualitySignal, payload: dict) -> bool:
        found, value = self._resolve(signal.field, payload)
        op = signal.operator
        if op is QualityOperator.EXISTS:
            return found
        if op is QualityOperator.NOT_EXISTS:
            return not found
        if not found:
            return False
        return self._apply(op, value, signal.value)

    @staticmethod
    def _apply(op: QualityOperator, actual: Any, expected: Any) -> bool:
        if op is QualityOperator.EQUALS:
            return bool(actual == expected)
        if op is QualityOperator.NOT_EQUALS:
            return bool(actual != expected)
        if op is QualityOperator.CONTAINS:
            try:
                return expected in actual
            except TypeError:
                return False
        # Ordered comparisons: numeric only, fail closed otherwise.
        try:
            a, b = float(actual), float(expected)
        except (TypeError, ValueError):
            return False
        if op is QualityOperator.GT:
            return a > b
        if op is QualityOperator.GTE:
            return a >= b
        if op is QualityOperator.LT:
            return a < b
        if op is QualityOperator.LTE:
            return a <= b
        return False

    def _eval_score(self, payload: dict) -> bool:
        field = self.score_field or "score"
        found, value = self._resolve(field, payload)
        if not found:
            return False
        # bool is an int subclass but is never a score.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        assert self.min_outcome_score is not None  # guarded by caller
        return value >= self.min_outcome_score


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
