"""Unit tests for the policy engine, the floor resolver and the audit logger.

Ported from syntri-core's test_policy_engine.py. The engine is a pure function
of its arguments and is tested as one. Core's real `Taxonomy` is private, so
these tests drive the engine through a minimal fake exposing the two things it
actually reads off a taxonomy: `intent(name)` and `floor_for(name)`.

Integration tests that go through HTTP or depend on core's taxonomy tier maths
are deliberately not ported — they live in syntri-core.
"""
from __future__ import annotations

import pytest

from experience.schema import Actor, Observation
from policy import (
    FAIL_CLOSED,
    POLICY_CHANNEL,
    POLICY_SOURCE,
    AuditLogger,
    PolicyDecision,
    PolicyEngine,
    Reason,
    is_risky,
    resolve_floor,
)


class _IntentSpec:
    def __init__(self, moves_money: bool) -> None:
        self.moves_money = moves_money


class FakeTaxonomy:
    """The taxonomy surface the policy layer reads: `intent` and `floor_for`.

    Built from a mapping of intent name -> (floor, moves_money). An intent
    that is not present is unknown: `intent()` returns None and `floor_for()`
    returns 0.75 — reproducing the elevated-tier default that `resolve_floor`
    is specifically written to override with FAIL_CLOSED.
    """

    UNKNOWN_DEFAULT = 0.75

    def __init__(self, intents: dict[str, tuple[float, bool]]) -> None:
        self._floors = {name: floor for name, (floor, _) in intents.items()}
        self._risky = {name: money for name, (_, money) in intents.items()}

    def intent(self, name: str) -> _IntentSpec | None:
        if name not in self._risky:
            return None
        return _IntentSpec(moves_money=self._risky[name])

    def floor_for(self, name: str) -> float:
        return self._floors.get(name, self.UNKNOWN_DEFAULT)


@pytest.fixture
def taxonomy() -> FakeTaxonomy:
    return FakeTaxonomy(
        {
            "check_balance": (0.65, False),   # low risk
            "flight_search": (0.55, False),   # no risk
            "send_money": (0.85, True),       # money
            "close_account": (0.90, True),    # irreversible
            "pay_bill": (0.95, True),         # money, explicit higher floor
        }
    )


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine()


class FakeStore:
    """The whole Store surface the audit logger needs."""

    def __init__(self, *, fail: bool = False) -> None:
        self.observations: list[Observation] = []
        self.fail = fail

    def append_observations(self, rows) -> int:
        if self.fail:
            raise RuntimeError("database is on fire")
        self.observations.extend(rows)
        return len(rows)


# -- resolve_floor ---------------------------------------------------------


def test_a_declared_intent_gets_its_floor(taxonomy):
    assert resolve_floor("send_money", taxonomy) == 0.85
    assert resolve_floor("close_account", taxonomy) == 0.90
    assert resolve_floor("check_balance", taxonomy) == 0.65
    assert resolve_floor("pay_bill", taxonomy) == 0.95


def test_an_undeclared_intent_fails_closed(taxonomy):
    """
    Not 0.75. `floor_for` answers the elevated-tier default for an intent it
    has never heard of, which is the one case where a plausible-looking
    number is worse than an impossible one.
    """
    assert taxonomy.floor_for("wire_transfer") == 0.75, "the behaviour being guarded"
    assert resolve_floor("wire_transfer", taxonomy) == FAIL_CLOSED == 1.0


def test_no_taxonomy_and_no_intent_fail_closed(taxonomy):
    assert resolve_floor("send_money", None) == FAIL_CLOSED
    assert resolve_floor("", taxonomy) == FAIL_CLOSED
    assert resolve_floor("send_money", object()) == FAIL_CLOSED


def test_the_default_is_configurable(taxonomy):
    assert resolve_floor("wire_transfer", taxonomy, default=0.5) == 0.5


# -- is_risky --------------------------------------------------------------


def test_risk_covers_money_and_irreversible_only(taxonomy):
    assert is_risky("send_money", taxonomy) is True
    assert is_risky("close_account", taxonomy) is True
    assert is_risky("check_balance", taxonomy) is False
    assert is_risky("flight_search", taxonomy) is False


def test_an_unknown_intent_counts_as_risky(taxonomy):
    assert is_risky("wire_transfer", taxonomy) is True
    assert is_risky("send_money", None) is True


# -- the floor check -------------------------------------------------------


def test_below_the_floor_is_refused(engine, taxonomy):
    decision = engine.evaluate("send_money", 0.84, "model", taxonomy)
    assert decision.permitted is False
    assert decision.reason == Reason.BELOW_FLOOR
    assert decision.floor == 0.85
    assert decision.confidence == 0.84


def test_exactly_at_the_floor_is_permitted(engine, taxonomy):
    """The floor is inclusive: `confidence < floor` refuses, not `<=`."""
    decision = engine.evaluate("send_money", 0.85, "model", taxonomy)
    assert decision.permitted is True
    assert decision.reason == Reason.PERMITTED


def test_above_the_floor_is_permitted(engine, taxonomy):
    decision = engine.evaluate("send_money", 0.97, "model", taxonomy)
    assert decision.permitted is True
    assert decision.reason == Reason.PERMITTED
    assert decision.floor == 0.85


def test_the_floor_is_per_intent_not_per_platform(engine, taxonomy):
    """0.90 clears send_money's 0.85 and misses pay_bill's declared 0.95."""
    assert engine.evaluate("send_money", 0.90, "model", taxonomy).permitted is True
    refused = engine.evaluate("pay_bill", 0.90, "model", taxonomy)
    assert refused.permitted is False
    assert refused.floor == 0.95


# -- the source check ------------------------------------------------------


def test_the_fallback_matcher_may_not_dispatch_a_money_intent(engine, taxonomy):
    decision = engine.evaluate("send_money", 0.31, "taxonomy", taxonomy)
    assert decision.permitted is False
    assert decision.reason == Reason.SOURCE_INSUFFICIENT


def test_a_confident_fallback_may_still_not_dispatch_a_money_intent(engine, taxonomy):
    """A token-overlap score of 0.99 is not evidence about money."""
    decision = engine.evaluate("send_money", 0.99, "taxonomy", taxonomy)
    assert decision.permitted is False
    assert decision.reason == Reason.SOURCE_INSUFFICIENT


def test_the_source_rule_covers_irreversible_too(engine, taxonomy):
    decision = engine.evaluate("close_account", 1.0, "taxonomy", taxonomy)
    assert decision.permitted is False
    assert decision.reason == Reason.SOURCE_INSUFFICIENT


def test_a_non_risky_intent_passes_on_the_fallback_source(engine, taxonomy):
    """Cold start has to work: search flights and read a balance, no money."""
    assert engine.evaluate("flight_search", 0.40, "taxonomy", taxonomy).permitted
    assert engine.evaluate("check_balance", 0.50, "taxonomy", taxonomy).permitted


def test_the_classifier_source_is_accepted_under_either_name(engine, taxonomy):
    assert engine.evaluate("send_money", 0.97, "model", taxonomy).permitted
    assert engine.evaluate("send_money", 0.97, "classifier", taxonomy).permitted


def test_an_unrecognised_source_is_untrusted(engine, taxonomy):
    decision = engine.evaluate("send_money", 0.99, "some_new_thing", taxonomy)
    assert decision.permitted is False
    assert decision.reason == Reason.SOURCE_INSUFFICIENT


def test_the_floor_is_not_applied_to_an_untrusted_source_by_default(engine, taxonomy):
    decision = engine.evaluate("check_balance", 0.10, "taxonomy", taxonomy)
    assert decision.permitted is True
    assert decision.floor == 0.65, "still resolved and reported, just not enforced"


def test_the_floor_can_be_enforced_on_untrusted_sources(taxonomy):
    strict = PolicyEngine(enforce_floor_on_untrusted=True)
    decision = strict.evaluate("check_balance", 0.10, "taxonomy", taxonomy)
    assert decision.permitted is False
    assert decision.reason == Reason.BELOW_FLOOR


# -- fail-closed defaults --------------------------------------------------


def test_an_intent_the_taxonomy_does_not_declare_is_refused(engine, taxonomy):
    decision = engine.evaluate("wire_transfer", 0.99, "model", taxonomy)
    assert decision.permitted is False
    assert decision.reason == Reason.BELOW_FLOOR
    assert decision.floor == 1.0


def test_only_an_impossible_confidence_clears_an_undeclared_intent(engine, taxonomy):
    assert engine.evaluate("wire_transfer", 1.0, "model", taxonomy).permitted is True
    assert engine.evaluate("wire_transfer", 0.999, "model", taxonomy).permitted is False


def test_an_empty_intent_is_refused(engine, taxonomy):
    decision = engine.evaluate("", 1.0, "model", taxonomy)
    assert decision.permitted is False
    assert decision.reason == Reason.NO_INTENT


def test_an_unreadable_confidence_is_zero_not_a_pass(engine, taxonomy):
    decision = engine.evaluate("send_money", None, "model", taxonomy)
    assert decision.permitted is False
    assert decision.confidence == 0.0


def test_evaluating_without_a_taxonomy_refuses_everything(engine):
    assert engine.evaluate("check_balance", 0.99, "model").permitted is False


# -- shadow mode -----------------------------------------------------------


def test_shadow_permits_what_it_would_otherwise_refuse(engine, taxonomy):
    decision = engine.evaluate("send_money", 0.31, "taxonomy", taxonomy, shadow=True)
    assert decision.permitted is True
    assert decision.shadow is True
    assert decision.reason == Reason.SOURCE_INSUFFICIENT, "the verdict is preserved"
    assert decision.would_have_blocked is True


def test_shadow_does_not_disguise_a_genuine_pass(engine, taxonomy):
    decision = engine.evaluate("send_money", 0.97, "model", taxonomy, shadow=True)
    assert decision.permitted is True
    assert decision.shadow is True
    assert decision.reason == Reason.PERMITTED
    assert decision.would_have_blocked is False


def test_shadow_is_off_unless_asked_for(engine, taxonomy):
    assert engine.evaluate("send_money", 0.31, "taxonomy", taxonomy).shadow is False


# -- the engine itself -----------------------------------------------------


def test_the_engine_needs_no_arguments():
    assert PolicyEngine().evaluate("x", 0.0, "model").permitted is False


def test_the_engine_holds_no_state(engine, taxonomy):
    first = engine.evaluate("send_money", 0.97, "model", taxonomy)
    for _ in range(50):
        engine.evaluate("send_money", 0.01, "taxonomy", taxonomy)
    second = engine.evaluate("send_money", 0.97, "model", taxonomy)
    assert first == second


def test_a_decision_is_frozen(engine, taxonomy):
    decision = engine.evaluate("send_money", 0.31, "model", taxonomy)
    with pytest.raises(Exception):
        decision.permitted = True  # type: ignore[misc]


def test_a_decision_serialises_everything_needed_to_explain_it(engine, taxonomy):
    payload = engine.evaluate("send_money", 0.31, "taxonomy", taxonomy).to_dict()
    assert payload == {
        "permitted": False,
        "reason": "source_insufficient",
        "floor": 0.85,
        "confidence": 0.31,
        "shadow": False,
        "intent": "send_money",
        "source": "taxonomy",
    }


# -- the audit logger ------------------------------------------------------


def test_a_refusal_is_written_to_the_store(engine, taxonomy):
    store = FakeStore()
    decision = engine.evaluate("send_money", 0.31, "taxonomy", taxonomy)
    assert AuditLogger(store).record(
        decision, instance_id="inst_1", session_id="s1"
    ) is True
    (row,) = store.observations
    assert row.instance_id == "inst_1"
    assert row.session_id == "s1"
    assert row.payload["kind"] == "policy_decision"
    assert row.payload["permitted"] is False
    assert row.payload["reason"] == "source_insufficient"
    assert row.payload["floor"] == 0.85
    assert row.payload["confidence"] == 0.31
    assert row.payload["intent"] == "send_money"
    assert row.payload["source"] == "taxonomy"


def test_a_permitted_decision_is_written_too(engine, taxonomy):
    store = FakeStore()
    decision = engine.evaluate("send_money", 0.97, "model", taxonomy)
    AuditLogger(store).record(decision, instance_id="inst_1", session_id="s1")
    (row,) = store.observations
    assert row.payload["permitted"] is True
    assert row.payload["reason"] == "permitted"


def test_an_audit_row_is_a_system_row_not_a_user_utterance(engine, taxonomy):
    store = FakeStore()
    decision = engine.evaluate("send_money", 0.31, "taxonomy", taxonomy)
    AuditLogger(store).record(decision, instance_id="inst_1", session_id="s1")
    (row,) = store.observations
    assert row.actor is Actor.SYSTEM
    assert row.channel == POLICY_CHANNEL
    assert row.source == POLICY_SOURCE
    assert row.text == "", "the utterance is stored once, by ingestion, tokenised"


def test_an_idempotency_key_is_captured_when_there_is_one(engine, taxonomy):
    store = FakeStore()
    decision = engine.evaluate("send_money", 0.97, "model", taxonomy)
    logger = AuditLogger(store)
    logger.record(decision, instance_id="i", session_id="s", idempotency_key="idem-1")
    logger.record(decision, instance_id="i", session_id="s")
    with_key, without_key = store.observations
    assert with_key.payload["idempotency_key"] == "idem-1"
    assert "idempotency_key" not in without_key.payload


def test_a_failed_audit_write_reports_false_and_does_not_raise(engine, taxonomy):
    decision = engine.evaluate("send_money", 0.31, "taxonomy", taxonomy)
    assert AuditLogger(FakeStore(fail=True)).record(
        decision, instance_id="i", session_id="s"
    ) is False


def test_the_logger_works_with_any_store_that_appends_observations(engine, taxonomy):
    written: list = []

    class Minimal:
        def append_observations(self, rows):
            written.extend(rows)
            return len(rows)

    AuditLogger(Minimal()).record(
        engine.evaluate("check_balance", 0.9, "model", taxonomy),
        instance_id="i", session_id="s",
    )
    assert len(written) == 1


def test_the_row_can_be_built_without_writing_it(engine, taxonomy):
    store = FakeStore()
    decision = engine.evaluate("send_money", 0.31, "taxonomy", taxonomy)
    row = AuditLogger(store).observation(
        decision, instance_id="i", session_id="s", episode_id="epi_1"
    )
    assert isinstance(row, Observation)
    assert row.episode_id == "epi_1"
    assert store.observations == [], "building a row writes nothing"


def test_a_shadow_decision_records_what_would_have_happened(engine, taxonomy):
    store = FakeStore()
    decision = engine.evaluate("send_money", 0.31, "taxonomy", taxonomy, shadow=True)
    AuditLogger(store).record(decision, instance_id="i", session_id="s")
    (row,) = store.observations
    assert row.payload["permitted"] is True
    assert row.payload["shadow"] is True
    assert row.payload["reason"] == "source_insufficient"


def test_a_policy_decision_is_constructible_directly():
    """The dataclass is part of the contract, not only what evaluate returns."""
    decision = PolicyDecision(
        permitted=False, reason=Reason.BELOW_FLOOR, floor=0.85, confidence=0.31
    )
    assert decision.shadow is False
    assert decision.would_have_blocked is True
