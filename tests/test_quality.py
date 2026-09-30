"""
Quality criteria: what makes an outcome a positive example.

`QualityConfig.evaluate` runs server-side against an outcome event's
payload. These tests pin the two gates (a signal on a field, a numeric
threshold on a score), how they combine, and the dot-notation resolution
they share — the behaviours a caller declaring "good" once relies on.
"""

from syntri_contracts.experience import (
    QualityConfig,
    QualityOperator,
    QualitySignal,
)


def test_good_signal_equals_true():
    config = QualityConfig(
        good_signal=QualitySignal(
            field="satisfied", operator=QualityOperator.EQUALS, value=True
        )
    )
    assert config.evaluate({"satisfied": True}) is True


def test_good_signal_equals_false():
    config = QualityConfig(
        good_signal=QualitySignal(
            field="satisfied", operator=QualityOperator.EQUALS, value=True
        )
    )
    assert config.evaluate({"satisfied": False}) is False


def test_good_signal_field_missing_returns_false():
    config = QualityConfig(
        good_signal=QualitySignal(
            field="satisfied", operator=QualityOperator.EQUALS, value=True
        )
    )
    assert config.evaluate({"something_else": True}) is False


def test_min_outcome_score_passes():
    config = QualityConfig(min_outcome_score=0.8)
    assert config.evaluate({"score": 0.9}) is True
    assert config.evaluate({"score": 0.8}) is True, "threshold is inclusive"


def test_min_outcome_score_fails():
    config = QualityConfig(min_outcome_score=0.8)
    assert config.evaluate({"score": 0.5}) is False


def test_min_outcome_score_non_numeric_returns_false():
    config = QualityConfig(min_outcome_score=0.8)
    assert config.evaluate({"score": "great"}) is False
    assert config.evaluate({"score": True}) is False, "bool is not a score"


def test_both_criteria_must_pass():
    config = QualityConfig(
        good_signal=QualitySignal(
            field="satisfied", operator=QualityOperator.EQUALS, value=True
        ),
        min_outcome_score=0.8,
    )
    assert config.evaluate({"satisfied": True, "score": 0.9}) is True
    assert config.evaluate({"satisfied": True, "score": 0.5}) is False
    assert config.evaluate({"satisfied": False, "score": 0.9}) is False


def test_no_criteria_always_passes():
    config = QualityConfig()
    assert config.evaluate({"anything": "at all"}) is True


def test_no_criteria_still_requires_all_three_by_default():
    config = QualityConfig()
    assert config.evaluate({}) is False, "empty outcome fails require_all_three"


def test_require_all_three_can_be_disabled():
    config = QualityConfig(require_all_three=False)
    assert config.evaluate({}) is True


def test_exists_operator():
    config = QualityConfig(
        good_signal=QualitySignal(field="handoff", operator=QualityOperator.EXISTS)
    )
    assert config.evaluate({"handoff": None}) is True, "present even if None"
    assert config.evaluate({"handoff": "human"}) is True
    assert config.evaluate({"other": 1}) is False


def test_not_exists_operator():
    config = QualityConfig(
        good_signal=QualitySignal(field="error", operator=QualityOperator.NOT_EXISTS)
    )
    assert config.evaluate({"score": 1}) is True
    assert config.evaluate({"error": "boom"}) is False


def test_dot_notation_nested_field():
    config = QualityConfig(
        good_signal=QualitySignal(
            field="payload.result.score", operator=QualityOperator.GTE, value=5
        )
    )
    assert config.evaluate({"payload": {"result": {"score": 7}}}) is True
    assert config.evaluate({"payload": {"result": {"score": 2}}}) is False
    assert config.evaluate({"payload": {"result": {}}}) is False


def test_score_field_dot_notation():
    config = QualityConfig(min_outcome_score=0.5, score_field="payload.metrics.csat")
    assert config.evaluate({"payload": {"metrics": {"csat": 0.6}}}) is True
    assert config.evaluate({"payload": {"metrics": {"csat": 0.4}}}) is False


def test_contains_operator():
    config = QualityConfig(
        good_signal=QualitySignal(
            field="tags", operator=QualityOperator.CONTAINS, value="resolved"
        )
    )
    assert config.evaluate({"tags": ["resolved", "billing"]}) is True
    assert config.evaluate({"tags": ["open"]}) is False
    assert config.evaluate({"tags": 42}) is False, "non-container fails closed"
