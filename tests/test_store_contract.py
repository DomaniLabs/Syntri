"""The `Store` contract, run against the backends the public package ships.

Ported from syntri-core's test_store_contract.py. There, each test is
parametrised over both `ExperienceStore` and `JsonlStore`. `ExperienceStore`
is private and not extracted, so here the parametrisation is over `JsonlStore`
alone — but the fixture keeps the parametrised shape, so re-adding a backend is
a one-line change.

These tests assert on behaviour, not storage. Backend-specific file mechanics
(atomic renames, tmp files, corruption handling) live in test_jsonl_store.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from experience.schema import (
    Actor,
    Episode,
    Interpretation,
    Judgment,
    JudgmentSource,
    Observation,
    Outcome,
)
from store import JsonlStore
from store.base import Store

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(params=["jsonl"])
def store(request, tmp_path) -> Store:
    """One store per backend. Public ships JsonlStore only."""
    if request.param == "jsonl":
        return JsonlStore(tmp_path / "jsonl")
    raise AssertionError(f"unknown backend {request.param!r}")


def obs(instance_id="inst_1", offset_minutes=0, **kwargs) -> Observation:
    base = {
        "instance_id": instance_id,
        "session_id": "s1",
        "text": "hello",
        "occurred_at": NOW + timedelta(minutes=offset_minutes),
    }
    return Observation(**{**base, **kwargs})


# -- the backend is a Store ------------------------------------------------


def test_the_backend_satisfies_the_interface(store):
    assert isinstance(store, Store)


def test_every_abstract_method_is_implemented(store):
    for name in sorted(Store.__abstractmethods__):
        assert callable(getattr(store, name)), name


def test_the_interface_is_the_methods_the_engine_uses(store):
    assert sorted(Store.__abstractmethods__) == [
        "append_episodes",
        "append_interpretations",
        "append_judgments",
        "append_observations",
        "append_subject_map",
        "episodes",
        "forget",
        "interpretations",
        "observations",
    ]


def test_both_halves_of_the_subject_lifecycle_are_on_the_interface():
    assert "forget" in Store.__abstractmethods__
    assert "append_subject_map" in Store.__abstractmethods__


def test_a_backend_that_cannot_record_subjects_is_not_a_store():
    """The gap closes at implementation time, not at runtime."""
    class Amnesiac(Store):
        def append_observations(self, rows): return len(rows)
        def append_interpretations(self, rows): return len(rows)
        def append_judgments(self, rows): return len(rows)
        def append_episodes(self, rows): return len(rows)
        def observations(self, instance_id, since=None, limit=1000): return []
        def interpretations(self, observation_ids, labeler=None): return []
        def episodes(self, instance_id, since=None, limit=1000): return []
        def forget(self, subject_id): return 0

    with pytest.raises(TypeError, match="append_subject_map"):
        Amnesiac()  # type: ignore[abstract]


def test_the_abstract_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Store()  # type: ignore[abstract]


def test_a_partial_implementation_is_refused():
    class Halfway(Store):
        def append_observations(self, rows): return len(rows)

    with pytest.raises(TypeError):
        Halfway()  # type: ignore[abstract]


# -- writes ----------------------------------------------------------------


def test_appending_observations_reports_the_count(store):
    assert store.append_observations([obs(), obs()]) == 2


def test_appending_nothing_writes_nothing(store):
    assert store.append_observations([]) == 0
    assert store.append_interpretations([]) == 0
    assert store.append_judgments([]) == 0
    assert store.append_episodes([]) == 0


def test_every_record_type_can_be_appended(store):
    written = obs()
    store.append_observations([written])
    assert store.append_interpretations([
        Interpretation(
            observation_id=written.observation_id,
            instance_id="inst_1", labeler="m", intent="check_balance",
        )
    ]) == 1
    assert store.append_judgments([
        Judgment(
            observation_id=written.observation_id, instance_id="inst_1",
            source=JudgmentSource.HUMAN, judge="ada",
        )
    ]) == 1
    assert store.append_episodes([
        Episode(instance_id="inst_1", session_id="s1")
    ]) == 1


def test_appends_accumulate(store):
    store.append_observations([obs(text="one")])
    store.append_observations([obs(text="two")])
    assert len(store.observations("inst_1")) == 2


# -- observations() --------------------------------------------------------


def test_observations_round_trip(store):
    written = obs(text="send 5k", actor=Actor.USER, subject_key="fp_1")
    store.append_observations([written])
    (read,) = store.observations("inst_1")
    assert read.observation_id == written.observation_id
    assert read.text == "send 5k"
    assert read.actor is Actor.USER
    assert read.subject_key == "fp_1"
    assert read.instance_id == "inst_1"


def test_observations_are_scoped_to_one_instance(store):
    store.append_observations([obs(instance_id="a"), obs(instance_id="b")])
    assert len(store.observations("a")) == 1
    assert store.observations("a")[0].instance_id == "a"


def test_an_unknown_instance_reads_as_empty(store):
    store.append_observations([obs()])
    assert store.observations("nobody") == []


def test_observations_come_back_oldest_first(store):
    store.append_observations([
        obs(text="third", offset_minutes=30),
        obs(text="first", offset_minutes=0),
        obs(text="second", offset_minutes=10),
    ])
    assert [o.text for o in store.observations("inst_1")] == [
        "first", "second", "third"
    ]


def test_since_filters_on_when_the_user_spoke(store):
    store.append_observations([
        obs(text="old", offset_minutes=0),
        obs(text="new", offset_minutes=60),
    ])
    found = store.observations("inst_1", since=NOW + timedelta(minutes=30))
    assert [o.text for o in found] == ["new"]


def test_since_is_inclusive(store):
    store.append_observations([obs(text="exactly", offset_minutes=0)])
    assert len(store.observations("inst_1", since=NOW)) == 1


def test_limit_caps_the_result(store):
    store.append_observations([obs(offset_minutes=i) for i in range(10)])
    assert len(store.observations("inst_1", limit=3)) == 3


def test_limit_takes_the_oldest_not_an_arbitrary_three(store):
    store.append_observations([
        obs(text=str(i), offset_minutes=i) for i in range(5)
    ])
    assert [o.text for o in store.observations("inst_1", limit=3)] == ["0", "1", "2"]


# -- interpretations() -----------------------------------------------------


def test_interpretations_are_fetched_in_one_batch(store):
    first, second = obs(text="a"), obs(text="b")
    store.append_observations([first, second])
    store.append_interpretations([
        Interpretation(
            observation_id=first.observation_id, instance_id="inst_1",
            labeler="m", intent="check_balance", intent_confidence=0.9,
        ),
        Interpretation(
            observation_id=second.observation_id, instance_id="inst_1",
            labeler="m", intent="send_money", intent_confidence=0.8,
        ),
    ])
    found = store.interpretations([first.observation_id, second.observation_id])
    assert {i.intent for i in found} == {"check_balance", "send_money"}


def test_interpretations_round_trip_their_fields(store):
    written = obs()
    store.append_observations([written])
    store.append_interpretations([
        Interpretation(
            observation_id=written.observation_id, instance_id="inst_1",
            labeler="model:v1", labeler_kind="model",
            intent="check_balance", intent_confidence=0.91,
        )
    ])
    (found,) = store.interpretations([written.observation_id])
    assert found.intent == "check_balance"
    assert found.intent_confidence == pytest.approx(0.91)
    assert found.labeler == "model:v1"


def test_asking_about_no_observations_returns_none_of_them(store):
    """An empty request must not read as 'everything'."""
    written = obs()
    store.append_observations([written])
    store.append_interpretations([
        Interpretation(
            observation_id=written.observation_id, instance_id="inst_1", labeler="m",
        )
    ])
    assert store.interpretations([]) == []


def test_interpretations_can_be_filtered_by_labeler(store):
    written = obs()
    store.append_observations([written])
    store.append_interpretations([
        Interpretation(
            observation_id=written.observation_id, instance_id="inst_1",
            labeler="model:v1", intent="a",
        ),
        Interpretation(
            observation_id=written.observation_id, instance_id="inst_1",
            labeler="human:ada", intent="b",
        ),
    ])
    found = store.interpretations([written.observation_id], labeler="human:ada")
    assert [i.intent for i in found] == ["b"]


def test_an_observation_with_no_interpretation_contributes_nothing(store):
    written = obs()
    store.append_observations([written])
    assert store.interpretations([written.observation_id]) == []


# -- episodes() ------------------------------------------------------------


def test_episodes_round_trip(store):
    store.append_episodes([
        Episode(
            instance_id="inst_1", session_id="s1",
            outcome=Outcome.COMPLETED, turn_count=4, started_at=NOW,
        )
    ])
    (found,) = store.episodes("inst_1")
    assert found.outcome is Outcome.COMPLETED
    assert found.turn_count == 4


def test_episodes_are_scoped_to_one_instance(store):
    store.append_episodes([
        Episode(instance_id="a", session_id="s", started_at=NOW),
        Episode(instance_id="b", session_id="s", started_at=NOW),
    ])
    assert len(store.episodes("a")) == 1


def test_episodes_come_back_oldest_first(store):
    store.append_episodes([
        Episode(instance_id="inst_1", session_id="late",
                started_at=NOW + timedelta(hours=1)),
        Episode(instance_id="inst_1", session_id="early", started_at=NOW),
    ])
    assert [e.session_id for e in store.episodes("inst_1")] == ["early", "late"]


def test_episodes_can_be_filtered_and_capped(store):
    store.append_episodes([
        Episode(instance_id="inst_1", session_id=str(i),
                started_at=NOW + timedelta(hours=i))
        for i in range(5)
    ])
    recent = store.episodes("inst_1", since=NOW + timedelta(hours=2))
    assert [e.session_id for e in recent] == ["2", "3", "4"]
    assert len(store.episodes("inst_1", limit=2)) == 2


# -- append_subject_map() --------------------------------------------------


def test_a_subject_entry_round_trips(store):
    assert store.append_subject_map({"__subject__fp_1": "fp_1"}, "inst_1") == 1
    assert store.forget("fp_1") == 1, "the row is there to be found"


def test_a_token_placeholder_round_trips(store):
    assert store.append_subject_map({"{PHONE_1}": "fp_2"}, "inst_1") == 1
    assert store.forget("fp_2") == 1


def test_several_links_are_written_in_one_call(store):
    written = store.append_subject_map(
        {
            "__subject__fp_1": "fp_1",
            "{PHONE_1}": "fp_1",
            "{ACCOUNT_NUMBER_1}": "fp_3",
        },
        "inst_1",
    )
    assert written == 3


def test_an_empty_map_writes_nothing(store):
    assert store.append_subject_map({}, "inst_1") == 0


def test_append_subject_records_a_subject_without_the_convention(store):
    assert store.append_subject("fp_1", instance_id="inst_1") == 1
    assert store.forget("fp_1") == 1


def test_append_subject_records_its_tokens_too(store):
    written = store.append_subject(
        "fp_1", {"{PHONE_1}": "fp_1", "{EMAIL_1}": "fp_9"}, "inst_1"
    )
    assert written == 3, "the subject, plus one row per token"


def test_append_subject_without_a_subject_writes_nothing(store):
    assert store.append_subject("", {"{PHONE_1}": "fp_1"}, "inst_1") == 0


# -- forget() --------------------------------------------------------------


def test_forget_keeps_the_corpus(store):
    """The person becomes unlinkable; the training data survives."""
    store.append_observations([
        obs(text="send 5k to {ACCOUNT_NUMBER_1}", subject_key="fp_1")
    ])
    store.append_subject_map({"__subject__fp_1": "fp_1"}, "inst_1")
    store.forget("fp_1")
    (survivor,) = store.observations("inst_1")
    assert survivor.text == "send 5k to {ACCOUNT_NUMBER_1}"


def test_forget_reports_how_many_links_it_dropped(store):
    store.append_subject_map({"__subject__fp_1": "fp_1"}, "inst_1")
    assert store.forget("fp_1") == 1


def test_forget_is_idempotent(store):
    store.append_subject_map({"__subject__fp_1": "fp_1"}, "inst_1")
    store.forget("fp_1")
    assert store.forget("fp_1") == 0


def test_forgetting_an_unknown_subject_drops_nothing(store):
    store.append_observations([obs(subject_key="fp_1")])
    store.append_subject_map({"__subject__fp_1": "fp_1"}, "inst_1")
    assert store.forget("fp_nobody") == 0
    assert store.forget("fp_1") == 1, "the real subject is still linked"


def test_nothing_recoverable_survives_forget(store):
    store.append_observations([
        obs(text="send 5k to {ACCOUNT_NUMBER_1}", subject_key="fp_1")
    ])
    store.append_subject(
        "fp_1", {"{PHONE_1}": "fp_1", "{ACCOUNT_NUMBER_1}": "fp_1"}, "inst_1"
    )
    dropped = store.forget("fp_1")
    assert dropped == 3, "the subject entry and both of its tokens"
    assert store.forget("fp_1") == 0, "nothing is left to drop"
    (survivor,) = store.observations("inst_1")
    assert survivor.text == "send 5k to {ACCOUNT_NUMBER_1}"
    assert survivor.subject_key != "fp_1"


def test_forget_leaves_other_subjects_linked(store):
    store.append_subject("fp_1", {"{PHONE_1}": "fp_1"}, "inst_1")
    store.append_subject("fp_2", {"{PHONE_2}": "fp_2"}, "inst_1")
    store.forget("fp_1")
    assert store.forget("fp_2") == 2, "the other subject is untouched"
