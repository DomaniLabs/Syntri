"""
The JSONL store backend: atomicity, batching and erasure.

Not the platform's store — see the note in `store/__init__.py`.
This is the append-only subset as flat files, for a corpus, an export or
an audit sink on a box with no database.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from syntri_contracts.experience.schema import (
    Actor,
    Episode,
    Interpretation,
    Judgment,
    JudgmentSource,
    Observation,
    Outcome,
)
from syntri_contracts.store import JsonlStore
from syntri_contracts.store.jsonl import (
    EPISODES,
    INTERPRETATIONS,
    JUDGMENTS,
    OBSERVATIONS,
    SUBJECTS,
    TMP_SUFFIX,
)


@pytest.fixture
def store(tmp_path) -> JsonlStore:
    return JsonlStore(tmp_path / "store")


def obs(**kwargs) -> Observation:
    base = {"instance_id": "inst_1", "session_id": "s1", "text": "hello"}
    return Observation(**{**base, **kwargs})


def lines(store: JsonlStore, name: str) -> list[dict]:
    path = store.file(name)
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_the_directory_is_created(tmp_path):
    path = tmp_path / "nested" / "store"
    assert not path.exists()

    JsonlStore(path)

    assert path.is_dir()


def test_an_existing_directory_is_reused(tmp_path):
    first = JsonlStore(tmp_path / "store")
    first.append_observations([obs()])

    second = JsonlStore(tmp_path / "store")

    assert len(list(second.iter_observations())) == 1


def test_a_missing_file_reads_as_empty(store):
    assert list(store.iter_observations()) == []
    assert list(store.iter_judgments()) == []
    assert store.subjects() == []


# --------------------------------------------------------------------------
# one file per record type
# --------------------------------------------------------------------------


def test_each_record_type_gets_its_own_file(store):
    store.append_observations([obs()])
    store.append_interpretations([
        Interpretation(observation_id="obs_1", instance_id="inst_1", labeler="m")
    ])
    store.append_judgments([
        Judgment(
            observation_id="obs_1", instance_id="inst_1",
            source=JudgmentSource.HUMAN, judge="ada",
        )
    ])
    store.append_episodes([Episode(instance_id="inst_1", session_id="s1")])

    for name in (OBSERVATIONS, INTERPRETATIONS, JUDGMENTS, EPISODES):
        assert store.file(name).exists(), name
        assert len(lines(store, name)) == 1


def test_records_round_trip_through_the_file(store):
    original = obs(text="send 5k", actor=Actor.USER, subject_key="fp_abc")

    store.append_observations([original])
    (restored,) = list(store.iter_observations())

    assert restored.observation_id == original.observation_id
    assert restored.text == "send 5k"
    assert restored.actor is Actor.USER
    assert restored.subject_key == "fp_abc"
    assert restored.occurred_at == original.occurred_at


def test_an_episode_round_trips_with_its_outcome(store):
    store.append_episodes([
        Episode(instance_id="inst_1", session_id="s1", outcome=Outcome.COMPLETED)
    ])

    (restored,) = list(store.iter_episodes())
    assert restored.outcome is Outcome.COMPLETED


def test_reads_can_be_filtered_by_instance(store):
    store.append_observations([obs(instance_id="a"), obs(instance_id="b")])

    assert len(list(store.iter_observations("a"))) == 1
    assert len(list(store.iter_observations())) == 2


# --------------------------------------------------------------------------
# appending
# --------------------------------------------------------------------------


def test_appends_accumulate_rather_than_overwrite(store):
    store.append_observations([obs(text="one")])
    store.append_observations([obs(text="two")])

    assert [o.text for o in store.iter_observations()] == ["one", "two"]


def test_a_batch_writes_every_record(store):
    written = store.append_observations([obs(text=str(i)) for i in range(20)])

    assert written == 20
    assert len(lines(store, OBSERVATIONS)) == 20


def test_an_empty_batch_writes_nothing_and_creates_no_file(store):
    assert store.append_observations([]) == 0
    assert not store.file(OBSERVATIONS).exists()


def test_every_row_is_one_line(store):
    """A reader that splits on newlines has to be right."""
    store.append_observations([obs(text="line one\nline two")])

    raw = store.file(OBSERVATIONS).read_text()
    assert raw.count("\n") == 1
    assert "line one\\nline two" in raw


# --------------------------------------------------------------------------
# atomicity
# --------------------------------------------------------------------------


def test_a_batch_that_cannot_serialise_writes_none_of_itself(store, monkeypatch):
    """
    All or nothing, and the "nothing" half. Every row is encoded before
    the file is opened, so a bad record leaves the file as it was rather
    than half-extended.
    """
    store.append_observations([obs(text="before")])

    import syntri_contracts.store.jsonl as jsonl

    real_dumps = jsonl.json.dumps
    calls = {"n": 0}

    def exploding_dumps(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise TypeError("not serialisable")
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(jsonl.json, "dumps", exploding_dumps)

    with pytest.raises(TypeError):
        store.append_observations([obs(text=str(i)) for i in range(5)])

    monkeypatch.undo()
    assert [o.text for o in store.iter_observations()] == ["before"]


def test_the_original_survives_a_crash_during_the_write(store, monkeypatch):
    """
    The rename is the commit. Failing before it leaves the previous file
    exactly as it was, whatever was staged in the temporary copy.
    """
    store.append_observations([obs(text="committed")])

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError):
        store.append_observations([obs(text="lost")])

    monkeypatch.undo()
    assert [o.text for o in store.iter_observations()] == ["committed"]


def test_no_temporary_file_is_left_behind(store):
    store.append_observations([obs()])
    store.append_observations([obs()])

    leftovers = list(Path(store.path).glob(f"*{TMP_SUFFIX}"))
    assert leftovers == []


def test_the_file_is_never_observed_half_written(store, monkeypatch):
    """
    The property the tmp-and-rename exists for: a reader between the two
    appends sees the first batch whole, never the second batch partly.
    """
    store.append_observations([obs(text="first")])
    seen: list[int] = []

    real_replace = os.replace

    def watching_replace(src, dst):
        # Mid-write: the staged copy exists, the live file has not moved.
        seen.append(len(lines(store, OBSERVATIONS)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", watching_replace)
    store.append_observations([obs(text="second"), obs(text="third")])
    monkeypatch.undo()

    assert seen == [1], "the live file still held only the first batch"
    assert len(lines(store, OBSERVATIONS)) == 3


# --------------------------------------------------------------------------
# the subject map
# --------------------------------------------------------------------------


def test_a_subject_map_entry_is_written(store):
    assert store.append_subject_map({"__subject__fp_1": "fp_1"}, "inst_1") == 1

    (row,) = store.subjects()
    assert row == {
        "placeholder": "fp_1", "fingerprint": "fp_1", "instance_id": "inst_1"
    }


def test_a_placeholder_entry_keeps_both_halves(store):
    """
    Same convention as ExperienceStore: without the `__subject__` prefix
    the key is a placeholder standing for a fingerprint, not one itself.
    """
    store.append_subject_map({"{PHONE_1}": "fp_2"}, "inst_1")

    (row,) = store.subjects()
    assert row["placeholder"] == "{PHONE_1}"
    assert row["fingerprint"] == "fp_2"


def test_an_empty_subject_map_writes_nothing(store):
    assert store.append_subject_map({}, "inst_1") == 0
    assert not store.file(SUBJECTS).exists()


# --------------------------------------------------------------------------
# forget
# --------------------------------------------------------------------------


def test_forget_drops_the_subject_map_entry(store):
    store.append_subject_map({"__subject__fp_1": "fp_1"}, "inst_1")
    store.append_subject_map({"__subject__fp_2": "fp_2"}, "inst_1")

    assert store.forget("fp_1") == 1

    assert [r["fingerprint"] for r in store.subjects()] == ["fp_2"]


def test_forget_keeps_the_corpus(store):
    """
    The erasure rule. The text was tokenised before it was written, so
    dropping the map leaves a sentence of placeholders with nothing to
    attach it to — and the training data survives the person.
    """
    store.append_observations([obs(text="send 5k to {ACCOUNT_NUMBER_1}",
                                   subject_key="fp_1")])
    store.append_subject_map({"__subject__fp_1": "fp_1"}, "inst_1")

    store.forget("fp_1")

    (row,) = list(store.iter_observations())
    assert row.text == "send 5k to {ACCOUNT_NUMBER_1}"


def test_forget_unlinks_the_observations_it_leaves(store):
    store.append_observations([
        obs(text="mine", subject_key="fp_1"),
        obs(text="theirs", subject_key="fp_2"),
    ])
    store.append_subject_map({"__subject__fp_1": "fp_1"}, "inst_1")

    store.forget("fp_1")

    keys = {o.text: o.subject_key for o in store.iter_observations()}
    assert keys == {"mine": None, "theirs": "fp_2"}


def test_forget_is_idempotent(store):
    store.append_subject_map({"__subject__fp_1": "fp_1"}, "inst_1")

    assert store.forget("fp_1") == 1
    assert store.forget("fp_1") == 0


def test_forgetting_an_unknown_subject_changes_nothing(store):
    store.append_observations([obs(subject_key="fp_1")])
    store.append_subject_map({"__subject__fp_1": "fp_1"}, "inst_1")

    assert store.forget("fp_nobody") == 0
    assert len(store.subjects()) == 1
    assert list(store.iter_observations())[0].subject_key == "fp_1"


def test_forgetting_nothing_is_a_no_op(store):
    assert store.forget("") == 0


def test_forget_rewrites_atomically(store):
    store.append_observations([obs(subject_key="fp_1") for _ in range(5)])
    store.append_subject_map({"__subject__fp_1": "fp_1"}, "inst_1")

    store.forget("fp_1")

    assert len(lines(store, OBSERVATIONS)) == 5, "every row is still there"
    assert list(Path(store.path).glob(f"*{TMP_SUFFIX}")) == []


# --------------------------------------------------------------------------
# resilience
# --------------------------------------------------------------------------


def test_a_corrupt_line_does_not_hide_the_rest(store, caplog):
    """
    Only reachable if something outside the class wrote the file, but a
    single bad line must not cost every record after it.
    """
    store.append_observations([obs(text="one"), obs(text="two")])
    with open(store.file(OBSERVATIONS), "a") as handle:
        handle.write("{not json\n")
    store.append_observations([obs(text="three")])

    texts = [o.text for o in store.iter_observations()]

    assert texts == ["one", "two", "three"]


def test_blank_lines_are_skipped(store):
    store.append_observations([obs(text="one")])
    with open(store.file(OBSERVATIONS), "a") as handle:
        handle.write("\n\n")

    assert len([o for o in store.iter_observations()]) == 1


# --------------------------------------------------------------------------
# batch parity with ExperienceStore
# --------------------------------------------------------------------------


def test_append_batch_writes_every_record_type(store):
    class Batch:
        observations = [obs()]
        interpretations = [
            Interpretation(observation_id="o", instance_id="inst_1", labeler="m")
        ]
        episodes = [Episode(instance_id="inst_1", session_id="s1")]
        judgments: list = []
        subject_map = {"__subject__fp_1": "fp_1"}

    counts = store.append_batch(Batch())

    assert counts["observations"] == 1
    assert counts["interpretations"] == 1
    assert counts["episodes"] == 1
    assert counts["judgments"] == 0
    assert counts["subject_map"] == 1


# --------------------------------------------------------------------------
# it satisfies the audit logger's duck type
# --------------------------------------------------------------------------


def test_the_audit_logger_can_write_to_it(store):
    """
    `AuditLogger` asks for `append_observations` and nothing else, so a
    deployment with no database can still keep a policy audit trail.
    """
    from syntri_contracts.policy import AuditLogger, PolicyEngine

    decision = PolicyEngine().evaluate("send_money", 0.31, "taxonomy")

    assert AuditLogger(store).record(
        decision, instance_id="inst_1", session_id="s1"
    ) is True

    (row,) = list(store.iter_observations())
    assert row.payload["reason"] == "source_insufficient"
