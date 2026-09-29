"""
The store contract: what the engine is allowed to assume about storage.

Everything above this line — the capture path, the policy audit logger,
the corpus builder, the training pipeline — depends on `Store` and on
nothing else. Which backend is underneath is a deployment decision made
once, in `main.py`, and is invisible everywhere afterwards.

Nine abstract methods: four appends, three reads, and the two halves of
the subject lifecycle — recording the links and dropping them. Keeping
the surface small is deliberate. A backend that grows a method the
interface does not name has grown something only one caller can use, and
the next backend has to reproduce it or the caller breaks.

`append_subject_map` is on the interface rather than left to convention
because `forget` is a statutory obligation, not a nicety. Erasure that
works only when a backend happens to have implemented the write side is
erasure that fails silently for the first backend that did not.


APPEND-ONLY, AND WHAT THAT LEAVES OUT
-------------------------------------

There is no update and no delete. A record that turns out to be wrong is
superseded by a later one, never edited: a Judgment supersedes a
Judgment, a fresh Interpretation outranks a stale one. The log is the
evidence, and evidence that can be rewritten is not evidence.

`forget` is the single exception and is not a delete of content. It drops
the subject-map entries that link a person to their records, which makes
them unlinkable while leaving the tokenised text — the training corpus —
intact. See `Store.forget`.


A NOTE ON RETURN TYPES
----------------------

The append methods and `forget` return `int`, not `bool`: the number of
rows written or dropped. A count answers "did it work" the same way a
boolean does — 0 is falsy, any success is truthy, so `if
store.append_observations(rows):` reads identically — and it also
answers "how many", which is what the ingest log, the erasure audit and
the training job report all need. A boolean would have to be widened to
an int the first time one of them asked, and both backends already
counted.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from syntri_contracts.experience.schema import (
    Episode,
    Interpretation,
    Judgment,
    Observation,
    SubjectMapEntry,
)


class Store(ABC):
    """
    The append-only experience store, as the engine sees it.

    Implemented by `ExperienceStore` (SQLite/Postgres, the production
    backend) and `JsonlStore` (flat files, the lightweight one). See
    `syntri/store/__init__.py` for which to deploy.
    """

    # -- writes ------------------------------------------------------------

    @abstractmethod
    def append_observations(self, rows: list[Observation]) -> int:
        """Append observations. Returns how many were written."""

    @abstractmethod
    def append_interpretations(self, rows: list[Interpretation]) -> int:
        """Append interpretations. Returns how many were written."""

    @abstractmethod
    def append_judgments(self, rows: list[Judgment]) -> int:
        """Append judgments. Returns how many were written."""

    @abstractmethod
    def append_episodes(self, rows: list[Episode]) -> int:
        """Append episodes. Returns how many were written."""

    # -- reads -------------------------------------------------------------

    @abstractmethod
    def observations(
        self,
        instance_id: str,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[Observation]:
        """
        One instance's observations, oldest first.

        Ordered by `occurred_at` so a corpus built twice from the same
        data comes out in the same order, which is half of what makes a
        training run reproducible. `since` is inclusive and compares
        against `occurred_at` — when the user sent the message, not when
        it was stored, so a backfill does not look like a spike of new
        traffic.

        `limit` caps the result, oldest first. It exists because a read
        over a year of a busy instance's data would otherwise try to
        hold all of it in memory at once. `limit=0` means no cap, which
        is what a corpus build passes — it genuinely does want all of
        it, and saying so explicitly is better than a caller guessing a
        number large enough.
        """

    @abstractmethod
    def interpretations(
        self,
        observation_ids: list[str],
        labeler: str | None = None,
    ) -> list[Interpretation]:
        """
        Every interpretation of the given observations, in one call.

        Batched rather than one lookup per observation: the corpus
        builder asks about every observation it just read, and a
        per-observation round trip turns one query into thousands.

        An empty `observation_ids` returns an empty list rather than
        everything — the degenerate case of "interpretations for no
        observations" is none of them, and the other reading would have
        a caller with an empty batch accidentally load the table.
        """

    @abstractmethod
    def episodes(
        self,
        instance_id: str,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[Episode]:
        """
        One instance's episodes, oldest first by `started_at`.

        Episodes carry the outcome a conversation reached, which is what
        weights a training example: a completed session with no
        corrections is worth more than an abandoned one.

        `since` and `limit` behave as in `observations`, including
        `limit=0` for no cap.
        """

    # -- the subject lifecycle ---------------------------------------------

    @abstractmethod
    def append_subject_map(
        self,
        entries: list[SubjectMapEntry],
        instance_id: str,
    ) -> int:
        """
        Record the links `forget` will later drop. Returns rows written.

        This is the other half of erasure, and it is on the interface for
        the same reason `forget` is. A store that can erase a subject but
        has no contractual way of having recorded one is a store where
        erasure works by convention: the first backend written by someone
        who has not read this docstring satisfies `Store`, accepts
        captures, and silently erases nothing. Both halves are declared,
        so that backend fails at implementation time instead.

        Each `SubjectMapEntry` carries the `observation_id` it came from,
        the `placeholder` (`{PHONE_1}`) and the `fingerprint` it stands
        for. **Never a raw identifier anywhere.** The fingerprint is what
        `Observation.subject_key` holds and what `forget` matches on; the
        raw phone number was turned into it at ingestion and does not
        exist in the store.

        WHY OBSERVATION-SCOPED, AND WHY A LIST

        The same placeholder means different people in different
        observations — `{PHONE_1}` in one message is not `{PHONE_1}` in
        the next — so a link is only meaningful together with the
        observation it was extracted from. Scoping every entry to its
        `observation_id` is what keeps two people from collapsing onto one
        row.

        A list because that is the shape the data has: a single ingest
        batch merges the token maps of every turn in every session it
        carries, so one call routinely spans several observations and
        several subjects. Use this when writing what a batch produced —
        the ingest path, a bulk import. Use `append_subject()` below when
        you have one subject in hand.
        """

    def append_subject(
        self,
        observation_id: str,
        subject_id: str,
        tokens: dict[str, str],
        instance_id: str,
    ) -> int:
        """
        Record one subject's tokens for a single observation.

        The per-observation view of `append_subject_map`, and the one to
        reach for when a caller has a single subject in hand: a connector
        translating one inbound message, a test, a script repairing one
        person's links.

        Concrete, not abstract, and deliberately so. It is defined in
        terms of the abstract method above, which means a backend
        implements one thing and gets both, and there is no way for the
        two to disagree about what was written.

        `subject_id` is a fingerprint. Passing a raw phone number here
        would write the raw phone number into the store, which is the
        one thing this whole mechanism exists to prevent — tokenise
        first, at ingestion, in the connector.
        """
        entries = [
            SubjectMapEntry(
                observation_id=observation_id,
                placeholder=placeholder,
                fingerprint=subject_id,
            )
            for placeholder in tokens
        ]
        return self.append_subject_map(entries, instance_id)

    # -- erasure -----------------------------------------------------------

    @abstractmethod
    def forget(self, subject_id: str) -> int:
        """
        Make one subject unlinkable. Returns how many links were dropped.

        `subject_id` is a fingerprint — the value `Observation.subject_key`
        holds — never a phone number or an email. Tokenisation at
        ingestion is what guarantees there is no raw identifier in the
        store to match against.

        What is dropped is the subject map. The observations stay, with
        their text as it was written: tokenised, placeholders where the
        identifiers were, and now nothing linking them to a person. That
        is the erasure rule — the corpus has to outlive the subject, or
        every erasure request would be a hole in the training data.
        """
