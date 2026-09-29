"""
Append-only records as JSONL files on disk.

One file per record type under a directory:

    <path>/observations.jsonl
    <path>/interpretations.jsonl
    <path>/judgments.jsonl
    <path>/episodes.jsonl
    <path>/subjects.jsonl

Public-repo material: nothing here knows about a customer, an intent or
a channel. It is the experience contract, serialised.


WHY ATOMICITY MATTERS FOR A LOG YOU ONLY EVER APPEND TO
-------------------------------------------------------

A plain `open(..., "a")` and a loop over the batch has two failure modes
that matter. A crash halfway leaves some of a batch on disk and the rest
gone, with no way to tell which from the file. And a partially-flushed
final line leaves a truncated JSON object that every future reader
trips over — one interrupted write makes the file unparseable from that
point on.

So a batch is serialised in full, and only then does it touch the file:

    1. copy the existing file to `<name>.jsonl.tmp`
    2. append the whole batch to the copy
    3. flush, `fsync`, `os.replace` over the original

`os.replace` is atomic on POSIX and on Windows. A reader at any instant
sees either every record of the batch or none of them, never a half
line, and a crash at any point leaves the original file untouched.

The cost is that a write is O(file), so this is a capture and export
backend, not a high-throughput one. When the copy becomes the
bottleneck, that is the signal to be using ExperienceStore — which is
what the server already does.


CONCURRENCY
-----------

A lock per instance serialises writers inside one process. It does not
coordinate across processes: two servers pointed at one directory will
lose records, because each one's copy-and-replace overwrites whatever
the other appended in between. Single-writer only, and that is checked
nowhere — a file path cannot tell you who else has it open.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from syntri_contracts.experience.schema import (
    Episode,
    Interpretation,
    Judgment,
    Observation,
    SubjectMapEntry,
)
from syntri_contracts.store.base import Store

log = logging.getLogger(__name__)

OBSERVATIONS = "observations.jsonl"
INTERPRETATIONS = "interpretations.jsonl"
JUDGMENTS = "judgments.jsonl"
EPISODES = "episodes.jsonl"
SUBJECTS = "subjects.jsonl"

#: Suffix for the copy a batch is staged in. Left behind only by a crash
#: mid-write, and overwritten by the next one.
TMP_SUFFIX = ".tmp"


class JsonlStore(Store):
    """
    The append-only record types, as files.

    Implements `Store` in full, plus `append_subject_map` and the
    streaming `iter_*` readers that the interface does not name.

    The `Store` read methods filter in memory after scanning the whole
    file, which is the trade this backend exists to make: no database,
    no schema, no migration, and a read cost linear in everything ever
    written. Fine for one WhatsApp bot; wrong for a busy tenant, which
    is what `ExperienceStore` is for.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        # Re-entrant: forget() holds it across several file operations
        # that each take it again.
        self._lock = threading.RLock()

    # -- writes ------------------------------------------------------------

    def append_observations(self, rows: Sequence[Observation]) -> int:
        return self._append(OBSERVATIONS, rows)

    def append_interpretations(self, rows: Sequence[Interpretation]) -> int:
        return self._append(INTERPRETATIONS, rows)

    def append_judgments(self, rows: Sequence[Judgment]) -> int:
        return self._append(JUDGMENTS, rows)

    def append_episodes(self, rows: Sequence[Episode]) -> int:
        return self._append(EPISODES, rows)

    def append_subject_map(
        self, entries: list[SubjectMapEntry], instance_id: str
    ) -> int:
        """
        Record observation-scoped subject links, so `forget` has something
        to find.

        Mirrors `ExperienceStore.append_subject_map`: each entry is written
        as its own row, keyed on the `observation_id` it came from, so the
        same placeholder in two observations never collapses onto one row.
        """
        if not entries:
            return 0
        created_at = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "observation_id": entry.observation_id,
                "placeholder": entry.placeholder,
                "fingerprint": entry.fingerprint,
                "instance_id": instance_id,
                "created_at": created_at,
            }
            for entry in entries
        ]
        return self._append_raw(SUBJECTS, rows)

    def append_batch(self, batch: Any) -> dict[str, int]:
        """
        A whole ProcessedBatch, for parity with ExperienceStore.

        Each file is atomic on its own; the set of them is not. A crash
        between two of these leaves observations without their
        interpretations, which is recoverable — the pair is joined by
        observation_id, and a missing interpretation reads as one that
        was never made.
        """
        with self._lock:
            counts = {
                "observations": self.append_observations(
                    getattr(batch, "observations", []) or []
                ),
                "interpretations": self.append_interpretations(
                    getattr(batch, "interpretations", []) or []
                ),
                "episodes": self.append_episodes(
                    getattr(batch, "episodes", []) or []
                ),
                "judgments": self.append_judgments(
                    getattr(batch, "judgments", []) or []
                ),
            }
            subject_map = getattr(batch, "subject_map", None) or []
            instance_id = ""
            observations = getattr(batch, "observations", []) or []
            if observations:
                instance_id = observations[0].instance_id
            counts["subject_map"] = self.append_subject_map(
                subject_map, instance_id
            )
            return counts

    # -- reads -------------------------------------------------------------

    def iter_observations(self, instance_id: str | None = None) -> Iterator[Observation]:
        for row in self._read(OBSERVATIONS):
            if instance_id and row.get("instance_id") != instance_id:
                continue
            yield Observation.model_validate(row)

    def iter_interpretations(
        self, instance_id: str | None = None
    ) -> Iterator[Interpretation]:
        for row in self._read(INTERPRETATIONS):
            if instance_id and row.get("instance_id") != instance_id:
                continue
            yield Interpretation.model_validate(row)

    def iter_judgments(self, instance_id: str | None = None) -> Iterator[Judgment]:
        for row in self._read(JUDGMENTS):
            if instance_id and row.get("instance_id") != instance_id:
                continue
            yield Judgment.model_validate(row)

    def iter_episodes(self, instance_id: str | None = None) -> Iterator[Episode]:
        for row in self._read(EPISODES):
            if instance_id and row.get("instance_id") != instance_id:
                continue
            yield Episode.model_validate(row)

    # -- the Store read interface ------------------------------------------
    #
    # Thin list-returning wrappers over the iterators above. They scan
    # the file and filter in memory; see the class docstring.

    def observations(
        self,
        instance_id: str,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[Observation]:
        rows = [
            row for row in self.iter_observations(instance_id)
            if since is None or row.occurred_at >= since
        ]
        rows.sort(key=lambda row: row.occurred_at)
        return rows[:limit] if limit else rows

    def interpretations(
        self,
        observation_ids: list[str],
        labeler: str | None = None,
    ) -> list[Interpretation]:
        if not observation_ids:
            return []
        wanted = set(observation_ids)
        return [
            row for row in self.iter_interpretations()
            if row.observation_id in wanted
            and (labeler is None or row.labeler == labeler)
        ]

    def episodes(
        self,
        instance_id: str,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[Episode]:
        rows = [
            row for row in self.iter_episodes(instance_id)
            if since is None or row.started_at >= since
        ]
        rows.sort(key=lambda row: row.started_at)
        return rows[:limit] if limit else rows

    def subjects(self) -> list[dict[str, Any]]:
        return list(self._read(SUBJECTS))

    def count(self, filename: str) -> int:
        return sum(1 for _ in self._read(filename))

    # -- erasure -----------------------------------------------------------

    def forget(self, subject_id: str) -> int:
        """
        Make one subject unlinkable. Returns how many rows were dropped.

        `subject_id` is a fingerprint — the value `subject_key` holds,
        not a phone number. There is no raw identifier anywhere in this
        store to match against, which is the property tokenisation at
        ingestion buys.

        What goes is the subject-map entry; that is the whole erasure
        rule, and it is why observation text survives. The text was
        tokenised before it was written, so without the map it is a
        sentence with placeholders in it and nothing to attach it to.
        Deleting the observations too would take the training corpus
        with the person, and the corpus is the thing that must outlive
        them.

        The observation's own `subject_key` and `session_id` are cleared
        in place, so a reader holding only this directory cannot re-group a
        subject's messages by either column after the map has gone. Both
        fields are nulled in the same rewrite pass, not two.
        """
        if not subject_id:
            return 0

        with self._lock:
            removed = self._rewrite(
                SUBJECTS,
                lambda row: row.get("fingerprint") != subject_id
                and row.get("placeholder") != subject_id,
            )

            def unlink(row: dict[str, Any]) -> dict[str, Any]:
                if row.get("subject_key") == subject_id:
                    row = dict(row)
                    row["subject_key"] = None
                    row["session_id"] = None
                return row

            self._map_rows(OBSERVATIONS, unlink)

        log.info("forget %s: %d subject-map rows dropped", subject_id, removed)
        return removed

    # -- the file layer ----------------------------------------------------

    def file(self, name: str) -> Path:
        return self.path / name

    def _append(self, name: str, rows: Sequence[Any]) -> int:
        if not rows:
            return 0
        return self._append_raw(
            name, [json.loads(r.model_dump_json()) for r in rows]
        )

    def _append_raw(self, name: str, rows: Sequence[dict]) -> int:
        """
        Add rows to a file, all or nothing.

        Serialising every row before opening anything matters: a record
        that will not encode raises here, with the file untouched,
        rather than after half the batch is already staged.
        """
        if not rows:
            return 0
        lines = "".join(
            json.dumps(row, separators=(",", ":"), default=str) + "\n"
            for row in rows
        )

        with self._lock:
            target = self.file(name)
            tmp = target.with_name(target.name + TMP_SUFFIX)

            existing = target.read_bytes() if target.exists() else b""
            with open(tmp, "wb") as handle:
                handle.write(existing)
                handle.write(lines.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
            self._sync_dir()

        return len(rows)

    def _read(self, name: str) -> Iterator[dict[str, Any]]:
        target = self.file(name)
        if not target.exists():
            return
        with open(target, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Only reachable if something outside this class
                    # wrote the file. Skip the line rather than abandon
                    # every record after it.
                    log.warning("%s:%d is not JSON; skipped", target, number)

    def _rewrite(self, name: str, keep) -> int:
        """Drop every row `keep` rejects. Returns the number dropped."""
        rows = list(self._read(name))
        survivors = [row for row in rows if keep(row)]
        if len(survivors) == len(rows):
            return 0
        self._replace_all(name, survivors)
        return len(rows) - len(survivors)

    def _map_rows(self, name: str, fn) -> int:
        """Rewrite every row through `fn`. Returns the number changed."""
        rows = list(self._read(name))
        mapped = [fn(row) for row in rows]
        changed = sum(1 for before, after in zip(rows, mapped) if before != after)
        if changed:
            self._replace_all(name, mapped)
        return changed

    def _replace_all(self, name: str, rows: Sequence[dict]) -> None:
        """Rewrite a whole file atomically, through the same tmp-and-rename."""
        lines = "".join(
            json.dumps(row, separators=(",", ":"), default=str) + "\n"
            for row in rows
        )
        with self._lock:
            target = self.file(name)
            tmp = target.with_name(target.name + TMP_SUFFIX)
            with open(tmp, "wb") as handle:
                handle.write(lines.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
            self._sync_dir()

    def _sync_dir(self) -> None:
        """
        fsync the directory, so the rename itself survives a power cut.

        Without it the file's contents are durable but the directory
        entry pointing at them may not be. Not supported everywhere;
        where it is not, the rename is still atomic, just not durable.
        """
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def __repr__(self) -> str:
        return f"<JsonlStore path={str(self.path)!r}>"
