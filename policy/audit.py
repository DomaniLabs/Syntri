"""
The audit trail: every policy decision, written down before anything acts.

One row per evaluation, permitted or refused, appended to the experience
store as an `Observation` — the same append-only log everything else in
the platform is recorded in, so a policy refusal sits in the timeline
next to the utterance that caused it and the session it happened in.

Refusals are the obvious reason to keep this. The less obvious one is
permissions: "the floor was 0.85, the classifier said 0.91, we let it
through" is the record you need when the transfer turns out to have been
wrong, and it does not exist unless it is written at the time. So both
are logged, always, before the pack is called.

WHEN THE WRITE FAILS. This writes, and writing can fail. A failed
audit write must not fail the turn, and it must not silently permit
either: `record()` returns False and logs the failure. The caller has
already got its decision from the engine and enforces it regardless of
whether the row landed. A dropped row is a gap in the audit trail, not
an open gate.

Backend-agnostic on purpose: it is typed against `Store`, so it writes
to SQLite, Postgres or flat files without knowing which. Only
`append_observations` is actually called, so an in-memory fake in a test
needs one method — but the annotation names the interface, because that
is the contract and not the subset one caller happens to use.
"""
from __future__ import annotations

import logging
from typing import Any

from experience.schema import Actor, Observation, Origin
from policy.engine import PolicyDecision
from store.base import Store

log = logging.getLogger(__name__)

#: Channel these rows are filed under, so they can be found and excluded.
POLICY_CHANNEL = "policy"

#: `source` on the Observation. Distinguishes an audit row from a real
#: utterance at a glance and in SQL.
POLICY_SOURCE = "policy.engine"


class AuditLogger:
    """
    Writes policy decisions into an experience store.

    Constructed per request or once per process; it holds nothing but the
    store reference.
    """

    def __init__(self, store: Store) -> None:
        self.store = store

    def record(
        self,
        decision: PolicyDecision,
        *,
        instance_id: str,
        session_id: str,
        idempotency_key: str | None = None,
        text: str = "",
        episode_id: str | None = None,
    ) -> bool:
        """
        Append one decision. Returns whether the row was written.

        `text` is left empty by default. These rows are keyed on the
        decision, not on what the user said, and the utterance is already
        stored — under tokenisation — by the ingestion path. Copying it
        here would duplicate PII into a second row that the erasure rule
        does not know to rewrite.
        """
        observation = self.observation(
            decision,
            instance_id=instance_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            text=text,
            episode_id=episode_id,
        )
        try:
            self.store.append_observations([observation])
        except Exception as exc:  # noqa: BLE001 - never fail the turn on audit
            log.error(
                "policy audit write failed instance=%s session=%s intent=%s "
                "reason=%s: %s",
                instance_id, session_id, decision.intent, decision.reason, exc,
            )
            return False

        # Refusals are warnings: each one is a conversation that stopped.
        # A run of them means a floor is wrong or a model has drifted, and
        # that should be visible without opening the database.
        if decision.would_have_blocked:
            log.warning(
                "policy %s instance=%s session=%s intent=%s confidence=%.4f "
                "floor=%s source=%s shadow=%s",
                decision.reason, instance_id, session_id, decision.intent,
                decision.confidence, decision.floor, decision.source,
                decision.shadow,
            )
        return True

    def observation(
        self,
        decision: PolicyDecision,
        *,
        instance_id: str,
        session_id: str,
        idempotency_key: str | None = None,
        text: str = "",
        episode_id: str | None = None,
    ) -> Observation:
        """
        The row `record()` would write.

        Separate so a caller can batch several decisions into one append,
        and so the shape can be asserted on directly in a test.
        """
        payload: dict[str, Any] = {
            "kind": "policy_decision",
            **decision.to_dict(),
        }
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key

        return Observation(
            instance_id=instance_id,
            session_id=session_id,
            episode_id=episode_id,
            # SYSTEM, not USER: this is the platform recording itself.
            # The corpus builder reads user turns; an audit row must never
            # become a training example.
            actor=Actor.SYSTEM,
            channel=POLICY_CHANNEL,
            text=text,
            payload=payload,
            origin=Origin.PRODUCTION,
            source=POLICY_SOURCE,
        )
