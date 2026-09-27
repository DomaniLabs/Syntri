"""
The policy engine: may this classification be acted on?

Sits between understanding and doing. The NLU layer says "I think this is
`send_money` and I am 0.31 sure"; the payments pack, given that, will
happily start assembling a transfer. Something has to stand between them
and say no. That is this.

    utterance -> NLUPipeline -> PolicyEngine -> CapabilityPack
                                     |
                                  AuditLogger

Stateless by construction. The engine holds no instance, no session, and
no store; `evaluate()` is a pure function of its arguments. Everything
with a side effect — writing the audit row, returning the error, deciding
whether the session advances — belongs to the caller. This is so the
engine can be reasoned about and tested on its own, and so that the
question "why was this refused" always has an answer that does not depend
on what else had happened that day.


TWO RULES, AND WHY THEY ARE NOT THE SAME RULE
---------------------------------------------

**The floor.** A trained classifier emits a calibrated probability. The
taxonomy declares, per intent, how much of one is required before acting:
0.85 for money, 0.90 for irreversible, lower down the risk tiers. Below
it, refuse — `below_floor`.

**The source.** The cold-start fallback in `nlu/pipeline.py` is not a
classifier. It scores token overlap against whatever example sentences
the developer happened to type into the taxonomy, and the number it
returns is not a probability of anything. It is reported as
`source="taxonomy"` precisely so that this layer can tell the difference.

Those two facts compose into the rule that matters: **an untrusted source
may never dispatch a risky intent, at any reported confidence**
(`source_insufficient`). Not 0.31, not 0.99. A token-overlap score of
0.99 means the user's sentence looked like an example, which is not
evidence that they want money moved.

It also means the floor comparison is applied where it means something —
to a source that emits probabilities — and not to a score from a
different scale entirely. Comparing an overlap ratio against 0.85 would
be arithmetic, not a safety property: it would read as strict (almost
nothing clears it) while resting on a number that was never calibrated,
and it would silently disable every non-risky cold-start flow on an
untrained instance, which is every instance before its first training
run. `enforce_floor_on_untrusted=True` turns that comparison on for
operators who want it anyway; the risk rule above does not depend on it
and cannot be switched off.

So, for an untrained instance: flight search works, balance checks work,
and `send_money` is refused until a classifier exists to ask. That is the
intended cold-start posture — the fallback can drive conversations, it
cannot authorise anything.


SHADOW MODE
-----------

An instance in shadow mode is meant to produce decisions and act on none
of them. A shadow evaluation therefore returns `permitted=True` with
`shadow=True` and the *underlying* reason intact, so the audit row
records what the engine would have done. The caller is responsible for
the second half of that contract — a shadow decision must not advance the
state machine — and `PolicyDecision.would_have_blocked` is how it tells.

Note that shadow is an argument, not something the engine reads off an
instance. `api/agent.py` deliberately does not pass the instance's
`shadow_mode` flag today: the turn path does not yet suppress anything
else for a shadow instance, so treating shadow as a policy bypass there
would turn the flag into a hole in the floor rather than a dry run. See
the comment at the call site.
"""
from __future__ import annotations

from dataclasses import dataclass

from policy.floors import FAIL_CLOSED, is_risky, resolve_floor


class Reason:
    """Machine-readable outcomes. Callers match on these, not on prose."""

    PERMITTED = "permitted"
    BELOW_FLOOR = "below_floor"
    SOURCE_INSUFFICIENT = "source_insufficient"
    NO_INTENT = "no_intent"


#: Reasons that describe a refusal rather than a pass.
BLOCKING_REASONS = frozenset({
    Reason.BELOW_FLOOR,
    Reason.SOURCE_INSUFFICIENT,
    Reason.NO_INTENT,
})

#: Classification sources whose confidence is a calibrated probability and
#: may therefore be measured against a floor. Everything else — the
#: taxonomy fallback, "none", anything a future source reports — is
#: untrusted until it is added here on purpose.
TRUSTED_SOURCES = frozenset({"model", "classifier"})


@dataclass(frozen=True)
class PolicyDecision:
    """
    The verdict on one proposed dispatch.

    `reason` always describes what the engine concluded, even when
    `permitted` is True because of shadow mode. Read `permitted` to decide
    whether to call the pack; read `would_have_blocked` to decide whether
    the state machine may advance.
    """

    permitted: bool
    reason: str
    floor: float | None
    confidence: float
    shadow: bool = False
    intent: str = ""
    source: str = ""

    @property
    def would_have_blocked(self) -> bool:
        """True when only shadow mode is keeping this permitted."""
        return self.reason in BLOCKING_REASONS

    def to_dict(self) -> dict:
        return {
            "permitted": self.permitted,
            "reason": self.reason,
            "floor": self.floor,
            "confidence": round(self.confidence, 4),
            "shadow": self.shadow,
            "intent": self.intent,
            "source": self.source,
        }


class PolicyEngine:
    """
    Evaluates a proposed dispatch. Holds no state; safe to share or rebuild.

    Constructible with no arguments — the taxonomy arrives per call,
    because one process serves many instances and each has its own.
    """

    def __init__(
        self,
        *,
        default_floor: float = FAIL_CLOSED,
        trusted_sources: frozenset[str] | set[str] = TRUSTED_SOURCES,
        enforce_floor_on_untrusted: bool = False,
    ) -> None:
        self.default_floor = float(default_floor)
        self.trusted_sources = frozenset(trusted_sources)
        self.enforce_floor_on_untrusted = bool(enforce_floor_on_untrusted)

    def evaluate(
        self,
        intent: str,
        confidence: float,
        source: str,
        taxonomy=None,
        *,
        shadow: bool = False,
    ) -> PolicyDecision:
        """
        Decide whether `intent` at `confidence` from `source` may dispatch.

        Returns a decision in every case, including for inputs that make
        no sense. It never raises: a caller that cannot get an answer out
        of the policy layer would have to invent one, and the one it
        invented would be permissive.
        """
        confidence = _as_float(confidence)
        floor = resolve_floor(intent or "", taxonomy, default=self.default_floor)
        trusted = (source or "") in self.trusted_sources

        def decide(permitted: bool, reason: str) -> PolicyDecision:
            return PolicyDecision(
                permitted=True if shadow else permitted,
                reason=reason,
                floor=floor,
                confidence=confidence,
                shadow=shadow,
                intent=intent or "",
                source=source or "",
            )

        # Nothing was classified. There is no dispatch to permit.
        if not intent:
            return decide(False, Reason.NO_INTENT)

        # The floor, where the number it is being compared against means
        # something. An intent the taxonomy does not declare resolves to
        # FAIL_CLOSED and is refused here.
        if (trusted or self.enforce_floor_on_untrusted) and confidence < floor:
            return decide(False, Reason.BELOW_FLOOR)

        # The source. A fallback matcher does not get to move money,
        # whatever number it reported.
        if not trusted and is_risky(intent, taxonomy):
            return decide(False, Reason.SOURCE_INSUFFICIENT)

        return decide(True, Reason.PERMITTED)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        # An unreadable confidence is zero confidence, not a pass.
        return 0.0
