"""
Policy: what the agent is allowed to act on.

The gate between understanding an utterance and doing something about it.
`engine.py` decides, `floors.py` says what the bar is, `audit.py` writes
down what happened. Nothing here is specific to a customer or a domain —
every number it enforces comes out of the instance's own taxonomy.
"""
from policy.audit import POLICY_CHANNEL, POLICY_SOURCE, AuditLogger
from policy.engine import (
    BLOCKING_REASONS,
    TRUSTED_SOURCES,
    PolicyDecision,
    PolicyEngine,
    Reason,
)
from policy.floors import FAIL_CLOSED, is_risky, resolve_floor

__all__ = [
    "AuditLogger",
    "BLOCKING_REASONS",
    "FAIL_CLOSED",
    "POLICY_CHANNEL",
    "POLICY_SOURCE",
    "PolicyDecision",
    "PolicyEngine",
    "Reason",
    "TRUSTED_SOURCES",
    "is_risky",
    "resolve_floor",
]
