"""Capability pack machinery: registration and fixture replay.

Both modules are built on the public `contracts` package and nothing else.
`CapabilityRegistry` runs the startup checks the platform would (duplicate
intents, money-on-agent, empty manifests); `replay_pack` drives a pack's
fixtures through `advance()` with no server, network or model. Together they
are what `syntri pack validate` runs.
"""

from capabilities.registry import (
    ENTRY_POINT_GROUP,
    CapabilityRegistry,
    default_registry,
)
from capabilities.replay import (
    FixtureOutcome,
    ReplayReport,
    TurnOutcome,
    replay_fixture,
    replay_pack,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "CapabilityRegistry",
    "FixtureOutcome",
    "ReplayReport",
    "TurnOutcome",
    "default_registry",
    "replay_fixture",
    "replay_pack",
]
