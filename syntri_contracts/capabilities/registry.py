"""
The capability registry — what this instance can do, resolved once at startup.

Every check in `register` is a startup check, and every one of them raises
rather than warns. The reason is that all three failure modes are silent
at runtime and expensive in production:

  duplicate intents   two packs claim `send_money`. Whichever won the
                      dict insert handles the money. That is decided by
                      import order, which is decided by entry-point
                      iteration order, which is not a thing anyone should
                      be reasoning about when money moves.

  money on the agent  a tool with moves_money=True and executor=AGENT is
                      a pack that transfers funds itself. The contract
                      rejects it in ToolSpec's own validator; this is the
                      second gate, because a pack can build a ToolSpec
                      dynamically and the registry is the last place to
                      look before the tool is reachable.

  empty manifest      a pack that claims no intents can never be
                      selected, so it is dead weight that looks installed.

A misconfigured pack therefore fails the process, loudly, at boot — which
is the cheapest possible moment to find out.
"""
from __future__ import annotations

import logging
from importlib.metadata import entry_points

from syntri_contracts.contracts import (
    CapabilityManifest,
    CapabilityPack,
    Executor,
    ToolSpec,
)

log = logging.getLogger(__name__)

#: The attributes the registry actually reads off a manifest.
_MANIFEST_FIELDS = ("name", "display_name", "version", "intents")


def _executor_value(tool: object) -> str:
    """
    A tool's executor as a plain string.

    Compared by value, never with `is`. A third-party pack resolves
    `Executor.AGENT` through its own installation of `syntri-contracts`,
    which may be a different version, a different site-packages, or the
    same module imported under a different name. Any of those makes it a
    member of a different enum class, so an identity check against an
    external pack's ToolSpec is always False — which would make this the
    most dangerous kind of safety check: one that always passes.
    """
    executor = getattr(tool, "executor", None)
    return str(getattr(executor, "value", executor) or "").lower()


def _is_manifest(candidate: object) -> bool:
    """
    Whether an object can be used as a manifest.

    Checked structurally rather than with isinstance. Both sides import
    CapabilityManifest from `contracts.capability` now, but that is not
    enough to make them one class: a third-party pack ships its own
    resolution of `syntri-contracts` and can easily be running a
    different version, or have the module loaded twice under different
    names. Either way isinstance is False for a pack that is perfectly
    correct.

    The same applies to CapabilityPack itself — which is why `register`
    takes any object with `manifest`, `tools` and `advance`, and never
    asserts the base class.
    """
    if isinstance(candidate, CapabilityManifest):
        return True
    return all(hasattr(candidate, field) for field in _MANIFEST_FIELDS)

#: The entry-point group third-party packs advertise themselves under.
ENTRY_POINT_GROUP = "syntri.capabilities"


class CapabilityRegistry:
    """
    Maps intents to the pack that handles them, and tool names to specs.

    One registry per process. It is populated at startup and read on every
    turn, so it is built to be written once and read concurrently: nothing
    mutates after `load_entry_points` returns except via an explicit
    `register` call.
    """

    def __init__(self) -> None:
        self._packs: dict[str, CapabilityPack] = {}
        self._by_intent: dict[str, CapabilityPack] = {}
        self._tools: dict[str, ToolSpec] = {}

    # -- registration -----------------------------------------------------

    def register(self, pack: CapabilityPack) -> None:
        """
        Add a pack, or raise ValueError explaining why it cannot be added.

        Validation happens before *any* mutation, so a rejected pack
        leaves the registry exactly as it was. A half-registered pack
        whose intents landed but whose tools did not would be worse than
        the crash.
        """
        manifest = getattr(pack, "manifest", None)
        if manifest is None or not _is_manifest(manifest):
            raise ValueError(
                f"{type(pack).__name__} has no usable manifest — a pack must "
                f"set the 'manifest' class variable to a CapabilityManifest"
            )

        for required in ("tools", "advance"):
            if not callable(getattr(pack, required, None)):
                raise ValueError(
                    f"{type(pack).__name__} does not implement {required}() — "
                    f"a pack must subclass CapabilityPack and implement both "
                    f"tools() and advance()"
                )

        if not manifest.intents:
            raise ValueError(
                f"pack '{manifest.name}' declares no intents; a pack that "
                f"claims no intent can never be selected"
            )

        if manifest.name in self._packs:
            raise ValueError(
                f"pack '{manifest.name}' is already registered "
                f"(installed twice, or two packs share a name)"
            )

        clashes = {
            intent: self._by_intent[intent].manifest.name
            for intent in manifest.intents
            if intent in self._by_intent
        }
        if clashes:
            detail = ", ".join(
                f"'{intent}' already claimed by '{owner}'"
                for intent, owner in sorted(clashes.items())
            )
            raise ValueError(
                f"pack '{manifest.name}' claims intents that are already "
                f"handled: {detail}. Two packs cannot own the same intent — "
                f"which one runs would depend on install order."
            )

        tools = list(pack.tools())
        for tool in tools:
            # The contract's own validator catches this at ToolSpec
            # construction. It is checked again here because a pack can
            # assemble a ToolSpec by other means (model_construct, a
            # mutated copy, a subclass that overrides validation), and the
            # registry is the last gate before the tool is callable.
            if tool.moves_money and _executor_value(tool) == Executor.AGENT.value:
                raise ValueError(
                    f"pack '{manifest.name}' declares tool '{tool.name}' with "
                    f"moves_money=True and executor=AGENT. A money-moving tool "
                    f"must be executed by the customer's backend "
                    f"(Executor.BACKEND) or a human (Executor.HUMAN), never by "
                    f"the agent."
                )

        tool_clashes = {
            tool.name: self._owner_of_tool(tool.name)
            for tool in tools
            if tool.name in self._tools
        }
        if tool_clashes:
            detail = ", ".join(
                f"'{name}' already declared by '{owner}'"
                for name, owner in sorted(tool_clashes.items())
            )
            raise ValueError(
                f"pack '{manifest.name}' declares tool names that are already "
                f"taken: {detail}. Tool names are namespaced to prevent this."
            )

        # Validation passed — commit.
        self._packs[manifest.name] = pack
        for intent in manifest.intents:
            self._by_intent[intent] = pack
        for tool in tools:
            self._tools[tool.name] = tool

        log.info(
            "capability pack registered name=%s version=%s intents=%s tools=%d",
            manifest.name, manifest.version,
            ",".join(manifest.intents), len(tools),
        )

    def _owner_of_tool(self, tool_name: str) -> str:
        for name, pack in self._packs.items():
            if any(t.name == tool_name for t in pack.tools()):
                return name
        return "unknown"

    # -- lookup -----------------------------------------------------------

    def for_intent(self, intent: str) -> CapabilityPack | None:
        """The pack that handles `intent`, or None if nothing claims it."""
        return self._by_intent.get(intent)

    def tool(self, name: str) -> ToolSpec | None:
        """The spec for a namespaced tool name, or None if unknown."""
        return self._tools.get(name)

    def pack(self, name: str) -> CapabilityPack | None:
        """A registered pack by manifest name."""
        return self._packs.get(name)

    def installed(self) -> list[CapabilityManifest]:
        """Every registered pack's manifest, ordered by name."""
        return [
            self._packs[name].manifest for name in sorted(self._packs)
        ]

    def intents(self) -> list[str]:
        """Every intent any registered pack claims."""
        return sorted(self._by_intent)

    def tools(self) -> list[ToolSpec]:
        """Every tool any registered pack declares, ordered by name."""
        return [self._tools[name] for name in sorted(self._tools)]

    def __len__(self) -> int:
        return len(self._packs)

    def __contains__(self, pack_name: object) -> bool:
        return pack_name in self._packs

    # -- promotion --------------------------------------------------------

    def can_promote(self, pack_name: str) -> tuple[bool, str]:
        """
        Whether a registered pack may be promoted from shadow to live.

        Two gates, and the fixtures one is the substantive gate. A pack
        with no fixtures has never been shown to produce a specific
        decision for a specific conversation, so there is nothing to
        regress against and no way to tell a behaviour change from a
        behaviour break. Shadow mode is where such a pack belongs.
        """
        pack = self._packs.get(pack_name)
        if pack is None:
            return False, f"pack '{pack_name}' is not registered"

        try:
            fixtures = list(pack.fixtures())
        except Exception as exc:  # noqa: BLE001 - a pack's own code
            return False, (
                f"pack '{pack_name}' raised loading its fixtures: "
                f"{type(exc).__name__}: {exc}"
            )

        if not fixtures:
            return False, (
                f"pack '{pack_name}' declares no fixtures — there is nothing "
                f"to replay, so a behaviour change cannot be told apart from "
                f"a behaviour break. Add at least one fixture."
            )

        # Duplicate intents are refused at register(), so a registered pack
        # cannot normally hold one. It is re-checked because a pack's
        # manifest is a mutable attribute: a pack that grew an intent after
        # registration would otherwise be promoted on a stale check.
        stolen = {
            intent: self._by_intent[intent].manifest.name
            for intent in pack.manifest.intents
            if intent in self._by_intent
            and self._by_intent[intent].manifest.name != pack_name
        }
        if stolen:
            detail = ", ".join(
                f"'{intent}' also claimed by '{owner}'"
                for intent, owner in sorted(stolen.items())
            )
            return False, (
                f"pack '{pack_name}' shares intents with another registered "
                f"pack: {detail}"
            )

        return True, "ok"

    # -- entry points -----------------------------------------------------

    def load_entry_points(self, *, group: str = ENTRY_POINT_GROUP,
                          strict: bool = True) -> list[str]:
        """
        Register every pack advertised under the entry-point group.

        Returns the names of the packs that were registered.

        `strict` controls what a bad third-party pack does to the process.
        It defaults to True — a pack that cannot load or cannot validate
        stops startup, which is the same policy as `register`. A host that
        would rather boot degraded than not at all can pass strict=False,
        and every failure is logged at ERROR with the entry point named.
        """
        registered: list[str] = []
        for ep in entry_points(group=group):
            try:
                cls = ep.load()
                pack = cls()
                self.register(pack)
            except Exception as exc:  # noqa: BLE001 - third-party code
                log.error(
                    "capability entry point %r (%s) failed to load: %s: %s",
                    ep.name, getattr(ep, "value", "?"),
                    type(exc).__name__, exc,
                )
                if strict:
                    raise
                continue
            registered.append(pack.manifest.name)
        return registered


def default_registry(*, load_entry_points: bool = True,
                     strict: bool = True) -> CapabilityRegistry:
    """
    A registry holding every capability pack installed on this machine.

    In syntri-core this also registers the first-party packs (payments,
    flights, commerce) in code, because they ship with the server. Those
    packs are private and not part of the public distribution, so here the
    registry is populated purely from installed entry points — which is
    exactly what a third-party developer's `syntri pack list` should show.
    """
    registry = CapabilityRegistry()
    if load_entry_points:
        registry.load_entry_points(strict=strict)
    return registry
