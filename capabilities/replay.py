"""
Fixture replay — the only automated check that a pack still behaves.

A fixture is a recorded conversation with the decisions the pack is
expected to make at each step. Replaying it drives `advance()` through the
whole workflow with no server, no network and no model: the tool results
are canned, so what is under test is the pack's decision-making and
nothing else.

This is what `syntri pack validate` runs, what `can_promote` requires to
exist, and what a third-party developer runs before submitting. The format
is the one the public example pack uses, because that is the file
developers copy:

    {
      "name": "basic check balance flow",
      "turns": [
        {
          "input":        {"text": "...", "intent": "...", "entities": {...},
                           "payload": {...}},
          "tool_calls":   [{"tool": "balance.fetch", "arguments": {}}],
          "tool_results": [{"success": true, "data": {...}}],
          "expected_reply": "Your balance is ₦5,000."
        }
      ]
    }

Within a turn, `tool_calls` are the requests the pack is expected to make
and `tool_results` are what the backend answers, paired by position. The
runner asserts the request, feeds the result, and calls `advance()` again —
so one fixture turn can cover several round trips, which is how a real
turn actually behaves.

Argument matching is subset matching: a fixture asserts the arguments it
cares about and stays silent about the rest. A fixture that pinned every
argument would fail the first time a pack passed an extra one, which is
not a behaviour change worth failing over.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from contracts import (
    CapabilityPack,
    Decision,
    SlotState,
    ToolResult,
)

log = logging.getLogger(__name__)

#: A fixture turn that never stops requesting tools is a bug in the pack,
#: not a fixture to keep feeding. This bounds one turn's round trips.
MAX_ROUND_TRIPS = 12


@dataclass
class TurnOutcome:
    """What happened on one fixture turn."""

    index: int
    failures: list[str] = field(default_factory=list)
    action: str = ""
    message: str | None = None

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass
class FixtureOutcome:
    """What happened replaying one fixture."""

    name: str
    source: str = ""
    turns: list[TurnOutcome] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and all(t.ok for t in self.turns)

    @property
    def failures(self) -> list[str]:
        if self.error:
            return [self.error]
        return [
            f"turn {t.index}: {failure}"
            for t in self.turns for failure in t.failures
        ]


@dataclass
class ReplayReport:
    """The result of replaying every fixture a pack declares."""

    pack_name: str
    outcomes: list[FixtureOutcome] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.outcomes) and all(o.ok for o in self.outcomes)

    @property
    def passed(self) -> int:
        return sum(1 for o in self.outcomes if o.ok)

    @property
    def failed(self) -> int:
        return sum(1 for o in self.outcomes if not o.ok)

    @property
    def total(self) -> int:
        return len(self.outcomes)


def _is_subset(expected: dict, actual: dict) -> tuple[bool, str]:
    """Whether every key in `expected` matches in `actual`, and what missed."""
    for key, want in expected.items():
        if key not in actual:
            return False, f"argument {key!r} was not passed"
        got = actual[key]
        if str(got) != str(want):
            return False, f"argument {key!r} was {got!r}, expected {want!r}"
    return True, ""


def _result_for(spec: dict, call_id: str, tool: str) -> ToolResult:
    """Build a ToolResult from a fixture's canned result."""
    return ToolResult(
        call_id=call_id,
        tool=spec.get("tool") or tool,
        success=bool(spec.get("success", True)),
        data=spec.get("data"),
        error_code=spec.get("error_code"),
        error_message=spec.get("error_message"),
    )


def _check_expectations(turn: dict, decision: Decision) -> list[str]:
    """Compare a fixture turn's expectations against the final decision."""
    failures: list[str] = []

    if "expected_action" in turn and decision.action != turn["expected_action"]:
        failures.append(
            f"expected action {turn['expected_action']!r}, "
            f"got {decision.action!r}"
        )

    if "expected_reply" in turn:
        want = turn["expected_reply"]
        if decision.message != want:
            failures.append(
                f"expected reply {want!r}, got {decision.message!r}"
            )

    if "expected_message_contains" in turn:
        want = turn["expected_message_contains"]
        if want not in (decision.message or ""):
            failures.append(
                f"expected message containing {want!r}, "
                f"got {decision.message!r}"
            )

    if "expected_missing" in turn:
        want = list(turn["expected_missing"])
        if sorted(decision.missing) != sorted(want):
            failures.append(
                f"expected missing slots {sorted(want)}, "
                f"got {sorted(decision.missing)}"
            )

    if "expected_status" in turn:
        want = turn["expected_status"]
        got = getattr(decision.status, "value", decision.status)
        if got != want:
            failures.append(f"expected status {want!r}, got {got!r}")

    if "expected_options" in turn:
        want = turn["expected_options"]
        if isinstance(want, int):
            if len(decision.options) != want:
                failures.append(
                    f"expected {want} options, got {len(decision.options)}"
                )
        else:
            if len(decision.options) != len(want):
                failures.append(
                    f"expected {len(want)} options, got {len(decision.options)}"
                )
            else:
                for position, expected_option in enumerate(want):
                    ok, why = _is_subset(
                        expected_option, decision.options[position]
                    )
                    if not ok:
                        failures.append(f"option {position}: {why}")

    return failures


def replay_fixture(pack: CapabilityPack, fixture: dict) -> FixtureOutcome:
    """
    Drive one fixture through the pack and report what did not match.

    Every turn accumulates into the same slots, scratch and tool-result
    list, because that is what a real conversation does — a fixture whose
    turns were independent would not test the workflow, only its steps.
    """
    outcome = FixtureOutcome(
        name=str(fixture.get("name") or "unnamed"),
        source=str(fixture.get("_source") or ""),
    )

    turns = fixture.get("turns")
    if not isinstance(turns, list) or not turns:
        outcome.error = "fixture declares no turns"
        return outcome

    session_id = str(fixture.get("session_id") or f"fixture-{outcome.name}")
    slots = SlotState()
    scratch: dict[str, Any] = {}
    tool_results: list[ToolResult] = []
    turn_count = 0

    for index, turn in enumerate(turns, start=1):
        record = TurnOutcome(index=index)
        outcome.turns.append(record)

        if not isinstance(turn, dict):
            record.failures.append("turn is not an object")
            return outcome

        turn_input = turn.get("input") or {}
        for name, value in (turn_input.get("entities") or {}).items():
            slots.fill(str(name), str(value))

        understanding: dict[str, Any] = {
            "intent": turn_input.get("intent") or "",
            "confidence": float(turn_input.get("confidence", 1.0)),
            "entities": dict(turn_input.get("entities") or {}),
            "alternatives": [],
            "text": turn_input.get("text") or "",
            "payload": dict(turn_input.get("payload") or {}),
            "scratch": dict(scratch),
        }

        turn_count += 1
        try:
            decision = pack.advance(
                session_id, slots, understanding, list(tool_results), turn_count
            )
        except Exception as exc:  # noqa: BLE001 - the pack's own code
            record.failures.append(
                f"advance() raised {type(exc).__name__}: {exc}"
            )
            return outcome

        scratch.update(decision.scratch or {})

        expected_calls = turn.get("tool_calls") or []
        canned_results = turn.get("tool_results") or []

        for position, expected_call in enumerate(expected_calls):
            if decision.action != "request_tool":
                record.failures.append(
                    f"expected a request for tool "
                    f"{expected_call.get('tool')!r}, but the pack returned "
                    f"action {decision.action!r}"
                )
                return outcome

            want_tool = expected_call.get("tool")
            if want_tool and decision.tool != want_tool:
                record.failures.append(
                    f"expected tool {want_tool!r}, got {decision.tool!r}"
                )
                return outcome

            ok, why = _is_subset(
                expected_call.get("arguments") or {}, decision.arguments
            )
            if not ok:
                record.failures.append(f"{decision.tool}: {why}")
                return outcome

            if position >= len(canned_results):
                record.failures.append(
                    f"fixture declares {len(expected_calls)} tool_calls but "
                    f"only {len(canned_results)} tool_results"
                )
                return outcome

            call_id = f"call-{index}-{position}"
            tool_results.append(
                _result_for(
                    canned_results[position], call_id, decision.tool or ""
                )
            )

            turn_count += 1
            try:
                decision = pack.advance(
                    session_id, slots, understanding,
                    list(tool_results), turn_count,
                )
            except Exception as exc:  # noqa: BLE001
                record.failures.append(
                    f"advance() raised after {decision.tool}: "
                    f"{type(exc).__name__}: {exc}"
                )
                return outcome

            scratch.update(decision.scratch or {})

            if position > MAX_ROUND_TRIPS:
                record.failures.append(
                    f"pack made more than {MAX_ROUND_TRIPS} tool requests in "
                    f"one turn — likely a loop"
                )
                return outcome

        record.action = decision.action
        record.message = decision.message
        record.failures.extend(_check_expectations(turn, decision))

    return outcome


def replay_pack(pack: CapabilityPack) -> ReplayReport:
    """Replay every fixture a pack declares."""
    report = ReplayReport(pack_name=pack.manifest.name)
    try:
        fixtures = list(pack.fixtures())
    except Exception as exc:  # noqa: BLE001
        report.outcomes.append(
            FixtureOutcome(
                name="<loading fixtures>",
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        return report

    for fixture in fixtures:
        report.outcomes.append(replay_fixture(pack, fixture))
    return report
