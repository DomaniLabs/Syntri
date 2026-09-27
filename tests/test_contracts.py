"""Tests for the public capability-pack contract."""

import pytest

from contracts.capability import (
    CapabilityManifest,
    CapabilityPack,
    Decision,
    Executor,
    SlotState,
    ToolResult,
    ToolSpec,
    WorkflowStatus,
)
from contracts.example_pack.pack import CheckBalancePack


def test_money_moving_tool_cannot_run_on_agent():
    with pytest.raises(ValueError):
        ToolSpec(
            name="pay.send",
            description="send money",
            input_schema={},
            output_schema={},
            executor=Executor.AGENT,
            moves_money=True,
        )


def test_money_moving_tool_allows_backend_and_human():
    for executor in (Executor.BACKEND, Executor.HUMAN):
        spec = ToolSpec(
            name="pay.send",
            description="send money",
            input_schema={},
            output_schema={},
            executor=executor,
            moves_money=True,
        )
        assert spec.executor is executor


def test_tool_name_without_dot_raises():
    with pytest.raises(ValueError):
        ToolSpec(
            name="search",  # no namespace dot
            description="search",
            input_schema={},
            output_schema={},
        )


def test_tool_name_with_dot_ok():
    spec = ToolSpec(
        name="flights.search",
        description="search flights",
        input_schema={},
        output_schema={},
    )
    assert spec.name == "flights.search"
    # frozen models reject mutation
    with pytest.raises(Exception):
        spec.name = "other.name"


def test_slotstate_missing_returns_correct_fields():
    slots = SlotState()
    slots.fill("origin", "Lagos")
    slots.fill("blank", "")  # empty counts as missing
    missing = slots.missing(["origin", "destination", "blank"])
    assert missing == ["destination", "blank"]
    assert slots.plain() == {"origin": "Lagos", "blank": ""}


def test_decision_defaults():
    d = Decision(action="reply")
    assert d.action == "reply"
    assert d.message is None
    assert d.status == WorkflowStatus.COLLECTING
    assert d.tool is None
    assert d.arguments == {}
    assert d.options == []
    assert d.missing == []
    assert d.scratch == {}
    # mutable defaults are not shared between instances
    other = Decision(action="ask")
    d.arguments["x"] = 1
    assert other.arguments == {}


def test_subclass_without_manifest_raises_typeerror():
    with pytest.raises(TypeError):

        class NoManifestPack(CapabilityPack):
            def tools(self):
                return []

            def advance(self, session_id, slots, understanding, tool_results, turn_count):
                return Decision(action="reply")


def test_example_pack_instantiates_and_advances():
    pack = CheckBalancePack()

    tools = pack.tools()
    assert len(tools) == 1
    assert tools[0].name == "balance.fetch"
    assert tools[0].executor is Executor.BACKEND
    assert tools[0].moves_money is False

    # No results yet -> requests the tool.
    first = pack.advance("s1", SlotState(), {}, [], 0)
    assert isinstance(first, Decision)
    assert first.action == "request_tool"
    assert first.tool == "balance.fetch"
    assert first.status == WorkflowStatus.EXECUTING

    # With a successful result -> replies with the formatted balance.
    result = ToolResult(
        call_id="c1",
        tool="balance.fetch",
        success=True,
        data={"balance": "5000", "currency": "NGN"},
    )
    reply = pack.advance("s1", SlotState(), {}, [result], 1)
    assert reply.action == "reply"
    assert reply.status == WorkflowStatus.COMPLETED
    assert reply.message == "Your balance is ₦5,000."


def test_example_pack_failure_escalates():
    pack = CheckBalancePack()
    failed = ToolResult(call_id="c1", tool="balance.fetch", success=False,
                        error_code="upstream", error_message="down")
    decision = pack.advance("s1", SlotState(), {}, [failed], 1)
    assert decision.action == "escalate"
    assert decision.status == WorkflowStatus.ESCALATED


def test_example_pack_fixture_loads_and_validates():
    pack = CheckBalancePack()
    fixtures = pack.fixtures()
    assert fixtures, "pack must ship at least one fixture to be promotable"

    fixture = fixtures[0]
    assert fixture["name"] == "basic check balance flow"
    turn = fixture["turns"][0]

    # Replay the recorded turn through the pack and check the expected reply.
    results = [
        ToolResult(call_id="c1", tool=tc["tool"], success=tr["success"], data=tr.get("data"))
        for tc, tr in zip(turn["tool_calls"], turn["tool_results"])
    ]
    decision = pack.advance("s1", SlotState(), turn["input"], results, 1)
    assert decision.message == turn["expected_reply"]


def test_manifest_yaml_matches_pack_manifest():
    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "contracts" / "example_pack" / "manifest.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    from_yaml = CapabilityManifest(**data)
    assert from_yaml == CheckBalancePack.manifest
