# Building Syntri capability packs

A **capability pack** is a small Python package that teaches a Syntri agent how
to handle one job — checking a balance, booking a flight, opening a support
ticket. You write the *logic*; the customer's backend does the *execution*.

This guide takes you from zero to a tested pack. You do not need to have used
Syntri before.

```bash
pip install syntri-contracts
```

That gives you everything in this folder: the contract (`contracts.capability`)
and a working example (`contracts/example_pack`).

---

## What is a capability pack?

Think of the agent as a dispatcher. It understands what the user wants
("check my balance"), then looks for a pack that claims that job. Your pack
answers two questions, over and over, until the job is done:

- **What tools do I offer?** — `tools()`
- **Given where we are, what happens next?** — `advance()`

A pack is pure decision logic. It holds no database connections, calls no
external APIs, and moves no money. When it needs something done in the real
world, it *requests* a tool and hands that request to the backend.

---

## The five things every pack declares

Every pack fills in a `CapabilityManifest`:

| Field | What it is |
| --- | --- |
| **name / version** | A unique id and a semver string. |
| **intents** | Which user intents this pack claims (e.g. `["check_balance"]`). |
| **required_entities** | Entities the agent must extract before your pack can run. |
| **tools** | The operations your pack can request (declared in `tools()`). |
| **entry_point** | Where Python finds your pack class, e.g. `my_package.pack:MyPack`. |

---

## The execution boundary: packs request, backends execute

This is the most important rule in Syntri.

Every tool declares an **executor**:

- `AGENT` — safe enough for the agent to run directly.
- `BACKEND` — must run inside the customer's own systems.
- `HUMAN` — needs a person.

A pack never runs a `BACKEND` or `HUMAN` tool itself. It returns a `Decision`
with `action="request_tool"`, and the runtime routes that request to the right
place. The results come back to your `advance()` on the next call, in
`tool_results`.

Money-moving tools are locked down at construction time:

```python
ToolSpec(name="pay.send", moves_money=True, executor=Executor.AGENT)
# ValueError: a money-moving tool cannot have executor=AGENT
```

`moves_money=True` forces `BACKEND` or `HUMAN`. No pack can opt out.

---

## Build your first pack

### 1. A tool

```python
from contracts.capability import ToolSpec, Executor

TOOL = ToolSpec(
    name="balance.fetch",              # must be namespaced with a dot
    description="Fetch the account balance.",
    input_schema={"type": "object", "properties": {}},
    output_schema={
        "type": "object",
        "properties": {"balance": {"type": "string"}, "currency": {"type": "string"}},
    },
    executor=Executor.BACKEND,
    moves_money=False,
)
```

### 2. The pack

```python
from contracts.capability import (
    CapabilityManifest, CapabilityPack, Decision, SlotState,
    ToolResult, ToolSpec, WorkflowStatus,
)


class MyPack(CapabilityPack):
    manifest = CapabilityManifest(
        name="my-check-balance",
        display_name="Check Balance",
        version="0.1.0",
        description="Reports the user's balance.",
        author="you@example.com",
        intents=["check_balance"],
        required_entities=[],
        entry_point="my_package.pack:MyPack",
    )

    def tools(self) -> list[ToolSpec]:
        return [TOOL]

    def advance(self, session_id, slots, understanding, tool_results, turn_count) -> Decision:
        if not tool_results:
            return Decision(action="request_tool", tool="balance.fetch",
                            arguments={}, status=WorkflowStatus.EXECUTING)
        data = tool_results[-1].data or {}
        return Decision(action="reply",
                        message=f"Your balance is {data.get('balance')}.",
                        status=WorkflowStatus.COMPLETED)
```

Every concrete pack **must** set `manifest` and implement `tools()` and
`advance()`. Omitting `manifest` raises `TypeError` when the class is defined.

### 3. What `advance()` returns

A `Decision`. Its `action` is one of:

| action | Meaning |
| --- | --- |
| `reply` | Say something; the job is done or informational. |
| `ask` | Ask the user for a missing slot. |
| `present` | Offer `options` for the user to choose from. |
| `request_tool` | Ask the runtime to run `tool` with `arguments`. |
| `request_confirm` | Ask the user to confirm before a sensitive step. |
| `escalate` | Send to a human (the default in `on_failure`). |
| `handoff` | Transfer the conversation elsewhere. |

Two optional hooks are worth overriding:

- **`validate_slot(field, value)`** — reject bad values as they come in.
- **`on_failure(tool_result)`** — recover from a failed tool call. The default
  escalates to a human.

---

## Register your pack (entry points)

Syntri discovers packs through a Python entry point. In your `pyproject.toml`:

```toml
[project.entry-points."syntri.capability_packs"]
my-check-balance = "my_package.pack:MyPack"
```

The key is your pack's `name`; the value matches the manifest's `entry_point`.
Once your package is installed, Syntri can find and load the pack by name.

---

## Test your pack

Ship at least one **fixture** — a recorded run the validator replays. A pack
with no fixtures cannot be promoted to production (`fixtures()` returning an
empty list is a hard stop).

```json
{
  "name": "basic check balance flow",
  "turns": [
    {
      "input": {"text": "what is my balance", "intent": "check_balance"},
      "tool_calls": [{"tool": "balance.fetch", "arguments": {}}],
      "tool_results": [{"success": true, "data": {"balance": "5000", "currency": "NGN"}}],
      "expected_reply": "Your balance is ₦5,000."
    }
  ]
}
```

Then run the validator:

```bash
syntri pack validate ./my-pack/
```

It checks the manifest, constructs every `ToolSpec` (so the money/executor and
naming rules are enforced), and replays each fixture through `advance()`.

---

## Submitting to the Syntri marketplace

*Coming soon.* The marketplace lets other teams install your pack by name. When
it opens, this section will cover packaging, review, and publishing. Until then,
build and self-host — the contract will not change underneath you.
