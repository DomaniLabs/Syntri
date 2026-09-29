<p align="center">
  <strong>Agent infrastructure for production AI.</strong><br/>
  Build, train, deploy and continuously improve domain-specific AI agents.
</p>

<p align="center">
  <a href="https://syntri.ai">syntri.ai</a> &nbsp;·&nbsp;
  <a href="mailto:hello@domani.studio">Request access</a> &nbsp;·&nbsp;
  Built by <a href="https://domani.studio">Domani</a>
</p>

---

## What it is

Most AI integrations are wrappers around a language model. You call an API, get a completion, and hope it does the right thing. When it doesn't, you have no visibility into why — and no way to make it better without starting over.

Syntri is different. It is infrastructure for building AI agents that are **measurable, improvable, and production-safe** — agents that get better the longer they run, without requiring you to trust a black box.

## Two modules, one platform

**Syntri Agent** — Build and operate domain-specific AI agents. Define what your agent understands, train models on your own data, connect your systems through capability packs, and enforce policies that keep sensitive operations safe. Language models are optional; custom-trained models are first-class.

**Syntri Observe** — AI-native observability. Not just logs — decision logs. See intent accuracy over time, entity extraction failure rates by field, outcome-weighted performance across capabilities, and where your agent is degrading before your users tell you.

## How it works

Your data → Train your models → Deploy your agent → Collect experience → Improve


Every agent instance trains on its own data. Syntri provides the infrastructure — the training pipeline, the evaluation harness, the runtime, the observability layer. You bring the domain knowledge.

```yaml
# An agent instance is configuration, not code
agent:
  name: your-agent
  locales: [en, sw, pcm]

capabilities:
  - payments
  - flights
  - customer-support

learning:
  level: 3          # classical ML — intent + entity models
  retrain_every_days: 14
```

## What makes it different

**Decision-level observability.** Existing tools tell you an API call succeeded. Syntri tells you the agent routed to `send_money` at 0.61 confidence, the bank extractor has an 18% error rate on three-word institution names, and session abandonment on the flight booking capability jumped 12 points this week.

**Custom-trained models.** The failure modes that matter most — local bank aliases, pidgin shorthand, currency formats, SMS abbreviations — are coverage problems. They are fixed by training on your distribution, not by engineering prompts. Syntri makes that trainable.

**Learns from experience.** Every production turn is a training candidate. Corrections, completions, abandonments — all become supervision signal. The agent improves on a gated, offline cadence. Nothing updates live.

**LLM-optional.** Language models are one component in the stack, not the foundation. An agent running at learning level 3 uses a trained intent classifier (~10ms, no API cost) and falls back to an LLM only for low-confidence or unmapped cases. At level 4 and above, a fine-tuned transformer handles everything.

**Safe by design.** Money-moving operations are typed as `irreversible` in the taxonomy. The runtime refuses to execute them locally — they are requests to your backend, not commands. No configuration or capability pack can override this.

## Status

Private beta. Infinitswap is our first production deployment — a WhatsApp-native financial platform operating across Nigeria, Ghana, Tanzania and South Africa.

[Request access →](mailto:hello@domani.studio)

---

## Connecting to Syntri

Syntri runs as a standalone service. Your application calls it over HTTP — no Python required on your side.

```bash
# Understand a message
POST /v1/understand
{ "text": "send 5k to my uba account", "session_id": "...", "user_id": "..." }

# Full agent turn
POST /v1/turn
{ "text": "...", "session_id": "...", "channel": "whatsapp" }

# Return tool results
POST /v1/tool-results
{ "session_id": "...", "results": [...] }

# Send outcome feedback (closes the learning loop)
POST /v1/feedback
{ "session_id": "...", "outcome": "completed" }
```

An SDK and CLI are in development. See [`/contracts`](/contracts) for the full API schema.

## Building capability packs

A **capability pack** is a small Python package that teaches an agent how to
handle one job — checking a balance, booking a flight, opening a ticket. Packs
declare tools and decision logic; the customer's backend does the execution.
Money-moving operations can never run in the agent — the contract enforces it.

```bash
pip install syntri-contracts
```

```python
from syntri_contracts.contracts.capability import (
    CapabilityManifest, CapabilityPack, Decision, WorkflowStatus,
)

class CheckBalancePack(CapabilityPack):
    manifest = CapabilityManifest(
        name="my-check-balance", display_name="Check Balance",
        version="0.1.0", description="Reports the user's balance.",
        author="you@example.com", intents=["check_balance"],
        required_entities=[], entry_point="my_pack.pack:CheckBalancePack",
    )

    def tools(self):
        ...  # declare a ToolSpec, e.g. "balance.fetch" (executor=BACKEND)

    def advance(self, session_id, slots, understanding, tool_results, turn_count):
        if not tool_results:
            return Decision(action="request_tool", tool="balance.fetch",
                            arguments={}, status=WorkflowStatus.EXECUTING)
        data = tool_results[-1].data or {}
        return Decision(action="reply",
                        message=f"Your balance is {data.get('balance')}.",
                        status=WorkflowStatus.COMPLETED)
```

Full guide: [`contracts/README.md`](/contracts/README.md). Working template:
[`contracts/example_pack/`](/contracts/example_pack). Validate a pack with
`syntri pack validate ./my-pack/`.

## Deployment

Self-hosted first. Syntri runs inside your infrastructure — your data never leaves your environment.

Your infrastructure
├── Your application (Node, Python, anything)
├── Syntri Agent ← runs here, calls back to your app for tool execution
└── Your database


Docker image and Helm chart coming with the beta release.

---

## Roadmap

- [x] Experience schema — Observation, Interpretation, Judgment, Episode
- [x] Config-driven taxonomy with risk tiers and slot rules
- [x] PII tokenisation at ingestion
- [x] Plugin discovery — connectors, capabilities, noise packs
- [x] Model registry with lineage and promotion gates
- [ ] Corpus builder and synthetic augmentation
- [ ] Intent classifier + entity extractor training pipeline
- [ ] Infinitswap connector (first production adapter)
- [ ] Shadow mode deployment on Infinitswap traffic
- [ ] Flight booking capability pack
- [x] Public capability-pack contract (`syntri-contracts`) — *Phase 5, in progress*
- [ ] `syntri pack validate` and marketplace submission
- [ ] CLI: `syntri init`, `syntri train`, `syntri evaluate`, `syntri serve`
- [ ] Syntri Observe dashboard

---

<p align="center">
  <sub>Built by <a href="https://domani.studio">Domani</a> · © 2026 Domani Labs · All rights reserved</sub>
</p>
