"""Public capability-pack contract for Syntri.

This module is the single source of truth for third-party developers who want
to build capability packs. It is fully standalone: the only dependencies are
the Python standard library and pydantic (>=2.9).

A capability pack declares *what it can do* (tools) and *what to do next*
(advance). It never executes money-moving or irreversible operations itself —
those are requests handed to the customer's backend. See ``contracts/README.md``
for the plain-English guide.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "WorkflowStatus",
    "Executor",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "SlotState",
    "Decision",
    "CapabilityManifest",
    "CapabilityPack",
]


class WorkflowStatus(str, Enum):
    """Where a conversation stands within a single capability's workflow."""

    IDLE = "idle"
    COLLECTING = "collecting"
    RESOLVING = "resolving"
    AWAITING_SELECTION = "awaiting_selection"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_AUTHORIZATION = "awaiting_authorization"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class Executor(str, Enum):
    """Who is allowed to run a tool.

    AGENT   — the pack may call this tool directly.
    BACKEND — must be executed by the customer's backend.
    HUMAN   — requires a human operator.
    """

    AGENT = "agent"
    BACKEND = "backend"
    HUMAN = "human"


class ToolSpec(BaseModel):
    """A tool a capability pack exposes.

    ``name`` is namespaced, e.g. ``"flights.search"``. Money-moving tools may
    never be executed by the agent directly — the constructor enforces this.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict
    output_schema: dict
    executor: Executor = Executor.BACKEND
    requires_confirmation: bool = False
    requires_authorization: bool = False
    moves_money: bool = False
    idempotent: bool = True

    @field_validator("name")
    @classmethod
    def _name_is_namespaced(cls, value: str) -> str:
        if "." not in value:
            raise ValueError(
                f"tool name must be namespaced with a dot, e.g. 'flights.search' "
                f"(got {value!r})"
            )
        return value

    @model_validator(mode="after")
    def _money_never_runs_on_agent(self) -> "ToolSpec":
        if self.moves_money and self.executor is Executor.AGENT:
            raise ValueError(
                "a money-moving tool cannot have executor=AGENT; "
                "it must be BACKEND or HUMAN"
            )
        return self


class ToolCall(BaseModel):
    """A request from a pack to run a tool."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    tool: str
    arguments: dict
    idempotency_key: str | None = None
    session_id: str
    reasoning: str = Field(default="", max_length=150)


class ToolResult(BaseModel):
    """The outcome of a tool call, handed back to the pack."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    tool: str
    success: bool
    data: dict | None = None
    error_code: str | None = None
    error_message: str | None = None
    latency_ms: float = 0.0


class SlotState(BaseModel):
    """The collected values for a workflow's slots (fields)."""

    values: dict[str, str] = Field(default_factory=dict)

    def fill(self, field: str, value: str) -> None:
        """Set a slot value."""
        self.values[field] = value

    def missing(self, required: list[str]) -> list[str]:
        """Return the required fields that have no non-empty value yet."""
        return [f for f in required if not self.values.get(f)]

    def plain(self) -> dict[str, str]:
        """Return a shallow copy of the collected values."""
        return dict(self.values)


@dataclass
class Decision:
    """What the pack wants to happen next, returned from ``advance``.

    ``action`` is one of: "reply", "ask", "present", "request_tool",
    "request_confirm", "escalate", "handoff".
    """

    action: str
    message: str | None = None
    status: WorkflowStatus = WorkflowStatus.COLLECTING
    tool: str | None = None
    arguments: dict = field(default_factory=dict)
    options: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    scratch: dict = field(default_factory=dict)


class CapabilityManifest(BaseModel):
    """Metadata a pack publishes about itself."""

    name: str
    display_name: str
    version: str
    description: str
    author: str
    license: str = "Proprietary"
    intents: list[str]
    required_entities: list[str]
    required_syntri_version: str = ">=0.1.0"
    homepage: str | None = None
    entry_point: str


class CapabilityPack(ABC):
    """Base class every capability pack subclasses.

    A concrete pack MUST set the ``manifest`` class variable and implement
    ``tools`` and ``advance``. Failing to set ``manifest`` on a concrete
    subclass raises ``TypeError`` at class-definition time.
    """

    manifest: CapabilityManifest

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Still-abstract intermediate bases are allowed to skip the manifest.
        if getattr(cls, "__abstractmethods__", None):
            return
        if "manifest" not in cls.__dict__ and not any(
            "manifest" in base.__dict__ for base in cls.__mro__[1:]
            if base is not CapabilityPack
        ):
            raise TypeError(
                f"{cls.__name__} must define a 'manifest' class variable"
            )

    @abstractmethod
    def tools(self) -> list[ToolSpec]:
        """The tools this pack exposes."""

    @abstractmethod
    def advance(
        self,
        session_id: str,
        slots: SlotState,
        understanding: dict,
        tool_results: list[ToolResult],
        turn_count: int,
    ) -> Decision:
        """Decide what to do next given the current conversation state."""

    def validate_slot(self, field: str, value: str) -> bool:
        """Pack-specific slot validation. Override as needed; default accepts."""
        return True

    def on_failure(self, tool_result: ToolResult) -> Decision:
        """React to a failed tool call. Default escalates to a human.

        Override for recoverable failures (retry, ask again, fall back).
        """
        return Decision(
            action="escalate",
            message="Sorry, something went wrong. Let me hand you to a human.",
            status=WorkflowStatus.ESCALATED,
        )

    def fixtures(self) -> list[dict]:
        """Replay fixtures used to validate the pack.

        An empty list means the pack cannot be promoted to production.
        """
        return []
