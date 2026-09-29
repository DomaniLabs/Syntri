"""A complete, working capability pack: check_balance.

This is the smallest useful pack. A user asks "what is my balance"; the pack
requests the ``balance.fetch`` tool from the backend, then replies with the
figure the backend returns. It moves no money and needs no confirmation, so it
is the ideal template to copy from.
"""

from __future__ import annotations

import json
from pathlib import Path

from syntri_contracts.contracts.capability import (
    CapabilityManifest,
    CapabilityPack,
    Decision,
    Executor,
    SlotState,
    ToolResult,
    ToolSpec,
    WorkflowStatus,
)

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Currency symbols used when formatting the reply. Extend as you support more.
_CURRENCY_SYMBOLS = {
    "NGN": "₦",  # ₦
    "GHS": "GH₵",  # GH₵
    "KES": "KSh",
    "ZAR": "R",
    "USD": "$",
}


def _format_amount(balance: str, currency: str) -> str:
    """Render a balance like '5000'/'NGN' as '₦5,000'."""
    symbol = _CURRENCY_SYMBOLS.get(currency, currency + " ")
    try:
        grouped = f"{int(str(balance).replace(',', '')):,}"
    except (TypeError, ValueError):
        grouped = str(balance)
    return f"{symbol}{grouped}"


class CheckBalancePack(CapabilityPack):
    """Fetches and reports an account balance."""

    manifest = CapabilityManifest(
        name="example-check-balance",
        display_name="Check Balance",
        version="0.1.0",
        description="Fetches and reports the user's account balance.",
        author="Syntri",
        license="MIT",
        intents=["check_balance"],
        required_entities=[],
        required_syntri_version=">=0.1.0",
        homepage="https://github.com/DomaniLabs/Syntri",
        entry_point="contracts.example_pack.pack:CheckBalancePack",
    )

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="balance.fetch",
                description="Fetch the current account balance for the session's user.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "balance": {"type": "string"},
                        "currency": {"type": "string"},
                    },
                    "required": ["balance", "currency"],
                },
                executor=Executor.BACKEND,
                moves_money=False,
                idempotent=True,
            )
        ]

    def advance(
        self,
        session_id: str,
        slots: SlotState,
        understanding: dict,
        tool_results: list[ToolResult],
        turn_count: int,
    ) -> Decision:
        # No result yet — ask the backend to fetch the balance.
        if not tool_results:
            return Decision(
                action="request_tool",
                tool="balance.fetch",
                arguments={},
                status=WorkflowStatus.EXECUTING,
            )

        result = tool_results[-1]
        if not result.success:
            return self.on_failure(result)

        data = result.data or {}
        balance = data.get("balance", "0")
        currency = data.get("currency", "")
        return Decision(
            action="reply",
            message=f"Your balance is {_format_amount(balance, currency)}.",
            status=WorkflowStatus.COMPLETED,
        )

    def fixtures(self) -> list[dict]:
        loaded: list[dict] = []
        for path in sorted(_FIXTURES_DIR.glob("*.json")):
            loaded.append(json.loads(path.read_text(encoding="utf-8")))
        return loaded
