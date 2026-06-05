from __future__ import annotations

from dataclasses import dataclass

from market_support_crewai_agent.schemas import ReplyRequest
from market_support_crewai_agent.settings import Settings


@dataclass(frozen=True)
class InputGuardrailError(ValueError):
    """Raised when a request is rejected before any LLM/CrewAI stage."""

    code: str
    message: str
    metadata: dict

    def __str__(self) -> str:
        return self.message


def validate_reply_request_input(
    request: ReplyRequest,
    settings: Settings,
) -> None:
    """Apply request-level guardrails before policy, evidence, or CrewAI calls."""
    max_chars = settings.agent_input_max_message_chars
    if max_chars is not None and len(request.message) > max_chars:
        raise InputGuardrailError(
            code="message_too_long",
            message=(
                "message exceeds configured input guardrail limit "
                f"({len(request.message)}>{max_chars})"
            ),
            metadata={
                "message_length": len(request.message),
                "max_message_chars": max_chars,
            },
        )
