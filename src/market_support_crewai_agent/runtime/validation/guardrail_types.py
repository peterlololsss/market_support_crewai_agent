from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, JsonValue

from market_support_crewai_agent.schemas.base import StrictModel

GuardrailOutcome = Literal[
    "allow",
    "block",
    "require_clarification",
    "require_confirmation",
    "abstain",
]
GuardrailPhase = Literal[
    "input",
    "retrieval_source",
    "execution_tool",
    "output",
]
GuardrailMetadata: TypeAlias = dict[str, JsonValue]

HANDOFF_TEXT_METADATA_KEY = "handoff_text"
HANDOFF_UNAVAILABLE_TEXT_METADATA_KEY = "handoff_unavailable_text"


class GuardrailDecision(StrictModel):
    outcome: GuardrailOutcome
    phase: GuardrailPhase
    reason_code: str = Field(min_length=1)
    human_readable_reason: str = ""
    capability_id: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    source_scopes: list[GuardrailMetadata] = Field(default_factory=list)
    evidence_required: list[str] = Field(default_factory=list)
    evidence_seen: list[str] = Field(default_factory=list)
    metadata: GuardrailMetadata = Field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.outcome == "allow"


def make_decision(
    outcome: GuardrailOutcome,
    phase: GuardrailPhase,
    reason_code: str,
    *,
    human_reason: str = "",
    capability_id: str | None = None,
    artifact_ids: list[str] | None = None,
    source_scopes: list[GuardrailMetadata] | None = None,
    evidence_required: list[str] | None = None,
    evidence_seen: list[str] | None = None,
    metadata: GuardrailMetadata | None = None,
) -> GuardrailDecision:
    return GuardrailDecision(
        outcome=outcome,
        phase=phase,
        reason_code=reason_code,
        human_readable_reason=human_reason,
        capability_id=capability_id,
        artifact_ids=artifact_ids or [],
        source_scopes=source_scopes or [],
        evidence_required=evidence_required or [],
        evidence_seen=evidence_seen or [],
        metadata=metadata or {},
    )
