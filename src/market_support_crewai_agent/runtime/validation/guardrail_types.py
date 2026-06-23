from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from market_support_crewai_agent.runtime.domain.capabilities import CapabilityName
from market_support_crewai_agent.runtime.domain.ontology import ArtifactType
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.schemas import SideEffectActionType, StrictModel

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
DestinationType = Literal[
    "none",
    "current_channel",
    "channel",
    "strategy",
    "unknown",
]
SensitivityLevel = Literal["public", "internal", "sensitive", "unknown"]


class RequestedScope(StrictModel):
    """Structured user-request scope emitted by the planner."""

    capability: CapabilityName | None = None
    action: SideEffectActionType | None = None
    destination_type: DestinationType = "none"
    destination_id: str | None = None
    destination_name: str | None = None
    artifact_type: ArtifactType = "unknown"
    artifact_id: str | None = None
    strategy_id: str | None = None
    strategy_name: str | None = None
    period: str | None = None
    time_range_start: str | None = None
    time_range_end: str | None = None
    sensitivity: SensitivityLevel = "unknown"
    requires_user_confirmation: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def has_destination(self) -> bool:
        return self.destination_type not in {"none", "current_channel"}


class SendScopePolicy(StrictModel):
    allowed_capabilities: list[CapabilityName] = Field(default_factory=list)
    allowed_artifact_types: list[ArtifactType] = Field(default_factory=list)
    allowed_destinations: list[str] = Field(default_factory=list)
    allowed_actions: list[SideEffectActionType] = Field(default_factory=list)
    required_user_confirmation: list[str] = Field(default_factory=list)
    redaction_policy: dict[str, object] = Field(default_factory=dict)


class GuardrailDecision(StrictModel):
    outcome: GuardrailOutcome
    phase: GuardrailPhase
    reason_code: str = Field(min_length=1)
    human_readable_reason: str = ""
    capability_id: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    source_scopes: list[dict] = Field(default_factory=list)
    evidence_required: list[str] = Field(default_factory=list)
    evidence_seen: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.outcome == "allow"


@dataclass(frozen=True)
class EvidenceSelection:
    accepted: tuple[EvidenceFact, ...]
    rejected: tuple[EvidenceFact, ...]
    decisions: tuple[GuardrailDecision, ...]

    @property
    def has_evidence(self) -> bool:
        return bool(self.accepted)


def abstention_response_text() -> str:
    return "老师，这个信息我这边暂时无法确认，先不回答避免信息不准确。"


def make_decision(
    outcome: GuardrailOutcome,
    phase: GuardrailPhase,
    reason_code: str,
    *,
    human_reason: str = "",
    capability_id: str | None = None,
    artifact_ids: list[str] | None = None,
    source_scopes: list[dict] | None = None,
    evidence_required: list[str] | None = None,
    evidence_seen: list[str] | None = None,
    metadata: dict[str, object] | None = None,
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
