from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from market_support_crewai_agent.runtime.domain.capabilities import (
    ArtifactKind,
    CapabilityName,
    ResponseMode,
)
from market_support_crewai_agent.runtime.domain.compliance_policy import (
    ComplianceReasonCode,
)
from market_support_crewai_agent.runtime.domain.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
    RequestedScope,
)
from market_support_crewai_agent.schemas import (
    AdapterResolveType,
    SideEffectActionType,
    StrictModel,
)

ActionReportScope = Literal["channel_all", "strategy", "none"]
PlanValidationSeverity = Literal["error", "fatal"]
PlanValidationCode = Literal[
    "response_mode_not_allowed",
    "capability_not_allowed",
    "adapter_resolve_not_allowed",
    "too_many_evidence_calls",
    "action_not_allowed",
    "action_capability_mismatch",
    "action_missing_required_resolve",
    "report_action_selector_missing",
    "report_action_strategy_selector_missing_strategy",
    "report_action_channel_all_selector_has_strategy",
    "non_compliant_plan_has_actions",
    "non_compliant_plan_not_refusal",
    "unknown_compliance_has_actions",
    "ambiguous_plan_has_actions",
    "ambiguous_plan_not_clarification",
    "knowledge_answer_missing_capability",
    "plan_spec_capability_not_found",
    "plan_spec_runtime_capability_not_allowed",
]


class ComplianceDecision(StrictModel):
    is_compliant: bool | None = Field(
        default=None,
        description=(
            "True when the request is compliant, false when it must be refused, "
            "null when not enough context."
        ),
    )
    reason_code: ComplianceReasonCode = "unknown"
    reason: str = Field(default="", max_length=400)


class AdapterResolveSpec(StrictModel):
    resolve_type: AdapterResolveType
    strategy: str | None = None
    artifact_id: str | None = None


class ActionIntentSpec(StrictModel):
    action_type: SideEffectActionType
    capability: CapabilityName
    report_scope: ActionReportScope = "none"
    strategy: str | None = None


class ExecutionPlan(StrictModel):
    contract_version: Literal["execution-plan"] = "execution-plan"
    user_need: str = Field(min_length=1, max_length=500)
    artifact_kind: ArtifactKind
    response_mode: ResponseMode
    compliance: ComplianceDecision
    evidence_query: str | None = Field(default=None, max_length=200)
    capabilities: list[CapabilityName] = Field(default_factory=list, max_length=8)
    answer_capabilities: list[CapabilityName] = Field(default_factory=list, max_length=4)
    adapter_resolves: list[AdapterResolveSpec] = Field(default_factory=list, max_length=8)
    action_intents: list[ActionIntentSpec] = Field(default_factory=list, max_length=4)
    selected_strategy: str | None = None
    requested_scope: RequestedScope | None = None
    plan_spec: PlanSpec | None = None
    guardrail_decisions: list[GuardrailDecision] = Field(default_factory=list)
    ambiguity_slots: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


@dataclass(frozen=True)
class PlanValidationIssue:
    code: PlanValidationCode
    message: str
    severity: PlanValidationSeverity = "error"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanValidationResult:
    valid: bool
    issues: tuple[PlanValidationIssue, ...] = ()

    @property
    def fatal(self) -> bool:
        return any(issue.severity == "fatal" for issue in self.issues)
