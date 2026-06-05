from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field, model_validator

from market_support_crewai_agent.runtime.compliance_policy import ComplianceReasonCode
from market_support_crewai_agent.runtime.policy import (
    BusinessCheck,
    PolicyManifest,
    ReadCapability,
)
from market_support_crewai_agent.schemas import (
    AdapterResolveType,
    SideEffectActionType,
    StrictModel,
)

PlannedIntent = Literal[
    "send_material_pack",
    "send_weekly_report",
    "send_monthly_report",
    "knowledge_qa",
    "human_handoff",
    "clarification",
    "refusal",
    "no_reply",
]
EntityType = Literal["strategy", "product", "channel", "person", "company", "period"]
ReportActionScope = Literal["channel_all", "strategy", "unknown"]
PlanValidationSeverity = Literal["error", "fatal"]
PlanValidationCode = Literal[
    "planner_output_invalid_contract",
    "evidence_capability_not_allowed",
    "business_check_not_allowed",
    "adapter_resolve_not_allowed",
    "too_many_evidence_requests",
    "action_not_allowed",
    "action_missing_required_resolve",
    "report_action_selector_missing",
    "report_action_strategy_selector_missing_strategy",
    "report_action_channel_all_selector_has_strategy",
    "non_compliant_plan_has_actions",
    "non_compliant_plan_not_refusal",
    "unknown_compliance_has_actions",
]

_REQUIRED_RESOLVE_FOR_ACTION: dict[SideEffectActionType, AdapterResolveType] = {
    "send_material_pack": "material_pack",
    "send_weekly_report": "weekly_report",
    "send_monthly_report": "monthly_report",
}


class ComplianceDecision(StrictModel):
    is_compliant: bool | None = Field(
        default=None,
        description="True when the request is compliant, false when it must be refused, null when not enough context.",
    )
    reason_code: ComplianceReasonCode = "unknown"
    reason: str = Field(default="", max_length=400)


class DetectedEntity(StrictModel):
    type: EntityType
    raw_text: str = Field(min_length=1)
    canonical_name: str | None = None


class EvidenceRequest(StrictModel):
    capability: ReadCapability
    reason: str = Field(default="", max_length=300)
    strategy: str | None = None
    period: str | None = None


class BusinessCheckRequest(StrictModel):
    check: BusinessCheck
    reason: str = Field(default="", max_length=300)


class CandidateAction(StrictModel):
    type: SideEffectActionType
    strategy: str | None = None
    report_scope: ReportActionScope = Field(
        default="unknown",
        description=(
            "Internal selector for report actions only: channel_all sends the "
            "adapter-resolved channel report package, strategy means the user "
            "asked about one confirmed strategy/product covered by that report."
        ),
    )


class ReplyPlan(StrictModel):
    contract_version: Literal["reply-plan.v1"] = "reply-plan.v1"
    user_need: str = Field(min_length=1, max_length=500)
    intent: PlannedIntent
    compliance: ComplianceDecision
    detected_entities: list[DetectedEntity] = Field(default_factory=list, max_length=12)
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list, max_length=8)
    business_checks: list[BusinessCheckRequest] = Field(default_factory=list, max_length=8)
    required_adapter_resolves: list[AdapterResolveType] = Field(
        default_factory=list,
        max_length=8,
    )
    candidate_actions: list[CandidateAction] = Field(default_factory=list, max_length=4)
    ambiguity: bool = False
    ambiguity_reason: str = Field(default="", max_length=300)
    unsupported_request_note: str = Field(default="", max_length=300)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_non_compliant_shape(self):
        if self.compliance.is_compliant is False and self.intent != "refusal":
            raise ValueError("non-compliant plans must use intent=refusal")
        return self


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


def validate_plan(plan: ReplyPlan, policy: PolicyManifest) -> PlanValidationResult:
    issues: list[PlanValidationIssue] = []

    if len(plan.evidence_requests) > policy.evidence_call_limit:
        issues.append(
            PlanValidationIssue(
                code="too_many_evidence_requests",
                message="plan exceeds policy evidence call limit",
                metadata={
                    "limit": policy.evidence_call_limit,
                    "actual": len(plan.evidence_requests),
                },
            )
        )

    for request in plan.evidence_requests:
        if request.capability not in policy.allowed_read_capabilities:
            issues.append(
                PlanValidationIssue(
                    code="evidence_capability_not_allowed",
                    message=f"evidence capability {request.capability} is not allowed",
                    severity="fatal",
                    metadata={"capability": request.capability},
                )
            )

    for check in plan.business_checks:
        if check.check not in policy.allowed_business_checks:
            issues.append(
                PlanValidationIssue(
                    code="business_check_not_allowed",
                    message=f"business check {check.check} is not allowed",
                    severity="fatal",
                    metadata={"check": check.check},
                )
            )

    for resolve_type in plan.required_adapter_resolves:
        if resolve_type not in policy.required_adapter_resolves:
            issues.append(
                PlanValidationIssue(
                    code="adapter_resolve_not_allowed",
                    message=f"adapter resolve {resolve_type} is not allowed",
                    severity="fatal",
                    metadata={"resolve_type": resolve_type},
                )
            )

    required_resolves = set(plan.required_adapter_resolves)
    for action in plan.candidate_actions:
        if action.type not in policy.allowed_side_effect_actions:
            issues.append(
                PlanValidationIssue(
                    code="action_not_allowed",
                    message=f"candidate action {action.type} is not allowed",
                    severity="fatal",
                    metadata={"action_type": action.type},
                )
            )
            continue

        expected_resolve = _REQUIRED_RESOLVE_FOR_ACTION[action.type]
        if expected_resolve not in required_resolves:
            issues.append(
                PlanValidationIssue(
                    code="action_missing_required_resolve",
                    message=f"{action.type} must require {expected_resolve}",
                    metadata={
                        "action_type": action.type,
                        "resolve_type": expected_resolve,
                    },
                )
            )

        if action.type in {"send_weekly_report", "send_monthly_report"}:
            if action.report_scope == "unknown":
                issues.append(
                    PlanValidationIssue(
                        code="report_action_selector_missing",
                        message=f"{action.type} must declare report_scope",
                        metadata={"action_type": action.type},
                    )
                )
            elif action.report_scope == "strategy" and not action.strategy:
                issues.append(
                    PlanValidationIssue(
                        code="report_action_strategy_selector_missing_strategy",
                        message=(
                            f"{action.type} with report_scope=strategy must "
                            "include a strategy"
                        ),
                        metadata={"action_type": action.type},
                    )
                )
            elif action.report_scope == "channel_all" and action.strategy:
                issues.append(
                    PlanValidationIssue(
                        code="report_action_channel_all_selector_has_strategy",
                        message=(
                            f"{action.type} with report_scope=channel_all must "
                            "not include a strategy"
                        ),
                        metadata={
                            "action_type": action.type,
                            "strategy": action.strategy,
                        },
                    )
                )

    if plan.compliance.is_compliant is False:
        if plan.intent != "refusal":
            issues.append(
                PlanValidationIssue(
                    code="non_compliant_plan_not_refusal",
                    message="non-compliant plan must use refusal intent",
                    severity="fatal",
                )
            )
        if plan.candidate_actions:
            issues.append(
                PlanValidationIssue(
                    code="non_compliant_plan_has_actions",
                    message="non-compliant plan must not propose side-effect actions",
                    severity="fatal",
                )
            )

    if plan.compliance.is_compliant is None and plan.candidate_actions:
        issues.append(
            PlanValidationIssue(
                code="unknown_compliance_has_actions",
                message="unknown compliance status cannot propose side-effect actions",
                severity="fatal",
            )
        )

    return PlanValidationResult(valid=not issues, issues=tuple(issues))


def fallback_plan(reason: str = "planner unavailable") -> ReplyPlan:
    return ReplyPlan(
        user_need=reason,
        intent="no_reply",
        compliance=ComplianceDecision(
            is_compliant=None,
            reason_code="unknown",
            reason=reason,
        ),
        ambiguity=True,
        ambiguity_reason=reason,
        confidence=0.0,
    )


def invalid_plan_contract_validation(
        reason: str = "planner output did not match ReplyPlan contract",
) -> PlanValidationResult:
    return PlanValidationResult(
        valid=False,
        issues=(
            PlanValidationIssue(
                code="planner_output_invalid_contract",
                message=reason,
                severity="fatal",
            ),
        ),
    )
