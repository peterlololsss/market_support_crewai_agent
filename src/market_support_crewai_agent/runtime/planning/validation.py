from __future__ import annotations

from market_support_crewai_agent.runtime.planning.models import (
    ExecutionPlanV2,
    PlanValidationIssue,
    PlanValidationResult,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2


def validate_execution_plan_v2(
    plan: ExecutionPlanV2,
    policy: PolicyManifestV2,
) -> PlanValidationResult:
    issues: list[PlanValidationIssue] = []
    eligible_refs = policy.eligible_capabilities
    if plan.response_mode not in policy.allowed_reply_modes:
        issues.append(
            PlanValidationIssue(
                code="response_mode_not_allowed",
                message="execution plan response mode is outside the final policy",
                severity="fatal",
            )
        )
    for unit in plan.units:
        if unit.manifest_ref not in eligible_refs:
            issues.append(
                PlanValidationIssue(
                    code="capability_not_allowed",
                    message="execution unit manifest is outside the final policy",
                    severity="fatal",
                    metadata={"unit_id": unit.unit_id},
                )
            )
        for resolve in unit.adapter_resolves:
            if resolve.resolve_type not in policy.allowed_adapter_resolves:
                issues.append(
                    PlanValidationIssue(
                        code="adapter_resolve_not_allowed",
                        message="execution unit resolve is outside the final policy",
                        severity="fatal",
                        metadata={"unit_id": unit.unit_id},
                    )
                )
        for action in unit.action_intents:
            if action.action_type not in policy.allowed_outbound_actions:
                issues.append(
                    PlanValidationIssue(
                        code="action_not_allowed",
                        message="execution unit action is outside the final policy",
                        severity="fatal",
                        metadata={"unit_id": unit.unit_id},
                    )
                )
    return PlanValidationResult(valid=not issues, issues=tuple(issues))
