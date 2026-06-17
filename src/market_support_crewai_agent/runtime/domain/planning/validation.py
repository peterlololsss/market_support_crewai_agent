from __future__ import annotations

from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    capability_by_action_type,
)
from market_support_crewai_agent.runtime.domain.planning.models import (
    ActionIntentSpec,
    ExecutionPlan,
    PlanValidationIssue,
    PlanValidationResult,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest


def validate_execution_plan(
    plan: ExecutionPlan,
    policy: PolicyManifest,
) -> PlanValidationResult:
    issues: list[PlanValidationIssue] = []
    if plan.plan_spec is not None:
        manifest = CAPABILITY_MANIFEST_REGISTRY.find(
            plan.plan_spec.selected_capability_id
        )
        if manifest is None:
            issues.append(
                PlanValidationIssue(
                    code="plan_spec_capability_not_found",
                    message="PlanSpec selected capability does not exist",
                    severity="fatal",
                    metadata={
                        "selected_capability_id": (
                            plan.plan_spec.selected_capability_id
                        )
                    },
                )
            )
        elif (
            manifest.runtime_capability is not None
            and manifest.runtime_capability not in policy.allowed_capabilities
        ):
            issues.append(
                PlanValidationIssue(
                    code="plan_spec_runtime_capability_not_allowed",
                    message="PlanSpec selected capability is outside policy",
                    severity="fatal",
                    metadata={
                        "selected_capability_id": manifest.id,
                        "runtime_capability": manifest.runtime_capability,
                    },
                )
            )

    if plan.response_mode not in policy.allowed_reply_modes:
        issues.append(
            PlanValidationIssue(
                code="response_mode_not_allowed",
                message=f"response mode {plan.response_mode} is not allowed",
                severity="fatal",
                metadata={"response_mode": plan.response_mode},
            )
        )

    for capability in plan.capabilities:
        if capability not in policy.allowed_capabilities:
            issues.append(
                PlanValidationIssue(
                    code="capability_not_allowed",
                    message=f"capability {capability} is not allowed",
                    severity="fatal",
                    metadata={"capability": capability},
                )
            )

    evidence_call_count = len(plan.adapter_resolves) + (
        1 if "document_context" in plan.capabilities else 0
    )
    if evidence_call_count > policy.evidence_call_limit:
        issues.append(
            PlanValidationIssue(
                code="too_many_evidence_calls",
                message="plan exceeds policy evidence call limit",
                metadata={
                    "limit": policy.evidence_call_limit,
                    "actual": evidence_call_count,
                },
            )
        )

    adapter_resolve_types = {item.resolve_type for item in plan.adapter_resolves}
    for resolve_spec in plan.adapter_resolves:
        if resolve_spec.resolve_type not in policy.allowed_adapter_resolves:
            issues.append(
                PlanValidationIssue(
                    code="adapter_resolve_not_allowed",
                    message=f"adapter resolve {resolve_spec.resolve_type} is not allowed",
                    severity="fatal",
                    metadata={"resolve_type": resolve_spec.resolve_type},
                )
            )

    for action in plan.action_intents:
        if action.action_type not in policy.allowed_side_effect_actions:
            issues.append(
                PlanValidationIssue(
                    code="action_not_allowed",
                    message=f"action {action.action_type} is not allowed",
                    severity="fatal",
                    metadata={"action_type": action.action_type},
                )
            )
            continue

        capability = capability_by_action_type(action.action_type)
        expected_capability = capability.name if capability is not None else None
        expected_resolve = capability.resolve_type if capability is not None else None
        if expected_capability != action.capability:
            issues.append(
                PlanValidationIssue(
                    code="action_capability_mismatch",
                    message="action capability does not match registry",
                    severity="fatal",
                    metadata={
                        "action_type": action.action_type,
                        "capability": action.capability,
                        "expected_capability": expected_capability,
                    },
                )
            )
            continue
        if expected_resolve is not None and expected_resolve not in adapter_resolve_types:
            issues.append(
                PlanValidationIssue(
                    code="action_missing_required_resolve",
                    message=f"{action.action_type} must require {expected_resolve}",
                    metadata={
                        "action_type": action.action_type,
                        "resolve_type": expected_resolve,
                    },
                )
            )
        if capability is not None and capability.is_report:
            issues.extend(_validate_report_action_selector(action))

    if plan.compliance.is_compliant is False:
        if plan.response_mode != "refusal":
            issues.append(
                PlanValidationIssue(
                    code="non_compliant_plan_not_refusal",
                    message="non-compliant plan must use refusal mode",
                    severity="fatal",
                )
            )
        if plan.capabilities or plan.adapter_resolves or plan.action_intents:
            issues.append(
                PlanValidationIssue(
                    code="non_compliant_plan_has_actions",
                    message="non-compliant plan must not propose capabilities, resolves, or actions",
                    severity="fatal",
                )
            )

    if plan.compliance.is_compliant is None and plan.action_intents:
        issues.append(
            PlanValidationIssue(
                code="unknown_compliance_has_actions",
                message="unknown compliance status cannot propose side-effect actions",
                severity="fatal",
            )
        )

    if plan.ambiguity_slots:
        if plan.response_mode != "clarification":
            issues.append(
                PlanValidationIssue(
                    code="ambiguous_plan_not_clarification",
                    message="plans with ambiguity slots must use clarification mode",
                    severity="fatal",
                    metadata={"ambiguity_slots": list(plan.ambiguity_slots)},
                )
            )
        if plan.action_intents:
            issues.append(
                PlanValidationIssue(
                    code="ambiguous_plan_has_actions",
                    message="ambiguous plans must not propose side-effect actions",
                    severity="fatal",
                    metadata={"ambiguity_slots": list(plan.ambiguity_slots)},
                )
            )

    if (
        plan.response_mode == "knowledge_answer"
        and "document_context" not in plan.capabilities
        and not plan.adapter_resolves
    ):
        issues.append(
            PlanValidationIssue(
                code="knowledge_answer_missing_capability",
                message="knowledge_answer mode requires document_context or report resolve evidence",
                severity="fatal",
            )
        )

    return PlanValidationResult(valid=not issues, issues=tuple(issues))


def _validate_report_action_selector(
    action: ActionIntentSpec,
) -> list[PlanValidationIssue]:
    if action.report_scope == "none":
        return [
            PlanValidationIssue(
                code="report_action_selector_missing",
                message=f"{action.action_type} must declare report_scope",
                metadata={"action_type": action.action_type},
            )
        ]
    if action.report_scope == "strategy" and not action.strategy:
        return [
            PlanValidationIssue(
                code="report_action_strategy_selector_missing_strategy",
                message=f"{action.action_type} with report_scope=strategy must include a strategy",
                metadata={"action_type": action.action_type},
            )
        ]
    if action.report_scope == "channel_all" and action.strategy:
        return [
            PlanValidationIssue(
                code="report_action_channel_all_selector_has_strategy",
                message=f"{action.action_type} with report_scope=channel_all must not include a strategy",
                metadata={
                    "action_type": action.action_type,
                    "strategy": action.strategy,
                },
            )
        ]
    return []
