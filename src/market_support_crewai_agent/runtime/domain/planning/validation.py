from __future__ import annotations

from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    capability_by_action_type,
)
from market_support_crewai_agent.runtime.domain.planning.models import (
    ExecutionPlan,
    PlanValidationIssue,
    PlanValidationResult,
)
from market_support_crewai_agent.runtime.domain.planning.clarification import (
    CLARIFICATION_PRIORITY,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest


def validate_execution_plan(
    plan: ExecutionPlan,
    policy: PolicyManifest,
) -> PlanValidationResult:
    issues: list[PlanValidationIssue] = []
    if plan.plan_spec is not None:
        for unit in plan.plan_spec.plan_units:
            manifest = CAPABILITY_MANIFEST_REGISTRY.find(unit.selected_capability_id)
            if manifest is None:
                issues.append(
                    PlanValidationIssue(
                        code="plan_spec_capability_not_found",
                        message="PlanSpec selected capability does not exist",
                        severity="fatal",
                        metadata={
                            "unit_id": unit.unit_id,
                            "selected_capability_id": unit.selected_capability_id,
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
                            "unit_id": unit.unit_id,
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

    issues.extend(_validate_material_pack_scope(plan, policy))

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

    if plan.response_mode == "clarification" and not plan.ambiguity_slots:
        issues.append(
            PlanValidationIssue(
                code="clarification_missing_supported_slot",
                message=(
                    "clarification requires a user-resolvable ambiguity slot: "
                    + ", ".join(CLARIFICATION_PRIORITY)
                ),
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


def _validate_material_pack_scope(
    plan: ExecutionPlan,
    policy: PolicyManifest,
) -> list[PlanValidationIssue]:
    if "material_pack" not in plan.capabilities:
        return []
    selected = {plan.material_pack_option or ""}
    selected.update(
        action.material_pack_option or ""
        for action in plan.action_intents
        if action.capability == "material_pack"
    )
    selected.discard("")
    if not selected:
        return []
    allowed = set(policy.material_pack_options)
    invalid = sorted(value for value in selected if value not in allowed)
    if not invalid:
        return []
    return [
        PlanValidationIssue(
            code="material_pack_scope_not_allowed",
            message="material_pack_option must be one of available_artifacts material_pack.options",
            severity="fatal",
            metadata={
                "invalid_scope": invalid,
                "material_pack_options": list(policy.material_pack_options),
            },
        )
    ]
