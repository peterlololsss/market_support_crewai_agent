from __future__ import annotations

from typing import Final

from market_support_crewai_agent.runtime.planning.compiler_inputs import (
    DeterministicPlanUnitV1,
)
from market_support_crewai_agent.runtime.planning.models import (
    CanonicalAdapterResolveV1,
    ExecutionPlanUnitV2,
)
from market_support_crewai_agent.runtime.planning.plan_spec import (
    AnswerabilityPolicy,
    PlanSpec,
    PlanUnit,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ArtifactKind,
    ManifestRefV1,
    ResponseMode,
)
from market_support_crewai_agent.runtime.policy.capabilities.runtime_projection import (
    capability_by_name,
)
from market_support_crewai_agent.runtime.policy.compliance import ComplianceReasonCode

RUNTIME_CAPABILITY_BY_MANIFEST_ID: Final = {
    "material_pack.send": "material_pack",
    "weekly_report.send": "weekly_report",
    "monthly_report.send": "monthly_report",
    "sales.handoff": "sales_mention",
    "weekly_report.product_list": "weekly_report",
    "monthly_report.product_list": "monthly_report",
    "answer_internal_company_knowledge": "document_context",
}
_COMPLIANCE_REASON_CODE_BY_VALUE: Final[dict[str, ComplianceReasonCode]] = {
    "compliant_product_request": "compliant_product_request",
    "customer_service_request": "customer_service_request",
    "expected_or_target_return": "expected_or_target_return",
    "principal_or_risk_guarantee": "principal_or_risk_guarantee",
    "peer_or_competitor_comparison": "peer_or_competitor_comparison",
    "private_contact_request": "private_contact_request",
    "contract_or_restricted_document": "contract_or_restricted_document",
    "restricted_internal_document": "restricted_internal_document",
    "fee_waiver_request": "fee_waiver_request",
    "qualified_investor_or_threshold": "qualified_investor_or_threshold",
    "unrelated_request": "unrelated_request",
    "unknown": "unknown",
}


def response_mode_for_plan_spec(units: list[PlanUnit]) -> ResponseMode:
    return response_mode_for_answerability_policies(
        {unit.answerability_policy for unit in units}
    )


def response_mode_for_answerability_units(
    units: tuple[DeterministicPlanUnitV1, ...],
) -> ResponseMode:
    return response_mode_for_answerability_policies(
        {unit.answerability_policy for unit in units}
    )


def response_mode_for_answerability_policies(
    policies: set[AnswerabilityPolicy],
) -> ResponseMode:
    if "refuse" in policies:
        return "refusal"
    if "send" in policies:
        return "action"
    if "clarify" in policies:
        return "clarification"
    if "answer" in policies:
        return "knowledge_answer"
    if "handoff" in policies:
        return "handoff"
    if "smalltalk" in policies:
        return "smalltalk"
    if "no_reply" in policies:
        return "no_reply"
    return "unable"


def artifact_kind_for_execution_units(
    units: tuple[ExecutionPlanUnitV2, ...],
    planner_units: list[PlanUnit],
    response_mode: ResponseMode,
) -> ArtifactKind:
    if planner_units:
        return artifact_kind_for_plan_spec(planner_units, response_mode)
    if response_mode == "action":
        action_units = tuple(
            unit for unit in units if unit.answerability_policy == "send"
        )
        if len(action_units) == 1:
            return action_units[0].artifact_kind
        return "multi_action"
    return artifact_kind_for_answerability(
        next(
            unit.answerability_policy
            for unit in units
            if response_mode_for_answerability_policies({unit.answerability_policy})
            == response_mode
        )
    )


def artifact_kind_for_answerability(
    answerability: AnswerabilityPolicy,
) -> ArtifactKind:
    mapping: dict[AnswerabilityPolicy, ArtifactKind] = {
        "answer": "knowledge_answer",
        "send": "multi_action",
        "clarify": "unclear",
        "abstain": "unclear",
        "refuse": "refusal",
        "handoff": "human_support",
        "smalltalk": "smalltalk",
        "no_reply": "smalltalk",
    }
    return mapping[answerability]


def compliance_reason_code_for_plan_spec(spec: PlanSpec) -> ComplianceReasonCode:
    if not any(unit.answerability_policy == "refuse" for unit in spec.plan_units):
        return "compliant_product_request"
    for flag in [
        *spec.risk_flags,
        *(flag for unit in spec.plan_units for flag in unit.risk_flags),
    ]:
        if (reason_code := _COMPLIANCE_REASON_CODE_BY_VALUE.get(flag)) is not None:
            return reason_code
    return "unknown"


def selected_refs_v2(
    units: tuple[ExecutionPlanUnitV2, ...],
) -> tuple[ManifestRefV1, ...]:
    selected: list[ManifestRefV1] = []
    seen: set[str] = set()
    for unit in units:
        if unit.manifest_ref.manifest_id not in seen:
            seen.add(unit.manifest_ref.manifest_id)
            selected.append(unit.manifest_ref)
    return tuple(selected)


def unit_resolves_v2(
    units: tuple[ExecutionPlanUnitV2, ...],
) -> tuple[CanonicalAdapterResolveV1, ...]:
    output: list[CanonicalAdapterResolveV1] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for unit in units:
        for resolve in unit.adapter_resolves:
            key = (
                resolve.resolve_type,
                resolve.material_pack_option,
                resolve.artifact_id,
            )
            if key not in seen:
                seen.add(key)
                output.append(resolve)
    return tuple(output)


def artifact_kind_for_plan_spec(
    units: list[PlanUnit],
    response_mode: ResponseMode,
) -> ArtifactKind:
    if response_mode == "action":
        send_units = [unit for unit in units if unit.answerability_policy == "send"]
        if len(send_units) > 1:
            return "multi_action"
        for unit in send_units:
            manifest = CAPABILITY_MANIFEST_REGISTRY.find(unit.selected_capability_id)
            capability = (
                capability_by_name(
                    RUNTIME_CAPABILITY_BY_MANIFEST_ID.get(manifest.manifest_id, "")
                )
                if manifest is not None
                else None
            )
            if capability is not None:
                return capability.artifact_kind
        return "multi_action"
    if response_mode == "knowledge_answer":
        return "knowledge_answer"
    if response_mode == "handoff":
        return "human_support"
    if response_mode == "refusal":
        return "refusal"
    if response_mode == "smalltalk":
        return "smalltalk"
    if response_mode == "no_reply":
        return "smalltalk"
    return "unclear"
