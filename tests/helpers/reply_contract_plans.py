from __future__ import annotations

from collections.abc import Mapping

from pydantic import JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.planning.plan_spec import (
    AnswerabilityPolicy,
    DistributionPlanDomainScopeV2,
    PlanDomainScopeV2,
    PlanSpec,
    PlanSpecEvidenceContractV1,
    PlanStep,
    PlanUnit,
    UnscopedPlanDomainScopeV2,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
)
from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityManifestIdV2,
)
from market_support_crewai_agent.schemas.conversation import DistributionScopeV1
from tests.helpers.reply_contract_json import (
    optional_str_value,
    str_list_value,
    str_value,
)
from tests.helpers.reply_contract_requests import make_v2_envelope

_ALL_EVIDENCE_ARTIFACTS: tuple[str, ...] = (
    "material_pack",
    "weekly_report",
    "monthly_report",
    "document_context",
    "adapter_context",
    "history",
    "user_upload",
    "unknown",
)


def _compliance_value(
    payload: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue] | None:
    value = payload.get("compliance")
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    raise AssertionError("compliance_must_be_mapping")


def _capability_id_from_payload(
    payload: Mapping[str, JsonValue],
) -> CapabilityManifestIdV2:
    selected_capability_id = optional_str_value(payload, "selected_capability_id")
    if selected_capability_id is not None:
        manifest = CAPABILITY_MANIFEST_REGISTRY.find(selected_capability_id)
        if manifest is None:
            raise AssertionError("selected_capability_id_must_be_known")
        return manifest.manifest_id
    compliance = _compliance_value(payload)
    if compliance is not None and compliance.get("is_compliant") is False:
        return "general.refusal"
    artifact_kind = str_value(payload, "artifact_kind", "unclear")
    action_intent = str_value(payload, "action_intent", "none")
    requested = str_list_value(payload, "requested_capabilities") or []
    if str_list_value(payload, "ambiguity_slots"):
        return "general.clarification"
    if action_intent == "send":
        if artifact_kind == "material_pack":
            return "material_pack.send"
        if artifact_kind == "weekly_report":
            return "weekly_report.send"
        if artifact_kind == "monthly_report":
            return "monthly_report.send"
    if action_intent == "handoff" or artifact_kind == "human_support":
        return "sales.handoff"
    if action_intent == "refuse" or artifact_kind == "refusal":
        return "general.refusal"
    if action_intent == "answer" or artifact_kind == "knowledge_answer":
        if "document_context" in requested:
            return "answer_internal_company_knowledge"
        if "weekly_report" in requested:
            return "weekly_report.product_list"
        if "monthly_report" in requested:
            return "monthly_report.product_list"
    if artifact_kind == "smalltalk":
        return "general.smalltalk"
    return "general.abstention"


def _answerability_from_capability_id(
    capability_id: str,
    payload: Mapping[str, JsonValue],
) -> AnswerabilityPolicy:
    if capability_id.endswith(".send"):
        return "send"
    if capability_id == "sales.handoff":
        return "handoff"
    if capability_id == "general.clarification":
        return "clarify"
    if capability_id == "general.refusal":
        return "refuse"
    if capability_id == "general.smalltalk":
        return "smalltalk"
    if capability_id == "general.no_reply":
        return "no_reply"
    if capability_id == "general.abstention":
        return "abstain"
    if str_value(payload, "action_intent", "none") == "send":
        return "send"
    return "answer"


def _answerability_value(
    capability_id: str,
    payload: Mapping[str, JsonValue],
) -> AnswerabilityPolicy:
    value = optional_str_value(payload, "answerability_policy")
    if value is None:
        return _answerability_from_capability_id(capability_id, payload)
    adapter: TypeAdapter[AnswerabilityPolicy] = TypeAdapter(AnswerabilityPolicy)
    return adapter.validate_python(value)


def _domain_scope_payload(
    request: KernelReplyRequestV1,
    material_pack_option: str | None,
    capability_id: str,
) -> PlanDomainScopeV2:
    scope_authority = business_scope_authority_v1(request.business_scope)
    selected_option = (
        material_pack_option if capability_id == "material_pack.send" else None
    )
    scope = request.business_scope
    if isinstance(scope, DistributionScopeV1):
        return DistributionPlanDomainScopeV2(
            business_scope_ref=scope_authority.business_scope_ref,
            channel_kind=scope.channel_type,
            material_pack_option=selected_option,
        )
    return UnscopedPlanDomainScopeV2()


def _risk_flags(payload: Mapping[str, JsonValue]) -> list[str]:
    flags = str_list_value(payload, "risk_flags") or []
    flags.extend(str_list_value(payload, "ambiguity_slots") or [])
    compliance = _compliance_value(payload)
    reason_code = compliance.get("reason_code") if compliance is not None else None
    if isinstance(reason_code, str):
        flags.append(reason_code)
    return flags


def plan_from_payload(
    request: KernelReplyRequestV1 | None,
    payload: Mapping[str, JsonValue],
) -> PlanSpec:
    request = request or make_v2_envelope().request
    capability_id = _capability_id_from_payload(payload)
    manifest = CAPABILITY_MANIFEST_REGISTRY.get(capability_id)
    required_artifacts = [
        str(artifact) for artifact in manifest.evidence_contract.required_artifact_types
    ]
    allowed_artifacts = [
        str(artifact) for artifact in manifest.evidence_contract.allowed_artifact_types
    ]
    forbidden_artifacts = [
        artifact_type
        for artifact_type in _ALL_EVIDENCE_ARTIFACTS
        if artifact_type not in allowed_artifacts
    ]
    unit = PlanUnit(
        unit_id=str_value(payload, "unit_id", "unit-1"),
        selected_capability_id=capability_id,
        domain_scope=_domain_scope_payload(
            request,
            optional_str_value(payload, "material_pack_option"),
            capability_id,
        ),
        required_artifacts=required_artifacts,
        allowed_artifacts=allowed_artifacts,
        forbidden_artifacts=forbidden_artifacts,
        required_tools=[],
        answerability_policy=_answerability_value(capability_id, payload),
        output_schema_ref=f"{manifest.manifest_id}:output_schema",
        evidence_contract_ref=f"{manifest.manifest_id}:evidence_contract",
        evidence_contract=PlanSpecEvidenceContractV1(
            required_fact_types=tuple(manifest.evidence_contract.required_fact_types),
            any_of_fact_types=tuple(manifest.evidence_contract.any_of_fact_types),
            allowed_source_types=tuple(manifest.evidence_contract.allowed_source_types),
            forbidden_source_types=tuple(
                manifest.evidence_contract.forbidden_source_types
            ),
            min_facts=manifest.evidence_contract.min_facts,
        ),
        steps=[
            PlanStep(
                step_id="step-1",
                description=str_value(
                    payload,
                    "user_intent_summary",
                    str_value(payload, "user_need", "handle current support request"),
                ),
                uses_artifacts=required_artifacts,
                required_artifacts=required_artifacts,
                allowed_artifacts=allowed_artifacts,
                forbidden_artifacts=forbidden_artifacts,
                required_tools=[],
                evidence_query=optional_str_value(payload, "evidence_query"),
            )
        ],
        acceptance_criteria=["satisfy selected capability contract"],
        abstention_cases=(
            [manifest.abstention_policy.guidance]
            if manifest.abstention_policy.guidance
            else []
        ),
        risk_flags=_risk_flags(payload),
    )
    return PlanSpec(
        plan_id=str_value(payload, "plan_id", "plan-test"),
        user_intent_summary=str_value(
            payload,
            "user_intent_summary",
            str_value(payload, "user_need", "handle current support request"),
        ),
        plan_units=[unit],
        risk_flags=_risk_flags(payload),
    )
