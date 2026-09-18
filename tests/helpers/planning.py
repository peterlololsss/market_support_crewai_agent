from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
)
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
from market_support_crewai_agent.runtime.policy.capabilities.semantics import (
    forbidden_artifacts_for_manifest,
)
from market_support_crewai_agent.schemas.conversation import DistributionScopeV1


def make_plan_spec(
    request: KernelReplyRequestV1 | None = None,
    *,
    selected_capability_id: str | None = None,
    answerability_policy: str | None = None,
    user_intent_summary: str | None = None,
    evidence_query: str | None = None,
    material_pack_option: str | None = None,
    plan_units: Sequence[Mapping[str, JsonValue]] | None = None,
    **intent_like: JsonValue,
) -> PlanSpec:
    request = request or make_request()
    base_payload: dict[str, JsonValue] = {
        **intent_like,
        "selected_capability_id": selected_capability_id,
        "answerability_policy": answerability_policy,
        "user_intent_summary": user_intent_summary,
        "evidence_query": evidence_query,
        "material_pack_option": material_pack_option,
    }
    unit_payloads = plan_units or [base_payload]
    units = [
        _plan_unit_payload(
            request,
            index=index,
            payload={**base_payload, **unit_payload},
        )
        for index, unit_payload in enumerate(unit_payloads, start=1)
    ]
    summary = (
        user_intent_summary
        or _string_value(intent_like, "user_need")
        or units[0].steps[0].description
    )
    risk_flags = _string_values(intent_like, "risk_flags")
    risk_flags.extend(_string_values(intent_like, "ambiguity_slots"))
    reason_code = _compliance_reason_code(intent_like)
    if reason_code is not None:
        risk_flags.append(reason_code)
    return PlanSpec(
        plan_id=_string_value(intent_like, "plan_id") or "plan-test",
        user_intent_summary=summary,
        plan_units=units,
        risk_flags=risk_flags,
    )


def _plan_unit_payload(
    request: KernelReplyRequestV1,
    *,
    index: int,
    payload: Mapping[str, JsonValue],
) -> PlanUnit:
    capability_id = _capability_id_from_payload(payload)
    manifest = CAPABILITY_MANIFEST_REGISTRY.get(capability_id)
    answerability = _answerability_from_capability_id(capability_id)
    evidence_contract = manifest.evidence_contract
    required_artifacts = [
        str(value) for value in evidence_contract.required_artifact_types
    ]
    allowed_artifacts = [
        str(value) for value in evidence_contract.allowed_artifact_types
    ]
    forbidden_artifacts = [
        str(artifact) for artifact in forbidden_artifacts_for_manifest(manifest)
    ]
    selected_option = _material_pack_option_for_scope(
        capability_id,
        _string_value(payload, "material_pack_option"),
        payload,
    )
    step = PlanStep(
        step_id=f"step-{index}",
        description=(
            _string_value(payload, "user_intent_summary")
            or _string_value(payload, "user_need")
            or "handle current support request"
        ),
        uses_artifacts=required_artifacts,
        required_artifacts=required_artifacts,
        allowed_artifacts=allowed_artifacts,
        forbidden_artifacts=forbidden_artifacts,
        evidence_query=_string_value(payload, "evidence_query"),
    )
    risk_flags = _string_values(payload, "risk_flags")
    risk_flags.extend(_string_values(payload, "ambiguity_slots"))
    reason_code = _compliance_reason_code(payload)
    if reason_code is not None:
        risk_flags.append(reason_code)
    return PlanUnit(
        unit_id=_string_value(payload, "unit_id") or f"unit-{index}",
        selected_capability_id=capability_id,
        domain_scope=_domain_scope_payload(request, selected_option),
        required_artifacts=required_artifacts,
        allowed_artifacts=allowed_artifacts,
        forbidden_artifacts=forbidden_artifacts,
        required_tools=[],
        answerability_policy=answerability,
        output_schema_ref=f"{manifest.manifest_id}:output_schema",
        evidence_contract_ref=f"{manifest.manifest_id}:evidence_contract",
        evidence_contract=PlanSpecEvidenceContractV1(
            required_fact_types=tuple(evidence_contract.required_fact_types),
            any_of_fact_types=tuple(evidence_contract.any_of_fact_types),
            allowed_source_types=tuple(evidence_contract.allowed_source_types),
            forbidden_source_types=tuple(evidence_contract.forbidden_source_types),
            min_facts=evidence_contract.min_facts,
        ),
        steps=[step],
        acceptance_criteria=["satisfy selected capability contract"],
        abstention_cases=[manifest.abstention_policy.guidance]
        if manifest.abstention_policy.guidance
        else [],
        risk_flags=risk_flags,
    )


def make_request(**overrides: JsonValue) -> KernelReplyRequestV1:
    from tests.helpers.reply_contract_requests import make_v2_envelope

    message = _string_value(overrides, "message") or ""
    envelope_overrides = {
        key: value for key, value in overrides.items() if key != "message"
    }
    return make_v2_envelope(message=message, **envelope_overrides).request


def _capability_id_from_payload(
    payload: Mapping[str, JsonValue],
) -> CapabilityManifestIdV2:
    selected_capability_id = _string_value(payload, "selected_capability_id")
    if selected_capability_id is not None:
        return _registered_capability_id(selected_capability_id)
    compliance = payload.get("compliance")
    if isinstance(compliance, dict) and compliance.get("is_compliant") is False:
        return "general.refusal"
    artifact_kind = _string_value(payload, "artifact_kind") or "unclear"
    action_intent = _string_value(payload, "action_intent") or "none"
    requested = _string_values(payload, "requested_capabilities")
    if _string_values(payload, "ambiguity_slots"):
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
    capability_id: CapabilityManifestIdV2,
) -> AnswerabilityPolicy:
    answerability_by_capability: dict[CapabilityManifestIdV2, AnswerabilityPolicy] = {
        "material_pack.send": "send",
        "weekly_report.send": "send",
        "monthly_report.send": "send",
        "sales.handoff": "handoff",
        "general.handoff": "handoff",
        "general.clarification": "clarify",
        "general.refusal": "refuse",
        "general.smalltalk": "smalltalk",
        "general.no_reply": "no_reply",
        "general.abstention": "abstain",
        "answer_internal_company_knowledge": "answer",
        "weekly_report.product_list": "answer",
        "monthly_report.product_list": "answer",
    }
    return answerability_by_capability[capability_id]


def _material_pack_option_for_scope(
    capability_id: CapabilityManifestIdV2,
    material_pack_option: str | None,
    payload: Mapping[str, JsonValue],
) -> str | None:
    if capability_id != "material_pack.send":
        return None
    option = material_pack_option or _string_value(payload, "material_pack_option")
    return option.strip() if option else None


def _registered_capability_id(value: str) -> CapabilityManifestIdV2:
    manifest = CAPABILITY_MANIFEST_REGISTRY.find(value)
    if manifest is None:
        raise AssertionError("unknown_capability_manifest")
    return manifest.manifest_id


def _string_value(payload: Mapping[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _string_values(payload: Mapping[str, JsonValue], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _compliance_reason_code(payload: Mapping[str, JsonValue]) -> str | None:
    compliance = payload.get("compliance")
    if not isinstance(compliance, dict):
        return None
    reason_code = compliance.get("reason_code")
    return reason_code if isinstance(reason_code, str) else None


def _domain_scope_payload(
    request: KernelReplyRequestV1,
    material_pack_option: str | None,
) -> PlanDomainScopeV2:
    scope_authority = business_scope_authority_v1(request.business_scope)
    if isinstance(request.business_scope, DistributionScopeV1):
        return DistributionPlanDomainScopeV2(
            business_scope_ref=scope_authority.business_scope_ref,
            channel_kind=request.business_scope.channel_type,
            material_pack_option=material_pack_option,
            product_ids=(),
        )
    return UnscopedPlanDomainScopeV2()
