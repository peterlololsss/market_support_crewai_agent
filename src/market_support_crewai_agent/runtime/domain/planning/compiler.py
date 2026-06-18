from __future__ import annotations

from typing import get_args

from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ArtifactKind,
    CapabilityName,
    ResponseMode,
    capability_by_name,
)
from market_support_crewai_agent.runtime.domain.compliance_policy import (
    ComplianceReasonCode,
)
from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.plan_spec import (
    AnswerabilityPolicy,
    PlanSpec,
    PlanUnit,
)
from market_support_crewai_agent.runtime.domain.planning.models import (
    ActionIntentSpec,
    AdapterResolveSpec,
    ComplianceDecision,
    ExecutionPlan,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.schemas import ReplyRequest


_AMBIGUITY_SLOT_FLAGS = frozenset(
    {
        "artifact",
        "material_pack_option",
        "report_query",
        "request_meaning",
    }
)


def compile_plan_spec(
    spec: PlanSpec,
    request: ReplyRequest,
    canonical_context: CanonicalContext,
    policy: PolicyManifest,
    domain_context: DomainContext | None = None,
) -> ExecutionPlan:
    del request, canonical_context, policy, domain_context
    units = list(spec.plan_units)
    response_mode = _response_mode_for_plan_spec(units)
    material_pack_option = _material_pack_option_for_plan_spec(units)
    compliance_reason_code = _compliance_reason_code_for_plan_spec(spec)
    compliance = ComplianceDecision(
        is_compliant=False
        if any(unit.answerability_policy == "refuse" for unit in units)
        else True,
        reason_code=compliance_reason_code,
        reason="PlanSpec answerability policy",
    )
    capabilities: list[CapabilityName] = []
    answer_capabilities: list[CapabilityName] = []
    adapter_resolves: list[AdapterResolveSpec] = []
    action_intents: list[ActionIntentSpec] = []

    if response_mode not in {
        "clarification",
        "refusal",
        "smalltalk",
        "no_reply",
        "unable",
    }:
        for unit in units:
            manifest = CAPABILITY_MANIFEST_REGISTRY.find(unit.selected_capability_id)
            if manifest is None:
                raise ValueError(
                    f"PlanSpec capability not found: {unit.selected_capability_id}"
                )
            capability = capability_by_name(str(manifest.runtime_capability or ""))
            if capability is None:
                continue
            answerability = unit.answerability_policy
            capabilities.append(capability.name)
            if answerability == "answer":
                answer_capabilities.append(capability.name)
            if answerability in {"answer", "send", "handoff"}:
                adapter_resolves.extend(
                    _adapter_resolves_from_plan_unit(
                        unit,
                        capability.name,
                        unit.domain_scope.material_pack_option,
                    )
                )
            if answerability == "send" and capability.side_effect_action_type is not None:
                action_intents.append(
                    ActionIntentSpec(
                        action_type=capability.side_effect_action_type,
                        capability=capability.name,
                        material_pack_option=_action_material_pack_option(
                            capability.name,
                            unit.domain_scope.material_pack_option,
                        ),
                    )
                )
            elif answerability == "handoff":
                adapter_resolves.extend(_adapter_resolves("sales_mention", None))

    if action_intents:
        adapter_resolves.extend(_adapter_resolves("sales_mention", None))

    return ExecutionPlan(
        user_need=spec.user_intent_summary,
        artifact_kind=_artifact_kind_for_plan_spec(units, response_mode),
        response_mode=response_mode,
        compliance=compliance,
        evidence_query=_evidence_query_from_plan_spec(spec),
        capabilities=_unique_capabilities(capabilities),
        answer_capabilities=_unique_capabilities(answer_capabilities),
        adapter_resolves=_unique_adapter_resolves(adapter_resolves),
        action_intents=action_intents,
        material_pack_option=material_pack_option,
        plan_spec=spec,
        ambiguity_slots=_ambiguity_slots_from_plan_spec(spec),
    )


def _response_mode_for_plan_spec(units: list[PlanUnit]) -> ResponseMode:
    policies = {unit.answerability_policy for unit in units}
    if "clarify" in policies:
        return "clarification"
    if "refuse" in policies:
        return "refusal"
    if "send" in policies:
        return "action"
    if "answer" in policies:
        return "knowledge_answer"
    if "handoff" in policies:
        return "handoff"
    if "smalltalk" in policies:
        return "smalltalk"
    if "no_reply" in policies:
        return "no_reply"
    return "unable"


def _compliance_reason_code_for_plan_spec(spec: PlanSpec) -> ComplianceReasonCode:
    if not any(unit.answerability_policy == "refuse" for unit in spec.plan_units):
        return "compliant_product_request"
    allowed_codes = set(get_args(ComplianceReasonCode))
    for flag in [
        *spec.risk_flags,
        *(flag for unit in spec.plan_units for flag in unit.risk_flags),
    ]:
        if flag in allowed_codes:
            return flag  # type: ignore[return-value]
    return "unknown"


def _artifact_kind_for_plan_spec(
    units: list[PlanUnit],
    response_mode: ResponseMode,
) -> ArtifactKind:
    if response_mode == "action":
        send_units = [
            unit for unit in units if unit.answerability_policy == "send"
        ]
        if len(send_units) > 1:
            return "multi_action"
        for unit in send_units:
            manifest = CAPABILITY_MANIFEST_REGISTRY.find(unit.selected_capability_id)
            capability = capability_by_name(str(manifest.runtime_capability or "")) if manifest is not None else None
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


def _adapter_resolves_from_plan_unit(
    unit: PlanUnit,
    capability_name: CapabilityName,
    material_pack_option: str | None,
) -> list[AdapterResolveSpec]:
    tools = list(unit.required_tools)
    if not tools:
        manifest = CAPABILITY_MANIFEST_REGISTRY.find(unit.selected_capability_id)
        tools = list(manifest.required_tools if manifest is not None else ())
    resolves: list[AdapterResolveSpec] = []
    for tool in tools:
        if not tool.startswith("adapter_resolve."):
            continue
        resolve_type = tool.removeprefix("adapter_resolve.")
        capability = capability_by_name(capability_name)
        option = (
            material_pack_option
            if capability is not None
            and capability.supports_material_pack_option
            else None
        )
        resolves.append(
            AdapterResolveSpec(
                resolve_type=resolve_type,  # type: ignore[arg-type]
                material_pack_option=option,
            )
        )
    if not resolves:
        resolves.extend(_adapter_resolves(capability_name, material_pack_option))
    return resolves


def _evidence_query_from_plan_spec(spec: PlanSpec) -> str | None:
    for unit in spec.plan_units:
        for step in unit.steps:
            if step.evidence_query:
                return step.evidence_query
    return None


def _ambiguity_slots_from_plan_spec(spec: PlanSpec) -> list[str]:
    if not any(unit.answerability_policy == "clarify" for unit in spec.plan_units):
        return []
    output: list[str] = []
    values = [
        *spec.risk_flags,
        *(flag for unit in spec.plan_units for flag in unit.risk_flags),
        *(case for unit in spec.plan_units for case in unit.abstention_cases),
    ]
    for value in values:
        slot = str(value).strip()
        if slot in _AMBIGUITY_SLOT_FLAGS and slot not in output:
            output.append(slot)
    return output


def _material_pack_option_for_plan_spec(units: list[PlanUnit]) -> str | None:
    for unit in units:
        option = unit.domain_scope.material_pack_option
        if option:
            return option
    return None


def _action_material_pack_option(
    capability_name: CapabilityName,
    material_pack_option: str | None,
) -> str | None:
    capability = capability_by_name(capability_name)
    if capability is None:
        return None
    return material_pack_option if capability.supports_material_pack_option else None


def _adapter_resolves(
    capability_name: CapabilityName,
    material_pack_option: str | None,
) -> list[AdapterResolveSpec]:
    capability = capability_by_name(capability_name)
    if capability is None or capability.resolve_type is None:
        return []
    return [
        AdapterResolveSpec(
            resolve_type=capability.resolve_type,
            material_pack_option=(
                material_pack_option if capability.supports_material_pack_option else None
            ),
        )
    ]


def _unique_capabilities(
    values: list[CapabilityName] | tuple[CapabilityName, ...],
    *extra: CapabilityName,
) -> list[CapabilityName]:
    seen = set()
    output: list[CapabilityName] = []
    for value in [*values, *extra]:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _unique_adapter_resolves(
    values: list[AdapterResolveSpec],
) -> list[AdapterResolveSpec]:
    seen = set()
    output: list[AdapterResolveSpec] = []
    for value in values:
        key = (value.resolve_type, value.material_pack_option)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output
