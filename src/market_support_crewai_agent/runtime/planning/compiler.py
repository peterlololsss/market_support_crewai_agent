from __future__ import annotations

from typing import TYPE_CHECKING

from market_support_crewai_agent.runtime.planning.compiler_derivations import (
    artifact_kind_for_execution_units,
    compliance_reason_code_for_plan_spec,
    response_mode_for_answerability_units,
    response_mode_for_plan_spec,
    selected_refs_v2,
    unit_resolves_v2,
)
from market_support_crewai_agent.runtime.planning.compiler_inputs import (
    DeterministicPlanOriginInputV1,
    DeterministicPlanUnitV1,
    ExecutionPlanCompilationError,
)
from market_support_crewai_agent.runtime.planning.compiler_units import (
    finalize_deterministic_unit_v2,
    finalize_plan_spec_unit_v2,
)
from market_support_crewai_agent.runtime.planning.models import (
    ComplianceDecisionV1,
    ExecutionPlanOriginV1,
    ExecutionPlanV2,
)
from market_support_crewai_agent.runtime.planning.plan_hash import execution_plan_id_v2
from market_support_crewai_agent.runtime.planning.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)

if TYPE_CHECKING:
    from market_support_crewai_agent.runtime.evidence.scope_authority import (
        BusinessScopeAuthorityV1,
    )

__all__ = [
    "DeterministicPlanOriginInputV1",
    "DeterministicPlanUnitV1",
    "ExecutionPlanCompilationError",
    "finalize_execution_plan_v2",
]


def finalize_execution_plan_v2(
    source: PlanSpec | DeterministicPlanOriginInputV1,
    policy: PolicyManifestV2,
    scope_authority: BusinessScopeAuthorityV1,
    *,
    origin: ExecutionPlanOriginV1,
    confidence: float | None = None,
) -> ExecutionPlanV2:
    is_planner_source = isinstance(source, PlanSpec)
    if (origin == "planner") != is_planner_source:
        raise ExecutionPlanCompilationError("execution_plan_v2_origin_source_mismatch")
    policy_refs = {ref.manifest_id: ref for ref in policy.eligible_capabilities}
    if is_planner_source:
        units = tuple(
            finalize_plan_spec_unit_v2(unit, policy_refs, scope_authority, policy)
            for unit in source.plan_units
        )
        response_mode = response_mode_for_plan_spec(list(source.plan_units))
        compliance_reason_code = compliance_reason_code_for_plan_spec(source)
        guardrail_decisions: tuple[GuardrailDecision, ...] = ()
        plan_spec: PlanSpec | None = source
        plan_confidence = 0.0 if confidence is None else confidence
        user_need = source.user_intent_summary
        artifact_units = list(source.plan_units)
    else:
        units = tuple(
            finalize_deterministic_unit_v2(unit, policy_refs, scope_authority, policy)
            for unit in source.units
        )
        response_mode = response_mode_for_answerability_units(source.units)
        compliance_reason_code = source.compliance_reason_code
        guardrail_decisions = source.guardrail_decisions
        plan_spec = None
        plan_confidence = source.confidence if confidence is None else confidence
        user_need = source.user_need
        artifact_units = []
    compliance = ComplianceDecisionV1(
        is_compliant=False if response_mode == "refusal" else True,
        reason_code=compliance_reason_code,
        reason="",
    )
    draft = ExecutionPlanV2.model_construct(
        execution_plan_id="epl1:" + "0" * 64,
        origin=origin,
        user_need=user_need,
        artifact_kind=artifact_kind_for_execution_units(
            units, artifact_units, response_mode
        ),
        response_mode=response_mode,
        compliance=compliance,
        units=units,
        selected_manifest_refs=selected_refs_v2(units),
        adapter_resolves=unit_resolves_v2(units),
        action_intents=tuple(
            intent for unit in units for intent in unit.action_intents
        ),
        guardrail_decisions=guardrail_decisions,
        confidence=plan_confidence,
        plan_spec=plan_spec,
    )
    return ExecutionPlanV2(
        execution_plan_id=execution_plan_id_v2(draft),
        origin=draft.origin,
        user_need=draft.user_need,
        artifact_kind=draft.artifact_kind,
        response_mode=draft.response_mode,
        compliance=draft.compliance,
        units=draft.units,
        selected_manifest_refs=draft.selected_manifest_refs,
        adapter_resolves=draft.adapter_resolves,
        action_intents=draft.action_intents,
        guardrail_decisions=draft.guardrail_decisions,
        confidence=draft.confidence,
        plan_spec=draft.plan_spec,
    )
