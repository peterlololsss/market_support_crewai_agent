from __future__ import annotations

from typing import TYPE_CHECKING

from market_support_crewai_agent.runtime.planning.clarification import (
    supported_clarification_slots,
)
from market_support_crewai_agent.runtime.planning.compiler_derivations import (
    RUNTIME_CAPABILITY_BY_MANIFEST_ID,
    artifact_kind_for_answerability,
)
from market_support_crewai_agent.runtime.planning.compiler_inputs import (
    DeterministicPlanUnitV1,
    ExecutionPlanCompilationError,
)
from market_support_crewai_agent.runtime.planning.models import (
    CanonicalActionIntentV1,
    CanonicalAdapterResolveV1,
    DistributionExecutionDomainScopeV2,
    ExecutionPlanUnitV2,
    UnscopedExecutionDomainScopeV2,
)
from market_support_crewai_agent.runtime.planning.plan_spec import (
    AnswerabilityPolicy,
    DistributionPlanDomainScopeV2,
    PlanDomainScopeV2,
    PlanUnit,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.policy.capabilities.runtime_projection import (
    CurrentRuntimeCapabilityProjection,
    capability_by_name,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2

if TYPE_CHECKING:
    from market_support_crewai_agent.runtime.evidence.scope_authority import (
        BusinessScopeAuthorityV1,
    )


def finalize_plan_spec_unit_v2(
    unit: PlanUnit,
    policy_refs: dict[str, ManifestRefV1],
    scope_authority: BusinessScopeAuthorityV1,
    policy: PolicyManifestV2,
) -> ExecutionPlanUnitV2:
    evidence_query = _single_unit_evidence_query(unit)
    if (
        unit.selected_capability_id
        in {"weekly_report.product_list", "monthly_report.product_list"}
        and evidence_query is not None
        and evidence_query.strip()
    ):
        raise ExecutionPlanCompilationError("product_list_evidence_query_forbidden")
    return _finalize_execution_unit_v2(
        unit_id=unit.unit_id,
        manifest_id=unit.selected_capability_id,
        answerability_policy=unit.answerability_policy,
        material_pack_option=_plan_scope_material_pack_option(unit.domain_scope),
        evidence_query=evidence_query,
        ambiguity_slots=tuple(
            supported_clarification_slots([*unit.risk_flags, *unit.abstention_cases])
        ),
        risk_flags=tuple(
            flag
            for flag in unit.risk_flags
            if flag == "weekly_report_rationale_required"
        ),
        policy_refs=policy_refs,
        scope_authority=scope_authority,
        policy=policy,
        plan_scope=unit.domain_scope,
    )


def finalize_deterministic_unit_v2(
    unit: DeterministicPlanUnitV1,
    policy_refs: dict[str, ManifestRefV1],
    scope_authority: BusinessScopeAuthorityV1,
    policy: PolicyManifestV2,
) -> ExecutionPlanUnitV2:
    return _finalize_execution_unit_v2(
        unit_id=unit.unit_id,
        manifest_id=unit.manifest_id,
        answerability_policy=unit.answerability_policy,
        material_pack_option=unit.material_pack_option,
        evidence_query=unit.evidence_query,
        ambiguity_slots=unit.ambiguity_slots,
        risk_flags=unit.risk_flags,
        policy_refs=policy_refs,
        scope_authority=scope_authority,
        policy=policy,
        plan_scope=None,
    )


def _plan_scope_material_pack_option(scope: PlanDomainScopeV2) -> str | None:
    if scope.kind == "distribution":
        return scope.material_pack_option
    return None


def _finalize_execution_unit_v2(
    *,
    unit_id: str,
    manifest_id: str,
    answerability_policy: AnswerabilityPolicy,
    material_pack_option: str | None,
    evidence_query: str | None,
    ambiguity_slots: tuple[str, ...],
    risk_flags: tuple[str, ...],
    policy_refs: dict[str, ManifestRefV1],
    scope_authority: BusinessScopeAuthorityV1,
    policy: PolicyManifestV2,
    plan_scope: PlanDomainScopeV2 | None,
) -> ExecutionPlanUnitV2:
    ref = policy_refs.get(manifest_id)
    if ref is None:
        raise ExecutionPlanCompilationError("plan_spec_capability_not_policy_eligible")
    manifest = CAPABILITY_MANIFEST_REGISTRY.find(manifest_id)
    if manifest is None or manifest.manifest_version != ref.manifest_version:
        raise ExecutionPlanCompilationError("plan_spec_manifest_ref_unknown")
    scope = _execution_scope(plan_scope, scope_authority, material_pack_option)
    runtime_capability = capability_by_name(
        RUNTIME_CAPABILITY_BY_MANIFEST_ID.get(manifest.manifest_id, "")
    )
    runtime_capabilities = (
        () if runtime_capability is None else (runtime_capability.name,)
    )
    resolves = (
        ()
        if runtime_capability is None or runtime_capability.resolve_type is None
        else (
            CanonicalAdapterResolveV1(
                resolve_type=runtime_capability.resolve_type,
                material_pack_option=material_pack_option
                if runtime_capability.supports_material_pack_option
                else None,
            ),
        )
    )
    intents = _action_intents(
        answerability_policy, runtime_capability, policy, material_pack_option
    )
    if answerability_policy == "send" and policy.scene == "direct":
        raise ExecutionPlanCompilationError("execution_plan_direct_send_forbidden")
    if answerability_policy == "send" and _sales_fallback_is_allowed(policy):
        resolves = (*resolves, CanonicalAdapterResolveV1(resolve_type="sales_mention"))
    return ExecutionPlanUnitV2(
        unit_id=unit_id,
        manifest_ref=ref,
        answerability_policy=answerability_policy,
        artifact_kind=runtime_capability.artifact_kind
        if runtime_capability is not None
        else artifact_kind_for_answerability(answerability_policy),
        runtime_capabilities=runtime_capabilities,
        answer_capabilities=runtime_capabilities
        if answerability_policy == "answer"
        else (),
        adapter_resolves=resolves,
        action_intents=intents,
        scope=scope,
        evidence_query=evidence_query,
        ambiguity_slots=tuple(supported_clarification_slots(ambiguity_slots)),
        risk_flags=tuple(
            flag for flag in risk_flags if flag == "weekly_report_rationale_required"
        ),
    )


def _execution_scope(
    plan_scope: PlanDomainScopeV2 | None,
    scope_authority: BusinessScopeAuthorityV1,
    material_pack_option: str | None,
) -> DistributionExecutionDomainScopeV2 | UnscopedExecutionDomainScopeV2:
    if scope_authority.scope.kind == "distribution":
        distribution_scope = _distribution_plan_scope(plan_scope)
        if (
            distribution_scope is not None
            and distribution_scope.business_scope_ref
            != scope_authority.business_scope_ref
        ):
            raise ExecutionPlanCompilationError("plan_spec_business_scope_mismatch")
        return DistributionExecutionDomainScopeV2(
            business_scope_ref=scope_authority.business_scope_ref,
            channel_kind=scope_authority.scope.channel_type,
            material_pack_option=material_pack_option,
            product_ids=()
            if distribution_scope is None
            else tuple(distribution_scope.product_ids),
        )
    _require_unscoped_plan_scope(plan_scope)
    return UnscopedExecutionDomainScopeV2()


def _action_intents(
    answerability_policy: AnswerabilityPolicy,
    runtime_capability: CurrentRuntimeCapabilityProjection | None,
    policy: PolicyManifestV2,
    material_pack_option: str | None,
) -> tuple[CanonicalActionIntentV1, ...]:
    if (
        answerability_policy != "send"
        or runtime_capability is None
        or runtime_capability.outbound_action_type is None
    ):
        return ()
    if runtime_capability.outbound_action_type not in policy.allowed_outbound_actions:
        raise ExecutionPlanCompilationError("execution_plan_action_not_allowed")
    return (
        CanonicalActionIntentV1(
            action_type=runtime_capability.outbound_action_type,
            capability=runtime_capability.name,
            material_pack_option=material_pack_option
            if runtime_capability.supports_material_pack_option
            else None,
        ),
    )


def _distribution_plan_scope(
    scope: PlanDomainScopeV2 | None,
) -> DistributionPlanDomainScopeV2 | None:
    if scope is None:
        return None
    if scope.kind == "distribution":
        return scope
    raise ExecutionPlanCompilationError("plan_spec_business_scope_mismatch")


def _require_unscoped_plan_scope(scope: PlanDomainScopeV2 | None) -> None:
    if scope is None or scope.kind == "unscoped":
        return
    raise ExecutionPlanCompilationError("plan_spec_business_scope_mismatch")


def _single_unit_evidence_query(unit: PlanUnit) -> str | None:
    queries = tuple(
        step.evidence_query for step in unit.steps if step.evidence_query is not None
    )
    if len(queries) > 1:
        raise ExecutionPlanCompilationError("plan_spec_multiple_evidence_queries")
    return queries[0] if queries else None


def _sales_fallback_is_allowed(policy: PolicyManifestV2) -> bool:
    return (
        "sales_mention" in policy.allowed_adapter_resolves
        and "sales" in policy.allowed_mention_types
    )
