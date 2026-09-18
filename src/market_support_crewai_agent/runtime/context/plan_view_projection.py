from __future__ import annotations

from typing import assert_never

from market_support_crewai_agent.runtime.context.plan_view_models import (
    ActionIntentViewV1,
    DistributionPlanScopeViewV1,
    PlanScopeViewV1,
    PlanTimeRangeViewV1,
    UnscopedPlanScopeViewV1,
    ValidatedPlanUnitViewV1,
    ValidatedPlanViewV1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.planning.models import (
    CanonicalActionIntentV1,
    DistributionExecutionDomainScopeV2,
    ExecutionDomainScopeV2,
    ExecutionPlanTimeRangeV1,
    ExecutionPlanUnitV2,
    ExecutionPlanV2,
    UnscopedExecutionDomainScopeV2,
)
from market_support_crewai_agent.schemas.conversation import (
    DistributionScopeV1,
    UnscopedScopeV1,
)


def project_plan_scope_view_v1(
    scope: ExecutionDomainScopeV2,
    authority: BusinessScopeAuthorityV1,
) -> PlanScopeViewV1:
    match scope:
        case DistributionExecutionDomainScopeV2():
            match authority.scope:
                case DistributionScopeV1():
                    if scope.business_scope_ref != authority.business_scope_ref:
                        raise ContextViewInvariantError(
                            "plan_scope_authority_ref_mismatch"
                        )
                    if scope.channel_kind != authority.scope.channel_type:
                        raise ContextViewInvariantError(
                            "plan_scope_authority_channel_mismatch"
                        )
                    return DistributionPlanScopeViewV1(
                        channel_type=authority.scope.channel_type,
                        dist_channel_name=authority.scope.dist_channel_name,
                        material_pack_option=scope.material_pack_option,
                        time_range=_project_time_range(scope.time_range),
                        product_count=len(scope.product_ids),
                    )
                case UnscopedScopeV1():
                    raise ContextViewInvariantError(
                        "plan_scope_authority_kind_mismatch"
                    )
                case unreachable:
                    assert_never(unreachable)
        case UnscopedExecutionDomainScopeV2():
            match authority.scope:
                case UnscopedScopeV1():
                    return UnscopedPlanScopeViewV1()
                case DistributionScopeV1():
                    raise ContextViewInvariantError(
                        "plan_scope_authority_kind_mismatch"
                    )
                case unreachable:
                    assert_never(unreachable)
        case unreachable:
            assert_never(unreachable)


def project_validated_plan_view_v1(
    plan: ExecutionPlanV2,
    authority: BusinessScopeAuthorityV1,
) -> ValidatedPlanViewV1:
    units = tuple(_project_plan_unit(unit, authority) for unit in plan.units)
    return ValidatedPlanViewV1(
        execution_plan_id=plan.execution_plan_id,
        user_need=plan.user_need,
        response_mode=plan.response_mode,
        artifact_kind=plan.artifact_kind,
        units=units,
        selected_manifest_refs=plan.selected_manifest_refs,
        action_intents=tuple(
            action for unit in units for action in unit.action_intents
        ),
        is_compliant=plan.compliance.is_compliant,
        compliance_reason_code=plan.compliance.reason_code,
        compliance_reason=plan.compliance.reason,
        confidence=plan.confidence,
    )


def project_action_intent_view_v1(
    intent: CanonicalActionIntentV1,
) -> ActionIntentViewV1:
    return ActionIntentViewV1(
        action_type=intent.action_type,
        capability_id=intent.capability,
        material_pack_option=intent.material_pack_option,
    )


def _project_plan_unit(
    unit: ExecutionPlanUnitV2,
    authority: BusinessScopeAuthorityV1,
) -> ValidatedPlanUnitViewV1:
    return ValidatedPlanUnitViewV1(
        unit_id=unit.unit_id,
        manifest_ref=unit.manifest_ref,
        answerability=unit.answerability_policy,
        scope=project_plan_scope_view_v1(unit.scope, authority),
        evidence_query=unit.evidence_query,
        action_intents=tuple(
            project_action_intent_view_v1(intent) for intent in unit.action_intents
        ),
        answer_capability_ids=unit.answer_capabilities,
        ambiguity_slots=unit.ambiguity_slots,
        risk_flags=unit.risk_flags,
    )


def _project_time_range(
    value: ExecutionPlanTimeRangeV1 | None,
) -> PlanTimeRangeViewV1 | None:
    if value is None:
        return None
    return PlanTimeRangeViewV1(
        period=value.period,
        start=value.start,
        end=value.end,
        label=value.label,
    )
