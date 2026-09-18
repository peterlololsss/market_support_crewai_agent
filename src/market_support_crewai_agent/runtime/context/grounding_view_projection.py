from __future__ import annotations

from typing import TYPE_CHECKING

from market_support_crewai_agent.runtime.context.business_view_projection import (
    project_business_facts_view_v1,
    project_guardrail_decision_view_v1,
)
from market_support_crewai_agent.runtime.context.evidence_view_projection import (
    project_evidence_fact_view_v1,
)
from market_support_crewai_agent.runtime.context.grounding_projection_context import (
    GroundingProjectionContextV1,
)
from market_support_crewai_agent.runtime.context.models import (
    UnitGroundingViewV1,
)
from market_support_crewai_agent.runtime.context.plan_view_projection import (
    project_action_intent_view_v1,
    project_plan_scope_view_v1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.planning.models import (
    ExecutionPlanUnitV2,
    ExecutionPlanV2,
)

if TYPE_CHECKING:
    from market_support_crewai_agent.runtime.evidence.grounding import (
        ExecutionUnitGroundingV1,
    )


def project_unit_grounding_views_v1(
    plan: ExecutionPlanV2,
    groundings: tuple[ExecutionUnitGroundingV1, ...],
    context: GroundingProjectionContextV1,
) -> tuple[UnitGroundingViewV1, ...]:
    if len(plan.units) != len(groundings):
        raise ContextViewInvariantError("unit_grounding_count_mismatch")
    return tuple(
        _project_grounding(unit, grounding, context)
        for unit, grounding in zip(plan.units, groundings, strict=True)
    )


def _project_grounding(
    unit: ExecutionPlanUnitV2,
    grounding: ExecutionUnitGroundingV1,
    context: GroundingProjectionContextV1,
) -> UnitGroundingViewV1:
    _validate_grounding_binding(unit, grounding)
    evidence_ids = tuple(fact.evidence_id for fact in grounding.allowed_evidence)
    if grounding.allowed_evidence_ids != evidence_ids:
        raise ContextViewInvariantError("unit_grounding_evidence_binding_mismatch")
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ContextViewInvariantError("unit_grounding_duplicate_evidence_id")
    evidence = tuple(
        project_evidence_fact_view_v1(fact, context)
        for fact in grounding.allowed_evidence
    )
    return UnitGroundingViewV1(
        unit_id=grounding.unit_id,
        manifest_ref=grounding.manifest_ref,
        answerability=grounding.answerability,
        scope=project_plan_scope_view_v1(
            grounding.scope,
            context.business_scope_authority,
        ),
        evidence_query=grounding.evidence_query,
        action_intents=tuple(
            project_action_intent_view_v1(intent) for intent in grounding.action_intents
        ),
        allowed_evidence_ids=grounding.allowed_evidence_ids,
        allowed_evidence=evidence,
        business_facts=project_business_facts_view_v1(
            grounding.business_facts,
            context.evaluation_epoch_seconds,
        ),
        guardrail_decisions=tuple(
            project_guardrail_decision_view_v1(decision)
            for decision in grounding.guardrail_decisions
        ),
    )


def _validate_grounding_binding(
    unit: ExecutionPlanUnitV2,
    grounding: ExecutionUnitGroundingV1,
) -> None:
    if grounding.unit_id != unit.unit_id:
        raise ContextViewInvariantError("unit_grounding_unit_mismatch")
    if grounding.manifest_ref != unit.manifest_ref:
        raise ContextViewInvariantError("unit_grounding_manifest_ref_mismatch")
    if grounding.scope != unit.scope:
        raise ContextViewInvariantError("unit_grounding_scope_mismatch")
    if grounding.evidence_query != unit.evidence_query:
        raise ContextViewInvariantError("unit_grounding_query_mismatch")
    if grounding.action_intents != unit.action_intents:
        raise ContextViewInvariantError("unit_grounding_action_intents_mismatch")
    if grounding.answerability != unit.answerability_policy:
        raise ContextViewInvariantError("unit_grounding_answerability_mismatch")
