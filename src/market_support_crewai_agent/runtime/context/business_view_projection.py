from __future__ import annotations

from market_support_crewai_agent.runtime.context.business_view_models import (
    BusinessFactsViewV1,
    GuardrailDecisionViewV1,
    ReportStateViewV1,
    ResolvableStateViewV1,
)
from market_support_crewai_agent.runtime.context.models import (
    RecentExecutedActionSummaryViewV1,
)
from market_support_crewai_agent.runtime.decisions.business_fact_models import (
    CanonicalExecutedActionStateV1,
    CanonicalReportStateV1,
    CanonicalResolvableStateV1,
    UnitBusinessFactsV1,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)


def project_business_facts_view_v1(
    facts: UnitBusinessFactsV1,
    evaluation_epoch_seconds: int | None,
) -> BusinessFactsViewV1:
    return BusinessFactsViewV1(
        material_pack=_project_resolvable_state(facts.material_pack),
        weekly_report=_project_report_state(facts.weekly_report),
        monthly_report=_project_report_state(facts.monthly_report),
        sales_mention=_project_resolvable_state(facts.sales_mention),
        recent_executed_actions=tuple(
            _project_recent_action(action, evaluation_epoch_seconds)
            for action in facts.recent_executed_actions
        ),
        requested_material_pack_option_status=(
            facts.requested_material_pack_option_status
        ),
        user_permission=facts.user_permission,
        evidence_fact_count=facts.evidence_fact_count,
    )


def project_guardrail_decision_view_v1(
    decision: GuardrailDecision,
) -> GuardrailDecisionViewV1:
    return GuardrailDecisionViewV1(
        outcome=decision.outcome,
        phase=decision.phase,
        reason_code=decision.reason_code,
        capability_id=decision.capability_id,
        artifact_ids=tuple(sorted(set(decision.artifact_ids))),
        evidence_required=tuple(sorted(set(decision.evidence_required))),
        evidence_seen=tuple(sorted(set(decision.evidence_seen))),
    )


def _project_resolvable_state(
    state: CanonicalResolvableStateV1,
) -> ResolvableStateViewV1:
    return ResolvableStateViewV1(
        availability=state.availability,
        candidate_labels=state.candidate_labels,
        reason_code=state.reason_code,
        source_available=state.source_record_ref is not None,
        resolve_ref_available=state.resolve_ref is not None,
        material_pack_option=state.material_pack_option,
    )


def _project_report_state(
    state: CanonicalReportStateV1,
) -> ReportStateViewV1:
    resolvable = _project_resolvable_state(state)
    return ReportStateViewV1(
        availability=resolvable.availability,
        candidate_labels=resolvable.candidate_labels,
        reason_code=resolvable.reason_code,
        source_available=resolvable.source_available,
        resolve_ref_available=resolvable.resolve_ref_available,
        material_pack_option=resolvable.material_pack_option,
        period=state.period,
        report_date=(
            state.report_date.isoformat() if state.report_date is not None else None
        ),
        period_start=(
            state.period_start.isoformat() if state.period_start is not None else None
        ),
        period_end=(
            state.period_end.isoformat() if state.period_end is not None else None
        ),
        period_label=state.period_label,
        scope_complete=state.scope_complete,
        expected_product_count=state.expected_product_count,
        generated_product_count=state.generated_product_count,
        missing_product_count=state.missing_product_count,
        report_section_labels=tuple(section.name for section in state.report_sections),
    )


def _project_recent_action(
    action: CanonicalExecutedActionStateV1,
    evaluation_epoch_seconds: int | None,
) -> RecentExecutedActionSummaryViewV1:
    age_seconds = (
        max(0, evaluation_epoch_seconds - action.received_at_epoch_seconds)
        if evaluation_epoch_seconds is not None
        else None
    )
    return RecentExecutedActionSummaryViewV1(
        action_type=action.action_type,
        artifact_type=action.artifact_type,
        material_pack_option=action.material_pack_option,
        period=action.period,
        report_date=(
            action.report_date.isoformat() if action.report_date is not None else None
        ),
        age_seconds=age_seconds,
    )
