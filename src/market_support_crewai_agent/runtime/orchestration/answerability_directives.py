from __future__ import annotations

from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityAssessment,
)


def directive_from_answerability(
    assessment: AnswerabilityAssessment,
    plan: ExecutionPlan,
) -> ResponseDirective | None:
    if plan.response_mode != "knowledge_answer":
        return None
    if assessment.recommended_response_mode == "answer":
        return None
    if assessment.recommended_response_mode == "clarify":
        return ResponseDirective(
            mode="clarification",
            reply_kind="clarification",
            text=assessment.user_facing_reason
            or "我需要再确认一下具体需求后再处理。",
            requires_knowledge_composer=True,
            composer_stage="knowledge_composer",
            reason_code=f"answerability_{assessment.ambiguity}",
        )
    return ResponseDirective(
        mode="unable",
        reply_kind="unable_to_answer",
        text=assessment.user_facing_reason
        or "当前上下文缺少该问题所需证据，我先不展开。",
        reason_code="answerability_missing_evidence",
    )


def default_answerability_for_plan(plan: ExecutionPlan) -> AnswerabilityAssessment:
    return AnswerabilityAssessment(
        can_answer=True,
        capability_id=_answer_capability_id(plan),
        recommended_response_mode="answer",
    )


def _answer_capability_id(plan: ExecutionPlan) -> str:
    if plan.plan_spec is None:
        return "unknown"
    for unit in plan.plan_spec.plan_units:
        if unit.answerability_policy == "answer":
            return unit.selected_capability_id
    return plan.plan_spec.plan_units[0].selected_capability_id
