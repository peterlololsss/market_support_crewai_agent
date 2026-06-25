from __future__ import annotations

from market_support_crewai_agent.runtime.domain.ontology import DomainContextBuilder
from market_support_crewai_agent.runtime.domain.planning import (
    AdapterResolveSpec,
    ComplianceDecision,
    ExecutionPlan,
    plan_spec_for_execution_plan,
)
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityGate,
)
from market_support_crewai_agent.schemas import ReplyRequest


def make_request(**overrides) -> ReplyRequest:
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": "材料包里有哪些产品",
        "is_group": True,
        "group_name": "test group",
        "dist_channel_name": "test channel",
        "sender_nickname": "test user",
        "available_artifacts": [
            {"type": "material_pack", "options": ["指增"]},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def knowledge_plan(
    request: ReplyRequest,
    capability: str,
    *,
    material_pack_option: str | None = "指增",
    evidence_query: str | None = None,
) -> ExecutionPlan:
    resolve_type = capability if capability in {"weekly_report", "monthly_report"} else None
    plan = ExecutionPlan(
        user_need=request.message,
        artifact_kind="knowledge_answer",
        response_mode="knowledge_answer",
        compliance=ComplianceDecision(
            is_compliant=True,
            reason_code="compliant_product_request",
            reason="normal support request",
        ),
        evidence_query=evidence_query,
        capabilities=[capability],  # type: ignore[list-item]
        answer_capabilities=[capability],  # type: ignore[list-item]
        adapter_resolves=(
            [AdapterResolveSpec(resolve_type=resolve_type)]  # type: ignore[list-item]
            if resolve_type
            else []
        ),
        material_pack_option=material_pack_option,
    )
    return plan.model_copy(
        update={
            "plan_spec": plan_spec_for_execution_plan(
                plan,
                domain_context=DomainContextBuilder().build(request),
            )
        }
    )


def weekly_report_fact() -> EvidenceFact:
    return EvidenceFact(
        fact_type="weekly_report_resolvable",
        value=True,
        source_type="adapter_resolve",
        source_id="weekly_report",
        resolve_type="weekly_report",
        metadata={
            "status": "resolved",
            "period": "20260612",
            "report_date": "2026-06-12",
            "resolve_ref": "weekly:ref",
        },
        artifact_type="weekly_report",
    )


def assess(
    request: ReplyRequest,
    plan: ExecutionPlan,
    facts: list[EvidenceFact],
):
    return AnswerabilityGate().assess(
        request=request,
        domain_context=DomainContextBuilder().build(
            request,
            available_artifacts=facts,
        ),
        plan=plan,
        policy=compile_policy(request, doc_mcp_enabled=True),
        evidence_facts=facts,
    )


def test_weekly_report_performance_question_answers_when_weekly_report_exists():
    request = make_request(message="这个周报表现怎么样")
    plan = knowledge_plan(
        request,
        "weekly_report",
        material_pack_option=None,
        evidence_query="performance",
    )
    assessment = assess(request, plan, [weekly_report_fact()])

    assert assessment.can_answer is True
    assert assessment.recommended_response_mode == "answer"
    assert assessment.allowed_evidence_ids == [
        "adapter_resolve:weekly_report:weekly_report_resolvable"
    ]


def test_missing_weekly_report_evidence_uses_plain_abstention_text():
    request = make_request(message="这个周报表现怎么样")
    plan = knowledge_plan(
        request,
        "weekly_report",
        material_pack_option=None,
        evidence_query="performance",
    )
    assessment = assess(request, plan, [])

    assert assessment.recommended_response_mode == "abstain"
    assert "安全" not in assessment.user_facing_reason
    assert assessment.user_facing_reason == "老师，这个问题需要以周报里的准确信息为准，我这边暂时无法确认。"
