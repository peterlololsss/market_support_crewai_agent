from __future__ import annotations

from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.capabilities.registry import (
    CapabilityManifest,
    CapabilityRegistry,
    EvidenceContract,
)
from market_support_crewai_agent.runtime.domain.ontology import (
    ArtifactScope,
    DomainContextBuilder,
)
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.domain.sources.precedence import (
    evidence_facts_for_plan,
)
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.orchestration.answerability_directives import (
    directive_from_answerability,
)
from market_support_crewai_agent.runtime.orchestration.decision import DecisionEngine
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityAssessment,
    AnswerabilityGate,
)
from market_support_crewai_agent.runtime.validation.capability_manifest_verifier import (
    verify_capability_contracts,
)
from market_support_crewai_agent.runtime.validation.evidence_source_guard import (
    retrieval_source_guard,
)
from market_support_crewai_agent.runtime.validation.plan_spec_verifier import (
    verify_plan_spec,
)
from market_support_crewai_agent.schemas import ReplyRequest
from tests.helpers.planning import compile_test_plan, make_plan_spec


def make_request(**overrides) -> ReplyRequest:
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": "材料包里有哪些产品",
        "is_group": True,
        "group_name": "测试群",
        "dist_channel_name": "测试渠道",
        "sender_nickname": "测试用户",
        "available_artifacts": [
            {"type": "material_pack", "options": ["策略S1"]},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def report_plan(request: ReplyRequest, capability: str):
    return compile_test_plan(
        request,
        user_need=f"answer {capability} performance",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=[capability],
        evidence_query="performance",
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal report question",
        },
    )


def unrelated_document_fact() -> EvidenceFact:
    return EvidenceFact(
        fact_type="document_context",
        value=True,
        source_type="document_mcp",
        source_id="doc",
        artifact_type="document_context",
    )


def report_products(resolve_type: str, *products: str) -> EvidenceFact:
    return EvidenceFact(
        fact_type="report_scope_products",
        value=True,
        source_type="adapter_report_scope",
        source_id=resolve_type,
        resolve_type=resolve_type,  # type: ignore[arg-type]
        artifact_type=resolve_type,  # type: ignore[arg-type]
        metadata={
            "products": [{"product_name": product} for product in products],
            "period": "202606",
        },
    )


def assess(request: ReplyRequest, plan, facts: list[EvidenceFact]):
    domain_context = DomainContextBuilder().build(request, available_artifacts=facts)
    return AnswerabilityGate().assess(
        request=request,
        domain_context=domain_context,
        plan=plan,
        policy=compile_policy(request, doc_mcp_enabled=True),
        evidence_facts=facts,
    )


def test_mixed_intent_partial_evidence_does_not_force_global_abstention():
    request = make_request(message="????\n??????")
    plan = compile_test_plan(
        request,
        doc_mcp_enabled=True,
        plan_units=[
            {
                "selected_capability_id": "channel.strategy_summary",
                "answerability_policy": "answer",
                "evidence_query": "????",
            },
            {
                "selected_capability_id": "general.abstention",
                "answerability_policy": "abstain",
            },
        ],
    )
    assessment = AnswerabilityAssessment(
        can_answer=False,
        capability_id="channel.strategy_summary",
        recommended_response_mode="abstain",
        allowed_evidence_ids=["document_mcp:doc-1:document_context"],
        user_facing_reason="partial evidence exists",
    )

    directive = directive_from_answerability(assessment, plan)

    assert directive is None


def test_unavailable_send_does_not_discard_supported_answer_part():
    request = make_request(message="intro strategy and send weekly")
    plan = compile_test_plan(
        request,
        doc_mcp_enabled=True,
        plan_units=[
            {
                "selected_capability_id": "channel.strategy_summary",
                "answerability_policy": "answer",
                "evidence_query": "strategy intro",
            },
            {
                "selected_capability_id": "weekly_report.send",
                "answerability_policy": "send",
            },
        ],
    )
    facts = [
        EvidenceFact(
            fact_type="document_context",
            value=True,
            source_type="document_mcp",
            source_id="doc-1",
            artifact_type="document_context",
        )
    ]
    domain_context = DomainContextBuilder().build(request, available_artifacts=facts)

    directive = DecisionEngine().decide(
        plan,
        derive_business_facts(facts, request),
        facts,
        request,
        compile_policy(request, doc_mcp_enabled=True),
        domain_context,
    )

    assert directive.mode == "knowledge_answer"
    assert directive.requires_knowledge_composer is True
    assert directive.action_intents == []


def test_non_bank_material_pack_option_is_not_inferred_into_domain_scope():
    request = make_request(
        channel_type="non_bank",
        available_artifacts=[{"type": "material_pack", "options": ["策略S1"]}, {"type": "weekly_report"}, {"type": "monthly_report"}],
    )
    domain_context = DomainContextBuilder().build(request)

    assert domain_context.channel.kind == "non_bank"
    assert domain_context.strategies == ()
    assert domain_context.metadata["material_pack_options"] == ("策略S1",)


def test_regression_original_bug_F_monthly_performance_uses_monthly_report_not_material_pack():
    request = make_request(message="月报里产品A表现怎么样")
    plan = report_plan(request, "monthly_report")
    facts = [
        material_products("产品B", material_pack_option="策略S1"),
        report_products("monthly_report", "产品A"),
    ]

    selected = evidence_facts_for_plan(plan, facts)
    assessment = assess(request, plan, facts)

    assert selected == [facts[1]]
    assert assessment.can_answer is True


def test_regression_original_bug_G_weekly_performance_uses_weekly_report_not_material_pack():
    request = make_request(message="周报里产品A表现怎么样")
    plan = report_plan(request, "weekly_report")
    facts = [
        material_products("产品B", material_pack_option="策略S1"),
        report_products("weekly_report", "产品A"),
    ]

    selected = evidence_facts_for_plan(plan, facts)
    assessment = assess(request, plan, facts)

    assert selected == [facts[1]]
    assert assessment.can_answer is True


def test_regression_original_bug_J_dummy_capability_uses_manifest_verifier_not_bespoke_code():
    registry = CapabilityRegistry([dummy_manifest()])
    result = verify_capability_contracts(
        ["dummy.echo"],
        registry=registry,
        output_payload={"reply": {"kind": "answer", "text": "dummy"}},
        runtime_inputs={"request": {"dist_channel_name": "测试渠道"}},
        evidence_facts=[
            EvidenceFact(
                fact_type="document_context",
                value=True,
                source_type="adapter_context",
                source_id="dummy",
                artifact_type="unknown",
                metadata={"artifact_type": "dummy_artifact"},
            )
        ],
    )

    assert result.valid is True


def dummy_manifest() -> CapabilityManifest:
    return CapabilityManifest.model_validate(
        {
            "id": "dummy.echo",
            "version": "test.1",
            "display_name": "Dummy echo",
            "description": "Dummy manifest proves verifier primitives are generic.",
            "capability_type": "answer",
            "domain_entities": ["channel", "artifact"],
            "required_inputs": ["request.dist_channel_name"],
            "optional_inputs": [],
            "required_artifacts": ["dummy_artifact"],
            "allowed_artifacts": ["dummy_artifact"],
            "forbidden_artifacts": [],
            "required_tools": ["dummy.echo"],
            "output_schema": {
                "type": "object",
                "required": ["reply"],
                "properties": {
                    "reply": {
                        "type": "object",
                        "required": ["kind", "text"],
                        "properties": {
                            "kind": {"type": "string"},
                            "text": {"type": "string"},
                        },
                    }
                },
            },
            "evidence_contract": EvidenceContract(
                required_fact_types=["document_context"],
                allowed_source_types=["adapter_context"],
                allowed_artifact_types=["dummy_artifact"],
                min_facts=1,
            ),
            "abstention_policy": {
                "requires_abstention_when_evidence_missing": True,
                "abstention_reply_kinds": ["unable_to_answer"],
            },
            "planner_guidance": "Use only when explicitly selected.",
            "agent_guidance": "Answer only from dummy evidence.",
            "verifier_checks": [
                "output_schema",
                "required_runtime_input_present",
                "required_evidence_present",
                "evidence_artifact_type_allowed",
                "forbidden_source_not_used",
                "abstention_correctness",
            ],
            "examples_positive": ["dummy"],
            "examples_negative": ["weekly report"],
        }
    )
