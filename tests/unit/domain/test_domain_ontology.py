from __future__ import annotations

from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.ontology import DomainContextBuilder
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.domain.sources.precedence import (
    evidence_facts_for_plan,
)
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.orchestration.decision import DecisionEngine
from market_support_crewai_agent.schemas import ReplyRequest
from tests.helpers.planning import compile_test_plan


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
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["策略A", "策略B"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def material_product_fact(*products: str, strategy: str | None = None) -> EvidenceFact:
    metadata = {
        "status": "resolved",
        "products": [{"product_name": product} for product in products],
    }
    if strategy:
        metadata["strategy"] = strategy
    return EvidenceFact(
        fact_type="material_pack_product_list",
        value=True,
        source_type="adapter_material_pack_content",
        source_id="material_pack",
        resolve_type="material_pack",
        metadata=metadata,
    )


def report_products_fact(resolve_type: str, *products: str) -> EvidenceFact:
    return EvidenceFact(
        fact_type="report_scope_products",
        value=True,
        source_type="adapter_report_scope",
        source_id=resolve_type,
        resolve_type=resolve_type,  # type: ignore[arg-type]
        metadata={
            "status": "resolved",
            "period": "202606",
            "products": [{"product_name": product} for product in products],
            "product_total_count": len(products),
        },
    )


def material_answer_payload():
    return {
        "user_need": "answer material pack product list",
        "artifact_kind": "knowledge_answer",
        "action_intent": "answer",
        "requested_capabilities": ["material_pack"],
        "report_scope": "none",
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal material question",
        },
        "confidence": 0.9,
    }


def monthly_answer_payload():
    return {
        "user_need": "answer monthly report performance",
        "artifact_kind": "knowledge_answer",
        "action_intent": "answer",
        "requested_capabilities": ["monthly_report"],
        "evidence_query": "Product A performance",
        "report_scope": "none",
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal monthly report question",
        },
        "confidence": 0.9,
    }


def compile_plan(request: ReplyRequest, payload: dict, facts=None):
    facts = facts or []
    domain_context = DomainContextBuilder().build(request, facts)
    return compile_test_plan(
        request,
        policy=compile_policy(request),
        domain_context=domain_context,
        **payload,
    ), domain_context


def test_domain_context_tracks_bank_strategy_cardinality_and_explicit_material_artifacts():
    request = make_request(available_strategies=["策略A", "策略B"])

    _, domain_context = compile_plan(request, material_answer_payload())

    assert domain_context.channel.kind == "bank"
    assert len(domain_context.strategies) == 2

    _, resolved_context = compile_plan(
        request,
        material_answer_payload(),
        [material_product_fact("Product B", strategy="策略A")],
    )

    explicit_material_artifacts = [
        artifact
        for artifact in resolved_context.artifacts_by_type("material_pack")
        if artifact.source_type != "adapter_channel_payload"
    ]
    assert len(explicit_material_artifacts) == 1


def test_non_bank_single_strategy_is_represented_in_domain_context():
    request = make_request(
        channel_type="non_bank",
        available_strategies=["策略A"],
    )

    _, domain_context = compile_plan(request, material_answer_payload())

    assert domain_context.channel.kind == "non_bank"
    assert [strategy.name for strategy in domain_context.strategies] == ["策略A"]


def test_weekly_report_product_evidence_cannot_answer_material_pack_product_question():
    request = make_request(available_strategies=["策略A"])
    weekly_fact = report_products_fact("weekly_report", "Product A")
    plan, _ = compile_plan(request, material_answer_payload())

    directive = DecisionEngine().decide(
        plan,
        derive_business_facts([weekly_fact], request),
        [weekly_fact],
        request,
        compile_policy(request),
    )

    assert evidence_facts_for_plan(plan, [weekly_fact]) == []
    assert directive.mode == "unable"
    assert directive.reply_kind == "unable_to_answer"


def test_material_pack_product_answer_uses_material_pack_only_not_weekly_report():
    request = make_request(available_strategies=["策略A"])
    material_fact = material_product_fact("Product B", strategy="策略A")
    weekly_fact = report_products_fact("weekly_report", "Product A")
    plan, _ = compile_plan(request, material_answer_payload(), [material_fact, weekly_fact])

    directive = DecisionEngine().decide(
        plan,
        derive_business_facts([material_fact, weekly_fact], request),
        [material_fact, weekly_fact],
        request,
        compile_policy(request),
    )

    selected = evidence_facts_for_plan(plan, [material_fact, weekly_fact])
    assert selected == [material_fact]
    assert directive.mode == "knowledge_answer"
    assert "Product B" in directive.text
    assert "Product A" not in directive.text


def test_monthly_report_performance_uses_monthly_report_not_material_pack_evidence():
    request = make_request(
        message="月报里Product A表现怎么样",
        available_strategies=["策略A"],
    )
    material_fact = material_product_fact("Product B", strategy="策略A")
    monthly_fact = report_products_fact("monthly_report", "Product A")
    plan, _ = compile_plan(request, monthly_answer_payload(), [material_fact, monthly_fact])

    selected = evidence_facts_for_plan(plan, [material_fact, monthly_fact])
    assert selected == [monthly_fact]

    monthly_directive = DecisionEngine().decide(
        plan,
        derive_business_facts([material_fact, monthly_fact], request),
        [material_fact, monthly_fact],
        request,
        compile_policy(request),
    )
    material_only_directive = DecisionEngine().decide(
        plan,
        derive_business_facts([material_fact], request),
        [material_fact],
        request,
        compile_policy(request),
    )

    assert monthly_directive.mode == "knowledge_answer"
    assert monthly_directive.requires_knowledge_composer is True
    assert material_only_directive.mode == "unable"


def test_unknown_channel_kind_preserves_unknown_and_does_not_assume_strategy_cardinality():
    request = make_request(
        channel_type="unknown",
        available_strategies=["策略A", "策略B"],
    )

    plan, domain_context = compile_plan(request, material_answer_payload())

    assert domain_context.channel.kind == "unknown"
    assert len(domain_context.strategies) == 2
    assert plan.response_mode == "knowledge_answer"
    assert plan.ambiguity_slots == []
