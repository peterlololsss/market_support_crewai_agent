from __future__ import annotations

from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.capabilities.registry import (
    CapabilityManifest,
    CapabilityRegistry,
    EvidenceContract,
)
from market_support_crewai_agent.runtime.domain.canonicalization import (
    canonicalize_request,
)
from market_support_crewai_agent.runtime.domain.entity_resolution import (
    CanonicalEntityResolver,
)
from market_support_crewai_agent.runtime.domain.ontology import (
    ArtifactScope,
    DistributionChannel,
    DomainContext,
    DomainContextBuilder,
    Product,
    Strategy,
)
from market_support_crewai_agent.runtime.domain.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.domain.sources.precedence import (
    evidence_facts_for_plan,
)
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.orchestration.answerability_directives import (
    directive_from_answerability,
)
from market_support_crewai_agent.runtime.orchestration.decision import DecisionEngine
from market_support_crewai_agent.runtime.validation.answerability import AnswerabilityGate
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
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["策略S1"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def material_plan(request: ReplyRequest, selected_strategy: str | None = "策略S1"):
    plan = compile_test_plan(
        request,
        user_need="answer material pack product list",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        report_scope="none",
        requested_capabilities=["material_pack"],
        selected_strategy=selected_strategy,
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal material question",
        },
    )
    return plan.model_copy(update={"selected_strategy": selected_strategy})


def report_plan(request: ReplyRequest, capability: str):
    return compile_test_plan(
        request,
        user_need=f"answer {capability} performance",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        report_scope="none",
        requested_capabilities=[capability],
        evidence_query="performance",
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal report question",
        },
    )


def material_products(
    *products: str,
    strategy: str | None = "策略S1",
    source_id: str | None = None,
    source_type: str = "adapter_material_pack_content",
    artifact_type: str = "material_pack",
    channel_id: str = "unknown",
) -> EvidenceFact:
    metadata = {
        "products": [{"product_name": product} for product in products],
    }
    if strategy:
        metadata["strategy"] = strategy
    return EvidenceFact(
        fact_type="material_pack_product_list",
        value=True,
        source_type=source_type,  # type: ignore[arg-type]
        source_id=source_id or f"material_pack:{strategy or 'default'}",
        resolve_type="material_pack",
        artifact_type=artifact_type,  # type: ignore[arg-type]
        metadata=metadata,
        scope=ArtifactScope(channel_id=channel_id),
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
        canonical_context=canonicalize_request(request, domain_context=domain_context),
        domain_context=domain_context,
        plan=plan,
        policy=compile_policy(request, doc_mcp_enabled=True),
        evidence_facts=facts,
    )


def test_regression_original_bug_A_material_pack_question_abstains_without_material_pack_artifact():
    request = make_request()
    plan = material_plan(request)
    facts = [report_products("weekly_report", "产品A")]

    assessment = assess(request, plan, facts)
    directive = directive_from_answerability(assessment, plan)

    assert assessment.can_answer is False
    assert assessment.missing_artifacts == ["material_pack"]
    assert directive is not None
    assert directive.reason_code == "answerability_missing_evidence"
    assert "材料包" in directive.text
    assert "产品A" not in directive.text


def test_regression_original_bug_B_material_pack_answer_uses_material_pack_products_only():
    request = make_request()
    plan = material_plan(request)
    facts = [
        material_products("产品B", strategy="策略S1"),
        report_products("weekly_report", "产品A"),
    ]
    domain_context = DomainContextBuilder().build(request, available_artifacts=facts)

    directive = DecisionEngine().decide(
        plan,
        derive_business_facts(facts, request),
        facts,
        request,
        compile_policy(request),
        domain_context,
    )

    assert directive.text == "材料包包含：产品B。"
    assert "产品A" not in directive.text


def test_regression_original_bug_C_bank_material_pack_two_strategies_clarifies_strategy():
    request = make_request(
        available_strategies=["策略S1", "策略S2"],
        channel_type="bank",
    )
    plan = material_plan(request, selected_strategy=None)
    facts = [
        material_products("产品B", strategy="策略S1"),
        material_products("产品A", strategy="策略S2"),
    ]

    assessment = assess(request, plan, facts)
    directive = directive_from_answerability(assessment, plan)

    assert assessment.recommended_response_mode == "clarify"
    assert assessment.ambiguity == "missing_strategy"
    assert directive is not None
    assert directive.reply_kind == "clarification"
    assert "策略S1" in directive.text
    assert "策略S2" in directive.text


def test_regression_original_bug_D_bank_selected_strategy_uses_only_that_material_pack():
    request = make_request(
        available_strategies=["策略S1", "策略S2"],
        channel_type="bank",
    )
    plan = material_plan(request, selected_strategy="策略S1")
    facts = [
        material_products("产品B", strategy="策略S1"),
        material_products("产品A", strategy="策略S2"),
    ]
    domain_context = DomainContextBuilder().build(request, available_artifacts=facts)

    selected = evidence_facts_for_plan(plan, facts, domain_context)
    directive = DecisionEngine().decide(
        plan,
        derive_business_facts(facts, request),
        facts,
        request,
        compile_policy(request),
        domain_context,
    )

    assert selected == [facts[0]]
    assert directive.text == "材料包包含：产品B。"
    assert "产品A" not in directive.text


def test_regression_original_bug_E_non_bank_one_strategy_is_inferred_into_domain_scope():
    request = make_request(
        channel_type="non_bank",
        available_strategies=["策略S1"],
    )
    domain_context = DomainContextBuilder().build(request)
    canonical_context = canonicalize_request(request, domain_context=domain_context)

    assert domain_context.channel.kind == "non_bank"
    assert [strategy.name for strategy in domain_context.strategies] == ["策略S1"]
    assert canonical_context.strategy_status == "resolved"
    assert canonical_context.selected_strategy == "策略S1"


def test_regression_original_bug_F_monthly_performance_uses_monthly_report_not_material_pack():
    request = make_request(message="月报里产品A表现怎么样")
    plan = report_plan(request, "monthly_report")
    facts = [
        material_products("产品B", strategy="策略S1"),
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
        material_products("产品B", strategy="策略S1"),
        report_products("weekly_report", "产品A"),
    ]

    selected = evidence_facts_for_plan(plan, facts)
    assessment = assess(request, plan, facts)

    assert selected == [facts[1]]
    assert assessment.can_answer is True


def test_regression_original_bug_H_ambiguous_product_alias_across_strategies_clarifies_not_nearest_match():
    domain_context = product_domain_context(
        products=[
            ("product:s1:a", "产品A", ("strategy:s1",)),
            ("product:s2:a", "产品A", ("strategy:s2",)),
        ]
    )

    result = CanonicalEntityResolver().resolve_request(
        make_request(
            message="产品A表现怎么样",
            available_strategies=["策略S1", "策略S2"],
        ),
        domain_context=domain_context,
    )
    product_resolution = result.by_type("product")[0]

    assert product_resolution.status == "ambiguous"
    assert {candidate.entity_id for candidate in product_resolution.candidates} == {
        "product:s1:a",
        "product:s2:a",
    }


def test_regression_original_bug_I_unknown_product_mention_does_not_nearest_match():
    domain_context = product_domain_context(
        products=[
            ("product:s1:a", "产品A", ("strategy:s1",)),
            ("product:s1:b", "产品B", ("strategy:s1",)),
        ]
    )

    result = CanonicalEntityResolver().resolve_request(
        make_request(message="产品C表现怎么样", available_strategies=["策略S1"]),
        domain_context=domain_context,
    )

    assert result.by_type("product") == ()


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


def test_regression_original_bug_K_send_scope_blocks_wrong_channel_evidence():
    request = make_request()
    plan = material_plan(request)
    domain_context = DomainContextBuilder().build(request)
    fact = material_products("产品B", channel_id="channel:other")

    decision = retrieval_source_guard(
        plan=plan,
        policy=compile_policy(request),
        evidence_facts=[fact],
        domain_context=domain_context,
    )

    assert decision.outcome == "abstain"
    assert decision.reason_code == "channel_scope_mismatch"


def test_regression_original_bug_L_history_only_evidence_rejected_when_allow_history_false():
    spec = material_history_plan_spec()
    history_fact = material_products(
        "历史产品",
        source_type="conversation_history",
        artifact_type="history",
        source_id="history-material",
    )

    result = verify_plan_spec(
        spec,
        output_payload={
            "reply": {"kind": "answer", "text": "材料包包含：历史产品。"},
            "actions": [],
        },
        evidence_facts=[history_fact],
        cited_evidence_ids=["history-material"],
    )

    assert result.valid is False
    assert "history_evidence_not_allowed" in {issue.code for issue in result.issues}


def product_domain_context(
    *,
    products: list[tuple[str, str, tuple[str, ...]]],
) -> DomainContext:
    channel = DistributionChannel(
        id="channel:current",
        name="测试渠道",
        kind="bank",
        provenance="test",
    )
    strategies = (
        Strategy(
            id="strategy:s1",
            name="策略S1",
            channel_id=channel.id,
            provenance="test",
        ),
        Strategy(
            id="strategy:s2",
            name="策略S2",
            channel_id=channel.id,
            provenance="test",
        ),
    )
    return DomainContext(
        channel=channel,
        strategies=strategies,
        products=tuple(
            Product(
                id=product_id,
                name=name,
                channel_id=channel.id,
                strategy_ids=strategy_ids,
                provenance="test",
            )
            for product_id, name, strategy_ids in products
        ),
    )


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


def material_history_plan_spec() -> PlanSpec:
    request = make_request()
    spec = make_plan_spec(
        request,
        selected_capability_id="material_pack.product_list",
        selected_strategy="策略S1",
        answerability_policy="answer",
        user_intent_summary="answer material product list from current material pack",
    )
    return spec.model_copy(
        update={
            "evidence_contract": EvidenceContract(
                required_fact_types=["material_pack_product_list"],
                allowed_source_types=["conversation_history"],
                allowed_artifact_types=["material_pack"],
                min_facts=1,
                allow_history=False,
            ),
            "evidence_contract_ref": None,
        }
    )
