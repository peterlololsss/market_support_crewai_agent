from __future__ import annotations

import asyncio
from types import SimpleNamespace

from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.capabilities.registry import (
    EvidenceContract,
)
from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.domain.ontology import (
    ArtifactScope,
    DomainContextBuilder,
    TimeRange,
)
from market_support_crewai_agent.runtime.domain.planning import (
    AdapterResolveSpec,
    ComplianceDecision,
    ExecutionPlan,
    plan_spec_for_execution_plan,
)
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.domain.sources.metadata import SourceMetadata
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.orchestration.decision import DecisionEngine
from market_support_crewai_agent.runtime.orchestration.reply_agent import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.conversation_store import ConversationStore
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityGate,
)
from market_support_crewai_agent.runtime.validation.evidence_source_guard import (
    retrieval_source_guard,
)
from market_support_crewai_agent.schemas import ReplyRequest
from market_support_crewai_agent.settings import Settings


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
        "material_pack_options": ["指增"],
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


def material_product_fact(
    *products: str,
    material_pack_option: str | None = "指增",
    source_type: str = "adapter_material_pack_content",
    artifact_type: str = "material_pack",
) -> EvidenceFact:
    metadata = {
        "status": "resolved",
        "products": [{"product_name": product} for product in products],
    }
    if material_pack_option:
        metadata["material_pack_option"] = material_pack_option
    return EvidenceFact(
        fact_type="material_pack_product_list",
        value=True,
        source_type=source_type,  # type: ignore[arg-type]
        source_id="material_pack",
        resolve_type="material_pack",
        metadata=metadata,
        artifact_type=artifact_type,  # type: ignore[arg-type]
        scope=ArtifactScope(channel_id="unknown"),
    )


def weekly_products_fact() -> EvidenceFact:
    return EvidenceFact(
        fact_type="report_scope_products",
        value=True,
        source_type="adapter_report_scope",
        source_id="weekly_report",
        resolve_type="weekly_report",
        metadata={"products": [{"product_name": "Weekly Product"}]},
        artifact_type="weekly_report",
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
        canonical_context=canonicalize_request(request),
        domain_context=DomainContextBuilder().build(
            request,
            available_artifacts=facts,
        ),
        plan=plan,
        policy=compile_policy(request, doc_mcp_enabled=True),
        evidence_facts=facts,
    )


def test_material_pack_product_list_abstains_when_only_weekly_report_has_products():
    request = make_request()
    plan = knowledge_plan(request, "material_pack")
    assessment = assess(request, plan, [weekly_products_fact()])

    assert assessment.can_answer is False
    assert assessment.recommended_response_mode == "abstain"
    assert assessment.missing_artifacts == ["material_pack"]
    assert assessment.allowed_evidence_ids == []
    assert assessment.disallowed_evidence_ids[0].reason == "forbidden_source_type"
    assert "不能用周报" in assessment.user_facing_reason


def test_material_pack_product_list_answers_from_material_pack_artifact():
    request = make_request()
    plan = knowledge_plan(request, "material_pack")
    facts = [material_product_fact("Product A", "Product B")]
    assessment = assess(request, plan, facts)

    assert assessment.can_answer is True
    assert assessment.recommended_response_mode == "answer"
    assert assessment.allowed_evidence_ids == [
        "adapter_material_pack_content:material_pack:material_pack_product_list"
    ]

    directive = DecisionEngine().decide(
        plan,
        derive_business_facts(facts, request),
        facts,
        request,
        compile_policy(request),
        DomainContextBuilder().build(request, available_artifacts=facts),
    )
    assert directive.text == "材料包包含：Product A、Product B。"


def test_bank_material_pack_options_do_not_force_local_strategy_clarification():
    request = make_request(
        material_pack_options=["中证1000指增", "中证A500指增"],
        channel_type="bank",
    )
    plan = knowledge_plan(request, "material_pack", material_pack_option=None)
    facts = [material_product_fact("Product A", material_pack_option=None)]
    assessment = assess(request, plan, facts)

    assert assessment.can_answer is True
    assert assessment.recommended_response_mode == "answer"
    assert assessment.ambiguity == "none"


def test_non_bank_single_strategy_material_pack_can_answer():
    request = make_request(
        material_pack_options=["指增"],
        channel_type="non_bank",
    )
    plan = knowledge_plan(request, "material_pack", material_pack_option=None)
    assessment = assess(request, plan, [material_product_fact("Product A")])

    assert assessment.can_answer is True
    assert assessment.recommended_response_mode == "answer"


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
    assert assessment.user_facing_reason == "当前上下文没有可用于回答该问题的周报证据，我先不展开。"


def test_old_history_material_pack_does_not_satisfy_current_material_question():
    request = make_request()
    plan = knowledge_plan(request, "material_pack")
    facts = [
        material_product_fact(
            "Old Product",
            source_type="action_ledger",
            artifact_type="history",
        )
    ]
    assessment = assess(request, plan, facts)

    assert assessment.can_answer is False
    assert assessment.recommended_response_mode == "abstain"
    assert assessment.allowed_evidence_ids == []
    assert assessment.disallowed_evidence_ids[0].reason == (
        "history_source_not_current_artifact"
    )


def test_history_product_list_without_current_material_pack_abstains():
    request = make_request()
    plan = knowledge_plan(request, "material_pack")
    facts = [
        material_product_fact(
            "Old Product",
            source_type="conversation_history",
            artifact_type="material_pack",
        )
    ]

    assessment = assess(request, plan, facts)

    assert assessment.can_answer is False
    assert assessment.recommended_response_mode == "abstain"
    assert assessment.allowed_evidence_ids == []
    assert assessment.disallowed_evidence_ids[0].reason == (
        "history_source_not_current_artifact"
    )


def test_weekly_report_history_is_not_material_pack_evidence():
    request = make_request()
    plan = knowledge_plan(request, "material_pack")
    facts = [
        EvidenceFact(
            fact_type="report_scope_products",
            value=True,
            source_type="conversation_history",
            source_id="weekly_report",
            resolve_type="weekly_report",
            metadata={"products": [{"product_name": "Weekly Product"}]},
            artifact_type="weekly_report",
        )
    ]

    assessment = assess(request, plan, facts)

    assert assessment.can_answer is False
    assert assessment.recommended_response_mode == "abstain"
    assert assessment.missing_artifacts == ["material_pack"]
    assert assessment.allowed_evidence_ids == []


def test_previous_material_pack_history_for_different_channel_is_rejected():
    request = make_request()
    domain_context = DomainContextBuilder().build(request)
    plan = _allow_history_material_plan(request)
    fact = material_product_fact(
        "Old Product",
        source_type="conversation_history",
        artifact_type="material_pack",
    )
    fact = fact.__class__(
        **{
            **fact.__dict__,
            "scope": ArtifactScope(
                channel_id="adapter_channel:bank:other channel",
                provenance="conversation_store",
            ),
            "source_metadata": SourceMetadata(
                source_id="history-material-pack",
                source_type="assistant_message",
                artifact_type="material_pack",
                channel_id="adapter_channel:bank:other channel",
                provenance="conversation_store",
                evidence_allowed_by_default=False,
            ),
        }
    )

    decision = retrieval_source_guard(
        plan=plan,
        policy=compile_policy(request),
        evidence_facts=[fact],
        domain_context=domain_context,
    )

    assert decision.outcome == "abstain"
    assert decision.reason_code == "channel_scope_mismatch"


def test_allow_history_uses_history_only_when_scope_matches():
    request = make_request()
    domain_context = DomainContextBuilder().build(request)
    plan = _allow_history_material_plan(
        request,
        channel_id=domain_context.channel.id,
        material_pack_option="指增",
        time_range=TimeRange(period="202606"),
    )
    fact = material_product_fact(
        "Historical Product",
        source_type="conversation_history",
        artifact_type="material_pack",
    )
    matching = fact.__class__(
        **{
            **fact.__dict__,
            "metadata": {
                **fact.metadata,
                "material_pack_option": "指增",
            },
            "scope": ArtifactScope(
                channel_id=domain_context.channel.id,
                time_range=TimeRange(period="202606"),
                provenance="conversation_store",
            ),
            "source_metadata": SourceMetadata(
                source_id="history-material-pack",
                source_type="assistant_message",
                artifact_type="material_pack",
                channel_id=domain_context.channel.id,
                time_range=TimeRange(period="202606"),
                provenance="conversation_store",
                evidence_allowed_by_default=False,
            ),
        }
    )
    mismatched_time = matching.__class__(
        **{
            **matching.__dict__,
            "scope": ArtifactScope(
                channel_id=domain_context.channel.id,
                time_range=TimeRange(period="202605"),
                provenance="conversation_store",
            ),
            "source_metadata": SourceMetadata(
                source_id="history-material-pack-old",
                source_type="assistant_message",
                artifact_type="material_pack",
                channel_id=domain_context.channel.id,
                time_range=TimeRange(period="202605"),
                provenance="conversation_store",
                evidence_allowed_by_default=False,
            ),
        }
    )

    allowed = retrieval_source_guard(
        plan=plan,
        policy=compile_policy(request),
        evidence_facts=[matching],
        domain_context=domain_context,
    )
    rejected = retrieval_source_guard(
        plan=plan,
        policy=compile_policy(request),
        evidence_facts=[mismatched_time],
        domain_context=domain_context,
    )

    assert allowed.outcome == "allow"
    assert allowed.evidence_seen == [
        "conversation_history:material_pack:material_pack_product_list"
    ]
    assert rejected.outcome == "abstain"
    assert rejected.reason_code == "time_range_scope_mismatch"


def test_runtime_abstains_before_composer_when_material_pack_content_missing():
    request = make_request()
    plan = knowledge_plan(request, "material_pack")
    composer_called = False

    class FakePlanner:
        async def kickoff_async(self, prompt, response_format):
            del prompt, response_format
            return SimpleNamespace(pydantic=plan.plan_spec, raw="")

    class FakeComposer:
        async def kickoff_async(self, prompt, response_format):
            del prompt, response_format
            nonlocal composer_called
            composer_called = True
            raise AssertionError("composer should not be called on abstain")

    class FakeEvidenceExecutor:
        async def execute(self, request, canonical_context, plan, policy, action_history=None):
            del canonical_context, plan, policy, action_history
            facts = [weekly_products_fact()]
            return SimpleNamespace(
                preflight=AdapterPreflightSnapshot.empty(),
                evidence_facts=facts,
                business_facts=derive_business_facts(facts, request),
                domain_context=DomainContextBuilder().build(
                    request,
                    available_artifacts=facts,
                ),
                guardrail_decisions=[],
            )

    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False),
        conversation_store=ConversationStore(),
        evidence_executor=FakeEvidenceExecutor(),
    )
    runtime._build_planner_agent = lambda: FakePlanner()  # type: ignore[method-assign]
    runtime._build_agent = lambda *_args, **_kwargs: FakeComposer()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(request))

    assert composer_called is False
    assert response.reply.kind == "unable_to_answer"
    assert "不能用周报" in response.reply.text
    assert response.actions == []


def _allow_history_material_plan(
    request: ReplyRequest,
    *,
    channel_id: str | None = None,
    material_pack_option: str | None = None,
    time_range: TimeRange | None = None,
) -> ExecutionPlan:
    plan = knowledge_plan(request, "material_pack")
    plan_spec = plan_spec_for_execution_plan(
        plan,
        domain_context=DomainContextBuilder().build(request),
    )
    scope = plan_spec.domain_scope.model_copy(
        update={
            "channel_id": channel_id or plan_spec.domain_scope.channel_id,
            "material_pack_option": material_pack_option,
            "time_range": time_range,
        }
    )
    return plan.model_copy(
        update={
            "plan_spec": plan_spec.model_copy(
                update={
                    "domain_scope": scope,
                    "evidence_contract": EvidenceContract(
                        required_fact_types=["material_pack_product_list"],
                        allowed_source_types=["conversation_history"],
                        required_artifact_types=["material_pack"],
                        allowed_artifact_types=["material_pack"],
                        required_scope_match=[
                            "channel_id",
                            *(
                                ["material_pack_option"]
                                if material_pack_option is not None
                                else []
                            ),
                            *(["time_range"] if time_range is not None else []),
                        ],
                        allow_history=True,
                        min_facts=1,
                    ),
                    "evidence_contract_ref": None,
                }
            )
        }
    )
