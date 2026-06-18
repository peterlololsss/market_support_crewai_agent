from __future__ import annotations

import asyncio
from types import SimpleNamespace

from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.ontology import (
    ArtifactScope,
    DomainContextBuilder,
)
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.orchestration.reply_agent import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.conversation_store import ConversationStore
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas import ReplyRequest
from market_support_crewai_agent.settings import Settings
from tests.helpers.planning import make_plan_spec


def test_regression_original_bug_full_planner_answerability_output_verifier_path_is_deterministic():
    request = ReplyRequest.model_validate(
        {
            "context_id": "msg-1",
            "conversation_key": "wecom:group-1:sender-1",
            "group_id": "group-1",
            "sender_id": "sender-1",
            "message": "策略S1材料包里有哪些产品",
            "is_group": True,
            "group_name": "测试群",
            "dist_channel_name": "测试渠道",
            "sender_nickname": "测试用户",
            "available_materials": ["material", "weekly", "monthly"],
            "material_pack_options": ["策略S1", "策略S2"],
            "channel_type": "bank",
        }
    )
    facts = [
        material_products("产品B", material_pack_option="策略S1"),
        report_products("weekly_report", "产品A"),
    ]
    verifier = RecordingAlignmentVerifier()
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=True),
        conversation_store=ConversationStore(),
        evidence_executor=FakeEvidenceExecutor(facts),
        alignment_verifier=verifier,
    )
    runtime._build_planner_agent = lambda: FakePlanner(  # type: ignore[method-assign]
        make_plan_spec(
            request,
            selected_capability_id="material_pack.product_list",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["material_pack"],
            material_pack_option="策略S1",
            user_intent_summary="answer material pack product list",
        )
    )
    runtime._build_agent = lambda *_args, **_kwargs: ComposerShouldNotRun()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(request))

    assert response.reply.kind == "answer"
    assert response.reply.text == "材料包包含：产品B。"
    assert "产品A" not in response.reply.text
    assert response.actions == []
    assert len(verifier.calls) == 1
    assert verifier.calls[0]["plan"].plan_spec.plan_units[0].selected_capability_id == (
        "material_pack.product_list"
    )


class FakePlanner:
    def __init__(self, plan_spec):
        self.plan_spec = plan_spec

    async def kickoff_async(self, prompt, response_format):
        del prompt, response_format
        return SimpleNamespace(
            pydantic=self.plan_spec,
            raw="",
            agent_role="fake-planner",
            usage_metrics={"total_tokens": 0},
            todos=[],
        )


class ComposerShouldNotRun:
    async def kickoff_async(self, prompt, response_format):
        del prompt, response_format
        raise AssertionError("deterministic material-pack product answer needs no composer")


class FakeEvidenceExecutor:
    def __init__(self, facts: list[EvidenceFact]):
        self.facts = facts

    async def execute(self, request, canonical_context, plan, policy, action_history=None):
        del canonical_context, plan, policy, action_history
        domain_context = DomainContextBuilder().build(
            request,
            available_artifacts=self.facts,
        )
        return SimpleNamespace(
            preflight=AdapterPreflightSnapshot.empty(),
            evidence_facts=self.facts,
            business_facts=derive_business_facts(self.facts, request),
            domain_context=domain_context,
            guardrail_decisions=[],
        )


class RecordingAlignmentVerifier:
    def __init__(self):
        self.calls = []

    async def verify(self, **kwargs):
        self.calls.append(kwargs)
        return ReplyAlignmentVerdict(
            aligned=True,
            safe_to_return=True,
            confidence=1.0,
        )


def material_products(*products: str, material_pack_option: str) -> EvidenceFact:
    return EvidenceFact(
        fact_type="material_pack_product_list",
        value=True,
        source_type="adapter_material_pack_content",
        source_id=f"material_pack:{material_pack_option}",
        resolve_type="material_pack",
        artifact_type="material_pack",
        metadata={
            "material_pack_option": material_pack_option,
            "products": [{"product_name": product} for product in products],
        },
        scope=ArtifactScope(channel_id="unknown"),
    )


def report_products(resolve_type: str, *products: str) -> EvidenceFact:
    return EvidenceFact(
        fact_type="report_scope_products",
        value=True,
        source_type="adapter_report_scope",
        source_id=resolve_type,
        resolve_type=resolve_type,  # type: ignore[arg-type]
        artifact_type=resolve_type,  # type: ignore[arg-type]
        metadata={"products": [{"product_name": product} for product in products]},
    )
