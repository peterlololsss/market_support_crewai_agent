from __future__ import annotations

import asyncio
import os
from typing import override

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

from market_support_crewai_agent.runtime.context.stage_inputs import (
    KnowledgeComposerPromptInputV1,
    SanitizedAlignmentVerifierInputV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentMcpCacheConfigV1,
)
from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    ComposerPromptInputV1,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    AlignmentRefetchRequestV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.crewai_adapter import make_completion_agent_adapter
from tests.helpers.reply_contract_agents import FakePlannerAgent
from tests.helpers.reply_contract_evidence import (
    document_evidence_fact,
    v2_evidence,
)
from tests.helpers.reply_contract_plan_fixtures import (
    make_support_plan_spec,
    make_weekly_plan_spec,
)
from tests.helpers.reply_contract_preflight import (
    EmptyPreflightService,
    ResolvedWeeklyPreflight,
)
from tests.helpers.reply_contract_runtime import (
    ReplyRequestBuilder,
    install_planner_agent_builder,
)

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_alignment_verifier_replan_path_includes_feedback():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local:23000",
        ),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    planner_prompts: list[str] = []
    frames = [
        make_weekly_plan_spec(),
        make_support_plan_spec(
            user_need="answer report format question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            evidence_query="月报 年化收益率 展示规则",
            ambiguity_slots=[],
        ),
    ]

    class FakeEvidenceExecutor(EvidenceExecutor):
        @override
        async def execute_v2(
            self,
            request: KernelReplyRequestV1,
            plan: ExecutionPlanV2,
            policy: PolicyManifestV2,
            *,
            scope_authority: BusinessScopeAuthorityV1,
            state_key_ref: str | None = None,
            document_cache_config: DocumentMcpCacheConfigV1 | None = None,
            alignment_refetch_request: AlignmentRefetchRequestV1 | None = None,
        ) -> CanonicalEvidenceExecutionResultV1:
            if plan.response_mode == "knowledge_answer":
                return v2_evidence(
                    request,
                    plan,
                    policy,
                    scope_authority,
                    facts=(document_evidence_fact(plan, "月报不展示年化收益率。"),),
                )
            return await EvidenceExecutor(ResolvedWeeklyPreflight()).execute_v2(
                request,
                plan,
                policy,
                scope_authority=scope_authority,
                alignment_refetch_request=alignment_refetch_request,
            )

    class FakeComposer:
        async def compose(
            self, input_value: ComposerPromptInputV1
        ) -> ComposerReplyOutput:
            assert isinstance(input_value, KnowledgeComposerPromptInputV1)
            return ComposerReplyOutput(
                response_mode="answer",
                reply=PrimaryReply(kind="answer", text="月报不展示年化收益率."),
            )

    class ReplanThenPassVerifier:
        def __init__(self):
            self.calls: int = 0

        async def verify(
            self,
            input_value: SanitizedAlignmentVerifierInputV1,
        ) -> ReplyAlignmentVerdict:
            self.calls += 1
            if self.calls == 1:
                assert input_value.candidate.actions[0].type == "send_weekly_report"
                return ReplyAlignmentVerdict(
                    aligned=False,
                    safe_to_return=False,
                    failure_code="wrong_artifact",
                    remediation="replan",
                    planner_feedback="This is a report-format knowledge question.",
                )
            return ReplyAlignmentVerdict(
                aligned=True, safe_to_return=True, confidence=0.9
            )

    verifier = ReplanThenPassVerifier()
    install_planner_agent_builder(
        runtime,
        lambda: make_completion_agent_adapter(
            lambda _response_format: frames.pop(0),
            role="planner",
            model="fake-planner",
            on_prompt=planner_prompts.append,
        ),
    )
    runtime.evidence_executor = FakeEvidenceExecutor(EmptyPreflightService())
    runtime.v2_composer = FakeComposer()
    runtime.alignment_verifier = verifier

    response = asyncio.run(
        runtime.reply(ReplyRequestBuilder("月报里为什么没有年化收益率").payload())
    )

    assert response.actions == []
    assert response.reply.text == "月报不展示年化收益率."
    assert len(planner_prompts) == 2


def test_alignment_replan_failure_returns_unable_instead_of_raising():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    planner_agents = [
        FakePlannerAgent(make_weekly_plan_spec()),
        make_completion_agent_adapter(
            lambda _response_format: "",
            role="planner",
            model="fake-planner",
        ),
    ]

    class ReplanVerifier:
        async def verify(
            self,
            input_value: SanitizedAlignmentVerifierInputV1,
        ) -> ReplyAlignmentVerdict:
            assert input_value.candidate.actions[0].type == "send_weekly_report"
            return ReplyAlignmentVerdict(
                aligned=False,
                safe_to_return=False,
                failure_code="wrong_intent",
                remediation="replan",
            )

    install_planner_agent_builder(runtime, lambda: planner_agents.pop(0))
    runtime.alignment_verifier = ReplanVerifier()

    response = asyncio.run(runtime.reply(ReplyRequestBuilder("报告发我一下").payload()))

    assert response.reply.kind == "unable_to_answer"
    assert not response.actions
