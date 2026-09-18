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
from tests.helpers.reply_contract_agents import install_fake_planner
from tests.helpers.reply_contract_composer import install_fake_clarification_composer
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
)

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_runtime_uses_composer_only_for_knowledge_answer():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local:23000",
            reply_alignment_verifier_enabled=False,
        ),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="answer knowledge question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            ambiguity_slots=[],
        ),
    )
    composer_inputs: list[KnowledgeComposerPromptInputV1] = []

    class FakeComposer:
        async def compose(
            self, input_value: ComposerPromptInputV1
        ) -> ComposerReplyOutput:
            assert isinstance(input_value, KnowledgeComposerPromptInputV1)
            composer_inputs.append(input_value)
            return ComposerReplyOutput(
                response_mode="answer",
                reply=PrimaryReply(kind="answer", text="文档证据回答"),
            )

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
            return v2_evidence(
                request,
                plan,
                policy,
                scope_authority,
                facts=(document_evidence_fact(plan, "文档证据"),),
            )

    runtime.evidence_executor = FakeEvidenceExecutor(EmptyPreflightService())
    runtime.v2_composer = FakeComposer()

    response = asyncio.run(runtime.reply(ReplyRequestBuilder("介绍一下衍复").payload()))

    assert response.reply.text == "文档证据回答"
    assert len(composer_inputs) == 1


def test_alignment_verifier_blocks_wrong_outbound_action():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
        ),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    install_fake_planner(runtime, make_weekly_plan_spec())

    class WrongActionVerifier:
        async def verify(
            self,
            input_value: SanitizedAlignmentVerifierInputV1,
        ) -> ReplyAlignmentVerdict:
            assert input_value.candidate.actions[0].type == "send_weekly_report"
            return ReplyAlignmentVerdict(
                aligned=False,
                safe_to_return=False,
                failure_code="wrong_artifact",
                remediation="return_unable",
                rationale="question asks why a monthly report omits a field",
            )

    runtime.alignment_verifier = WrongActionVerifier()

    response = asyncio.run(
        runtime.reply(ReplyRequestBuilder("月报里为什么没有年化收益率").payload())
    )

    assert response.reply.kind == "unable_to_answer"
    assert not response.actions


def test_alignment_verifier_allows_valid_action_response():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    install_fake_planner(runtime, make_weekly_plan_spec())
    verifier_calls: list[str] = []

    class PassingVerifier:
        async def verify(
            self,
            input_value: SanitizedAlignmentVerifierInputV1,
        ) -> ReplyAlignmentVerdict:
            verifier_calls.append(input_value.message.text)
            assert input_value.candidate.actions[0].type == "send_weekly_report"
            return ReplyAlignmentVerdict(
                aligned=True,
                safe_to_return=True,
                confidence=0.9,
            )

    runtime.alignment_verifier = PassingVerifier()

    response = asyncio.run(
        runtime.reply(ReplyRequestBuilder("想确认一下周报能不能发我").payload())
    )

    assert verifier_calls == ["想确认一下周报能不能发我"]
    assert response.reply.text == ""
    assert response.actions[0].type == "send_weekly_report"


def test_alignment_verifier_return_clarification_uses_composer():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    install_fake_planner(runtime, make_weekly_plan_spec())
    composer_stages: list[str] = []
    install_fake_clarification_composer(
        runtime,
        text="当前不确定你要周报还是月报，请确认一下。",
        stages=composer_stages,
    )

    class ClarifyingVerifier:
        async def verify(
            self,
            input_value: SanitizedAlignmentVerifierInputV1,
        ) -> ReplyAlignmentVerdict:
            assert input_value.candidate.actions[0].type == "send_weekly_report"
            return ReplyAlignmentVerdict(
                aligned=False,
                safe_to_return=False,
                failure_code="ambiguous_request",
                remediation="return_clarification",
                rationale="report artifact is ambiguous",
            )

    runtime.alignment_verifier = ClarifyingVerifier()

    response = asyncio.run(runtime.reply(ReplyRequestBuilder("报告发我一下").payload()))

    assert response.reply.kind == "clarification"
    assert response.reply.text == "老师，麻烦补充一下具体需求，我再继续处理。"
    assert not response.actions
    assert composer_stages == []
