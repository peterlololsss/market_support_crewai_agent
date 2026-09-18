from __future__ import annotations

import asyncio
import os
from typing import override

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

from market_support_crewai_agent.runtime.context.stage_inputs import (
    SmalltalkComposerPromptInputV1,
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
    V2ComposerOutputRejected,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    AlignmentRefetchRequestV1,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_agents import install_fake_planner
from tests.helpers.reply_contract_evidence import (
    document_evidence_fact,
    v2_evidence,
)
from tests.helpers.reply_contract_plan_fixtures import make_support_plan_spec
from tests.helpers.reply_contract_preflight import EmptyPreflightService
from tests.helpers.reply_contract_runtime import (
    ReplyRequestBuilder,
)

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_runtime_uses_smalltalk_composer_for_triggered_greeting():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="greeting",
            artifact_kind="smalltalk",
            action_intent="none",
            ambiguity_slots=[],
            compliance={
                "is_compliant": True,
                "reason_code": "unrelated_request",
                "reason": "greeting",
            },
        ),
    )
    prompts: list[str] = []
    stages: list[str] = []

    class FakeSmalltalkComposer:
        async def compose(
            self, input_value: ComposerPromptInputV1
        ) -> ComposerReplyOutput:
            assert isinstance(input_value, SmalltalkComposerPromptInputV1)
            stages.append("smalltalk_composer")
            prompts.append(input_value.model_dump_json())
            return ComposerReplyOutput(
                response_mode="answer",
                reply=PrimaryReply(kind="answer", text="smalltalk response"),
            )

    runtime.v2_composer = FakeSmalltalkComposer()

    response = asyncio.run(runtime.reply(ReplyRequestBuilder("hi").payload()))

    assert stages == ["smalltalk_composer"]
    assert response.reply.kind == "answer"
    assert response.reply.text == "smalltalk response"
    assert not response.actions
    assert prompts


def test_runtime_skips_knowledge_composer_without_document_evidence():
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

    class EmptyEvidenceExecutor(EvidenceExecutor):
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
            return v2_evidence(request, plan, policy, scope_authority)

    runtime.evidence_executor = EmptyEvidenceExecutor(EmptyPreflightService())

    response = asyncio.run(runtime.reply(ReplyRequestBuilder("介绍一下衍复").payload()))

    assert response.reply.kind == "unable_to_answer"
    assert not response.actions


def test_runtime_raises_when_composer_returns_invalid_reply_contract():
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

    class BadComposer:
        async def compose(
            self, input_value: ComposerPromptInputV1
        ) -> ComposerReplyOutput:
            del input_value
            raise V2ComposerOutputRejected("invalid composer output")

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
    runtime.v2_composer = BadComposer()

    response = asyncio.run(runtime.reply(ReplyRequestBuilder("介绍一下衍复").payload()))

    assert response.reply.kind == "unable_to_answer"
    assert not response.actions
