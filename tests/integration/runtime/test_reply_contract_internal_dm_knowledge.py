from __future__ import annotations

import asyncio
import base64
import os
from typing import override

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

import pytest
from pydantic import JsonValue

from market_support_crewai_agent.runtime.context.stage_inputs import (
    KnowledgeComposerPromptInputV1,
    PlannerPromptInputV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge import (
    InternalCompanyKnowledgeGatewayV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayDocumentContextV1,
    GatewayStaticContextV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightService,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DocumentMcpCacheAuthorityV1,
)
from market_support_crewai_agent.runtime.planning.planner_results import (
    PlannerFrameResult,
)
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
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
from market_support_crewai_agent.schemas.reply import PrimaryReply
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_plan_fixtures import make_support_plan_spec
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.helpers.reply_contract_runtime import next_request_id

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_enabled_internal_dm_answers_granted_knowledge_without_business_resolve(
    monkeypatch: pytest.MonkeyPatch,
):
    envelope = make_v2_envelope(
        "介绍一下公司",
        request_id=next_request_id("enabled-internal-dm"),
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:test",
            "direct_thread_ref": "direct:enabled-knowledge",
            "principal_ref": "principal:enabled-knowledge",
        },
        presentation={
            "contract_version": "direct-presentation.v1",
            "principal_name": "direct user",
        },
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"],
            "outbound_actions": [],
            "mention_types": [],
        },
    )

    class Preflight(AdapterPreflightService):
        def __init__(self) -> None:
            super().__init__()
            self.resolve_types: list[list[str]] = []

        @override
        async def collect(
            self,
            request: KernelReplyRequestV1,
            resolve_types: list[AdapterResolveType] | None = None,
            resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
        ) -> AdapterPreflightSnapshot:
            del request, resolve_material_pack_options
            self.resolve_types.append(list(resolve_types or ()))
            return AdapterPreflightSnapshot.empty()

    class DocumentProvider:
        def __init__(self) -> None:
            self.calls: int = 0

        async def collect(
            self,
            *,
            request: KernelReplyRequestV1,
            evidence_query: str,
            cache_authority: DocumentMcpCacheAuthorityV1 | None,
        ) -> tuple[GatewayDocumentContextV1, ...]:
            del request, evidence_query, cache_authority
            self.calls += 1
            return (
                GatewayDocumentContextV1(
                    document_id="company",
                    text="公司内部资料回答。",
                ),
            )

    class StaticProvider:
        def __init__(self) -> None:
            self.calls: int = 0

        async def collect(
            self,
            *,
            request: KernelReplyRequestV1,
            evidence_query: str,
        ) -> tuple[GatewayStaticContextV1, ...]:
            del request, evidence_query
            self.calls += 1
            return ()

    composer_inputs: list[KnowledgeComposerPromptInputV1] = []

    class Composer:
        async def compose(
            self, input_value: ComposerPromptInputV1
        ) -> ComposerReplyOutput:
            assert isinstance(input_value, KnowledgeComposerPromptInputV1)
            composer_inputs.append(input_value)
            return ComposerReplyOutput(
                response_mode="answer",
                reply=PrimaryReply(
                    kind="answer",
                    text="公司内部资料回答。",
                    mentions=[],
                ),
            )

    preflight = Preflight()
    document_provider = DocumentProvider()
    static_provider = StaticProvider()
    audit_key = base64.urlsafe_b64encode(b"d" * 32).decode("ascii").rstrip("=")
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            direct_audit_hmac_key=audit_key,
            reply_alignment_verifier_enabled=False,
        ),
        conversation_store=ConversationStore(),
        preflight_service=preflight,
        internal_company_knowledge_gateway=InternalCompanyKnowledgeGatewayV1(
            document_provider=document_provider,
            static_provider=static_provider,
        ),
        v2_composer=Composer(),
    )
    plan_spec = make_support_plan_spec(
        request=envelope.request,
        selected_capability_id="answer_internal_company_knowledge",
        answerability_policy="answer",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        evidence_query="company overview",
        ambiguity_slots=[],
    )
    planner_recall_modes: list[str] = []

    async def planner_kickoff(
        planner_agent: CrewAIAgentAdapterV1 | None,
        planner_program: PromptProgram,
        *,
        planner_input: PlannerPromptInputV1,
        timeout_seconds: float | None,
        retry_attempts: int,
        base_delay_seconds: float,
        journal: TurnLlmInvocationJournalV1 | None = None,
        settings: Settings | None = None,
    ) -> tuple[PlannerFrameResult, list[dict[str, JsonValue]]]:
        del (
            planner_agent,
            planner_program,
            timeout_seconds,
            retry_attempts,
            base_delay_seconds,
            journal,
            settings,
        )
        planner_recall_modes.append(planner_input.effective_policy.recall_mode)
        return PlannerFrameResult(pydantic=plan_spec, raw=""), []

    monkeypatch.setattr(
        "market_support_crewai_agent.runtime.planning_flow.run_planner_kickoff_with_retry",
        planner_kickoff,
    )

    response = asyncio.run(runtime.reply(envelope))

    assert response.reply.text == "公司内部资料回答。"
    assert response.reply.mentions == []
    assert not response.actions
    assert "%%" not in response.reply.text
    assert preflight.resolve_types == [[]]
    assert (document_provider.calls, static_provider.calls) == (1, 1)
    assert composer_inputs[0].scene == "direct"
    assert planner_recall_modes == ["off"]
