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


def test_alignment_verifier_refetches_document_context_with_refined_query():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local:23000",
        ),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="answer report format question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            evidence_query="年化收益率",
            ambiguity_slots=[],
        ),
    )
    queries: list[str | None] = []

    class RefetchEvidenceExecutor(EvidenceExecutor):
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
            refetch = alignment_refetch_request
            query = (
                refetch.refined_evidence_query
                if refetch is not None
                else plan.units[0].evidence_query
            )
            queries.append(query)
            facts = (
                (
                    document_evidence_fact(
                        plan, "月报采用区间收益展示，不展示年化收益率。"
                    ),
                )
                if query == "月报 年化收益率 展示规则"
                else ()
            )
            return v2_evidence(
                request,
                plan,
                policy,
                scope_authority,
                facts=facts,
            )

    class FakeComposer:
        async def compose(
            self, input_value: ComposerPromptInputV1
        ) -> ComposerReplyOutput:
            assert isinstance(input_value, KnowledgeComposerPromptInputV1)
            has_evidence = bool(input_value.unit_groundings[0].allowed_evidence_ids)
            if not has_evidence:
                return ComposerReplyOutput(
                    response_mode="abstain",
                    reply=PrimaryReply(
                        kind="unable_to_answer",
                        text="老师，这个信息我这边暂时无法确认，先不回答避免信息不准确。",
                    ),
                )
            return ComposerReplyOutput(
                response_mode="answer",
                reply=PrimaryReply(
                    kind="answer",
                    text="月报采用区间收益展示，不展示年化收益率。",
                ),
            )

    class RefetchThenPassVerifier:
        def __init__(self):
            self.calls: int = 0

        async def verify(
            self,
            input_value: SanitizedAlignmentVerifierInputV1,
        ) -> ReplyAlignmentVerdict:
            self.calls += 1
            if self.calls == 1:
                assert input_value.candidate.reply_kind == "unable_to_answer"
                return ReplyAlignmentVerdict(
                    aligned=False,
                    safe_to_return=False,
                    failure_code="missing_evidence",
                    remediation="refetch_document_context",
                    refined_evidence_query="月报 年化收益率 展示规则",
                )
            return ReplyAlignmentVerdict(
                aligned=True, safe_to_return=True, confidence=0.9
            )

    verifier = RefetchThenPassVerifier()
    runtime.evidence_executor = RefetchEvidenceExecutor(EmptyPreflightService())
    runtime.v2_composer = FakeComposer()
    runtime.alignment_verifier = verifier

    response = asyncio.run(
        runtime.reply(ReplyRequestBuilder("月报里为什么没有年化收益率").payload())
    )

    assert queries == ["年化收益率", "月报 年化收益率 展示规则"]
    assert response.reply.text == "月报采用区间收益展示，不展示年化收益率。"
    assert not response.actions
