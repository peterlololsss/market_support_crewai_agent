from __future__ import annotations

import asyncio
import os
from typing import override

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

from market_support_crewai_agent.runtime.context.stage_inputs import (
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
from market_support_crewai_agent.runtime.policy.compliance import (
    refusal_text_for_reason,
)
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
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
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


def test_alignment_verdict_rejects_free_text_report_scope_refetch_query():
    try:
        _ = ReplyAlignmentVerdict(
            aligned=False,
            safe_to_return=False,
            failure_code="missing_answer",
            remediation="refetch_report_scope",
            refined_evidence_query="weekly report product list scope",
        )
    except ValueError as exc:
        error = exc
    else:
        raise AssertionError("free-text report-scope refetch query must be rejected")

    assert "alignment_report_refetch_query_not_allowed" in str(error)


def test_alignment_verifier_failure_does_not_return_action():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    install_fake_planner(runtime, make_weekly_plan_spec())

    class FailingVerifier:
        async def verify(
            self,
            input_value: SanitizedAlignmentVerifierInputV1,
        ) -> ReplyAlignmentVerdict:
            del input_value
            raise AgentRuntimeError("alignment verifier transport failed")

    runtime.alignment_verifier = FailingVerifier()

    response = asyncio.run(
        runtime.reply(ReplyRequestBuilder("想确认一下周报能不能发我").payload())
    )

    assert response.reply.kind == "unable_to_answer"
    assert not response.actions


def test_alignment_verifier_failure_preserves_non_compliant_refusal_text():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="refuse private contact request",
            artifact_kind="refusal",
            action_intent="refuse",
            ambiguity_slots=[],
            requested_capabilities=[],
            compliance={
                "is_compliant": False,
                "reason_code": "private_contact_request",
                "reason": "asks to add private contact",
            },
        ),
    )

    class FailingVerifier:
        async def verify(
            self,
            input_value: SanitizedAlignmentVerifierInputV1,
        ) -> ReplyAlignmentVerdict:
            del input_value
            raise AgentRuntimeError("alignment verifier transport failed")

    runtime.alignment_verifier = FailingVerifier()

    response = asyncio.run(
        runtime.reply(ReplyRequestBuilder("加你微信了，通过一下").payload())
    )

    assert response.reply.kind == "unable_to_answer"
    assert response.reply.text == refusal_text_for_reason("private_contact_request")
    assert not response.actions


def test_alignment_verifier_recompose_once():
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
            user_need="answer knowledge question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            evidence_query="示例 公司介绍",
            ambiguity_slots=[],
        ),
    )
    composer_calls: list[str] = []

    class FakeComposer:
        async def compose(
            self, input_value: ComposerPromptInputV1
        ) -> ComposerReplyOutput:
            composer_calls.append(input_value.model_dump_json())
            if len(composer_calls) == 1:
                text = "这是一段没有回答问题的文字。"
            else:
                text = "示例是一家量化私募管理人。"
            return ComposerReplyOutput(
                response_mode="answer",
                reply=PrimaryReply(kind="answer", text=text),
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
                facts=(document_evidence_fact(plan, "示例是一家量化私募管理人。"),),
            )

    class RecomposeThenValidVerifier:
        def __init__(self):
            self.calls: int = 0

        async def verify(
            self,
            input_value: SanitizedAlignmentVerifierInputV1,
        ) -> ReplyAlignmentVerdict:
            del input_value
            self.calls += 1
            if self.calls == 1:
                return ReplyAlignmentVerdict(
                    aligned=False,
                    safe_to_return=False,
                    failure_code="composer_drift",
                    remediation="recompose",
                    composer_feedback="answer the company introduction question",
                )
            return ReplyAlignmentVerdict(
                aligned=True, safe_to_return=True, confidence=0.9
            )

    verifier = RecomposeThenValidVerifier()
    runtime.evidence_executor = FakeEvidenceExecutor(EmptyPreflightService())
    runtime.alignment_verifier = verifier
    runtime.v2_composer = FakeComposer()

    response = asyncio.run(runtime.reply(ReplyRequestBuilder("介绍一下示例").payload()))

    assert response.reply.text == "示例是一家量化私募管理人。"
    assert len(composer_calls) == 2
    assert verifier.calls == 2
