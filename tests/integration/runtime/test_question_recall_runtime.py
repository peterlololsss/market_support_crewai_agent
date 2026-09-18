from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
from pydantic import JsonValue

from market_support_crewai_agent.runtime import pipeline
from market_support_crewai_agent.runtime.context.models import (
    RecentExecutedActionSummaryViewV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology_models import DomainContextV1
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.context import IntentGateResult
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.prompts.profiles import ModelFamily
from market_support_crewai_agent.runtime.recall.question_models import (
    QuestionRecallMatch,
)
from market_support_crewai_agent.runtime.recall.service import (
    ApprovedStaticRecallCollectionV1,
)
from market_support_crewai_agent.runtime.recall.turn_state import RecallTurnStateV1
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.v2_attempt import V2AttemptResult
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from tests.integration.runtime._recall_pipeline_fixtures import (
    direct_context,
    document_candidate,
    group_context,
    planner_attempt_result,
    static_hit,
)
from tests.integration.runtime._recall_pipeline_runtime_fixtures import (
    DocumentCollector,
    PipelineContext,
    RecallPipelineRuntime,
    StaticCollector,
    run_pipeline,
)


def _group_off_context() -> PipelineContext:
    return group_context("off")


CONTEXT_FACTORIES: tuple[Callable[[], PipelineContext], ...] = (
    _group_off_context,
    direct_context,
)


class PlannerCallRecorder:
    def __init__(self) -> None:
        self.calls: int = 0
        self.snapshots: list[str] = []
        self.keyword_names: tuple[str, ...] = (
            "request",
            "domain_context",
            "policy",
            "model_family",
            "intent_gate",
            "history",
            "action_history",
            "prompt_programs",
            "llm_executions",
            "scope_authority",
            "state_key_ref",
            "alignment_verdict",
            "alignment_attempt",
            "recall_state",
            "llm_journal",
        )

    async def __call__(
        self,
        _runtime: RecallPipelineRuntime,
        *,
        request: KernelReplyRequestV1,
        domain_context: DomainContextV1,
        policy: PolicyManifestV2,
        model_family: ModelFamily,
        intent_gate: IntentGateResult,
        history: list[ConversationMessage],
        action_history: Sequence[RecentExecutedActionSummaryViewV1],
        prompt_programs: list[PromptProgram],
        llm_executions: list[dict[str, JsonValue]],
        scope_authority: BusinessScopeAuthorityV1,
        state_key_ref: str,
        alignment_verdict: ReplyAlignmentVerdict | None = None,
        alignment_attempt: int = 0,
        recall_state: RecallTurnStateV1,
        llm_journal: TurnLlmInvocationJournalV1 | None = None,
    ) -> V2AttemptResult:
        del _runtime, model_family, alignment_verdict, alignment_attempt, llm_journal
        self.calls += 1
        self.snapshots.append(
            repr(
                (
                    request,
                    domain_context,
                    policy,
                    intent_gate,
                    history,
                    action_history,
                    prompt_programs,
                    llm_executions,
                    scope_authority,
                    state_key_ref,
                    recall_state,
                )
            )
        )
        return planner_attempt_result(
            request=request,
            policy=policy,
            scope_authority=scope_authority,
            recall_state=recall_state,
        )


async def planner_must_not_run(
    _runtime: RecallPipelineRuntime,
    *,
    request: KernelReplyRequestV1,
    domain_context: DomainContextV1,
    policy: PolicyManifestV2,
    model_family: ModelFamily,
    intent_gate: IntentGateResult,
    history: list[ConversationMessage],
    action_history: Sequence[RecentExecutedActionSummaryViewV1],
    prompt_programs: list[PromptProgram],
    llm_executions: list[dict[str, JsonValue]],
    scope_authority: BusinessScopeAuthorityV1,
    state_key_ref: str,
    alignment_verdict: ReplyAlignmentVerdict | None = None,
    alignment_attempt: int = 0,
    recall_state: RecallTurnStateV1,
    llm_journal: TurnLlmInvocationJournalV1 | None = None,
) -> V2AttemptResult:
    del (
        _runtime,
        request,
        domain_context,
        policy,
        model_family,
        intent_gate,
        history,
        action_history,
        prompt_programs,
        llm_executions,
        scope_authority,
        state_key_ref,
        alignment_verdict,
        alignment_attempt,
        recall_state,
        llm_journal,
    )
    raise AssertionError("planner must not run for a validated static shortcut")


@pytest.mark.anyio
@pytest.mark.parametrize("context_factory", CONTEXT_FACTORIES)
async def test_active_v2_off_mode_reaches_planner_with_zero_recall_calls(
    monkeypatch: pytest.MonkeyPatch,
    context_factory: Callable[[], PipelineContext],
) -> None:
    # Given: an active group/direct V2 request whose compiled recall mode is off.
    envelope, policy, scope = context_factory()
    static = StaticCollector(static_hit(policy, with_payload=False))
    document = DocumentCollector(document_candidate())
    runtime = RecallPipelineRuntime(static, document)
    planner = PlannerCallRecorder()

    monkeypatch.setattr(pipeline, "build_candidate_via_planner", planner)

    # When: the active candidate pipeline handles the request.
    result = await run_pipeline(runtime, (envelope, policy, scope))

    # Then: no pre-planner provider runs and planner remains the origin.
    assert result.reason_code == "compliant_product_request"
    assert policy.recall_mode == "off"
    assert (static.calls, document.calls) == (0, 0)
    assert planner.calls == 1


@pytest.mark.anyio
async def test_active_v2_advisory_candidates_cannot_bypass_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: advisory mode with a static payload and a Document MCP candidate.
    envelope, policy, scope = group_context("advisory")
    static = StaticCollector(static_hit(policy, with_payload=True))
    document = DocumentCollector(document_candidate())
    runtime = RecallPipelineRuntime(static, document)
    planner = PlannerCallRecorder()

    monkeypatch.setattr(pipeline, "build_candidate_via_planner", planner)

    # When: the active candidate pipeline handles the advisory recall result.
    result = await run_pipeline(runtime, (envelope, policy, scope))

    # Then: both providers run once, planner runs, and no raw recall data enters it.
    assert result.reason_code == "compliant_product_request"
    assert (static.calls, document.calls) == (1, 1)
    assert planner.calls == 1
    assert "recall" not in planner.keyword_names
    planner_snapshot = planner.snapshots[0]
    assert "https://example.test" not in planner_snapshot
    assert "raw_selector_output" not in planner_snapshot
    assert "file:///provider" not in planner_snapshot


@pytest.mark.anyio
async def test_active_v2_shortcut_uses_validated_static_plan_without_document_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: shortcut mode with an exact policy-bound approved-static payload.
    envelope, policy, scope = group_context("shortcut")
    static = StaticCollector(static_hit(policy, with_payload=True))
    document = DocumentCollector(document_candidate())
    runtime = RecallPipelineRuntime(static, document)

    monkeypatch.setattr(pipeline, "build_candidate_via_planner", planner_must_not_run)

    # When: the active candidate pipeline handles the shortcut.
    result = await run_pipeline(runtime, (envelope, policy, scope))

    # Then: the observable plan origin is static and Document MCP stays untouched.
    assert result.plan is runtime.plans[0]
    assert result.plan.origin == "approved_static_shortcut"
    assert result.plan.response_mode == "knowledge_answer"
    assert (static.calls, document.calls) == (1, 0)


@pytest.mark.anyio
async def test_active_v2_static_candidate_without_answer_cannot_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: shortcut mode with a valid payload but no approved static answer.
    envelope, policy, scope = group_context("shortcut")
    static = StaticCollector(
        static_hit(policy, with_payload=True, answer_available=False)
    )
    document = DocumentCollector(document_candidate())
    runtime = RecallPipelineRuntime(static, document)
    planner = PlannerCallRecorder()

    monkeypatch.setattr(pipeline, "build_candidate_via_planner", planner)

    # When: the active pipeline evaluates the static candidate.
    result = await run_pipeline(runtime, (envelope, policy, scope))

    # Then: both recall branches remain advisory and Planner owns the plan.
    assert result.reason_code == "compliant_product_request"
    assert planner.calls == 1
    assert (static.calls, document.calls) == (1, 1)
    assert runtime.plans == []


@pytest.mark.anyio
async def test_active_v2_document_candidate_never_shortcuts_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: shortcut mode with no bound static hit and a high Document MCP score.
    envelope, policy, scope = group_context("shortcut")
    no_match = QuestionRecallMatch(
        status="no_match",
        decision="fail_open",
        reason_code="no_match",
    )
    static = StaticCollector(ApprovedStaticRecallCollectionV1(match=no_match))
    document = DocumentCollector(document_candidate())
    runtime = RecallPipelineRuntime(static, document)
    planner = PlannerCallRecorder()

    monkeypatch.setattr(pipeline, "build_candidate_via_planner", planner)

    # When: the active candidate pipeline handles the Document MCP result.
    result = await run_pipeline(runtime, (envelope, policy, scope))

    # Then: the provider remains advisory regardless of its score.
    assert result.reason_code == "compliant_product_request"
    assert planner.calls == 1
    assert (static.calls, document.calls) == (1, 1)
    assert runtime.plans == []
