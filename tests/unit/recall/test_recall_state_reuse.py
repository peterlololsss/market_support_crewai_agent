from __future__ import annotations

from typing import Literal, assert_never

import pytest

from market_support_crewai_agent.runtime import pipeline
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.runtime.policy.ontology import DomainContextV1Builder
from market_support_crewai_agent.runtime.prompts.context import IntentGateResult
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    KnowledgeQaMatch,
)
from market_support_crewai_agent.runtime.recall.flow import (
    PreplannerRecallRuntimeV1,
)
from market_support_crewai_agent.runtime.recall.question_models import (
    QuestionRecallMatch,
)
from market_support_crewai_agent.runtime.recall.turn_state import (
    PreplannerRecallTransitionV1,
)
from market_support_crewai_agent.runtime.recall.service import (
    ApprovedStaticRecallCollectionV1,
)
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.integration.runtime._recall_pipeline_fixtures import (
    document_candidate,
    group_context,
    static_hit,
)
from tests.integration.runtime._recall_pipeline_runtime_fixtures import (
    DocumentCollector,
    RecallPipelineRuntime,
    StaticCollector,
    run_pipeline,
)
from tests.unit.recall._recall_state_pipeline_spies import (
    PlannerStateRecorder,
    RecallCollectionRecorder,
)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("mode", "expected_decision"),
    (
        ("off", "disabled"),
        ("advisory", "advisory_candidates"),
        ("shortcut", "shortcut_match"),
    ),
)
async def test_pipeline_carries_one_collected_state_for_each_recall_outcome(
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["off", "advisory", "shortcut"],
    expected_decision: str,
) -> None:
    # Given: each active recall mode and a spy around the sole collection seam.
    envelope, policy, scope = group_context(mode)
    static = StaticCollector(static_hit(policy, with_payload=mode == "shortcut"))
    document = DocumentCollector(document_candidate())
    runtime = RecallPipelineRuntime(static, document)
    collected = RecallCollectionRecorder()
    planned = PlannerStateRecorder()

    monkeypatch.setattr(pipeline, "collect_preplanner_recall", collected)
    monkeypatch.setattr(pipeline, "build_candidate_via_planner", planned)

    # When: the active V2 pipeline builds the first candidate.
    result = await run_pipeline(runtime, (envelope, policy, scope))

    # Then: one frozen state and its original rch1 cross the chosen route.
    assert len(collected.states) == 1
    state = collected.states[0]
    assert state.outcome.decision == expected_decision
    assert state.outcome.trace_hash.startswith("rch1:")
    match mode:
        case "shortcut":
            routed_state = runtime.recall_states[0]
        case "off" | "advisory":
            routed_state = planned.states[0]
        case _:
            assert_never(mode)
    assert routed_state is state
    assert routed_state is not None
    assert routed_state.outcome is state.outcome
    assert routed_state.outcome.trace_hash == state.outcome.trace_hash
    match mode:
        case "shortcut":
            assert result.plan is runtime.plans[0]
            assert result.recall_state is state
        case "off" | "advisory":
            assert result.reason_code == "compliant_product_request"
            assert result.recall_state is state
        case _:
            assert_never(mode)


@pytest.mark.anyio
async def test_unavailable_recall_state_is_reused_without_recollection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: one failed static branch, one clean miss, and a planner re-entry spy.
    envelope, policy, scope = group_context("advisory")
    static = StaticCollector(
        ApprovedStaticRecallCollectionV1(
            match=QuestionRecallMatch(
                status="unavailable",
                decision="fail_open",
                reason_code="source_unavailable",
            )
        )
    )
    document = DocumentCollector(KnowledgeQaMatch(status="no_match"))
    runtime = RecallPipelineRuntime(static, document)
    planned = PlannerStateRecorder()

    monkeypatch.setattr(pipeline, "build_candidate_via_planner", planned)

    # When: the first planner pass is followed by a replan with its same state.
    _ = await run_pipeline(runtime, (envelope, policy, scope))
    _ = await pipeline.build_candidate_response(
        runtime,
        request=envelope.request,
        domain_context=DomainContextV1Builder().build(
            envelope.request,
            scope_authority=scope,
        ),
        policy=policy,
        model_family="generic",
        intent_gate=IntentGateResult(artifact_hint="unclear"),
        history=[],
        action_history=[],
        prompt_programs=[],
        llm_executions=[],
        scope_authority=scope,
        state_key_ref=envelope.state_key_ref,
        recall_state=planned.states[0],
    )

    # Then: providers ran once total and both planner passes share identity/hash.
    assert (static.calls, document.calls) == (1, 1)
    assert len(planned.states) == 2
    assert planned.states[0].outcome.decision == "unavailable"
    assert planned.states[1] is planned.states[0]
    assert planned.states[1].outcome is planned.states[0].outcome
    assert planned.states[1].outcome.trace_hash == planned.states[0].outcome.trace_hash


@pytest.mark.anyio
async def test_direct_deterministic_branch_does_not_collect_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an explicit direct handoff and a recall seam that must stay untouched.
    envelope = make_v2_envelope(
        "t0",
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:recall-reuse",
            "direct_thread_ref": "direct:recall-reuse",
            "principal_ref": "principal:recall-reuse",
        },
        presentation={"contract_version": "direct-presentation.v1"},
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"],
            "outbound_actions": [],
            "mention_types": [],
        },
    )
    scope = business_scope_authority_v1(envelope.request.business_scope)
    core = compile_policy_authority_core_v1(envelope.request, scope)
    policy = PolicyManifestV2.from_core(core, policy_ledger_summary_v1((), 0))
    static = StaticCollector(static_hit(policy, with_payload=False))
    document = DocumentCollector(document_candidate())
    runtime = RecallPipelineRuntime(static, document)
    collect_calls = 0

    async def collect_must_not_run(
        runtime_arg: PreplannerRecallRuntimeV1,
        *,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
        scope_authority: BusinessScopeAuthorityV1,
    ) -> PreplannerRecallTransitionV1:
        del runtime_arg, request, policy, scope_authority
        nonlocal collect_calls
        collect_calls += 1
        raise AssertionError("direct deterministic branch collected recall")

    monkeypatch.setattr(pipeline, "collect_preplanner_recall", collect_must_not_run)

    # When: the deterministic input-policy branch builds its candidate.
    result = await pipeline.build_candidate_response(
        runtime,
        request=envelope.request,
        domain_context=DomainContextV1Builder().build(
            envelope.request,
            scope_authority=scope,
        ),
        policy=policy,
        model_family="generic",
        intent_gate=IntentGateResult(artifact_hint="unclear"),
        history=[],
        action_history=[],
        prompt_programs=[],
        llm_executions=[],
        scope_authority=scope,
        state_key_ref=envelope.state_key_ref,
    )

    # Then: recall work stayed at zero and no per-turn recall state was invented.
    assert collect_calls == 0
    assert (static.calls, document.calls) == (0, 0)
    assert result.plan.origin == "input_policy"
    assert runtime.recall_states == [None]
