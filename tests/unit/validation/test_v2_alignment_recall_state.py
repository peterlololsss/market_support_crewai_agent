from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import final

import pytest

from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.planning import finalize_execution_plan_v2
from market_support_crewai_agent.runtime.planning.compiler import (
    DeterministicPlanOriginInputV1,
    DeterministicPlanUnitV1,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology import DomainContextV1Builder
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    KnowledgeQaMatch,
)
from market_support_crewai_agent.runtime.recall.flow import collect_preplanner_recall
from market_support_crewai_agent.runtime.recall.outcome_models import RecallOutcomeV1
from market_support_crewai_agent.runtime.recall.question_models import (
    QuestionRecallMatch,
)
from market_support_crewai_agent.runtime.recall.turn_state import RecallTurnStateV1
from market_support_crewai_agent.runtime.recall.service import (
    ApprovedStaticRecallCollectionV1,
)
from market_support_crewai_agent.runtime.v2_attempt import (
    V2AttemptResult,
    V2ReplyValidationResult,
)
from market_support_crewai_agent.runtime.validation import alignment_loop
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    AlignmentRefetchRequestV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from tests.integration.runtime._recall_pipeline_fixtures import group_context
from tests.integration.runtime._recall_pipeline_runtime_fixtures import (
    DocumentCollector,
    RecallPipelineRuntime,
    StaticCollector,
)


@final
class _ScriptedRemediator:
    def __init__(self) -> None:
        self.verdicts: Iterator[ReplyAlignmentVerdict] = iter(
            (
                ReplyAlignmentVerdict(
                    aligned=False,
                    safe_to_return=False,
                    failure_code="missing_evidence",
                    remediation="refetch_document_context",
                    refined_evidence_query="refined company website query",
                ),
                ReplyAlignmentVerdict(
                    aligned=False,
                    safe_to_return=False,
                    failure_code="composer_drift",
                    remediation="recompose",
                ),
                ReplyAlignmentVerdict(
                    aligned=True,
                    safe_to_return=True,
                    confidence=1.0,
                ),
            )
        )
        self.observed_states: list[RecallTurnStateV1] = []
        self.observed_outcomes: list[RecallOutcomeV1] = []
        self.observed_hashes: list[str] = []
        self.verify_attempts: list[alignment_loop.AlignmentAttemptV1] = []
        self.recompose_plan: ExecutionPlanV2 | None = None
        self.recompose_evidence: CanonicalEvidenceExecutionResultV1 | None = None
        self.refetch_request: AlignmentRefetchRequestV1 | None = None

    def _observe(self, candidate: V2AttemptResult) -> None:
        state = candidate.recall_state
        assert state is not None
        self.observed_states.append(state)
        self.observed_outcomes.append(state.outcome)
        self.observed_hashes.append(state.outcome.trace_hash)

    async def verdict_for(
        self,
        candidate: V2AttemptResult,
        attempt: alignment_loop.AlignmentAttemptV1,
    ) -> ReplyAlignmentVerdict:
        assert attempt == len(self.verify_attempts)
        self.verify_attempts.append(attempt)
        self._observe(candidate)
        return next(self.verdicts)

    async def replan(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: alignment_loop.AlignmentActionAttemptV1,
    ) -> V2AttemptResult:
        del candidate, verdict, attempt
        raise AssertionError("replan is not part of this bounded scenario")

    async def refetch(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: alignment_loop.AlignmentActionAttemptV1,
    ) -> alignment_loop.AlignmentRefetchOutcomeV1:
        assert verdict.remediation == "refetch_document_context"
        self._observe(candidate)
        request = AlignmentRefetchRequestV1(
            unit_id=candidate.plan.units[0].unit_id,
            manifest_ref=candidate.plan.units[0].manifest_ref,
            refined_evidence_query=verdict.refined_evidence_query or "",
            attempt=attempt,
        )
        self.refetch_request = request
        return alignment_loop.AlignmentRefetchOutcomeV1(
            request=request,
            candidate=replace(candidate, evidence=replace(candidate.evidence)),
        )

    async def recompose(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: alignment_loop.AlignmentActionAttemptV1,
    ) -> V2AttemptResult:
        assert attempt == 1
        assert verdict.remediation == "recompose"
        self._observe(candidate)
        self.recompose_plan = candidate.plan
        self.recompose_evidence = candidate.evidence
        return replace(
            candidate,
            response=ReplyResponse(
                reply=PrimaryReply(kind="answer", text="recomposed", mentions=[]),
                actions=[],
            ),
            reason_code="knowledge_answer_composer",
        )

    async def replace(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict | None,
        remediation: alignment_loop.AlignmentFallbackV1,
    ) -> V2AttemptResult:
        del candidate, verdict, remediation
        raise AssertionError("replacement is not part of this bounded scenario")


@pytest.mark.anyio
async def test_alignment_refetch_recompose_reuses_one_recall_state() -> None:
    # Given: one advisory collection and the maximum two-step remediation sequence.
    envelope, policy, scope = group_context("advisory")
    static = StaticCollector(
        ApprovedStaticRecallCollectionV1(
            match=QuestionRecallMatch(status="no_match", decision="fail_open")
        )
    )
    document = DocumentCollector(KnowledgeQaMatch(status="no_match"))
    recall_runtime = RecallPipelineRuntime(static, document)
    transition = await collect_preplanner_recall(
        recall_runtime,
        request=envelope.request,
        policy=policy,
        scope_authority=scope,
    )
    state = transition.turn_state
    candidate = _candidate(envelope.request, policy, scope, state)
    remediator = _ScriptedRemediator()

    # When: refetch and recompose run before the attempt-two aligned verdict.
    result = await alignment_loop.ensure_aligned_v2_response(candidate, remediator)

    # Then: providers ran once, rch1/identity never changed, and boundaries stayed narrow.
    assert (static.calls, document.calls) == (1, 1)
    assert result.recall_state is state
    result_state = result.recall_state
    assert result_state is not None
    assert result_state.outcome is state.outcome
    assert result_state.outcome.trace_hash == state.outcome.trace_hash
    assert all(item is state for item in remediator.observed_states)
    assert all(item is state.outcome for item in remediator.observed_outcomes)
    assert set(remediator.observed_hashes) == {state.outcome.trace_hash}
    assert result.plan.units[0].evidence_query == "initial company website query"
    assert remediator.refetch_request is not None
    assert remediator.refetch_request.refined_evidence_query == (
        "refined company website query"
    )
    assert result.response.reply.text == "recomposed"
    assert result.plan is remediator.recompose_plan
    assert result.evidence is remediator.recompose_evidence


def _candidate(
    request: KernelReplyRequestV1,
    policy: PolicyManifestV2,
    scope: BusinessScopeAuthorityV1,
    state: RecallTurnStateV1,
) -> V2AttemptResult:
    plan = finalize_execution_plan_v2(
        DeterministicPlanOriginInputV1(
            user_need="answer company question",
            units=(
                DeterministicPlanUnitV1(
                    unit_id="company-answer",
                    manifest_id="answer_internal_company_knowledge",
                    answerability_policy="answer",
                    evidence_query="initial company website query",
                ),
            ),
            compliance_reason_code="compliant_product_request",
            confidence=0.9,
        ),
        policy,
        scope,
        origin="deterministic",
    )
    evidence = CanonicalEvidenceExecutionResultV1(
        preflight=AdapterPreflightSnapshot.empty(),
        canonical_facts=(),
        resolve_bindings=(),
        groundings=(),
        domain_context=DomainContextV1Builder().build(request, scope_authority=scope),
    )
    return V2AttemptResult(
        plan=plan,
        evidence=evidence,
        response=ReplyResponse(
            reply=PrimaryReply(kind="answer", text="initial", mentions=[]),
            actions=[],
        ),
        reply_validation=V2ReplyValidationResult(valid=True),
        reason_code="compliant_product_request",
        recall_state=state,
    )
