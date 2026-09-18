from __future__ import annotations

from dataclasses import replace

import pytest

from market_support_crewai_agent.runtime.planning import finalize_execution_plan_v2
from market_support_crewai_agent.runtime.planning.compiler import (
    DeterministicPlanOriginInputV1,
    DeterministicPlanUnitV1,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.validation.alignment_loop import (
    AlignmentRefetchOutcomeV1,
    AlignmentRemediationLimitsV1,
    ensure_aligned_v2_response,
)
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    AlignmentRefetchRequestV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from tests.unit.llm._stage_input_widening_cases import (
    planner_finalizer_control,
    verifier_candidate,
)


def test_aligned_verdict_shape_requires_safe_none_fields():
    verdict = ReplyAlignmentVerdict(
        aligned=True,
        safe_to_return=True,
        failure_code="none",
        remediation="none",
    )

    assert verdict.contract_version == "reply-alignment-verdict"
    assert verdict.aligned is True


def test_aligned_verdict_rejects_non_none_failure_code():
    with pytest.raises(ValueError):
        ReplyAlignmentVerdict(
            aligned=True,
            safe_to_return=True,
            failure_code="wrong_action",
            remediation="none",
        )


def test_aligned_verdict_requires_safe_to_return():
    with pytest.raises(ValueError):
        ReplyAlignmentVerdict(
            aligned=True,
            safe_to_return=False,
            failure_code="none",
            remediation="none",
        )


def test_refetch_document_context_requires_refined_query():
    with pytest.raises(ValueError):
        ReplyAlignmentVerdict(
            aligned=False,
            safe_to_return=False,
            failure_code="missing_evidence",
            remediation="refetch_document_context",
        )


def test_refetch_report_scope_requires_refined_query():
    with pytest.raises(ValueError):
        ReplyAlignmentVerdict(
            aligned=False,
            safe_to_return=False,
            failure_code="missing_evidence",
            remediation="refetch_report_scope",
        )


@pytest.mark.anyio
async def test_mixed_remediations_stop_after_two_total_iterations() -> None:
    # Given: three different requested actions under a shared two-iteration budget.
    candidate = verifier_candidate()
    remediator = _MixedRemediator()

    # When: recompose and refetch consume the budget before the requested replan.
    result = await ensure_aligned_v2_response(
        candidate,
        remediator,
        AlignmentRemediationLimitsV1(
            max_replans=2,
            max_refetches=2,
            max_recomposes=2,
            max_total=2,
        ),
    )

    # Then: attempts are exactly 0..2 and no action counter exceeds the total.
    assert remediator.verify_attempts == [0, 1, 2]
    assert remediator.action_attempts == {
        "replan": [],
        "refetch": [1],
        "recompose": [1],
    }
    assert sum(len(values) for values in remediator.action_attempts.values()) == 2
    assert result.plan.origin == "remediation"
    assert result.plan.selected_manifest_refs[0].manifest_id == "general.abstention"


@pytest.mark.anyio
async def test_replan_must_replace_the_prior_plan_object() -> None:
    # Given: a verifier-requested replan whose remediator returns the prior candidate.
    candidate = verifier_candidate()
    remediator = _SamePlanReplanner()

    # When/Then: plan identity cannot be preserved through a replan remediation.
    with pytest.raises(AgentRuntimeError, match="replan_invalid_plan_authority"):
        await ensure_aligned_v2_response(
            candidate,
            remediator,
            AlignmentRemediationLimitsV1(1, 0, 0, 1),
        )


class _MixedRemediator:
    def __init__(self) -> None:
        self.verify_attempts: list[int] = []
        self.action_attempts: dict[str, list[int]] = {
            "replan": [],
            "refetch": [],
            "recompose": [],
        }

    async def verdict_for(self, candidate, attempt):
        del candidate
        self.verify_attempts.append(attempt)
        verdicts = (
            ReplyAlignmentVerdict(
                aligned=False,
                safe_to_return=False,
                failure_code="composer_drift",
                remediation="recompose",
            ),
            ReplyAlignmentVerdict(
                aligned=False,
                safe_to_return=False,
                failure_code="missing_evidence",
                remediation="refetch_document_context",
                refined_evidence_query="refined query",
            ),
            ReplyAlignmentVerdict(
                aligned=False,
                safe_to_return=False,
                failure_code="wrong_intent",
                remediation="replan",
            ),
        )
        return verdicts[attempt]

    async def replan(self, candidate, verdict, attempt):
        del verdict
        self.action_attempts["replan"].append(attempt)
        return replace(candidate, plan=candidate.plan.model_copy())

    async def refetch(self, candidate, verdict, attempt):
        self.action_attempts["refetch"].append(attempt)
        request = AlignmentRefetchRequestV1(
            unit_id=candidate.plan.units[0].unit_id,
            manifest_ref=candidate.plan.units[0].manifest_ref,
            refined_evidence_query=verdict.refined_evidence_query or "",
            attempt=attempt,
        )
        return AlignmentRefetchOutcomeV1(
            request=request,
            candidate=replace(candidate, evidence=replace(candidate.evidence)),
        )

    async def recompose(self, candidate, verdict, attempt):
        del verdict
        self.action_attempts["recompose"].append(attempt)
        return replace(
            candidate,
            response=ReplyResponse(
                reply=PrimaryReply(kind="answer", text="recomposed")
            ),
        )

    async def replace(self, candidate, verdict, remediation):
        del verdict
        return _replacement(candidate, remediation)


class _SamePlanReplanner(_MixedRemediator):
    async def verdict_for(self, candidate, attempt):
        del candidate, attempt
        return ReplyAlignmentVerdict(
            aligned=False,
            safe_to_return=False,
            failure_code="wrong_intent",
            remediation="replan",
        )

    async def replan(self, candidate, verdict, attempt):
        del verdict, attempt
        return candidate


def _replacement(candidate, remediation):
    control = planner_finalizer_control()
    is_clarification = remediation == "return_clarification"
    plan = finalize_execution_plan_v2(
        DeterministicPlanOriginInputV1(
            user_need=candidate.plan.user_need,
            units=(
                DeterministicPlanUnitV1(
                    unit_id="alignment-remediation",
                    manifest_id=(
                        "general.clarification"
                        if is_clarification
                        else "general.abstention"
                    ),
                    answerability_policy=("clarify" if is_clarification else "abstain"),
                ),
            ),
            compliance_reason_code=candidate.plan.compliance.reason_code,
        ),
        control.policy,
        control.scope,
        origin="remediation",
    )
    reply_kind = "clarification" if is_clarification else "unable_to_answer"
    return replace(
        candidate,
        plan=plan,
        response=ReplyResponse(reply=PrimaryReply(kind=reply_kind, text="fallback")),
    )
