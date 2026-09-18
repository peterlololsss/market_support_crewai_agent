from __future__ import annotations

from dataclasses import replace
from typing import Literal, assert_never

import pytest

from market_support_crewai_agent.runtime.planning import finalize_execution_plan_v2
from market_support_crewai_agent.runtime.planning.compiler import (
    DeterministicPlanOriginInputV1,
    DeterministicPlanUnitV1,
)
from market_support_crewai_agent.runtime.v2_attempt import V2AttemptResult
from market_support_crewai_agent.runtime.validation.alignment_loop import (
    AlignmentRemediationLimitsV1,
    ensure_aligned_v2_response,
)
from market_support_crewai_agent.runtime.validation.alignment_postconditions import (
    AlignmentRefetchOutcomeV1,
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

_LimitKind = Literal["total", "replan", "refetch", "recompose"]
_ActionKind = Literal["replan", "refetch", "recompose"]


@pytest.mark.anyio
@pytest.mark.parametrize("limit_kind", ("total", "replan", "refetch", "recompose"))
@pytest.mark.parametrize("limit_value", (0, 1, 2))
async def test_alignment_loop_executes_each_limit_at_zero_one_two(
    limit_kind: _LimitKind,
    limit_value: int,
) -> None:
    candidate = verifier_candidate()
    action_kind = _action_for_limit(limit_kind)
    remediator = _LimitMatrixRemediator(action_kind)

    result = await ensure_aligned_v2_response(
        candidate,
        remediator,
        _limits_for(limit_kind, limit_value),
    )

    expected_actions = limit_value
    assert len(remediator.action_attempts[action_kind]) == expected_actions
    assert remediator.action_attempts[action_kind] == list(
        range(1, expected_actions + 1)
    )
    assert remediator.verifier_attempts == list(range(expected_actions + 1))
    assert all(0 <= attempt <= 2 for attempt in remediator.verifier_attempts)
    assert result.plan.origin == "remediation"


def _limits_for(
    limit_kind: _LimitKind,
    limit_value: int,
) -> AlignmentRemediationLimitsV1:
    match limit_kind:
        case "total":
            return AlignmentRemediationLimitsV1(
                limit_value,
                limit_value,
                limit_value,
                limit_value,
            )
        case "replan":
            return AlignmentRemediationLimitsV1(limit_value, 2, 2, 2)
        case "refetch":
            return AlignmentRemediationLimitsV1(2, limit_value, 2, 2)
        case "recompose":
            return AlignmentRemediationLimitsV1(2, 2, limit_value, 2)
        case unreachable:
            assert_never(unreachable)


def _action_for_limit(limit_kind: _LimitKind) -> _ActionKind:
    match limit_kind:
        case "total":
            return "recompose"
        case "replan" | "refetch" | "recompose":
            return limit_kind
        case unreachable:
            assert_never(unreachable)


class _LimitMatrixRemediator:
    action_kind: _ActionKind
    verifier_attempts: list[int]
    action_attempts: dict[_ActionKind, list[int]]

    def __init__(self, action_kind: _ActionKind) -> None:
        self.action_kind = action_kind
        self.verifier_attempts: list[int] = []
        self.action_attempts: dict[_ActionKind, list[int]] = {
            "replan": [],
            "refetch": [],
            "recompose": [],
        }

    async def verdict_for(
        self,
        candidate: V2AttemptResult,
        attempt: int,
    ) -> ReplyAlignmentVerdict:
        del candidate
        self.verifier_attempts.append(attempt)
        match self.action_kind:
            case "refetch":
                return ReplyAlignmentVerdict(
                    aligned=False,
                    safe_to_return=False,
                    failure_code="missing_evidence",
                    remediation="refetch_document_context",
                    refined_evidence_query="refined query",
                )
            case "replan":
                return ReplyAlignmentVerdict(
                    aligned=False,
                    safe_to_return=False,
                    failure_code="wrong_intent",
                    remediation="replan",
                )
            case "recompose":
                return ReplyAlignmentVerdict(
                    aligned=False,
                    safe_to_return=False,
                    failure_code="wrong_intent",
                    remediation="recompose",
                )
            case unreachable:
                assert_never(unreachable)

    async def replan(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: int,
    ) -> V2AttemptResult:
        del verdict
        self.action_attempts["replan"].append(attempt)
        return replace(candidate, plan=candidate.plan.model_copy())

    async def refetch(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: int,
    ) -> AlignmentRefetchOutcomeV1:
        self.action_attempts["refetch"].append(attempt)
        request = AlignmentRefetchRequestV1(
            unit_id=candidate.plan.units[0].unit_id,
            manifest_ref=candidate.plan.units[0].manifest_ref,
            refined_evidence_query=verdict.refined_evidence_query or "refined query",
            attempt=attempt,
        )
        return AlignmentRefetchOutcomeV1(
            request=request,
            candidate=replace(candidate, evidence=replace(candidate.evidence)),
        )

    async def recompose(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: int,
    ) -> V2AttemptResult:
        del verdict
        self.action_attempts["recompose"].append(attempt)
        return replace(
            candidate,
            response=ReplyResponse(
                reply=PrimaryReply(kind="answer", text=f"recomposed-{attempt}")
            ),
        )

    async def replace(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict | None,
        remediation: Literal["return_clarification", "return_unable"],
    ) -> V2AttemptResult:
        del verdict
        control = planner_finalizer_control()
        match remediation:
            case "return_clarification":
                manifest_id = "general.clarification"
                answerability_policy = "clarify"
            case "return_unable":
                manifest_id = "general.abstention"
                answerability_policy = "abstain"
            case unreachable:
                assert_never(unreachable)
        plan = finalize_execution_plan_v2(
            DeterministicPlanOriginInputV1(
                user_need=candidate.plan.user_need,
                units=(
                    DeterministicPlanUnitV1(
                        unit_id="alignment-remediation",
                        manifest_id=manifest_id,
                        answerability_policy=answerability_policy,
                    ),
                ),
                compliance_reason_code=candidate.plan.compliance.reason_code,
            ),
            control.policy,
            control.scope,
            origin="remediation",
        )
        return replace(
            candidate,
            plan=plan,
            response=ReplyResponse(
                reply=PrimaryReply(kind="unable_to_answer", text="fallback")
            ),
        )
