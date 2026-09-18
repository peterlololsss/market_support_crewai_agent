from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, assert_never

from market_support_crewai_agent.runtime.recall.turn_state import RecallTurnStateV1
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.v2_attempt import V2AttemptResult
from market_support_crewai_agent.runtime.validation.alignment_postconditions import (
    AlignmentRefetchOutcomeV1,
    require_recompose_response_only,
    require_refetch_authority,
    require_replan_authority,
    require_replacement_authority,
    require_same_recall_state,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.settings_model import Settings

AlignmentAttemptV1 = Literal[0, 1, 2]
AlignmentActionAttemptV1 = Literal[1, 2]
AlignmentFallbackV1 = Literal["return_clarification", "return_unable"]
_ALIGNMENT_ATTEMPTS: tuple[AlignmentAttemptV1, ...] = (0, 1, 2)
_ACTION_ATTEMPTS: tuple[AlignmentActionAttemptV1, ...] = (1, 2)


@dataclass(frozen=True, slots=True)
class AlignmentLimitError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class AlignmentRemediationLimitsV1:
    max_replans: int
    max_refetches: int
    max_recomposes: int
    max_total: int

    def __post_init__(self) -> None:
        if any(
            value < 0 or value > 2
            for value in (
                self.max_replans,
                self.max_refetches,
                self.max_recomposes,
                self.max_total,
            )
        ):
            raise AlignmentLimitError("alignment_remediation_limit_out_of_range")
        if any(
            value > self.max_total
            for value in (
                self.max_replans,
                self.max_refetches,
                self.max_recomposes,
            )
        ):
            raise AlignmentLimitError("alignment_action_limit_exceeds_total")


class V2AlignmentRemediator(Protocol):
    async def verdict_for(
        self,
        candidate: V2AttemptResult,
        attempt: AlignmentAttemptV1,
    ) -> ReplyAlignmentVerdict: ...

    async def replan(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: AlignmentActionAttemptV1,
    ) -> V2AttemptResult: ...

    async def refetch(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: AlignmentActionAttemptV1,
    ) -> AlignmentRefetchOutcomeV1: ...

    async def recompose(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: AlignmentActionAttemptV1,
    ) -> V2AttemptResult: ...

    async def replace(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict | None,
        remediation: AlignmentFallbackV1,
    ) -> V2AttemptResult: ...


def alignment_limits_from_settings(
    settings: Settings,
) -> AlignmentRemediationLimitsV1:
    return AlignmentRemediationLimitsV1(
        max_replans=settings.reply_alignment_max_replans,
        max_refetches=settings.reply_alignment_max_evidence_refetches,
        max_recomposes=settings.reply_alignment_max_recomposes,
        max_total=settings.reply_alignment_max_total_remediations,
    )


async def ensure_aligned_v2_response(
    candidate: V2AttemptResult,
    remediator: V2AlignmentRemediator,
    limits: AlignmentRemediationLimitsV1 = AlignmentRemediationLimitsV1(
        max_replans=2,
        max_refetches=2,
        max_recomposes=2,
        max_total=2,
    ),
) -> V2AttemptResult:
    recall_state = candidate.recall_state
    if recall_state is None:
        return candidate

    replan_count = 0
    refetch_count = 0
    recompose_count = 0
    total = 0
    while True:
        attempt = _alignment_attempt(total)
        try:
            verdict = await remediator.verdict_for(candidate, attempt)
        except (AgentRuntimeError, TimeoutError):
            match candidate.plan.response_mode:
                case "refusal":
                    return candidate
                case (
                    "action"
                    | "clarification"
                    | "handoff"
                    | "unable"
                    | "knowledge_answer"
                    | "smalltalk"
                    | "no_reply"
                ):
                    return await _replace(
                        candidate,
                        remediator,
                        recall_state,
                        verdict=None,
                        remediation="return_unable",
                    )
                case unreachable:
                    assert_never(unreachable)
        if verdict.aligned and verdict.safe_to_return:
            return candidate
        if total >= limits.max_total:
            return await _replace(
                candidate,
                remediator,
                recall_state,
                verdict=verdict,
                remediation="return_unable",
            )

        previous = candidate
        match verdict.remediation:
            case "replan" if replan_count < limits.max_replans:
                replan_count += 1
                total += 1
                try:
                    candidate = await remediator.replan(
                        previous,
                        verdict,
                        _action_attempt(replan_count),
                    )
                except (AgentRuntimeError, TimeoutError):
                    return await _replace(
                        previous,
                        remediator,
                        recall_state,
                        verdict=verdict,
                        remediation="return_unable",
                    )
                require_replan_authority(previous, candidate)
            case "refetch_document_context" | "refetch_report_scope" if (
                refetch_count < limits.max_refetches
            ):
                refetch_count += 1
                total += 1
                outcome = await remediator.refetch(
                    previous,
                    verdict,
                    _action_attempt(refetch_count),
                )
                candidate = outcome.candidate
                require_refetch_authority(previous, outcome, verdict)
            case "recompose" if recompose_count < limits.max_recomposes:
                recompose_count += 1
                total += 1
                candidate = await remediator.recompose(
                    previous,
                    verdict,
                    _action_attempt(recompose_count),
                )
                require_recompose_response_only(previous, candidate)
            case "return_clarification":
                return await _replace(
                    previous,
                    remediator,
                    recall_state,
                    verdict=verdict,
                    remediation="return_clarification",
                )
            case (
                "none"
                | "return_unable"
                | "replan"
                | "refetch_document_context"
                | "refetch_report_scope"
                | "recompose"
            ):
                return await _replace(
                    previous,
                    remediator,
                    recall_state,
                    verdict=verdict,
                    remediation="return_unable",
                )
            case unreachable:
                assert_never(unreachable)
        require_same_recall_state(candidate, recall_state)
        if not candidate.reply_validation.valid:
            return candidate


async def _replace(
    candidate: V2AttemptResult,
    remediator: V2AlignmentRemediator,
    recall_state: RecallTurnStateV1,
    *,
    verdict: ReplyAlignmentVerdict | None,
    remediation: AlignmentFallbackV1,
) -> V2AttemptResult:
    replacement = await remediator.replace(candidate, verdict, remediation)
    require_replacement_authority(candidate, replacement, remediation)
    require_same_recall_state(replacement, recall_state)
    return replacement


def _alignment_attempt(value: int) -> AlignmentAttemptV1:
    if value < 0 or value > 2:
        raise AgentRuntimeError("alignment_attempt_out_of_range")
    return _ALIGNMENT_ATTEMPTS[value]


def _action_attempt(value: int) -> AlignmentActionAttemptV1:
    if value < 1 or value > 2:
        raise AgentRuntimeError("alignment_action_attempt_out_of_range")
    return _ACTION_ATTEMPTS[value - 1]


__all__ = [
    "AlignmentActionAttemptV1",
    "AlignmentAttemptV1",
    "AlignmentFallbackV1",
    "AlignmentRefetchOutcomeV1",
    "AlignmentRemediationLimitsV1",
    "V2AlignmentRemediator",
    "alignment_limits_from_settings",
    "ensure_aligned_v2_response",
]
