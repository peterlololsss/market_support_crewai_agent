from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from market_support_crewai_agent.runtime.hashing import canonical_json_bytes
from market_support_crewai_agent.runtime.planning.models import execution_plan_id_v2
from market_support_crewai_agent.runtime.recall.turn_state import RecallTurnStateV1
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.v2_attempt import V2AttemptResult
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    AlignmentRefetchRequestV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)


@dataclass(frozen=True, slots=True)
class AlignmentRefetchOutcomeV1:
    request: AlignmentRefetchRequestV1
    candidate: V2AttemptResult


def require_same_recall_state(
    candidate: V2AttemptResult,
    expected: RecallTurnStateV1,
) -> None:
    actual = candidate.recall_state
    if actual is None:
        raise AgentRuntimeError("alignment_replaced_recall_turn_state")
    if (
        actual is not expected
        or actual.outcome is not expected.outcome
        or actual.outcome.trace_hash != expected.outcome.trace_hash
    ):
        raise AgentRuntimeError("alignment_replaced_recall_turn_state")


def require_replan_authority(
    before: V2AttemptResult,
    after: V2AttemptResult,
) -> None:
    plan = after.plan
    if (
        plan is before.plan
        or plan.origin != "planner"
        or plan.plan_spec is None
        or plan.execution_plan_id != execution_plan_id_v2(plan)
    ):
        raise AgentRuntimeError("alignment_replan_invalid_plan_authority")


def require_refetch_authority(
    before: V2AttemptResult,
    outcome: AlignmentRefetchOutcomeV1,
    verdict: ReplyAlignmentVerdict,
) -> None:
    after = outcome.candidate
    request = outcome.request
    if (
        request.refined_evidence_query != verdict.refined_evidence_query
        or request.unit_id not in {unit.unit_id for unit in before.plan.units}
        or before.plan is not after.plan
        or _plan_bytes(before) != _plan_bytes(after)
        or before.evidence is after.evidence
    ):
        raise AgentRuntimeError("alignment_refetch_changed_plan_authority")


def require_recompose_response_only(
    before: V2AttemptResult,
    after: V2AttemptResult,
) -> None:
    if before.plan is not after.plan or before.evidence is not after.evidence:
        raise AgentRuntimeError("alignment_recompose_changed_plan_or_evidence")


def require_replacement_authority(
    before: V2AttemptResult,
    after: V2AttemptResult,
    remediation: Literal["return_clarification", "return_unable"],
) -> None:
    expected_mode = (
        "clarification" if remediation == "return_clarification" else "unable"
    )
    expected_manifest = (
        "general.clarification"
        if remediation == "return_clarification"
        else "general.abstention"
    )
    plan = after.plan
    if (
        plan.origin != "remediation"
        or plan.plan_spec is not None
        or plan.response_mode != expected_mode
        or tuple(ref.manifest_id for ref in plan.selected_manifest_refs)
        != (expected_manifest,)
        or plan.execution_plan_id == before.plan.execution_plan_id
        or plan.execution_plan_id != execution_plan_id_v2(plan)
    ):
        raise AgentRuntimeError("alignment_replacement_invalid_plan_authority")


def _plan_bytes(candidate: V2AttemptResult) -> bytes:
    return canonical_json_bytes(
        candidate.plan.model_dump(mode="json", exclude_none=False)
    )


__all__ = [
    "AlignmentRefetchOutcomeV1",
    "require_recompose_response_only",
    "require_refetch_authority",
    "require_replan_authority",
    "require_replacement_authority",
    "require_same_recall_state",
]
