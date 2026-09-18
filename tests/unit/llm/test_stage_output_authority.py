from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from market_support_crewai_agent.runtime.planning import finalize_execution_plan_v2
from market_support_crewai_agent.runtime.recall.approved_static_knowledge import (
    _validate_selection,
    approved_knowledge_manifest,
)
from market_support_crewai_agent.runtime.recall.approved_static_selector import (
    ApprovedKnowledgeSelection,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    V2ComposerOutputRejected,
    _validate_v2_composer_output,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.v2_attempt import (
    V2AttemptResult,
    _validate_v2_reply,
)
from market_support_crewai_agent.runtime.validation.alignment_loop import (
    ensure_aligned_v2_response,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas.reply import (
    PrimaryReply,
    ReplyMention,
    ReplyResponse,
    SendWeeklyReportAction,
)
from tests.unit.llm._stage_input_widening_cases import (
    composer_finalizer_control,
    planner_finalizer_control,
    selected_refs,
    verifier_candidate,
)


def test_planner_output_cannot_widen_policy_eligible_refs() -> None:
    # Given: one eligible neutral plan and a schema-valid plan selecting another ref.
    control = planner_finalizer_control()
    eligible = frozenset(
        ref.manifest_id for ref in control.policy.eligible_capabilities
    )

    # When: the deterministic planner finalizer processes both outputs.
    neutral = finalize_execution_plan_v2(
        control.neutral,
        control.policy,
        control.scope,
        origin="planner",
    )

    # Then: the neutral selection stays eligible and the widening output is rejected.
    assert selected_refs(neutral) <= eligible
    with pytest.raises(ValueError, match="plan_spec_capability_not_policy_eligible"):
        finalize_execution_plan_v2(
            control.widening,
            control.policy,
            control.scope,
            origin="planner",
        )
    assert (
        frozenset(ref.manifest_id for ref in control.policy.eligible_capabilities)
        == eligible
    )


def test_composer_output_cannot_widen_admitted_evidence() -> None:
    # Given: a composer input with one admitted evidence ID and two typed outputs.
    control = composer_finalizer_control()
    admitted = control.input_value.unit_groundings[0].allowed_evidence_ids

    # When: the deterministic composer finalizer processes the neutral output.
    _validate_v2_composer_output(control.neutral, control.input_value)

    # Then: the extra evidence citation rejects without changing admitted authority.
    with pytest.raises(V2ComposerOutputRejected, match="unselected_evidence"):
        _validate_v2_composer_output(control.evidence_widening, control.input_value)
    assert control.input_value.unit_groundings[0].allowed_evidence_ids == admitted


def test_composer_output_cannot_widen_reply_character_ceiling() -> None:
    # Given: one output inside the fixed ceiling and one output one character above it.
    control = composer_finalizer_control()
    ceiling = control.input_value.output_ceilings.max_reply_chars

    # When: the deterministic composer finalizer processes the neutral output.
    _validate_v2_composer_output(control.neutral, control.input_value)

    # Then: the oversized output rejects and the input ceiling stays byte-identical.
    with pytest.raises(V2ComposerOutputRejected, match="reply_text_too_long"):
        _validate_v2_composer_output(control.ceiling_widening, control.input_value)
    assert control.input_value.output_ceilings.max_reply_chars == ceiling


def test_selector_output_cannot_widen_candidate_or_media_authority() -> None:
    # Given: one selected catalog row and its owned image asset.
    candidate = next(
        item for item in approved_knowledge_manifest() if item.image_assets
    )
    asset_id = candidate.image_assets[0].asset_id
    neutral_output = ApprovedKnowledgeSelection(
        selected_entry_ids=(candidate.entry_id,),
        selected_image_asset_ids=(asset_id,),
        confidence="high",
        rationale="neutral",
    )
    widening_output = neutral_output.model_copy(
        update={
            "selected_entry_ids": (candidate.entry_id, "entry:not-a-candidate"),
            "selected_image_asset_ids": (asset_id, "asset:not-a-candidate"),
        }
    )

    # When: the deterministic selector finalizer filters both controlled outputs.
    neutral = _validate_selection(neutral_output, max_entries=1, max_images=1)
    widened = _validate_selection(widening_output, max_entries=1, max_images=1)

    # Then: attempted candidate and media widening preserves the neutral authority.
    assert widened == neutral


def test_reply_postconditions_reject_actions_mentions_and_unapproved_media() -> None:
    # Given: a valid direct answer and one output adding all three effects.
    control = planner_finalizer_control()
    plan = finalize_execution_plan_v2(
        control.neutral,
        control.policy,
        control.scope,
        origin="planner",
    )
    neutral = ReplyResponse(reply=PrimaryReply(kind="answer", text="grounded answer"))
    widening = ReplyResponse(
        reply=PrimaryReply(
            kind="answer",
            text="grounded answer %%unapproved.png%%",
            mentions=[ReplyMention(type="sales", reason="widen")],
        ),
        actions=[
            SendWeeklyReportAction(
                type="send_weekly_report",
                resolve_type="weekly_report",
                resolve_ref="weekly:forged",
                period="2026-W29",
                report_date="2026-07-18",
            )
        ],
    )

    # When: the final reply postcondition validator processes both outputs.
    neutral_effects = _validate_v2_reply(control.request, control.policy, plan, neutral)
    widening_effects = _validate_v2_reply(
        control.request, control.policy, plan, widening
    )

    # Then: neutral passes while action, mention, and media authority all reject.
    assert neutral_effects.valid is True
    assert {
        "reply_actions_outside_policy",
        "reply_mentions_outside_policy",
        "direct_reply_has_image_marker",
    } <= set(widening_effects.issues)


@dataclass(frozen=True, slots=True)
class _VerifierRemediator:
    widen_recompose: bool

    async def verdict_for(
        self,
        candidate: V2AttemptResult,
        attempt: int,
    ) -> ReplyAlignmentVerdict:
        del candidate, attempt
        if self.widen_recompose:
            return ReplyAlignmentVerdict(
                aligned=False,
                safe_to_return=False,
                failure_code="composer_drift",
                remediation="recompose",
            )
        return ReplyAlignmentVerdict(aligned=True, safe_to_return=True)

    async def replan(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: int,
    ) -> V2AttemptResult:
        del verdict, attempt
        return candidate

    async def refetch(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: int,
    ):
        del candidate, verdict, attempt
        raise AssertionError("refetch is not part of this scenario")

    async def recompose(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: int,
    ) -> V2AttemptResult:
        del verdict, attempt
        return replace(candidate, plan=candidate.plan.model_copy())

    async def replace(self, candidate, verdict, remediation):
        del verdict, remediation
        return candidate


@pytest.mark.anyio
async def test_verifier_output_cannot_relax_recompose_postconditions() -> None:
    # Given: one candidate with a stable recall state and neutral/malicious verdicts.
    candidate = verifier_candidate()

    # When: the neutral verifier output is finalized by the alignment loop.
    neutral = await ensure_aligned_v2_response(
        candidate, _VerifierRemediator(widen_recompose=False)
    )

    # Then: neutral preserves identity and malicious plan replacement is rejected.
    assert neutral is candidate
    with pytest.raises(AgentRuntimeError, match="recompose_changed_plan_or_evidence"):
        await ensure_aligned_v2_response(
            candidate, _VerifierRemediator(widen_recompose=True)
        )
