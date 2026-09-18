from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.planning.planner_input import (
    PlannerRuntimeInputSourceV1,
    build_runtime_planner_prompt_input_v1,
)
from market_support_crewai_agent.runtime.planning.planner_retry import (
    planner_alignment_replan_overlay,
    planner_schema_repair_allowed,
)
from market_support_crewai_agent.runtime.prompts.context import IntentGateResult
from market_support_crewai_agent.runtime.recall.flow import collect_preplanner_recall
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from tests.integration.runtime._recall_pipeline_fixtures import (
    direct_context,
    document_candidate,
    static_hit,
)
from tests.integration.runtime._recall_pipeline_runtime_fixtures import (
    DocumentCollector,
    RecallPipelineRuntime,
    StaticCollector,
)


@pytest.mark.parametrize(("phase_attempt", "overall_attempt"), ((1, 2), (2, 3)))
def test_alignment_replan_overlay_uses_closed_attempt_mapping(
    phase_attempt: int,
    overall_attempt: int,
) -> None:
    verdict = ReplyAlignmentVerdict(
        aligned=False,
        safe_to_return=False,
        failure_code="wrong_intent",
        remediation="replan",
        planner_feedback="select the eligible knowledge capability",
    )

    overlay = planner_alignment_replan_overlay(verdict, phase_attempt)

    assert overlay is not None
    assert (overlay.attempt, overlay.phase, overlay.phase_attempt) == (
        overall_attempt,
        "alignment_replan",
        phase_attempt,
    )
    assert not planner_schema_repair_allowed(overlay)


def test_alignment_replan_rejects_a_fourth_planner_attempt() -> None:
    verdict = ReplyAlignmentVerdict(
        aligned=False,
        safe_to_return=False,
        failure_code="wrong_intent",
        remediation="replan",
    )

    with pytest.raises(
        ContextViewInvariantError,
        match="planner_alignment_replan_mapping_invalid",
    ):
        _ = planner_alignment_replan_overlay(verdict, 3)


@pytest.mark.anyio
async def test_unscoped_direct_projection_has_no_sentinel_authority() -> None:
    envelope, policy, scope = direct_context()
    transition = await collect_preplanner_recall(
        RecallPipelineRuntime(
            StaticCollector(static_hit(policy, with_payload=False)),
            DocumentCollector(document_candidate()),
        ),
        request=envelope.request,
        policy=policy,
        scope_authority=scope,
    )
    now = datetime(2026, 7, 18, 15, 4, 5, tzinfo=ZoneInfo("Asia/Shanghai"))

    planner_input = build_runtime_planner_prompt_input_v1(
        PlannerRuntimeInputSourceV1(
            request=envelope.request,
            policy=policy,
            scope_authority=scope,
            intent_gate=IntentGateResult(artifact_hint="unclear"),
            history=(ConversationMessage("user", "x" * 1_500, now),),
            action_history=(),
            recall_state=transition.turn_state,
            now=now,
        )
    )
    payload = planner_input.model_dump(mode="json")

    assert payload["business_scope"] == {"kind": "unscoped"}
    assert planner_input.effective_policy.outbound_actions == ()
    assert planner_input.effective_policy.mention_types == ()
    assert planner_input.effective_policy.adapter_resolves == ()
    assert planner_input.runtime_clock.current_datetime == "2026-07-18T15:04:05+08:00"
    assert len(planner_input.history[0].text) == 1_200
    assert envelope.request.identity.subject_ref not in planner_input.model_dump_json()
