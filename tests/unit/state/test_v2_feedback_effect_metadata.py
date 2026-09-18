from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.runtime.state.effect_records import (
    FeedbackReceiptReplayV1,
)
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2
from market_support_crewai_agent.schemas.reply import ReplyResponse
from tests.unit.state.test_feedback_effect_coalescing import (
    _commit_response,
    _state_key,
)

_ACTIONS = {
    "send_material_pack": {
        "type": "send_material_pack",
        "resolve_type": "material_pack",
        "resolve_ref": "material:resolve",
        "material_pack_option": "growth",
    },
    "send_weekly_report": {
        "type": "send_weekly_report",
        "resolve_type": "weekly_report",
        "resolve_ref": "weekly:resolve",
        "period": "20260717",
        "report_date": "2026-07-17",
    },
    "send_monthly_report": {
        "type": "send_monthly_report",
        "resolve_type": "monthly_report",
        "resolve_ref": "monthly:resolve",
        "period": "202607",
        "report_date": "2026-07-01",
    },
}


def _feedback(response: ReplyResponse, feedback_id: str) -> ActionFeedbackRequestV2:
    action = response.actions[0]
    match action.type:
        case "send_material_pack":
            artifact = {
                "type": "material_pack",
                "resolve_ref": action.resolve_ref,
                "option": action.material_pack_option,
            }
        case "send_weekly_report" | "send_monthly_report":
            artifact = {
                "type": action.resolve_type,
                "resolve_ref": action.resolve_ref,
                "period": action.period,
                "report_date": action.report_date,
            }
        case unreachable:
            raise AssertionError(f"unexpected action type: {unreachable}")
    return ActionFeedbackRequestV2.model_validate(
        {
            "contract_version": "action-feedback.v2",
            "feedback_id": feedback_id,
            "request_id": "req:cross-version",
            "response_id": response.response_id,
            "identity": {
                "contract_version": "conversation-identity.v1",
                "surface": "wecom",
                "scene": "group",
                "tenant_ref": "tenant:test",
                "group_ref": "group:test",
                "principal_ref": "principal:test",
            },
            "executions": [
                {
                    "action_type": action.type,
                    "status": "failed",
                    "action_id": action.action_id,
                    "artifact": artifact,
                }
            ],
        }
    )


def _metadata_variant(
    feedback: ActionFeedbackRequestV2,
    field: str,
    value: str | None,
) -> ActionFeedbackRequestV2:
    execution = feedback.executions[0]
    assert execution.artifact is not None
    artifact = execution.artifact.model_copy(update={field: value})
    return feedback.model_copy(
        update={"executions": [execution.model_copy(update={"artifact": artifact})]}
    )


@pytest.mark.parametrize(
    ("action_type", "field", "substitute"),
    (
        ("send_material_pack", "resolve_ref", "material:other"),
        ("send_material_pack", "option", "value"),
        ("send_weekly_report", "resolve_ref", "weekly:other"),
        ("send_weekly_report", "period", "20260718"),
        ("send_weekly_report", "report_date", "2026-07-18"),
        ("send_monthly_report", "resolve_ref", "monthly:other"),
        ("send_monthly_report", "period", "202608"),
        ("send_monthly_report", "report_date", "2026-08-01"),
    ),
)
@pytest.mark.parametrize("value_kind", ("substituted", "omitted"))
def test_v2_feedback_rejects_each_frozen_issued_metadata_mismatch(
    action_type: str,
    field: str,
    substitute: str,
    value_kind: str,
) -> None:
    coordinator = ReplyStateTransactionCoordinatorV1()
    state_key = _state_key()
    response = _commit_response(coordinator, state_key, _ACTIONS[action_type])
    feedback = _feedback(response, "fb:metadata-mismatch")
    value = substitute if value_kind == "substituted" else None
    revision = coordinator.root_revision()

    with pytest.raises(CoordinatorError, match="feedback_effect_metadata_mismatch"):
        coordinator.prepare_feedback(
            state_key,
            _metadata_variant(feedback, field, value),
        )

    assert coordinator.root_revision() == revision


@pytest.mark.parametrize("action_type", tuple(_ACTIONS))
def test_v2_feedback_with_exact_issued_metadata_commits_and_replays(
    action_type: str,
) -> None:
    coordinator = ReplyStateTransactionCoordinatorV1()
    state_key = _state_key()
    response = _commit_response(coordinator, state_key, _ACTIONS[action_type])
    feedback = _feedback(response, "fb:metadata-replay")

    prepared = coordinator.prepare_feedback(state_key, feedback)
    assert coordinator.commit_feedback(prepared).stored == 1
    assert isinstance(
        coordinator.prepare_feedback(state_key, feedback), FeedbackReceiptReplayV1
    )


def test_v2_feedback_dto_rejects_duplicate_execution_effects() -> None:
    coordinator = ReplyStateTransactionCoordinatorV1()
    state_key = _state_key()
    response = _commit_response(coordinator, state_key, _ACTIONS["send_weekly_report"])
    feedback = _feedback(response, "fb:duplicate-dto")
    payload = feedback.model_dump(mode="json")
    payload["executions"] = [payload["executions"][0], payload["executions"][0]]

    with pytest.raises(
        ValidationError, match="feedback execution effects must be unique"
    ):
        ActionFeedbackRequestV2.model_validate(payload)


@pytest.mark.parametrize(
    "kind", ("duplicate", "response_with_action", "response_with_artifact")
)
def test_v2_feedback_prepare_rejects_bypassed_duplicate_or_malformed_effects(
    kind: str,
) -> None:
    coordinator = ReplyStateTransactionCoordinatorV1()
    state_key = _state_key()
    response = _commit_response(coordinator, state_key, _ACTIONS["send_weekly_report"])
    feedback = _feedback(response, "fb:duplicate-bypass")
    execution = feedback.executions[0]
    match kind:
        case "duplicate":
            bypassed = feedback.model_copy(
                update={"executions": [execution, execution]}
            )
            error_code = "feedback_effect_duplicate"
        case "response_with_action":
            bypassed = feedback.model_copy(
                update={
                    "executions": [
                        execution.model_copy(
                            update={"action_type": "send_text", "artifact": None}
                        )
                    ]
                }
            )
            error_code = "feedback_effect_mismatch"
        case "response_with_artifact":
            bypassed = feedback.model_copy(
                update={
                    "executions": [
                        execution.model_copy(
                            update={"action_type": "send_text", "action_id": None}
                        )
                    ]
                }
            )
            error_code = "feedback_effect_mismatch"
        case unreachable:
            raise AssertionError(f"unexpected bypass kind: {unreachable}")
    revision = coordinator.root_revision()

    with pytest.raises(CoordinatorError, match=error_code):
        coordinator.prepare_feedback(state_key, bypassed)

    assert coordinator.root_revision() == revision
    assert not coordinator._root.prepared_feedback_tokens
