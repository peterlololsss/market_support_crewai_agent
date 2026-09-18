from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.audit_records import (
    GroupAuditCommitProposalV1,
)
from market_support_crewai_agent.runtime.state.conversation_records import (
    AssistantTurnProposalV1,
    UserTurnProposalV1,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    FeedbackPreparationV1,
    FeedbackReceiptReplayV1,
    PreparedFeedbackV1,
)
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.runtime.state.transaction_records import (
    NewReplyReservationV1,
    ReplyCommitProposalV1,
)
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2
from market_support_crewai_agent.schemas.reply import (
    ReplyResponse,
    SendWeeklyReportAction,
)
from market_support_crewai_agent.schemas.type_ids import ActionExecutionStatus

_PREPARED_FEEDBACK_ADAPTER: TypeAdapter[PreparedFeedbackV1] = TypeAdapter(
    PreparedFeedbackV1
)


def make_feedback_state_key() -> ConversationStateKey:
    return ConversationStateKey(
        surface="wecom",
        adapter_namespace="test-adapter",
        tenant_ref="tenant:test",
        scene="group",
        subject_ref="group:test",
        principal_ref="principal:test",
    )


def _state_key() -> ConversationStateKey:
    return make_feedback_state_key()


def _commit_response(
    coordinator: ReplyStateTransactionCoordinatorV1,
    state_key: ConversationStateKey,
    action: dict[str, str] | None = None,
) -> ReplyResponse:
    reservation = coordinator.reserve_reply(
        state_key,
        "req:cross-version",
        "rqh1:test",
        True,
    )
    assert isinstance(reservation, NewReplyReservationV1)
    response = ReplyResponse.model_validate(
        {
            "reply": {"kind": "answer", "text": "已安排发送。"},
            "actions": [
                action
                or {
                    "type": "send_weekly_report",
                    "resolve_type": "weekly_report",
                    "resolve_ref": "weekly:resolve",
                    "period": "20260717",
                    "report_date": "2026-07-17",
                }
            ],
        }
    )
    return coordinator.commit_reply(
        ReplyCommitProposalV1(
            state_key=state_key,
            request_id=reservation.request_id,
            request_hash=reservation.request_hash,
            replay_eligible=reservation.replay_eligible,
            owner_token=reservation.owner_token,
            expected_revision_epoch=reservation.revision_epoch,
            expected_state_revision=reservation.state_revision,
            response=response,
            pol1="pol1:test",
            par1="par1:test",
            grh1="grh1:test",
            bsh1="bsh1:test",
            user_turn=UserTurnProposalV1(text="请发周报"),
            assistant_turn=AssistantTurnProposalV1(
                text="已安排发送。",
                reply_kind="answer",
                clarification_requested=False,
            ),
            clarification=None,
            audit=GroupAuditCommitProposalV1(
                reason_code="action_ready",
                manifest_refs=("weekly_report.send@2026-07-15.1",),
                program_attempts=(),
            ),
        )
    )


def _v2_feedback(
    response: ReplyResponse,
    feedback_id: str,
    status: ActionExecutionStatus,
) -> ActionFeedbackRequestV2:
    action = SendWeeklyReportAction.model_validate(response.actions[0].model_dump())
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
                    "action_type": "send_weekly_report",
                    "status": status,
                    "action_id": action.action_id,
                    "artifact": {
                        "type": "weekly_report",
                        "resolve_ref": action.resolve_ref,
                        "period": action.period,
                        "report_date": action.report_date,
                    },
                }
            ],
        }
    )


def _require_prepared(preparation: FeedbackPreparationV1) -> PreparedFeedbackV1:
    try:
        return _PREPARED_FEEDBACK_ADAPTER.validate_python(preparation)
    except ValidationError as exc:
        raise AssertionError("unexpected feedback replay") from exc


def test_v2_feedback_replay_is_idempotent_by_feedback_id() -> None:
    coordinator = ReplyStateTransactionCoordinatorV1()
    state_key = _state_key()
    response = _commit_response(coordinator, state_key)
    feedback = _v2_feedback(response, "fb:v2-failed", "failed")
    prepared = _require_prepared(coordinator.prepare_feedback(state_key, feedback))
    assert coordinator.commit_feedback(prepared).stored == 1

    snapshot = coordinator.read_turn_admission_snapshot(
        state_key,
        "par1:test",
        "grh1:test",
        "bsh1:test",
    )

    assert len(snapshot.ledger_rows) == 1
    assert snapshot.ledger_rows[0].contract_version == "action-ledger-record.v2"
    assert snapshot.ledger_rows[0].status == "failed"
    assert isinstance(
        coordinator.prepare_feedback(state_key, feedback),
        FeedbackReceiptReplayV1,
    )


def test_v2_conflict_and_unknown_issued_feedback_do_not_mutate_effect_history() -> None:
    coordinator = ReplyStateTransactionCoordinatorV1()
    state_key = _state_key()
    response = _commit_response(coordinator, state_key)
    prepared = _require_prepared(
        coordinator.prepare_feedback(
            state_key,
            _v2_feedback(response, "fb:v2-failed", "failed"),
        )
    )
    _ = coordinator.commit_feedback(prepared)
    revision = coordinator.root_revision()

    with pytest.raises(CoordinatorError, match="feedback_effect_conflict"):
        _ = coordinator.prepare_feedback(
            state_key,
            _v2_feedback(response, "fb:v2-skipped", "skipped"),
        )

    assert coordinator.root_revision() == revision
    unknown = _v2_feedback(response, "fb:v2-unknown", "failed").model_copy(
        update={"response_id": "resp-" + "0" * 32}
    )
    with pytest.raises(CoordinatorError, match="issued_response_not_found"):
        _ = coordinator.prepare_feedback(state_key, unknown)

    assert coordinator.root_revision() == revision
