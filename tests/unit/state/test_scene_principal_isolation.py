from __future__ import annotations

from dataclasses import fields
from typing import assert_never

import pytest

from market_support_crewai_agent.runtime.context.payload_store import (
    ScopedContextPayloadStoreV1,
)
from market_support_crewai_agent.runtime.identity import (
    ConversationStateKey,
    state_key_ref,
)
from market_support_crewai_agent.runtime.state.audit_records import (
    DirectAuditCommitProposalV1,
    DirectAuditLengthV1,
    DirectDependencyCountV1,
    GroupAuditCommitProposalV1,
)
from market_support_crewai_agent.runtime.state.conversation_records import (
    AssistantTurnProposalV1,
    PendingClarificationProposalV1,
    UserTurnProposalV1,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    FeedbackReceiptReplayV1,
)
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.runtime.state.transaction_records import (
    CompletedReplyReplayV1,
    NewReplyReservationV1,
    ReplyCommitProposalV1,
)
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2
from market_support_crewai_agent.schemas.reply import ReplyResponse

PAR1, GRH1, BSH1 = "par1:isolation", "grh1:isolation", "bsh1:isolation"


def _key(principal_ref: str, *, scene: str = "group") -> ConversationStateKey:
    return ConversationStateKey(
        surface="wecom",
        adapter_namespace="assistant-wecom",
        tenant_ref="tenant:primary",
        scene=scene,
        subject_ref="group:alpha" if scene == "group" else "direct:alice",
        principal_ref=principal_ref,
    )


def _commit_reply(
    coordinator: ReplyStateTransactionCoordinatorV1,
    state_key: ConversationStateKey,
    request_id: str,
    text: str,
    *,
    clarification: bool,
) -> ReplyResponse:
    request_hash = f"rqh1:{request_id}"
    reservation = coordinator.reserve_reply(state_key, request_id, request_hash, True)
    assert isinstance(reservation, NewReplyReservationV1)
    response = ReplyResponse.model_validate(
        {
            "reply": {
                "kind": "clarification" if clarification else "answer",
                "text": text,
            },
            "actions": [
                {
                    "type": "send_weekly_report",
                    "resolve_type": "weekly_report",
                    "resolve_ref": "weekly:isolation",
                    "period": "20260717",
                    "report_date": "2026-07-17",
                }
            ],
        }
    )
    match state_key.scene:
        case "group":
            audit = GroupAuditCommitProposalV1(
                reason_code="action_ready_with_clarification"
                if clarification
                else "action_ready",
                manifest_refs=("weekly_report.send@2026-07-15.1",),
                program_attempts=(),
            )
        case "direct":
            digest = "dah1:" + "0" * 64
            audit = DirectAuditCommitProposalV1(
                state_key_digest=digest,
                request_id_digest=digest,
                reason_code="knowledge_answer_composer",
                manifest_refs=("company.fact.answer@2026-07-15.1",),
                content_digests=(("reply", digest),),
                lengths=(DirectAuditLengthV1("reply_chars", len(text)),),
                dependency_counts=(DirectDependencyCountV1("evidence", 0),),
                static_registry_id="capability-registry.v2",
                static_registry_version="2026-07-15.1",
            )
        case unreachable:
            assert_never(unreachable)
    pending = (
        PendingClarificationProposalV1(
            kind="report_scope",
            slots=("period",),
            question=text,
            topic="weekly_report",
        )
        if clarification
        else None
    )
    return coordinator.commit_reply(
        ReplyCommitProposalV1(
            state_key=state_key,
            request_id=request_id,
            request_hash=request_hash,
            replay_eligible=True,
            owner_token=reservation.owner_token,
            expected_revision_epoch=reservation.revision_epoch,
            expected_state_revision=reservation.state_revision,
            response=response,
            pol1="pol1:isolation",
            par1=PAR1,
            grh1=GRH1,
            bsh1=BSH1,
            user_turn=UserTurnProposalV1(
                text=f"request from {state_key.principal_ref}"
            ),
            assistant_turn=AssistantTurnProposalV1(
                text=text,
                reply_kind=response.reply.kind,
                clarification_requested=clarification,
            ),
            clarification=pending,
            audit=audit,
        )
    )


def _feedback(
    response: ReplyResponse,
    state_key: ConversationStateKey,
    request_id: str,
    feedback_id: str,
) -> ActionFeedbackRequestV2:
    action = response.actions[0]
    identity = {
        "contract_version": "conversation-identity.v1",
        "surface": "wecom",
        "scene": state_key.scene,
        "tenant_ref": state_key.tenant_ref,
        "principal_ref": state_key.principal_ref,
    }
    identity["group_ref" if state_key.scene == "group" else "direct_thread_ref"] = (
        state_key.subject_ref
    )
    return ActionFeedbackRequestV2.model_validate(
        {
            "contract_version": "action-feedback.v2",
            "feedback_id": feedback_id,
            "request_id": request_id,
            "response_id": response.response_id,
            "identity": identity,
            "executions": [
                {
                    "action_type": action.type,
                    "status": "executed",
                    "action_id": action.action_id,
                    "artifact": {
                        "type": "weekly_report",
                        "resolve_ref": action.resolve_ref,
                        "artifact_ref": "weekly:delivered",
                        "period": action.period,
                        "report_date": action.report_date,
                    },
                }
            ],
        }
    )


def test_state_key_fields_and_known_csk1_audit_references_are_frozen() -> None:
    # Given: representative canonical group and direct identities.
    group = _key("principal:alice")
    direct = _key("principal:alice", scene="direct")

    # When: callers inspect the immutable key schema and stable references.
    field_names = tuple(item.name for item in fields(ConversationStateKey))

    # Then: the full six-field identity and csk1 frames remain exact.
    assert field_names == (
        "surface",
        "adapter_namespace",
        "tenant_ref",
        "scene",
        "subject_ref",
        "principal_ref",
    )
    assert (
        state_key_ref(group)
        == "csk1:dc52e6e4ea05acf823a145d13ab49baccae95afc18c96f5df202ae4c4a9372be"
    )
    assert (
        state_key_ref(direct)
        == "csk1:f932c573ab40ac720bb1dcb7bd9009949570bf2ea79354f51998f1bcef9b11ae"
    )


def test_same_group_principals_isolate_replay_payload_ledger_receipts_and_audit() -> (
    None
):
    # Given: Alice has a committed clarification, payload, issued action, and feedback receipt.
    coordinator = ReplyStateTransactionCoordinatorV1()
    alice = _key("principal:alice")
    bob = _key("principal:bob")
    response = _commit_reply(
        coordinator,
        alice,
        "req:alice-first",
        "请确认周报周期。",
        clarification=True,
    )
    payloads = ScopedContextPayloadStoreV1(
        conversation_ttl_seconds=60,
        direct_audit_ttl_seconds=60,
    )
    handle = payloads.put(alice, {"owner": "alice"})
    feedback = _feedback(response, alice, "req:alice-first", "fb:alice-first")
    coordinator.commit_feedback(coordinator.prepare_feedback(alice, feedback))

    # When: Bob attempts every correlation path using Alice's opaque identifiers.
    revision = coordinator.root_revision()
    cross_feedback = _feedback(response, bob, "req:alice-first", "fb:bob-forged")
    with pytest.raises(CoordinatorError, match="feedback_identity_mismatch"):
        coordinator.prepare_feedback(bob, cross_feedback)
    replay = coordinator.reserve_reply(
        alice, "req:alice-first", "rqh1:req:alice-first", True
    )
    bob_reservation = coordinator.reserve_reply(
        bob,
        "req:alice-first",
        "rqh1:req:alice-first",
        True,
    )

    # Then: only Alice can observe continuity, replay, payload, ledger, receipt, and audit.
    assert coordinator.root_revision() == revision + 1
    assert isinstance(replay, CompletedReplyReplayV1)
    assert replay.response == response
    assert isinstance(bob_reservation, NewReplyReservationV1)
    coordinator.abort_reply(
        bob, bob_reservation.request_id, bob_reservation.owner_token
    )
    alice_view = coordinator.read_turn_admission_snapshot(alice, PAR1, GRH1, BSH1)
    bob_view = coordinator.read_turn_admission_snapshot(bob, PAR1, GRH1, BSH1)
    assert len(alice_view.turns) == 2
    assert alice_view.pending_clarification is not None
    assert len(alice_view.ledger_rows) == 1
    assert bob_view.turns == ()
    assert bob_view.pending_clarification is None
    assert bob_view.ledger_rows == ()
    assert payloads.get(alice, handle).payload == {"owner": "alice"}
    assert payloads.get(bob, handle) is None
    assert isinstance(
        coordinator.prepare_feedback(alice, feedback),
        FeedbackReceiptReplayV1,
    )
    assert (
        coordinator.lookup_issued_response(bob, response.response_id).kind
        == "identity_mismatch"
    )
    assert {key[0] for key in coordinator._root.feedback_receipts} == {alice}
    assert {key[0] for key in coordinator._root.action_ledger_records} == {alice}
    assert {key[0] for key in coordinator._root.audit_records} == {alice}


def test_same_principal_cross_scene_continuity_and_replay_remain_separate() -> None:
    # Given: one coordinator contains group and direct turns for the same principal.
    coordinator = ReplyStateTransactionCoordinatorV1()
    group = _key("principal:alice")
    direct = _key("principal:alice", scene="direct")
    group_response = _commit_reply(
        coordinator,
        group,
        "req:shared",
        "group answer",
        clarification=False,
    )
    direct_response = _commit_reply(
        coordinator,
        direct,
        "req:shared",
        "direct answer",
        clarification=False,
    )

    # When: the same request ID replays against each full scene key.
    group_replay = coordinator.reserve_reply(
        group, "req:shared", "rqh1:req:shared", True
    )
    direct_replay = coordinator.reserve_reply(
        direct, "req:shared", "rqh1:req:shared", True
    )

    # Then: each scene returns only its own response, transcript, issued record, and audit.
    assert group != direct
    assert isinstance(group_replay, CompletedReplyReplayV1)
    assert isinstance(direct_replay, CompletedReplyReplayV1)
    assert group_replay.response == group_response
    assert direct_replay.response == direct_response
    assert (
        tuple(turn.text for turn in coordinator._root.conversation_turns[group])[-1]
        == "group answer"
    )
    assert (
        tuple(turn.text for turn in coordinator._root.conversation_turns[direct])[-1]
        == "direct answer"
    )
    assert (
        coordinator.lookup_issued_response(group, direct_response.response_id).kind
        == "identity_mismatch"
    )
    assert (
        coordinator.lookup_issued_response(direct, group_response.response_id).kind
        == "identity_mismatch"
    )
    assert {(key[0], key[1]) for key in coordinator._root.audit_records} == {
        (group, group_response.response_id),
        (direct, direct_response.response_id),
    }
