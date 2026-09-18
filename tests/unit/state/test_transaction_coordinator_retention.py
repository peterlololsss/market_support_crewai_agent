import pytest

from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.effect_records import PreparedFeedbackV1
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.runtime.state.transaction_records import (
    NewReplyReservationV1,
)
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2
from tests.unit.state.action_feedback_payloads import (
    make_issued_weekly_feedback_fixture,
)
from tests.unit.state.v2_audit_commit_fixtures import (
    commit_reply,
    group_audit,
    reply,
    state_key,
)


def test_stale_reply_commit_preserves_feedback_publication() -> None:
    # Given: a reply reservation followed by authoritative adapter feedback.
    issued = make_issued_weekly_feedback_fixture()
    reservation = issued.coordinator.reserve_reply(
        issued.state_key, "req:stale", "rqh1:stale", True
    )
    assert isinstance(reservation, NewReplyReservationV1)
    feedback = ActionFeedbackRequestV2.model_validate(issued.payload)
    prepared = issued.coordinator.prepare_feedback(issued.state_key, feedback)
    assert isinstance(prepared, PreparedFeedbackV1)
    _ = issued.coordinator.commit_feedback(prepared)
    authoritative_revision = issued.coordinator.root_revision()

    # When: the reservation attempts to commit against its stale state revision.
    with pytest.raises(CoordinatorError, match="conversation_state_changed"):
        _ = commit_reply(
            issued.coordinator,
            issued.state_key,
            reservation.request_id,
            reservation.request_hash,
            reply(),
            group_audit("unknown"),
            reservation,
        )

    # Then: the feedback root remains authoritative and the reservation stays pending.
    assert issued.coordinator.root_revision() == authoritative_revision
    with pytest.raises(CoordinatorError, match="request_in_progress"):
        _ = issued.coordinator.reserve_reply(
            issued.state_key, "req:stale", "rqh1:stale", True
        )


def test_issued_capacity_evicts_complete_before_accepting_new_reservation() -> None:
    # Given: the one-record issued store contains a completed response.
    coordinator = ReplyStateTransactionCoordinatorV1(
        issued_response_capacity=1, feedback_receipt_capacity=1
    )
    first_state = state_key("group")
    first = commit_reply(
        coordinator,
        first_state,
        "req:first",
        "rqh1:first",
        reply(),
        group_audit("unknown"),
    )
    revision_before = coordinator.root_revision()

    # When: another principal reserves while the store is at capacity.
    second_state = first_state.__class__(
        surface=first_state.surface,
        adapter_namespace=first_state.adapter_namespace,
        tenant_ref=first_state.tenant_ref,
        scene=first_state.scene,
        subject_ref=first_state.subject_ref,
        principal_ref="principal:other",
    )
    reservation = coordinator.reserve_reply(
        second_state, "req:second", "rqh1:second", True
    )

    # Then: eviction publishes before reservation and the pending record remains pinned.
    assert isinstance(reservation, NewReplyReservationV1)
    assert coordinator.root_revision() == revision_before + 2
    assert (
        coordinator.lookup_issued_response(first_state, first.response_id).kind
        == "not_found"
    )
    with pytest.raises(CoordinatorError, match="issued_response_store_unavailable"):
        _ = coordinator.reserve_reply(first_state, "req:third", "rqh1:third", True)
