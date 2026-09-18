from __future__ import annotations

import base64
from typing import get_args

import anyio
import pytest

from market_support_crewai_agent.runtime.state.audit_records import (
    DirectAuditRecordV1,
    GroupAuditRecordV1,
)
from market_support_crewai_agent.runtime.state.audit_types import (
    DirectAuditReasonCodeV1,
    ReasonCodeV1,
)
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.schemas.reply import ReplyResponse
from market_support_crewai_agent.settings_model import Settings
from tests.unit.state.v2_audit_commit_fixtures import (
    DIRECT_AUDIT_REASON_CODES,
    GROUP_CANDIDATE_REASON_CODES,
    GROUP_ONLY_REASON_CODES,
    action_reply_variants,
    commit_reply,
    direct_audit,
    direct_envelope,
    direct_runtime,
    group_audit,
    reply,
    state_key,
)


class _AuditPublicationFailure(RuntimeError):
    pass


@pytest.mark.parametrize(
    "response",
    action_reply_variants(),
)
def test_commit_server_issues_action_ids_with_strict_serialization(
    response: ReplyResponse,
) -> None:
    coordinator = ReplyStateTransactionCoordinatorV1()
    issued = commit_reply(
        coordinator,
        state_key("group"),
        "req:actions",
        "rqh1:actions",
        response,
        group_audit("knowledge_evidence_available"),
    )

    assert issued.model_dump_json(warnings="error")
    assert [(action.type, action.resolve_ref) for action in issued.actions] == [
        (action.type, action.resolve_ref) for action in response.actions
    ]
    assert [action.model_dump(exclude={"action_id"}) for action in issued.actions] == [
        action.model_dump(exclude={"action_id"}) for action in response.actions
    ]
    assert all(action.action_id.startswith("act-") for action in issued.actions)
    assert len({action.action_id for action in issued.actions}) == len(issued.actions)
    assert issued.response_id.startswith("resp-")


def test_commit_derives_group_audit_in_the_same_root_publication() -> None:
    coordinator = ReplyStateTransactionCoordinatorV1()
    state = state_key("group")
    response = commit_reply(
        coordinator,
        state,
        "req:group",
        "rqh1:group",
        reply(),
        group_audit("knowledge_evidence_available"),
    )

    journal = coordinator.last_journal()
    assert journal is not None
    record = journal.candidate_root.audit_records[(state, response.response_id)]
    assert isinstance(record, GroupAuditRecordV1)
    assert record.scene == "group"
    assert record.manifest_refs == ("general.smalltalk@2026-07-15.1",)
    assert record.response_id == response.response_id


def test_commit_derives_redacted_direct_audit_without_identity_or_content() -> None:
    coordinator = ReplyStateTransactionCoordinatorV1()
    state = state_key("direct")
    response = commit_reply(
        coordinator,
        state,
        "req:direct",
        "rqh1:direct",
        reply(),
        direct_audit("direct_human_handoff"),
    )

    journal = coordinator.last_journal()
    assert journal is not None
    record = journal.candidate_root.audit_records[(state, response.response_id)]
    assert isinstance(record, DirectAuditRecordV1)
    assert record.state_key_digest == "dah1:" + "0" * 64
    assert record.request_id_digest == "dah1:" + "0" * 64
    assert not hasattr(record, "state_key")
    assert "audit request" not in repr(record)
    assert "principal:test" not in repr(record)


def test_audit_publication_fault_keeps_the_reply_pending_without_audit() -> None:
    def fail_commit(operation: str) -> None:
        if operation == "reply_commit":
            raise _AuditPublicationFailure("audit publication failure")

    coordinator = ReplyStateTransactionCoordinatorV1(fault_hook=fail_commit)
    state = state_key("group")
    reservation = coordinator.reserve_reply(state, "req:fault", "rqh1:fault", True)

    with pytest.raises(_AuditPublicationFailure, match="audit publication failure"):
        _ = commit_reply(
            coordinator,
            state,
            "req:fault",
            "rqh1:fault",
            reply(),
            group_audit("smalltalk_requires_composer"),
            reservation=reservation,
        )

    journal = coordinator.last_journal()
    assert journal is not None
    assert journal.outcome == "rolled_back"
    assert journal.prior_root.audit_records == {}
    assert (
        journal.prior_root.issued_records[(state, reservation.request_id)].phase
        == "pending"
    )


@pytest.mark.parametrize("reason_code", GROUP_CANDIDATE_REASON_CODES)
def test_v2_candidate_audit_reason_codes_are_closed_and_accepted(
    reason_code: ReasonCodeV1,
) -> None:
    assert group_audit(reason_code).reason_code == reason_code


def test_group_audit_reason_code_type_excludes_unregistered_values() -> None:
    assert "prompt_injected_reason_code" not in get_args(ReasonCodeV1)


@pytest.mark.parametrize("reason_code", DIRECT_AUDIT_REASON_CODES)
def test_direct_v2_audit_reason_codes_commit_only_reachable_direct_outcomes(
    reason_code: DirectAuditReasonCodeV1,
) -> None:
    coordinator = ReplyStateTransactionCoordinatorV1()
    state = state_key("direct")
    response = commit_reply(
        coordinator,
        state,
        f"req:direct-reason:{reason_code}",
        f"rqh1:direct-reason:{reason_code}",
        reply(),
        direct_audit(reason_code),
    )

    journal = coordinator.last_journal()
    assert journal is not None
    record = journal.candidate_root.audit_records[(state, response.response_id)]
    assert isinstance(record, DirectAuditRecordV1)
    assert record.reason_code == reason_code


@pytest.mark.parametrize("reason_code", GROUP_ONLY_REASON_CODES)
def test_direct_v2_audit_type_excludes_group_only_reason_codes(
    reason_code: ReasonCodeV1,
) -> None:
    assert reason_code not in get_args(DirectAuditReasonCodeV1)


def test_direct_lifecycle_commits_hmac_only_audit_and_replays_before_work() -> None:
    key = base64.urlsafe_b64encode(b"d" * 32).decode("ascii").rstrip("=")
    runtime = direct_runtime(
        Settings(
            llm_api_key="test-key",
            direct_audit_hmac_key=key,
            reply_alignment_verifier_enabled=False,
        ),
        ReplyStateTransactionCoordinatorV1(),
    )
    envelope = direct_envelope(
        "T0请人工协助",
        "direct:thread-audit",
        "principal:direct-audit",
        "canary-direct-name",
    )

    response = anyio.run(runtime.reply, envelope)
    replay = anyio.run(runtime.reply, envelope)

    journal = runtime.coordinator.last_journal()
    assert journal is not None
    record = journal.candidate_root.audit_records[
        (envelope.state_key, response.response_id)
    ]
    assert replay == response
    assert isinstance(record, DirectAuditRecordV1)
    assert response.reply.kind == "human_handoff"
    assert response.actions == []
    assert response.reply.mentions == []
    assert "T0请人工协助" not in repr(record)
    assert "canary-direct-name" not in repr(record)


def test_direct_audit_key_failure_aborts_the_reservation_before_replay() -> None:
    coordinator = ReplyStateTransactionCoordinatorV1()
    envelope = direct_envelope(
        "T0 audit failure",
        "direct:thread-audit-failure",
        "principal:direct-audit-failure",
    )
    failing_runtime = direct_runtime(
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False),
        coordinator,
    )

    with pytest.raises(CoordinatorError, match="direct_audit_hmac_key_required"):
        _ = anyio.run(failing_runtime.reply, envelope)

    journal = coordinator.last_journal()
    assert journal is not None
    assert journal.operation == "reply_abort"
    assert journal.candidate_root.audit_records == {}
    assert journal.candidate_root.issued_records == {}
    key = base64.urlsafe_b64encode(b"e" * 32).decode("ascii").rstrip("=")
    recovered_runtime = direct_runtime(
        Settings(
            llm_api_key="test-key",
            direct_audit_hmac_key=key,
            reply_alignment_verifier_enabled=False,
        ),
        coordinator,
    )

    response = anyio.run(recovered_runtime.reply, envelope)

    assert response.response_id.startswith("resp-")
