from collections.abc import Callable
from dataclasses import replace

from market_support_crewai_agent.runtime.hashing import clarification_ref_hash
from market_support_crewai_agent.runtime.state.conversation_records import (
    ConversationTurnRecordV2,
    PendingClarificationRecordV1,
    StateRevisionRecordV1,
)
from market_support_crewai_agent.runtime.state.coordinator_audit import (
    build_audit_record,
)
from market_support_crewai_agent.runtime.state.coordinator_config import (
    CoordinatorConfigV1,
)
from market_support_crewai_agent.runtime.state.coordinator_effects import (
    derive_issued_effects,
)
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.coordinator_response_ids import (
    issue_response_ids,
)
from market_support_crewai_agent.runtime.state.coordinator_retention import (
    enforce_session_capacity,
)
from market_support_crewai_agent.runtime.state.coordinator_root_orders import key_bytes
from market_support_crewai_agent.runtime.state.coordinator_state import (
    CoordinatorStateRootV1,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    CompleteIssuedResponseRecordV1,
)
from market_support_crewai_agent.runtime.state.transaction_records import (
    ReplyCommitProposalV1,
)
from market_support_crewai_agent.schemas.reply import ReplyResponse


def commit_reply_candidate(
    root: CoordinatorStateRootV1,
    proposal: ReplyCommitProposalV1,
    *,
    monotonic_clock_ns: Callable[[], int],
    epoch_clock_ms: Callable[[], int],
    config: CoordinatorConfigV1,
) -> tuple[CoordinatorStateRootV1, ReplyResponse]:
    key = (proposal.state_key, proposal.request_id)
    pending = root.issued_records.get(key)
    revision = root.state_revisions.get(proposal.state_key)
    if (
        pending is None
        or pending.phase != "pending"
        or pending.owner_token != proposal.owner_token
    ):
        raise CoordinatorError("reservation_not_found")
    if (
        revision is None
        or revision.revision_epoch != proposal.expected_revision_epoch
        or revision.revision != proposal.expected_state_revision
    ):
        raise CoordinatorError("conversation_state_changed")
    if (
        pending.request_hash != proposal.request_hash
        or pending.replay_eligible != proposal.replay_eligible
    ):
        raise CoordinatorError("request_id_conflict")
    response = issue_response_ids(proposal.response)
    if response.response_id in root.response_index:
        raise CoordinatorError("state_store_unavailable")
    now_monotonic_ns = monotonic_clock_ns()
    now_epoch_ms = epoch_clock_ms()
    revised_state = StateRevisionRecordV1(
        revision_epoch=revision.revision_epoch,
        revision=revision.revision + 1,
        created_at_monotonic_ns=revision.created_at_monotonic_ns,
        updated_at_monotonic_ns=now_monotonic_ns,
        expires_at_monotonic_ns=now_monotonic_ns + config.issued_ttl_ns,
    )
    effects = derive_issued_effects(response)
    if (
        response.reply.kind == "clarification"
    ) != proposal.assistant_turn.clarification_requested:
        raise CoordinatorError("reply_commit_invalid")
    if (
        proposal.clarification is not None
    ) != proposal.assistant_turn.clarification_requested:
        raise CoordinatorError("reply_commit_invalid")
    prior_turns = root.conversation_turns.get(proposal.state_key, ())
    next_ordinal = len(prior_turns)
    clarification_ref = None
    clarification_record = None
    if proposal.clarification is not None:
        clarification_ref = clarification_ref_hash(
            proposal.state_key,
            next_ordinal + 1,
            proposal.clarification.question,
        )
        clarification_record = PendingClarificationRecordV1(
            clarification_ref=clarification_ref,
            kind=proposal.clarification.kind,
            slots=proposal.clarification.slots,
            question=proposal.clarification.question,
            topic=proposal.clarification.topic,
            originating_turn_ordinal=next_ordinal + 1,
            created_at_epoch_ms=now_epoch_ms,
            pol1=proposal.pol1,
            par1=proposal.par1,
            grh1=proposal.grh1,
            bsh1=proposal.bsh1,
        )
    turns = (
        *prior_turns,
        ConversationTurnRecordV2(
            "user",
            proposal.user_turn.text,
            next_ordinal,
            now_epoch_ms,
            proposal.pol1,
            proposal.par1,
            proposal.grh1,
            proposal.bsh1,
            None,
            None,
        ),
        ConversationTurnRecordV2(
            "assistant",
            proposal.assistant_turn.text,
            next_ordinal + 1,
            now_epoch_ms,
            proposal.pol1,
            proposal.par1,
            proposal.grh1,
            proposal.bsh1,
            proposal.assistant_turn.reply_kind,
            clarification_ref,
        ),
    )[-config.conversation_max_messages :]
    audit = build_audit_record(
        proposal.audit,
        state_key=proposal.state_key,
        response_id=response.response_id,
        request_id=proposal.request_id,
        reply_kind=response.reply.kind,
        created_at_epoch_ms=now_epoch_ms,
        expires_at_monotonic_ns=now_monotonic_ns + config.issued_ttl_ns,
    )
    audit_records = root.audit_records
    if len(audit_records) >= config.conversation_max_sessions:
        victim_key, _ = min(
            audit_records.items(),
            key=lambda item: (
                item[1].created_at_epoch_ms,
                key_bytes(item[0][0], item[0][1]),
            ),
        )
        audit_records = audit_records.without_keys(frozenset({victim_key}))
    record = CompleteIssuedResponseRecordV1(
        state_key=proposal.state_key,
        request_id=proposal.request_id,
        request_hash=proposal.request_hash,
        replay_eligible=proposal.replay_eligible,
        revision_epoch=revised_state.revision_epoch,
        committed_state_revision=revised_state.revision,
        record_revision=1,
        response=response,
        effects=effects,
        receipts=(),
        issued_at_epoch_ms=now_epoch_ms,
        completed_at_monotonic_ns=now_monotonic_ns,
        expires_at_monotonic_ns=now_monotonic_ns + config.issued_ttl_ns,
        pol1=proposal.pol1,
        par1=proposal.par1,
        grh1=proposal.grh1,
        bsh1=proposal.bsh1,
    )
    action_entries = {
        effect.action_id: (
            proposal.state_key,
            response.response_id,
            effect.effect_key,
        )
        for effect in effects
        if effect.action_id is not None
    }
    candidate = replace(
        root,
        root_revision=root.root_revision + 1,
        issued_records=root.issued_records.with_updates({key: record}),
        pending_request_by_state=root.pending_request_by_state.without_keys(
            frozenset({proposal.state_key})
        ),
        state_revisions=root.state_revisions.with_updates(
            {proposal.state_key: revised_state}
        ),
        response_index=root.response_index.with_updates({response.response_id: key}),
        action_index=root.action_index.with_updates(action_entries),
        conversation_turns=root.conversation_turns.with_updates(
            {proposal.state_key: turns}
        ),
        clarification_records=(
            root.clarification_records.with_updates(
                {proposal.state_key: clarification_record}
            )
            if clarification_record is not None
            else root.clarification_records.without_keys(
                frozenset({proposal.state_key})
            )
        ),
        audit_records=audit_records.with_updates(
            {(proposal.state_key, response.response_id): audit}
        ),
    )
    return enforce_session_capacity(
        candidate, config.conversation_max_sessions
    ), response
