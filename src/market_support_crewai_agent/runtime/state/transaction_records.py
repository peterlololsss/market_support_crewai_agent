from dataclasses import dataclass
from typing import Literal

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.audit_records import (
    SanitizedAuditCommitProposalV1,
)
from market_support_crewai_agent.runtime.state.conversation_records import (
    AssistantTurnProposalV1,
    ConversationTurnRecordV2,
    PendingClarificationProposalV1,
    PendingClarificationRecordV1,
    UserTurnProposalV1,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    ActionLedgerRecordV2,
)
from market_support_crewai_agent.schemas.reply import ReplyResponse


@dataclass(frozen=True, slots=True)
class TurnAdmissionSnapshotV1:
    state_key: ConversationStateKey
    par1: str
    grh1: str
    bsh1: str
    ledger_rows: tuple[ActionLedgerRecordV2, ...]
    turns: tuple[ConversationTurnRecordV2, ...]
    pending_clarification: PendingClarificationRecordV1 | None
    revision_epoch: bytes
    state_revision: int
    snapshot_at_epoch_ms: int
    snapshot_at_monotonic_ns: int
    pending_reply: bool
    contract_version: Literal["turn-admission-snapshot.v1"] = (
        "turn-admission-snapshot.v1"
    )


@dataclass(frozen=True, slots=True)
class NewReplyReservationV1:
    state_key: ConversationStateKey
    request_id: str
    request_hash: str
    replay_eligible: bool
    owner_token: bytes
    revision_epoch: bytes
    state_revision: int
    contract_version: Literal["reply-reservation-result.v1"] = (
        "reply-reservation-result.v1"
    )
    kind: Literal["new"] = "new"


@dataclass(frozen=True, slots=True)
class CompletedReplyReplayV1:
    state_key: ConversationStateKey
    request_id: str
    request_hash: str
    response: ReplyResponse
    issued_at_epoch_ms: int
    issued_record_revision: int
    replay_eligible: Literal[True] = True
    contract_version: Literal["reply-reservation-result.v1"] = (
        "reply-reservation-result.v1"
    )
    kind: Literal["replay"] = "replay"


ReservationResultV1 = NewReplyReservationV1 | CompletedReplyReplayV1


@dataclass(frozen=True, slots=True)
class ReplyCommitProposalV1:
    state_key: ConversationStateKey
    request_id: str
    request_hash: str
    replay_eligible: bool
    owner_token: bytes
    expected_revision_epoch: bytes
    expected_state_revision: int
    response: ReplyResponse
    pol1: str
    par1: str
    grh1: str
    bsh1: str
    user_turn: UserTurnProposalV1
    assistant_turn: AssistantTurnProposalV1
    clarification: PendingClarificationProposalV1 | None
    audit: SanitizedAuditCommitProposalV1
    contract_version: Literal["reply-commit-proposal.v1"] = "reply-commit-proposal.v1"

    def __post_init__(self) -> None:
        if self.state_key.scene != self.audit.kind:
            raise ValueError("reply commit audit scene does not match state key")
