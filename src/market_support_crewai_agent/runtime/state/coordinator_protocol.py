from typing import Protocol

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.transaction_records import (
    ReplyCommitProposalV1,
    ReservationResultV1,
    TurnAdmissionSnapshotV1,
)
from market_support_crewai_agent.schemas.reply import ReplyResponse


class ReplyTurnStateCoordinatorV1(Protocol):
    def read_turn_admission_snapshot(
        self, state_key: ConversationStateKey, par1: str, grh1: str, bsh1: str
    ) -> TurnAdmissionSnapshotV1: ...

    def reserve_reply(
        self,
        state_key: ConversationStateKey,
        request_id: str,
        request_hash: str,
        replay_eligible: bool,
    ) -> ReservationResultV1: ...

    def abort_reply(
        self, state_key: ConversationStateKey, request_id: str, owner_token: bytes
    ) -> None: ...

    def commit_reply(self, proposal: ReplyCommitProposalV1) -> ReplyResponse: ...
