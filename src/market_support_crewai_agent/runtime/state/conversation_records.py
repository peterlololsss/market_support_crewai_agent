from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class StateRevisionRecordV1:
    revision_epoch: bytes
    revision: int
    created_at_monotonic_ns: int
    updated_at_monotonic_ns: int
    expires_at_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class ConversationTurnRecordV2:
    role: Literal["user", "assistant"]
    text: str
    ordinal: int
    created_at_epoch_ms: int
    pol1: str
    par1: str
    grh1: str
    bsh1: str
    reply_kind: str | None
    clarification_ref: str | None
    contract_version: Literal["conversation-turn-record.v2"] = (
        "conversation-turn-record.v2"
    )


@dataclass(frozen=True, slots=True)
class PendingClarificationRecordV1:
    clarification_ref: str
    kind: Literal["material_pack_option", "report_scope", "destination", "other"]
    slots: tuple[str, ...]
    question: str
    topic: Literal["material_pack", "weekly_report", "monthly_report"] | None
    originating_turn_ordinal: int
    created_at_epoch_ms: int
    pol1: str
    par1: str
    grh1: str
    bsh1: str
    contract_version: Literal["pending-clarification-record.v1"] = (
        "pending-clarification-record.v1"
    )


@dataclass(frozen=True, slots=True)
class UserTurnProposalV1:
    text: str
    contract_version: Literal["turn-proposal.v1"] = "turn-proposal.v1"
    role: Literal["user"] = "user"


@dataclass(frozen=True, slots=True)
class AssistantTurnProposalV1:
    text: str
    reply_kind: str
    clarification_requested: bool
    contract_version: Literal["turn-proposal.v1"] = "turn-proposal.v1"
    role: Literal["assistant"] = "assistant"


@dataclass(frozen=True, slots=True)
class PendingClarificationProposalV1:
    kind: Literal["material_pack_option", "report_scope", "destination", "other"]
    slots: tuple[str, ...]
    question: str
    topic: Literal["material_pack", "weekly_report", "monthly_report"] | None
