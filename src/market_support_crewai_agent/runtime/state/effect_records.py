from dataclasses import dataclass
from typing import Literal

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.schemas.reply import ReplyResponse
from market_support_crewai_agent.schemas.type_ids import (
    ActionExecutionStatus,
    ActionExecutionType,
)


@dataclass(frozen=True, slots=True)
class PendingIssuedResponseRecordV1:
    state_key: ConversationStateKey
    request_id: str
    request_hash: str
    replay_eligible: bool
    owner_token: bytes
    revision_epoch: bytes
    reserved_state_revision: int
    created_at_epoch_ms: int
    created_at_monotonic_ns: int
    expires_at_monotonic_ns: int
    contract_version: Literal["issued-response-record.v1"] = "issued-response-record.v1"
    phase: Literal["pending"] = "pending"
    record_revision: Literal[0] = 0


@dataclass(frozen=True, slots=True)
class IssuedEffectV1:
    effect_key: str
    effect_type: ActionExecutionType
    action_id: str | None
    resolve_ref: str | None
    material_pack_option: str | None
    period: str | None
    report_date: str | None
    status: ActionExecutionStatus | None
    artifact_ref: str | None
    status_history: tuple[ActionExecutionStatus, ...]
    contract_version: Literal["issued-effect.v1"] = "issued-effect.v1"


@dataclass(frozen=True, slots=True)
class FeedbackReceiptV1:
    receipt_id: str
    state_key: ConversationStateKey
    feedback_id: str
    feedback_hash: str
    response_id: str
    issued_record_revision: int
    stored: int
    created_at_epoch_ms: int
    created_at_monotonic_ns: int
    expires_at_monotonic_ns: int
    contract_version: Literal["feedback-receipt.v1"] = "feedback-receipt.v1"


@dataclass(frozen=True, slots=True)
class CompleteIssuedResponseRecordV1:
    state_key: ConversationStateKey
    request_id: str
    request_hash: str
    replay_eligible: bool
    revision_epoch: bytes
    committed_state_revision: int
    record_revision: int
    response: ReplyResponse
    effects: tuple[IssuedEffectV1, ...]
    receipts: tuple[FeedbackReceiptV1, ...]
    issued_at_epoch_ms: int
    completed_at_monotonic_ns: int
    expires_at_monotonic_ns: int
    pol1: str
    par1: str
    grh1: str
    bsh1: str
    contract_version: Literal["issued-response-record.v1"] = "issued-response-record.v1"
    phase: Literal["complete"] = "complete"


IssuedResponseRecordV1 = PendingIssuedResponseRecordV1 | CompleteIssuedResponseRecordV1


@dataclass(frozen=True, slots=True)
class PreparedEffectTransitionV1:
    effect_key: str
    effect_type: ActionExecutionType
    expected_status: ActionExecutionStatus | None
    requested_status: ActionExecutionStatus
    expected_artifact_ref: str | None
    requested_artifact_ref: str | None
    is_noop: bool
    contract_version: Literal["prepared-effect-transition.v1"] = (
        "prepared-effect-transition.v1"
    )


@dataclass(frozen=True, slots=True)
class PreparedFeedbackV1:
    state_key: ConversationStateKey
    feedback_id: str
    feedback_hash: str
    request_id: str
    response_id: str
    commit_token: bytes
    revision_epoch: bytes
    expected_state_revision: int
    expected_issued_record_revision: int
    transitions: tuple[PreparedEffectTransitionV1, ...]
    expires_at_monotonic_ns: int
    contract_version: Literal["feedback-preparation.v1"] = "feedback-preparation.v1"
    kind: Literal["prepared"] = "prepared"


@dataclass(frozen=True, slots=True)
class FeedbackReceiptReplayV1:
    state_key: ConversationStateKey
    feedback_id: str
    feedback_hash: str
    request_id: str
    response_id: str
    receipt_id: str
    issued_record_revision: int
    stored: Literal[0] = 0
    contract_version: Literal["feedback-preparation.v1"] = "feedback-preparation.v1"
    kind: Literal["replay"] = "replay"


FeedbackPreparationV1 = PreparedFeedbackV1 | FeedbackReceiptReplayV1


@dataclass(frozen=True, slots=True)
class PreparedFeedbackTokenRecordV1:
    preparation: PreparedFeedbackV1
    created_at_monotonic_ns: int
    consumed: Literal[False] = False


@dataclass(frozen=True, slots=True)
class IssuedFoundV1:
    record: CompleteIssuedResponseRecordV1
    contract_version: Literal["issued-response-lookup.v1"] = "issued-response-lookup.v1"
    kind: Literal["found"] = "found"


@dataclass(frozen=True, slots=True)
class IssuedNotFoundV1:
    contract_version: Literal["issued-response-lookup.v1"] = "issued-response-lookup.v1"
    kind: Literal["not_found"] = "not_found"


@dataclass(frozen=True, slots=True)
class IssuedIdentityMismatchV1:
    contract_version: Literal["issued-response-lookup.v1"] = "issued-response-lookup.v1"
    kind: Literal["identity_mismatch"] = "identity_mismatch"


IssuedResponseLookupV1 = IssuedFoundV1 | IssuedNotFoundV1 | IssuedIdentityMismatchV1


@dataclass(frozen=True, slots=True)
class FeedbackCommitResultV1:
    receipt_id: str
    stored: int
    replayed: bool
    contract_version: Literal["feedback-commit-result.v1"] = "feedback-commit-result.v1"


@dataclass(frozen=True, slots=True)
class ActionLedgerRecordV2:
    state_key: ConversationStateKey
    response_id: str
    action_id: str
    action_type: Literal[
        "send_material_pack", "send_weekly_report", "send_monthly_report"
    ]
    status: ActionExecutionStatus
    artifact_ref: str | None
    status_revision: int
    received_at_epoch_ms: int
    expires_at_monotonic_ns: int
    pol1: str
    par1: str
    grh1: str
    bsh1: str
    contract_version: Literal["action-ledger-record.v2"] = "action-ledger-record.v2"
