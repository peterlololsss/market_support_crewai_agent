from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectOutboundDraft,
)
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.schemas import (
    PrepareOutboundMessageAction,
    ReplyResponse,
)


class _RuntimeHistoryPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    contract_version: Literal["reply-runtime-history"]
    reply_response: ReplyResponse
    pending_plan: dict[str, object] | None = None
    pending_outbound_draft: DirectOutboundDraft | None = None


@dataclass(frozen=True, slots=True)
class PendingOutboundConfirmation:
    response_id: str
    action: PrepareOutboundMessageAction

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "response_id": self.response_id,
            "action_id": self.action.action_id,
            "action_type": self.action.type,
            "target": {
                "kind": self.action.target.kind,
                "name": self.action.target.name,
            },
            "content": self.action.content.model_dump(mode="json", exclude_none=True),
        }


@dataclass(frozen=True, slots=True)
class PendingUserAnswer:
    assistant_question: str
    pending_plan: dict[str, object] | None
    pending_outbound_draft: DirectOutboundDraft | None
    pending_confirmation: PendingOutboundConfirmation | None
    user_messages_after_question: tuple[str, ...]

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "status": "awaiting_user_answer",
            "assistant_question": self.assistant_question,
            "pending_plan": self.pending_plan,
            "pending_outbound_draft": (
                self.pending_outbound_draft.model_dump(mode="json", exclude_none=True)
                if self.pending_outbound_draft is not None
                else None
            ),
            "pending_confirmation": (
                self.pending_confirmation.to_prompt_dict()
                if self.pending_confirmation is not None
                else None
            ),
            "unresolved_fields": _unresolved_fields(self.pending_outbound_draft),
            "user_messages_after_question": list(self.user_messages_after_question),
            "instruction": _continuation_instruction(self.pending_confirmation),
        }


def pending_user_answer(history: list[ConversationMessage]) -> PendingUserAnswer | None:
    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if message.role != "assistant":
            continue
        parsed = _runtime_history_payload(message.content)
        if parsed is None:
            continue
        response, pending_plan, pending_draft = parsed
        if response.reply.kind != "clarification":
            return None
        confirmation = _prepared_confirmation(response)
        if pending_plan is None and confirmation is None:
            return None
        return PendingUserAnswer(
            assistant_question=response.reply.text,
            pending_plan=pending_plan,
            pending_outbound_draft=pending_draft,
            pending_confirmation=confirmation,
            user_messages_after_question=tuple(
                item.content for item in history[index + 1 :] if item.role == "user"
            ),
        )
    return None


def pending_outbound_confirmation(
    history: list[ConversationMessage],
) -> PendingOutboundConfirmation | None:
    pending = pending_user_answer(history)
    return pending.pending_confirmation if pending is not None else None


def _runtime_history_payload(
    content: str,
) -> tuple[ReplyResponse, dict[str, object] | None, DirectOutboundDraft | None] | None:
    try:
        payload = _RuntimeHistoryPayload.model_validate_json(content)
    except ValidationError:
        return None
    return (
        payload.reply_response,
        payload.pending_plan,
        payload.pending_outbound_draft,
    )


def _prepared_confirmation(
    response: ReplyResponse,
) -> PendingOutboundConfirmation | None:
    prepared = [
        action
        for action in response.actions
        if isinstance(action, PrepareOutboundMessageAction)
    ]
    if len(prepared) != 1 or not response.response_id or not prepared[0].action_id:
        return None
    return PendingOutboundConfirmation(response_id=response.response_id, action=prepared[0])


def _unresolved_fields(draft: DirectOutboundDraft | None) -> list[str]:
    if draft is None:
        return []
    unresolved: list[str] = []
    if draft.target is None:
        unresolved.append("target")
    elif draft.target.kind is None:
        unresolved.append("target.kind")
    if draft.content is None:
        unresolved.append("content")
    return unresolved


def _continuation_instruction(
    confirmation: PendingOutboundConfirmation | None,
) -> str:
    if confirmation is not None:
        return (
            "Resolve the current message against this exact latest prepared action. "
            "An unambiguous confirmation continues it; a correction creates a new prepare; "
            "a cancellation or topic switch does neither. Never revive an older prepared action."
        )
    return (
        "Resolve the current message as a reply to the assistant question before treating it "
        "as unrelated. A direct answer supplies exactly one unresolved field literally. "
        "Cancellation, a new question, or another clear topic switch does not fill the field."
    )
