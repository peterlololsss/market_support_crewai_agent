from __future__ import annotations

from uuid import uuid4

from market_support_crewai_agent.schemas import ReplyResponse

_PLAIN_TEXT_MARKERS = ("**", "```")


def ensure_response_ids(
    response: ReplyResponse,
) -> ReplyResponse:
    response_id = response.response_id.strip() or f"resp-{uuid4().hex}"
    reply = response.reply
    if reply.text_format == "plain_text":
        reply = reply.model_copy(update={"text": _sanitize_plain_text(reply.text)})
    actions = []
    for index, action in enumerate(response.actions, start=1):
        action_id = action.action_id.strip() or f"act-{index}"
        actions.append(action.model_copy(update={"action_id": action_id}))
    return response.model_copy(
        update={
            "response_id": response_id,
            "reply": reply,
            "actions": actions,
        }
    )


def _sanitize_plain_text(text: str) -> str:
    sanitized = text
    for marker in _PLAIN_TEXT_MARKERS:
        sanitized = sanitized.replace(marker, "")
    return sanitized
