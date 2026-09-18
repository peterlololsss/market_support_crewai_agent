from uuid import uuid4

from market_support_crewai_agent.schemas.reply import ReplyResponse


def issue_response_ids(response: ReplyResponse) -> ReplyResponse:
    response_id = response.response_id or f"resp-{uuid4().hex}"
    actions = [
        action.model_copy(
            update={"action_id": action.action_id or f"act-{uuid4().hex}"}
        )
        for action in response.actions
    ]
    return response.model_copy(update={"response_id": response_id, "actions": actions})
