from __future__ import annotations

import json
from typing import cast

from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectOutboundDraft,
)
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.schemas import ReplyResponse


def compact_assistant_result(
    response: ReplyResponse,
    plan: ExecutionPlan,
    pending_outbound_draft: DirectOutboundDraft | None = None,
) -> str:
    return json.dumps(
        {
            "contract_version": "reply-runtime-history",
            "reply_response": response.model_dump(mode="json", exclude_none=True),
            "pending_plan": _compact_pending_plan(plan),
            "pending_outbound_draft": pending_outbound_draft.model_dump(
                mode="json", exclude_none=True
            )
            if pending_outbound_draft is not None
            else None,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def pending_direct_outbound_draft(
    history: list[ConversationMessage],
) -> DirectOutboundDraft | None:
    for message in reversed(history):
        if message.role != "assistant":
            continue
        try:
            parsed: object = json.loads(message.content)
        except ValueError:
            continue
        if not isinstance(parsed, dict):
            continue
        payload = cast(dict[str, object], parsed)
        if payload.get("contract_version") != "reply-runtime-history":
            continue
        value = payload.get("pending_outbound_draft")
        if value is None:
            return None
        try:
            return DirectOutboundDraft.model_validate(value)
        except ValueError:
            return None
    return None


def _compact_pending_plan(plan: ExecutionPlan) -> dict[str, object] | None:
    if plan.response_mode != "clarification" and not plan.ambiguity_slots:
        return None
    return {
        "artifact_kind": plan.artifact_kind,
        "response_mode": plan.response_mode,
        "ambiguity_slots": list(plan.ambiguity_slots),
        "material_pack_option": plan.material_pack_option,
        "capabilities": list(plan.capabilities),
    }
