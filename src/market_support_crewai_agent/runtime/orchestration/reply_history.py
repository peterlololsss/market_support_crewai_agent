from __future__ import annotations

import json

from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.schemas import ReplyResponse


def compact_assistant_result(response: ReplyResponse, plan: ExecutionPlan) -> str:
    return json.dumps(
        {
            "contract_version": "reply-runtime-history",
            "reply_response": response.model_dump(mode="json", exclude_none=True),
            "pending_plan": _compact_pending_plan(plan),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _compact_pending_plan(plan: ExecutionPlan) -> dict[str, object] | None:
    if plan.response_mode != "clarification" and not plan.ambiguity_slots:
        return None
    return {
        "artifact_kind": plan.artifact_kind,
        "response_mode": plan.response_mode,
        "ambiguity_slots": list(plan.ambiguity_slots),
        "selected_strategy": plan.selected_strategy,
        "report_scope": getattr(plan, "report_scope", "none"),
        "capabilities": list(plan.capabilities),
    }
