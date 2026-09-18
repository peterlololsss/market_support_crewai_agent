from __future__ import annotations

import json

from pydantic import JsonValue

from market_support_crewai_agent.runtime.planning import ExecutionPlanV2
from market_support_crewai_agent.schemas.reply import ReplyResponse


def compact_assistant_result(response: ReplyResponse, plan: ExecutionPlanV2) -> str:
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


def _compact_pending_plan(plan: ExecutionPlanV2) -> dict[str, JsonValue] | None:
    ambiguity_slots = list(
        dict.fromkeys(slot for unit in plan.units for slot in unit.ambiguity_slots)
    )
    if plan.response_mode != "clarification" and not ambiguity_slots:
        return None
    ambiguity_slots_json: list[JsonValue] = list(ambiguity_slots)
    capabilities_json: list[JsonValue] = list(
        dict.fromkeys(
            capability
            for unit in plan.units
            for capability in unit.runtime_capabilities
        )
    )
    payload: dict[str, JsonValue] = {
        "artifact_kind": plan.artifact_kind,
        "response_mode": plan.response_mode,
        "ambiguity_slots": ambiguity_slots_json,
        "material_pack_option": next(
            (
                intent.material_pack_option
                for intent in plan.action_intents
                if intent.material_pack_option is not None
            ),
            None,
        ),
        "capabilities": capabilities_json,
    }
    return payload
