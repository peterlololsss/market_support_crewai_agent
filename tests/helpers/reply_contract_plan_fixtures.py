from __future__ import annotations

from pydantic import JsonValue

from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.planning.plan_spec import PlanSpec
from tests.helpers.reply_contract_json import JsonInput, json_mapping
from tests.helpers.reply_contract_plans import plan_from_payload


def make_support_plan_spec(
    request: KernelReplyRequestV1 | None = None,
    **overrides: JsonInput,
) -> PlanSpec:
    payload: dict[str, JsonValue] = {
        "user_need": "answer current market support request",
        "artifact_kind": "unclear",
        "action_intent": "none",
        "ambiguity_slots": ["artifact"],
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal product or support request",
        },
        "confidence": 0.8,
    }
    payload.update(json_mapping(overrides))
    return plan_from_payload(request, payload)


def make_weekly_plan_spec(
    request: KernelReplyRequestV1 | None = None,
    **overrides: JsonInput,
) -> PlanSpec:
    payload: dict[str, JsonValue] = {
        "user_need": "send weekly report",
        "artifact_kind": "weekly_report",
        "action_intent": "send",
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal weekly report request",
        },
        "confidence": 0.8,
    }
    payload.update(json_mapping(overrides))
    return plan_from_payload(request, payload)


def make_monthly_plan_spec(
    request: KernelReplyRequestV1 | None = None,
    **overrides: JsonInput,
) -> PlanSpec:
    payload: dict[str, JsonValue] = {
        "user_need": "send monthly report",
        "artifact_kind": "monthly_report",
        "action_intent": "send",
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal monthly report request",
        },
        "confidence": 0.8,
    }
    payload.update(json_mapping(overrides))
    return plan_from_payload(request, payload)
