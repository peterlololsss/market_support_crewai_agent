from __future__ import annotations

import json
from types import SimpleNamespace

from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.domain.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.orchestration.reply_agent import CrewAIReplyRuntime
from market_support_crewai_agent.schemas import AdapterResolveResult
from tests.helpers.planning import make_plan_spec


class FakePlannerAgent:
    def __init__(self, plan_spec: PlanSpec | None = None, prompts: list[str] | None = None):
        self.plan_spec = plan_spec or make_plan_spec(
            artifact_kind="unclear",
            action_intent="none",
            ambiguity_slots=["request_meaning"],
        )
        self.prompts = prompts

    async def kickoff_async(self, prompt, response_format):
        if self.prompts is not None:
            self.prompts.append(prompt)
        return SimpleNamespace(pydantic=self.plan_spec, raw="")


def make_support_plan_spec(**overrides) -> PlanSpec:
    payload = {
        "user_need": "answer current market support request",
        "artifact_kind": "unclear",
        "action_intent": "none",
        "ambiguity_slots": ["request_meaning"],
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal product or support request",
        },
        "confidence": 0.8,
    }
    payload.update(overrides)
    return make_plan_spec(**payload)


def make_weekly_plan_spec(**overrides) -> PlanSpec:
    payload = {
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
    payload.update(overrides)
    return make_plan_spec(**payload)


def make_monthly_plan_spec(**overrides) -> PlanSpec:
    payload = {
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
    payload.update(overrides)
    return make_plan_spec(**payload)


def install_fake_planner(runtime: CrewAIReplyRuntime, plan_spec: PlanSpec | None = None):
    runtime._build_planner_agent = lambda: FakePlannerAgent(plan_spec)  # type: ignore[method-assign]


def make_payload(message: str = "hello", **overrides):
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": message,
        "is_group": True,
        "group_name": "test group",
        "dist_channel_name": "test channel",
        "sender_nickname": "test user",
        "available_materials": ["material", "weekly", "monthly"],
        "material_pack_options": [],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return payload


def _assistant_history_with_pending(
    *,
    text: str,
    pending_plan: dict,
) -> str:
    return json.dumps(
        {
            "contract_version": "reply-runtime-history",
            "reply_response": {
                "contract_version": "reply",
                "response_id": "resp-history",
                "reply": {"kind": "clarification", "text": text, "mentions": []},
                "actions": [],
            },
            "pending_plan": pending_plan,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def resolved_item(resolve_type: str, **overrides) -> AdapterPreflightItem:
    payload = {
        "contract_version": "adapter-resolve",
        "resolve_type": resolve_type,
        "status": "resolved",
        "display_name": "测试渠道",
        "reason_code": "ok",
        "candidates": [],
        "channel_type": "bank",
        "available_materials": ["material", "weekly", "monthly"],
        "material_pack_options": ["指增"],
        "resolved_at": 1,
        "resolve_ref": f"{resolve_type}:ref",
    }
    payload.update(overrides)
    return AdapterPreflightItem(
        resolve_type=resolve_type,
        result=AdapterResolveResult.model_validate(payload),
    )


class ResolvedWeeklyPreflight:
    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_material_pack_options=None,
    ):
        del request, canonical_context, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot(
            items=[
                resolved_item(
                    "weekly_report",
                    resolve_ref="weekly:ref",
                    period="20260529",
                    report_date="2026-05-29",
                )
            ]
        )


class ResolvedMonthlyPreflight:
    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_material_pack_options=None,
    ):
        del request, canonical_context, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot(
            items=[
                resolved_item(
                    "monthly_report",
                    resolve_ref="monthly:ref",
                    period="202605",
                    report_date="2026-05-31",
                )
            ]
        )


class ResolvedWeeklyMonthlyPreflight:
    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_material_pack_options=None,
    ):
        del request, canonical_context, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot(
            items=[
                resolved_item(
                    "weekly_report",
                    resolve_ref="weekly:ref",
                    period="20260529",
                    report_date="2026-05-29",
                ),
                resolved_item(
                    "monthly_report",
                    resolve_ref="monthly:ref",
                    period="202605",
                    report_date="2026-05-31",
                ),
            ]
        )


class CapturingResolvedWeeklyPreflight:
    def __init__(self):
        self.resolve_material_pack_options = None

    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_material_pack_options=None,
    ):
        del request, canonical_context, resolve_types
        self.resolve_material_pack_options = resolve_material_pack_options or {}
        return AdapterPreflightSnapshot(
            items=[
                resolved_item(
                    "weekly_report",
                    resolve_ref="weekly:ref",
                    period="20260529",
                    report_date="2026-05-29",
                )
            ]
        )


class CapturingResolvedMaterialPreflight:
    def __init__(self):
        self.resolve_material_pack_options = None

    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_material_pack_options=None,
    ):
        del request, canonical_context, resolve_types
        self.resolve_material_pack_options = resolve_material_pack_options or {}
        material_pack_option = self.resolve_material_pack_options.get("material_pack")
        return AdapterPreflightSnapshot(
            items=[
                resolved_item(
                    "material_pack",
                    resolve_ref="material:ref",
                    material_pack_option=material_pack_option,
                )
            ]
        )


class MissingWeeklyWithSalesPreflight:
    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_material_pack_options=None,
    ):
        del request, canonical_context, resolve_types, resolve_material_pack_options
        missing = {
            "contract_version": "adapter-resolve",
            "resolve_type": "weekly_report",
            "status": "missing",
            "display_name": "测试渠道",
            "reason_code": "weekly_report_unavailable",
            "candidates": [],
            "channel_type": "bank",
            "available_materials": [],
            "material_pack_options": [],
            "resolved_at": 1,
        }
        return AdapterPreflightSnapshot(
            items=[
                AdapterPreflightItem(
                    resolve_type="weekly_report",
                    result=AdapterResolveResult.model_validate(missing),
                ),
                resolved_item("sales_mention", resolve_ref="sales:ref"),
            ]
        )


class EmptyPreflightService:
    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_material_pack_options=None,
    ):
        del request, canonical_context, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot.empty()


class CapturingEmptyPreflightService:
    def __init__(self):
        self.calls = []

    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_material_pack_options=None,
    ):
        self.calls.append(
            {
                "resolve_types": list(resolve_types or []),
                "resolve_material_pack_options": dict(
                    resolve_material_pack_options or {}
                ),
            }
        )
        del request, canonical_context
        return AdapterPreflightSnapshot.empty()
