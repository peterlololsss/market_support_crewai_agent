from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace

from market_support_crewai_agent.runtime.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.audit import AuditStore
from market_support_crewai_agent.runtime.conversation_store import ConversationStore
from market_support_crewai_agent.runtime.planning import ReplyPlan
from market_support_crewai_agent.runtime.reply_agent import CrewAIReplyRuntime
from market_support_crewai_agent.schemas import (
    AdapterResolveResult,
    AdapterResolveType,
    PrimaryReply,
    ReplyRequest,
    ReplyResponse,
    SendMaterialPackAction,
    SendWeeklyReportAction,
)
from market_support_crewai_agent.settings import Settings


@dataclass(frozen=True)
class SmokeScenario:
    name: str
    request: ReplyRequest
    plan: ReplyPlan
    composer_response: ReplyResponse


class FakeCrewAgent:
    def __init__(self, pydantic_result):
        self.pydantic_result = pydantic_result

    async def kickoff_async(self, prompt, response_format):
        del prompt, response_format
        return SimpleNamespace(
            pydantic=self.pydantic_result,
            raw="",
            agent_role="smoke",
            usage_metrics={"total_tokens": 0},
            todos=[],
        )


class FakePreflightService:
    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_strategies=None,
    ):
        del canonical_context
        requested = resolve_types or [
            "material_pack",
            "weekly_report",
            "monthly_report",
            "sales_mention",
        ]
        resolve_strategies = resolve_strategies or {}
        return AdapterPreflightSnapshot(
            items=[
                _resolved_item(
                    resolve_type,
                    request,
                    strategy=resolve_strategies.get(resolve_type),
                )
                for resolve_type in requested
            ]
        )


def _resolved_item(
    resolve_type: AdapterResolveType,
    request: ReplyRequest,
    *,
    strategy: str | None = None,
) -> AdapterPreflightItem:
    return AdapterPreflightItem(
        resolve_type=resolve_type,
        result=AdapterResolveResult.model_validate(
            {
                "contract_version": "adapter-resolve.v1",
                "resolve_type": resolve_type,
                "status": "resolved",
                "display_name": request.dist_channel_name,
                "reason_code": "ok",
                "candidates": request.available_strategies,
                "channel_type": request.channel_type,
                "available_materials": request.available_materials,
                "available_strategies": request.available_strategies,
                "resolved_at": 1,
                "strategy": strategy,
                "period": "20260529" if resolve_type == "weekly_report" else None,
                "scope_status": "unknown",
            }
        ),
    )


def _request(message: str, **overrides) -> ReplyRequest:
    payload = {
        "context_id": "smoke-msg-1",
        "conversation_key": "wecom:smoke-group:smoke-sender",
        "group_id": "smoke-group",
        "sender_id": "smoke-sender",
        "message": message,
        "is_group": True,
        "group_name": "smoke group",
        "dist_channel_name": "测试渠道",
        "sender_nickname": "测试用户",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["中证500", "中证1000"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def _plan(**overrides) -> ReplyPlan:
    payload = {
        "user_need": "smoke test",
        "intent": "clarification",
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "smoke test compliant request",
        },
        "confidence": 0.8,
    }
    payload.update(overrides)
    return ReplyPlan.model_validate(payload)


def _weekly_scenario() -> SmokeScenario:
    request = _request("请发一下周报", available_strategies=[])
    return SmokeScenario(
        name="weekly_report_action",
        request=request,
        plan=_plan(
            user_need="send weekly report",
            intent="send_weekly_report",
            required_adapter_resolves=["weekly_report"],
            evidence_requests=[
                {
                    "capability": "resolve_weekly_report",
                    "reason": "confirm weekly report can be sent",
                }
            ],
            candidate_actions=[
                {"type": "send_weekly_report", "report_scope": "channel_all"}
            ],
        ),
        composer_response=ReplyResponse(
            response_id="smoke-weekly",
            reply=PrimaryReply(kind="answer", text=""),
            actions=[
                SendWeeklyReportAction(
                    type="send_weekly_report",
                    action_id="send-weekly",
                )
            ],
        ),
    )


def _bank_material_clarification_scenario() -> SmokeScenario:
    request = _request("发一下材料包")
    return SmokeScenario(
        name="bank_material_strategy_clarification",
        request=request,
        plan=_plan(
            user_need="send material pack but strategy is unclear",
            intent="send_material_pack",
            required_adapter_resolves=["material_pack"],
            evidence_requests=[
                {
                    "capability": "resolve_material_pack",
                    "reason": "confirm material pack can be sent",
                }
            ],
            candidate_actions=[
                {"type": "send_material_pack"}
            ],
        ),
        composer_response=ReplyResponse(
            response_id="smoke-material",
            reply=PrimaryReply(kind="answer", text=""),
            actions=[
                SendMaterialPackAction(
                    type="send_material_pack",
                    action_id="send-material",
                )
            ],
        ),
    )


async def _run_scenario(scenario: SmokeScenario) -> dict:
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="smoke-key", adapter_preflight_enabled=True),
        conversation_store=ConversationStore(),
        action_ledger=ActionLedger(),
        preflight_service=FakePreflightService(),
        audit_store=AuditStore(),
    )
    runtime._build_planner_agent = lambda: FakeCrewAgent(scenario.plan)  # type: ignore[method-assign]
    runtime._build_agent = lambda: FakeCrewAgent(scenario.composer_response)  # type: ignore[method-assign]

    response = await runtime.reply(scenario.request)
    return {
        "scenario": scenario.name,
        "response": response.model_dump(mode="json", exclude_none=True),
    }


async def main() -> None:
    results = [
        await _run_scenario(scenario)
        for scenario in (
            _weekly_scenario(),
            _bank_material_clarification_scenario(),
        )
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
