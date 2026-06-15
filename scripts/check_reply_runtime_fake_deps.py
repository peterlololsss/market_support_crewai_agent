from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace

from market_support_crewai_agent.runtime.state.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.state.audit import AuditStore
from market_support_crewai_agent.runtime.state.conversation_store import ConversationStore
from market_support_crewai_agent.runtime.domain.planning import IntentFrame
from market_support_crewai_agent.runtime.orchestration.reply_agent import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas import (
    AdapterResolveResult,
    AdapterResolveStatus,
    AdapterResolveType,
    ReplyRequest,
)
from market_support_crewai_agent.settings import Settings


@dataclass(frozen=True)
class RuntimeScenario:
    name: str
    request: ReplyRequest
    intent_frame: IntentFrame


class FakeCrewAgent:
    def __init__(self, pydantic_result):
        self.pydantic_result = pydantic_result

    async def kickoff_async(self, prompt, response_format):
        del prompt, response_format
        return SimpleNamespace(
            pydantic=self.pydantic_result,
            raw="",
            agent_role="fake-deps-check",
            usage_metrics={"total_tokens": 0},
            todos=[],
        )


class ComposerShouldNotRun:
    async def kickoff_async(self, prompt, response_format):
        del prompt, response_format
        raise AssertionError("fake-deps scenarios should be deterministic")


class PassingAlignmentVerifier:
    async def verify(self, **kwargs):
        del kwargs
        return ReplyAlignmentVerdict(
            aligned=True,
            safe_to_return=True,
            confidence=1.0,
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
                _resolve_item(
                    resolve_type,
                    request,
                    strategy=resolve_strategies.get(resolve_type),
                )
                for resolve_type in requested
            ]
        )


def _resolve_item(
    resolve_type: AdapterResolveType,
    request: ReplyRequest,
    *,
    strategy: str | None = None,
) -> AdapterPreflightItem:
    status: AdapterResolveStatus = "resolved"
    resolve_ref = f"{resolve_type}:runtime-check-ref"
    if (
        resolve_type == "material_pack"
        and request.channel_type == "bank"
        and not strategy
        and len(request.available_strategies) > 1
    ):
        status = "ambiguous"
        resolve_ref = None

    payload = {
        "contract_version": "adapter-resolve",
        "resolve_type": resolve_type,
        "status": status,
        "display_name": request.dist_channel_name,
        "reason_code": "ok" if status == "resolved" else "multiple_candidates",
        "candidates": request.available_strategies if status == "ambiguous" else [],
        "channel_type": request.channel_type,
        "available_materials": request.available_materials,
        "available_strategies": request.available_strategies,
        "resolved_at": 1,
        "resolve_ref": resolve_ref,
        "strategy": strategy,
        "period": "20260529" if resolve_type == "weekly_report" else None,
        "report_date": "2026-05-29" if resolve_type == "weekly_report" else None,
        "scope_status": "included" if resolve_type == "weekly_report" else "unknown",
        "contains_strategy": True if resolve_type == "weekly_report" else None,
    }
    return AdapterPreflightItem(
        resolve_type=resolve_type,
        result=AdapterResolveResult.model_validate(payload),
    )


def _request(message: str, **overrides) -> ReplyRequest:
    payload = {
        "context_id": "runtime-check-msg-1",
        "conversation_key": "wecom:runtime-check-group:runtime-check-sender",
        "group_id": "runtime-check-group",
        "sender_id": "runtime-check-sender",
        "message": message,
        "is_group": True,
        "group_name": "runtime check group",
        "dist_channel_name": "测试渠道",
        "sender_nickname": "测试用户",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["中证500", "中证1000"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def _intent_frame(**overrides) -> IntentFrame:
    payload = {
        "user_need": "runtime check",
        "artifact_kind": "unclear",
        "action_intent": "none",
        "report_scope": "none",
        "ambiguity_slots": ["request_meaning"],
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "runtime check compliant request",
        },
        "confidence": 0.8,
    }
    payload.update(overrides)
    return IntentFrame.model_validate(payload)


def _weekly_scenario() -> RuntimeScenario:
    request = _request("请发一下周报", available_strategies=[])
    return RuntimeScenario(
        name="weekly_report_action",
        request=request,
        intent_frame=_intent_frame(
            user_need="send weekly report",
            artifact_kind="weekly_report",
            action_intent="send",
            report_scope="channel_all",
            ambiguity_slots=[],
            requested_capabilities=["weekly_report"],
        ),
    )


def _bank_material_clarification_scenario() -> RuntimeScenario:
    request = _request("发一下材料包")
    return RuntimeScenario(
        name="bank_material_strategy_clarification",
        request=request,
        intent_frame=_intent_frame(
            user_need="send material pack but strategy is unclear",
            artifact_kind="material_pack",
            action_intent="send",
            report_scope="none",
            ambiguity_slots=[],
            requested_capabilities=["material_pack"],
        ),
    )


async def _run_scenario(scenario: RuntimeScenario) -> dict:
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="check-key"),
        conversation_store=ConversationStore(),
        action_ledger=ActionLedger(),
        preflight_service=FakePreflightService(),
        audit_store=AuditStore(),
        alignment_verifier=PassingAlignmentVerifier(),
    )
    runtime._build_planner_agent = lambda: FakeCrewAgent(scenario.intent_frame)  # type: ignore[method-assign]
    runtime._build_agent = lambda: ComposerShouldNotRun()  # type: ignore[method-assign]

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
