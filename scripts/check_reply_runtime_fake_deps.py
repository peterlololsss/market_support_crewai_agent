from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace

from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
)
from market_support_crewai_agent.runtime.domain.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.state.audit import AuditStore
from market_support_crewai_agent.runtime.state.conversation_store import ConversationStore
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
    plan_spec: PlanSpec


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


def _plan_spec(request: ReplyRequest, **overrides) -> PlanSpec:
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
    capability_id = _capability_id_from_payload(payload)
    manifest = CAPABILITY_MANIFEST_REGISTRY.get(capability_id)
    strategy = payload.get("selected_strategy")
    if (
        manifest.runtime_capability == "material_pack"
        and request.channel_type == "bank"
        and strategy is None
        and len(request.available_strategies) == 1
    ):
        strategy = request.available_strategies[0]
    step = {
        "step_id": "step-1",
        "description": payload["user_need"],
        "uses_artifacts": list(manifest.required_artifacts),
        "required_artifacts": list(manifest.required_artifacts),
        "allowed_artifacts": list(manifest.allowed_artifacts),
        "forbidden_artifacts": list(manifest.forbidden_artifacts),
        "required_tools": list(manifest.required_tools),
        "evidence_query": payload.get("evidence_query"),
    }
    return PlanSpec.model_validate(
        {
            "plan_id": f"plan-{capability_id}",
            "selected_capability_id": capability_id,
            "user_intent_summary": payload["user_need"],
            "domain_scope": {
                "channel_id": request.group_id or request.conversation_key,
                "channel_kind": request.channel_type,
                "strategy_id": strategy,
                "strategy_name": strategy,
                "product_ids": [],
            },
            "required_artifacts": list(manifest.required_artifacts),
            "allowed_artifacts": list(manifest.allowed_artifacts),
            "forbidden_artifacts": list(manifest.forbidden_artifacts),
            "required_tools": list(manifest.required_tools),
            "answerability_policy": "send"
            if capability_id.endswith(".send")
            else "clarify",
            "output_schema_ref": f"{manifest.id}:output_schema",
            "evidence_contract_ref": f"{manifest.id}:evidence_contract",
            "evidence_contract": manifest.evidence_contract,
            "steps": [step],
            "acceptance_criteria": ["satisfy selected capability contract"],
            "abstention_cases": [manifest.abstention_policy.guidance]
            if manifest.abstention_policy.guidance
            else [],
            "risk_flags": list(payload.get("ambiguity_slots") or []),
        }
    )


def _capability_id_from_payload(payload: dict) -> str:
    artifact_kind = payload.get("artifact_kind", "unclear")
    action_intent = payload.get("action_intent", "none")
    if payload.get("ambiguity_slots") or payload.get("report_scope") == "ambiguous":
        return "general.clarification"
    if action_intent == "send":
        if artifact_kind == "material_pack":
            return "material_pack.send"
        if artifact_kind == "weekly_report":
            return "weekly_report.send"
        if artifact_kind == "monthly_report":
            return "monthly_report.send"
    return "general.clarification"


def _weekly_scenario() -> RuntimeScenario:
    request = _request("请发一下周报", available_strategies=[])
    return RuntimeScenario(
        name="weekly_report_action",
        request=request,
        plan_spec=_plan_spec(
            request,
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
        plan_spec=_plan_spec(
            request,
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
    runtime._build_planner_agent = lambda: FakeCrewAgent(scenario.plan_spec)  # type: ignore[method-assign]
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
