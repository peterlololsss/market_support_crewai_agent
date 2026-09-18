from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Final, Literal

import anyio
from pydantic import ConfigDict, JsonValue
from serve_reply_fake_deps import (
    _CURRENT_ENVELOPE,
    CounterStore,
    FakeComposer,
    PlannerAgentLike,
    _plan_for,
)

from market_support_crewai_agent.runtime.context.stage_inputs import (
    PlannerPromptInputV1,
)
from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    VerifiedRequestEnvelopeV1,
    kernel_distribution_name,
    normalize_reply_request_v2,
)
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.schemas.adapter import AdapterResolveResult
from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.schemas.conversation import ReplyRequestV2
from market_support_crewai_agent.schemas.reply import ReplyResponse
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from market_support_crewai_agent.settings_model import Settings

TENANT: Final = "tenant:runtime-check"


@dataclass(frozen=True, slots=True)
class RuntimeCheckError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


class ScenarioResult(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_contract_version: Literal["reply-request.v2"]
    request_id: str
    response: ReplyResponse


class FakePreflightService:
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del resolve_material_pack_options
        if resolve_types != ["weekly_report"]:
            return AdapterPreflightSnapshot.empty()
        result = AdapterResolveResult(
            contract_version="adapter-resolve",
            resolve_type="weekly_report",
            status="resolved",
            display_name=kernel_distribution_name(request),
            reason_code="ok",
            candidates=[],
            channel_type="non_bank",
            available_artifacts=[{"type": "weekly_report", "options": []}],
            resolved_at=1,
            resolve_ref="weekly_report:runtime-check",
            period="2026-W29",
            report_date="2026-07-17",
        )
        return AdapterPreflightSnapshot(
            items=[AdapterPreflightItem(resolve_type="weekly_report", result=result)]
        )


def _request(request_id: str, *, action: bool) -> VerifiedRequestEnvelopeV1:
    artifacts = [{"type": "weekly_report", "options": []}] if action else []
    reads = ["resolve_weekly_report"] if action else []
    actions = ["send_weekly_report"] if action else []
    request = ReplyRequestV2.model_validate(
        {
            "contract_version": "reply-request.v2",
            "request_id": request_id,
            "message": "服务运行检查，请简短回答。",
            "context_id": f"ctx:{request_id.removeprefix('req:')}",
            "identity": {
                "contract_version": "conversation-identity.v1",
                "surface": "wecom",
                "scene": "group",
                "tenant_ref": TENANT,
                "group_ref": "group:runtime-check",
                "principal_ref": "principal:runtime-check",
            },
            "presentation": {
                "contract_version": "group-presentation.v1",
                "conversation_name": "runtime check",
                "principal_name": "runtime check",
            },
            "business_scope": {
                "kind": "distribution",
                "dist_channel_name": "runtime check",
                "channel_type": "non_bank",
                "available_artifacts": artifacts,
            },
            "grants": {
                "contract_version": "principal-grants.v1",
                "read_capabilities": reads,
                "outbound_actions": actions,
                "mention_types": [],
            },
        }
    )
    return normalize_reply_request_v2(request, adapter_namespace="assistant-wecom")


async def _run() -> tuple[ScenarioResult, ...]:
    from market_support_crewai_agent.runtime import planning_flow

    counters = CounterStore()
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="fixture-llm-key",
            group_recall_mode="off",
            reply_alignment_verifier_enabled=False,
        ),
        preflight_service=FakePreflightService(),
        coordinator=ReplyStateTransactionCoordinatorV1(),
        v2_composer=FakeComposer(counters),
    )

    async def fake_planner(
        _planner_agent: PlannerAgentLike | None,
        _planner_program: PromptProgram,
        *,
        planner_input: PlannerPromptInputV1,
        timeout_seconds: float | None,
        retry_attempts: int,
        base_delay_seconds: float,
        journal: TurnLlmInvocationJournalV1 | None = None,
        settings: Settings | None = None,
    ) -> tuple[SimpleNamespace, list[dict[str, JsonValue]]]:
        del timeout_seconds, retry_attempts, base_delay_seconds, journal, settings
        envelope = _CURRENT_ENVELOPE.get()
        if envelope is None:
            raise RuntimeCheckError("runtime check request context is unavailable")
        counters.set_history_count(
            envelope.request.request_id, len(planner_input.history)
        )
        plan = _plan_for(envelope)
        return SimpleNamespace(pydantic=plan, raw=plan.model_dump_json()), []

    planning_flow.run_planner_kickoff_with_retry = fake_planner
    results: list[ScenarioResult] = []
    for envelope in (
        _request("req:http-group-safe", action=False),
        _request("req:http-group-action", action=True),
    ):
        token = _CURRENT_ENVELOPE.set(envelope)
        try:
            response = await runtime.reply(envelope)
        finally:
            _CURRENT_ENVELOPE.reset(token)
        results.append(
            ScenarioResult(
                request_contract_version="reply-request.v2",
                request_id=envelope.request.request_id,
                response=response,
            )
        )
    if results[0].response.actions:
        raise RuntimeCheckError("safe runtime check unexpectedly proposed an action")
    if len(results[1].response.actions) != 1:
        raise RuntimeCheckError("runtime check did not produce one typed action")
    if results[1].response.actions[0].type != "send_weekly_report":
        raise RuntimeCheckError("runtime check produced the wrong action type")
    return tuple(results)


def main() -> None:
    results = anyio.run(_run)
    print(
        json.dumps(
            [result.model_dump(mode="json") for result in results],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
