from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import final

import pytest
from typing_extensions import override

from market_support_crewai_agent.runtime import planning_flow
from market_support_crewai_agent.runtime.context.stage_inputs import (
    PlannerPromptInputV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology import DomainContextV1Builder
from market_support_crewai_agent.runtime.prompts.context import IntentGateResult
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.prompts.profiles import ModelFamily
from market_support_crewai_agent.runtime.recall.flow import collect_preplanner_recall
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.crewai_adapter import make_agent_adapter
from tests.integration.runtime._recall_pipeline_fixtures import (
    document_candidate,
    group_context,
    static_hit,
)
from tests.integration.runtime._recall_pipeline_runtime_fixtures import (
    DocumentCollector,
    RecallPipelineRuntime,
    StaticCollector,
)


@dataclass(frozen=True, slots=True)
class _Profile:
    stage: str = "planner_intent"


@dataclass(frozen=True, slots=True)
class _Program:
    name: str = "target"
    profile: _Profile = _Profile()
    scene_key: str = "wecom_group.v1"


@dataclass(frozen=True, slots=True)
class _KickoffCall:
    agent: str
    planner_input: PlannerPromptInputV1
    settings: Settings | None


@final
class _Runtime(RecallPipelineRuntime):
    def __init__(
        self,
        policy: PolicyManifestV2,
        planner_agent: CrewAIAgentAdapterV1,
    ) -> None:
        super().__init__(
            StaticCollector(static_hit(policy, with_payload=False)),
            DocumentCollector(document_candidate()),
        )
        self.planner_agent: CrewAIAgentAdapterV1 = planner_agent

    @override
    def build_planner_agent(self) -> CrewAIAgentAdapterV1:
        return self.planner_agent


@pytest.mark.anyio
async def test_planner_kickoff_receives_strict_advisory_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a policy-eligible advisory candidate and a fake planner boundary.
    envelope, policy, scope = group_context("advisory")
    recall_runtime = RecallPipelineRuntime(
        StaticCollector(static_hit(policy, with_payload=False)),
        DocumentCollector(document_candidate()),
    )
    recall_state = (
        await collect_preplanner_recall(
            recall_runtime,
            request=envelope.request,
            policy=policy,
            scope_authority=scope,
        )
    ).turn_state
    invalid_frame = SimpleNamespace(raw="invalid", pydantic=None)
    valid_frame = SimpleNamespace(raw="{}", pydantic=None)
    frames = iter((invalid_frame, valid_frame))
    plan = SimpleNamespace(
        response_mode="knowledge_answer",
        selected_manifest_refs=policy.eligible_capabilities[:1],
        adapter_resolves=(),
    )
    kickoff_calls: list[_KickoffCall] = []

    async def kickoff(
        agent: CrewAIAgentAdapterV1,
        _program: _Program,
        *,
        planner_input: PlannerPromptInputV1,
        timeout_seconds: float | None,
        retry_attempts: int,
        base_delay_seconds: float,
        journal: TurnLlmInvocationJournalV1 | None = None,
        settings: Settings | None = None,
    ) -> tuple[SimpleNamespace, list[dict[str, str]]]:
        del timeout_seconds, retry_attempts, base_delay_seconds, journal
        kickoff_calls.append(_KickoffCall(agent.role, planner_input, settings))
        return next(frames), []

    def select_program(
        _input: PlannerPromptInputV1,
        _model_family: ModelFamily,
    ) -> _Program:
        return _Program()

    def model_family_from_fake_settings(
        settings: Settings,
        *,
        stage: str | None = None,
    ) -> ModelFamily:
        del settings, stage
        return "generic"

    def coerce_plan(
        frame: SimpleNamespace,
        _policy: PolicyManifestV2,
        *,
        scope_authority: BusinessScopeAuthorityV1,
    ) -> tuple[SimpleNamespace | None, str]:
        del _policy, scope_authority
        return (plan, "") if frame is valid_frame else (None, "invalid-plan")

    monkeypatch.setattr(
        planning_flow, "select_stage_input_prompt_program", select_program
    )
    monkeypatch.setattr(
        planning_flow, "model_family_from_settings", model_family_from_fake_settings
    )
    monkeypatch.setattr(planning_flow, "run_planner_kickoff_with_retry", kickoff)
    monkeypatch.setattr(planning_flow, "coerce_planner_plan_with_error", coerce_plan)

    # When: the production planner flow performs its initial round.
    runtime = _Runtime(
        policy,
        make_agent_adapter(role="planner-target"),
    )
    _ = await planning_flow.build_candidate_via_planner(
        runtime,
        request=envelope.request,
        domain_context=DomainContextV1Builder().build(
            envelope.request,
            scope_authority=scope,
        ),
        policy=policy,
        model_family="generic",
        intent_gate=IntentGateResult(artifact_hint="unclear"),
        history=[],
        action_history=(),
        prompt_programs=[],
        llm_executions=[],
        scope_authority=scope,
        state_key_ref=envelope.state_key_ref,
        recall_state=recall_state,
    )

    # Then: kickoff receives the strict DTO, not a raw runtime/request bag.
    assert len(kickoff_calls) == 2
    planner_input = kickoff_calls[0].planner_input
    assert isinstance(planner_input, PlannerPromptInputV1)
    assert (
        planner_input.eligible_capabilities.manifest_refs
        == policy.eligible_capabilities
    )
    assert planner_input.recall.candidates == recall_state.outcome.candidates
    assert "shortcut_payload" not in type(planner_input).model_fields
    assert "answer" not in planner_input.recall.model_dump_json()
    retry_input = kickoff_calls[1].planner_input
    assert retry_input is not planner_input
    assert retry_input.retry_overlay is not None
    assert (
        retry_input.retry_overlay.attempt,
        retry_input.retry_overlay.phase,
        retry_input.retry_overlay.phase_attempt,
    ) == (1, "schema_repair", 1)
    assert tuple(call.agent for call in kickoff_calls) == (
        "planner-target",
        "planner-target",
    )
    assert kickoff_calls[0].settings is runtime.settings
    assert kickoff_calls[1].settings is runtime.settings
