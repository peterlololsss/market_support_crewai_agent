from __future__ import annotations

from collections.abc import Callable

import pytest

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
from market_support_crewai_agent.runtime.planning import (
    ExecutionPlanV2,
    PlanSpec,
    finalize_execution_plan_v2,
)
from market_support_crewai_agent.runtime.planning.planner_input import (
    PlannerRuntimeInputSourceV1,
    build_runtime_planner_prompt_input_v1,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology import DomainContextV1Builder
from market_support_crewai_agent.runtime.prompts.context import (
    IntentGateResult,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.prompts.profiles import ModelFamily
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderFailureCodeV1,
    ProviderInvocationError,
)
from market_support_crewai_agent.runtime.recall.flow import collect_preplanner_recall
from market_support_crewai_agent.runtime.recall.turn_state import RecallTurnStateV1
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.settings_model import Settings
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
from tests.unit.planning._planner_recall_state_fixtures import (
    FakeFrame,
    FakeProgram,
    KickoffObservation,
    PlannerRuntime,
)


@pytest.mark.anyio
async def test_selected_planner_target_is_reused_for_schema_repair(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, str | int], None],
) -> None:
    # Given: planner and composer use different providers and planner output is empty.
    envelope, policy, scope = group_context("advisory")
    static = StaticCollector(static_hit(policy, with_payload=False))
    document = DocumentCollector(document_candidate())
    recall_runtime = RecallPipelineRuntime(static, document)
    transition = await collect_preplanner_recall(
        recall_runtime,
        request=envelope.request,
        policy=policy,
        scope_authority=scope,
    )
    state = transition.turn_state
    plan_spec = PlanSpec.model_validate(
        {
            "plan_id": "planner-state-reuse",
            "user_intent_summary": "answer company question",
            "plan_units": [
                {
                    "unit_id": "company-answer",
                    "selected_capability_id": "answer_internal_company_knowledge",
                    "domain_scope": {
                        "kind": "distribution",
                        "business_scope_ref": scope.business_scope_ref,
                        "channel_kind": "bank",
                    },
                    "answerability_policy": "answer",
                    "output_schema_ref": "answer_internal_company_knowledge:output_schema",
                    "steps": [
                        {
                            "step_id": "lookup",
                            "description": "answer company question",
                            "evidence_query": "company website",
                        }
                    ],
                }
            ],
        }
    )
    plan = finalize_execution_plan_v2(plan_spec, policy, scope, origin="planner")
    runtime = PlannerRuntime()
    empty_frame, valid_frame = (
        FakeFrame("empty"),
        FakeFrame("valid"),
    )
    frames = iter((empty_frame, valid_frame))
    kickoff_observations: list[KickoffObservation] = []
    bound_inputs: list[PlannerPromptInputV1] = []
    source_recall_states: list[RecallTurnStateV1] = []

    original_build_input = build_runtime_planner_prompt_input_v1

    def build_input(source: PlannerRuntimeInputSourceV1) -> PlannerPromptInputV1:
        source_recall_states.append(source.recall_state)
        return original_build_input(source)

    def select_program(
        planner_input: PlannerPromptInputV1,
        model_family: ModelFamily,
    ) -> FakeProgram:
        bound_inputs.append(planner_input)
        return FakeProgram(model_family)

    async def kickoff(
        agent: CrewAIAgentAdapterV1,
        program: FakeProgram,
        *,
        planner_input: PlannerPromptInputV1,
        timeout_seconds: float | None,
        retry_attempts: int,
        base_delay_seconds: float,
        journal: TurnLlmInvocationJournalV1 | None = None,
        settings: Settings | None = None,
    ) -> tuple[FakeFrame, list[dict[str, str]]]:
        del timeout_seconds, retry_attempts, base_delay_seconds
        kickoff_observations.append(
            KickoffObservation(agent, program, planner_input, settings)
        )
        if journal is not None:
            _ = journal.reserve(
                stage_kind="planner_intent",
                program_id="planner_intent.wecom_group.v1@1",
                target_slot="planner",
                purpose="planner_test",
            )
        return next(frames), []

    def coerce(
        frame: FakeFrame,
        _policy: PolicyManifestV2,
        *,
        scope_authority: BusinessScopeAuthorityV1,
    ) -> tuple[ExecutionPlanV2 | None, str]:
        del _policy, scope_authority
        return (plan, "") if frame is valid_frame else (None, "invalid-plan")

    def model_family_from_fake_settings(
        settings: Settings,
        *,
        stage: str | None = None,
    ) -> ModelFamily:
        del settings
        return "gpt" if stage == "planner_intent" else "generic"

    monkeypatch.setattr(
        planning_flow,
        "select_stage_input_prompt_program",
        select_program,
    )
    monkeypatch.setattr(
        planning_flow, "model_family_from_settings", model_family_from_fake_settings
    )
    monkeypatch.setattr(
        planning_flow,
        "build_runtime_planner_prompt_input_v1",
        build_input,
    )
    monkeypatch.setattr(planning_flow, "run_planner_kickoff_with_retry", kickoff)

    monkeypatch.setattr(planning_flow, "coerce_planner_plan_with_error", coerce)
    journal = TurnLlmInvocationJournalV1()

    # When: the selected planner needs one bounded schema repair.
    result = await planning_flow.build_candidate_via_planner(
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
        action_history=[],
        prompt_programs=[],
        llm_executions=[],
        scope_authority=scope,
        state_key_ref=envelope.state_key_ref,
        recall_state=state,
        llm_journal=journal,
    )

    # Then: both invocations stay on the selected planner target and journal slot.
    assert (static.calls, document.calls) == (1, 1)
    assert len(kickoff_observations) == 2
    assert len(bound_inputs) == 2
    assert len(source_recall_states) == 2
    assert all(observed_state is state for observed_state in source_recall_states)
    assert all(
        observation.planner_input.recall.candidates == state.outcome.candidates
        for observation in kickoff_observations
    )
    assert (
        kickoff_observations[1].planner_input
        is not kickoff_observations[0].planner_input
    )
    assert tuple(observation.agent.role for observation in kickoff_observations) == (
        "primary",
        "primary",
    )
    assert kickoff_observations[0].settings is runtime.settings
    assert kickoff_observations[1].settings is runtime.settings
    assert tuple(observation.program.name for observation in kickoff_observations) == (
        "gpt",
        "gpt",
    )
    assert kickoff_observations[1].planner_input.retry_overlay is not None
    assert kickoff_observations[1].planner_input.retry_overlay.phase == "schema_repair"
    assert len(journal.rows) == 2
    assert {row.target_slot for row in journal.rows} == {"planner"}
    record_property("selected_target", "gemini/gemini-planner")
    record_property("generic_target", "openai/generic-composer")
    record_property("invocation_count", len(kickoff_observations))
    record_property("journal_rows", len(journal.rows))
    record_property("typed_outcome", "schema_repair_success")
    assert runtime.received_states == [state]
    result_state = result.recall_state
    assert result_state is not None
    assert result_state is state
    assert result_state.outcome is state.outcome
    assert result_state.outcome.trace_hash == state.outcome.trace_hash


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("provider_code", "runtime_error"),
    (
        ("provider_timeout", "CrewAI planner timed out"),
        ("provider_auth_failed", "planner_provider_failure"),
    ),
)
async def test_selected_planner_failure_is_typed_without_target_switch(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, str | int], None],
    provider_code: ProviderFailureCodeV1,
    runtime_error: str,
) -> None:
    # Given: distinct planner/composer targets and one failing planner invocation.
    envelope, policy, scope = group_context("off")
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
    runtime = PlannerRuntime()
    invocations: list[str] = []
    journal = TurnLlmInvocationJournalV1()

    def select_program(
        _planner_input: PlannerPromptInputV1,
        model_family: ModelFamily,
    ) -> FakeProgram:
        return FakeProgram(model_family)

    async def kickoff(
        agent: CrewAIAgentAdapterV1,
        _program: FakeProgram,
        *,
        planner_input: PlannerPromptInputV1,
        timeout_seconds: float | None,
        retry_attempts: int,
        base_delay_seconds: float,
        journal: TurnLlmInvocationJournalV1 | None = None,
        settings: Settings | None = None,
    ) -> tuple[FakeFrame, list[dict[str, str]]]:
        del planner_input, timeout_seconds, retry_attempts, base_delay_seconds, settings
        invocations.append(agent.role)
        if journal is not None:
            _ = journal.reserve(
                stage_kind="planner_intent",
                program_id="planner_intent.wecom_group.v1@1",
                target_slot="planner",
                purpose="planner_failure_test",
            )
        raise ProviderInvocationError(provider_code)

    monkeypatch.setattr(
        planning_flow,
        "select_stage_input_prompt_program",
        select_program,
    )
    monkeypatch.setattr(planning_flow, "run_planner_kickoff_with_retry", kickoff)

    # When: the configured planner fails before producing a PlanSpec.
    with pytest.raises(AgentRuntimeError, match=f"^{runtime_error}$"):
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
            action_history=[],
            prompt_programs=[],
            llm_executions=[],
            scope_authority=scope,
            state_key_ref=envelope.state_key_ref,
            recall_state=recall_state,
            llm_journal=journal,
        )

    # Then: exactly one selected-target invocation and one planner journal row exist.
    assert invocations == ["primary"]
    assert len(journal.rows) == 1
    assert journal.rows[0].target_slot == "planner"
    record_property("selected_target", "gemini/gemini-planner")
    record_property("generic_target", "openai/generic-composer")
    record_property("invocation_count", len(invocations))
    record_property("journal_rows", len(journal.rows))
    record_property("typed_outcome", runtime_error)
