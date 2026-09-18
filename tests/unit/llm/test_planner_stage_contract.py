from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import BaseModel, JsonValue
from typing_extensions import override

from market_support_crewai_agent.runtime import planning_flow
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAICompletionValueV1,
    CrewAIKickoffOutputV1,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology import DomainContextV1Builder
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.context import (
    IntentGateResult,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.recall.flow import collect_preplanner_recall
from tests.helpers.crewai_adapter import (
    invoke_completion,
    make_agent_adapter,
    make_llm_adapter,
)
from tests.helpers.planning import make_plan_spec
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
async def test_planner_actual_dispatch_records_registry_program_in_turn_journal() -> (
    None
):
    # Given: a real planner flow with a fake external agent and a turn LLM journal.
    envelope, policy, scope = group_context("off")
    plan_spec = make_plan_spec(
        envelope.request,
        selected_capability_id="answer_internal_company_knowledge",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["document_context"],
        answerability_policy="answer",
        user_intent_summary="answer company website",
        evidence_query="公司网址",
    )
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
    provider_requests: list[Mapping[str, JsonValue]] = []

    def complete(
        *,
        params: Mapping[str, JsonValue],
        available_functions: JsonValue | None = None,
        from_task: JsonValue | None = None,
        from_agent: JsonValue | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> CrewAICompletionValueV1:
        del available_functions, from_task, from_agent
        assert response_model is not None
        assert response_model.__name__ == "PlanSpec"
        provider_requests.append(params)
        return plan_spec

    llm = make_llm_adapter(
        provider="openai",
        model="unit-test-model",
        base_url="https://unit.test/v1",
        api_key="unit-test-key",
        temperature=0.0,
        max_tokens=1200,
        completion=complete,
    )

    def kickoff(prompt: str, response_format: type[BaseModel]) -> CrewAIKickoffOutputV1:
        completion_value = invoke_completion(
            llm,
            params={
                "model": llm.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            response_model=response_format,
        )
        assert isinstance(completion_value, BaseModel)
        return CrewAIKickoffOutputV1(
            raw=completion_value.model_dump_json(),
            pydantic=completion_value,
        )

    runtime = _Runtime(
        policy,
        make_agent_adapter(role="planner", llm=llm, on_prompt=kickoff),
    )
    prompt_programs: list[PromptProgram] = []
    llm_executions: list[dict[str, JsonValue]] = []
    journal = TurnLlmInvocationJournalV1()

    # When: the production planner flow assembles, builds the agent, and dispatches.
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
        prompt_programs=prompt_programs,
        llm_executions=llm_executions,
        scope_authority=scope,
        state_key_ref=envelope.state_key_ref,
        recall_state=recall_state,
        llm_journal=journal,
    )

    # Then: actual dispatch used the packaged planner program and closed one row.
    assert len(provider_requests) == 1
    assert len(prompt_programs) == 1
    assert prompt_programs[0].program_id == "planner_intent.wecom_group.v1@1"
    assert len(llm_executions) == 1
    assert len(journal.rows) == 1
    assert journal.rows[0].program_id == prompt_programs[0].program_id
    assert journal.rows[0].status == "success"
    assert journal.rows[0].prh1 is not None
    assert journal.rows[0].output_digest is not None
