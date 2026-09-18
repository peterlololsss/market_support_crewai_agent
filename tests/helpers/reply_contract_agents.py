from __future__ import annotations

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.planning.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from tests.helpers.crewai_adapter import make_completion_agent_adapter
from tests.helpers.reply_contract_plan_fixtures import make_support_plan_spec


def FakePlannerAgent(
    plan_spec: PlanSpec | None = None,
    prompts: list[str] | None = None,
) -> CrewAIAgentAdapterV1:
    def remember(prompt: str) -> None:
        if prompts is not None:
            prompts.append(prompt)

    return make_completion_agent_adapter(
        lambda _response_format: plan_spec or make_support_plan_spec(),
        role="planner",
        model="fake-planner",
        on_prompt=remember,
    )


class FakePlannerAgentFactory:
    def __init__(
        self,
        plan_spec: PlanSpec | None = None,
        prompts: list[str] | None = None,
    ) -> None:
        self.plan_spec: PlanSpec | None = plan_spec
        self.prompts: list[str] | None = prompts

    def _build_agent(self) -> CrewAIAgentAdapterV1:
        plan_spec = self.plan_spec or make_support_plan_spec()

        def on_prompt(prompt: str) -> None:
            if self.prompts is not None:
                self.prompts.append(prompt)

        return make_completion_agent_adapter(
            lambda _response_format: plan_spec,
            role="planner",
            model="fake-planner",
            on_prompt=on_prompt,
        )

    def build_planner_agent(self) -> CrewAIAgentAdapterV1:
        return self._build_agent()

    def build_composer_agent(self, stage: str) -> CrewAIAgentAdapterV1:
        del stage
        return self._build_agent()

    def build_alignment_verifier_agent(self) -> CrewAIAgentAdapterV1:
        return self._build_agent()


def install_fake_planner(
    runtime: CrewAIReplyRuntime,
    plan_spec: PlanSpec | None = None,
) -> None:
    runtime.planner_agent_factory = FakePlannerAgentFactory(plan_spec)
