from __future__ import annotations

import asyncio
from types import SimpleNamespace

from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.prompting.profiles import prompt_profile_by_stage
from market_support_crewai_agent.runtime.orchestration.crewai_io import run_crewai_kickoff
from tests.helpers.reply_contract import make_weekly_plan_spec


def test_run_crewai_kickoff_uses_gemini_structured_schema():
    class FakeModels:
        def generate_content(self, *, model, contents, config):
            self.model = model
            self.contents = contents
            self.config = config
            return SimpleNamespace(
                text=make_weekly_plan_spec().model_dump_json(),
                usage_metadata=None,
            )

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    class FakeLlm:
        provider = "gemini"
        model = "gemini-3-flash-preview"
        temperature = 0.1
        top_p = None
        top_k = None
        max_output_tokens = 6000
        stop_sequences = []
        thinking_config = None

        def __init__(self):
            self.client = FakeClient()

        def _get_sync_client(self):
            return self.client

    class FakeAgent:
        role = "planner"
        llm = FakeLlm()

    agent = FakeAgent()

    result, _ = asyncio.run(run_crewai_kickoff(agent, _program(), timeout_seconds=10))

    config = agent.llm.client.models.config
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema["properties"]["plan_units"]["type"] == "array"
    assert result.pydantic is not None


def test_run_crewai_kickoff_keeps_response_format_for_default_provider():
    class FakeAgent:
        llm = SimpleNamespace(provider="openai")

        async def kickoff_async(self, prompt, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(raw="{}", pydantic=None, agent_role="", usage_metrics=None)

    agent = FakeAgent()

    asyncio.run(run_crewai_kickoff(agent, _program(), timeout_seconds=1))

    assert agent.kwargs["response_format"].__name__ == "PlanSpec"


def _program():
    return PromptProgram(
        profile=prompt_profile_by_stage("planner_intent"),
        fragment_ids=(),
        prompt_text="{}",
        prompt_hash="hash",
        fragment_hashes={},
        layers=(),
    )
