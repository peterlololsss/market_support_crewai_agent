from __future__ import annotations

from market_support_crewai_agent.runtime.llm.prompting.profiles import (
    PromptProfile,
    prompt_profile_by_stage,
)
from market_support_crewai_agent.runtime.llm.prompting.registry import (
    prompt_agent_spec_by_id,
)
from market_support_crewai_agent.runtime.llm.prompting.router import (
    model_family_from_settings,
)
from market_support_crewai_agent.settings import Settings

_UNSET = object()


class CrewAIAgentFactory:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_planner_agent(self):
        spec = prompt_agent_spec_by_id("agent.planner")
        return self._build_crewai_agent(
            role=spec.role,
            goal=spec.goal,
            backstory=spec.backstory,
            inject_date=True,
            prompt_profile=prompt_profile_by_stage(
                "planner_intent",
                model_family_from_settings(self.settings, stage="planner_intent"),
            ),
            llm_model=self.settings.planner_llm_model,
            llm_provider=self.settings.planner_llm_provider,
            llm_base_url=self.settings.planner_llm_base_url,
            llm_api_key=self.settings.planner_llm_api_key,
        )

    def build_planner_fallback_agent(self):
        spec = prompt_agent_spec_by_id("agent.planner")
        return self._build_crewai_agent(
            role=spec.role,
            goal=spec.goal,
            backstory=spec.backstory,
            inject_date=True,
            prompt_profile=prompt_profile_by_stage(
                "planner_intent",
                model_family_from_settings(self.settings),
            ),
        )

    def build_composer_agent(self, stage="knowledge_composer"):
        spec = prompt_agent_spec_by_id(
            "agent.direct_composer" if stage == "direct_composer" else "agent.composer"
        )
        return self._build_crewai_agent(
            role=spec.role,
            goal=spec.goal,
            backstory=spec.backstory,
            inject_date=True,
            prompt_profile=prompt_profile_by_stage(
                stage,
                model_family_from_settings(self.settings),
            ),
        )

    def build_alignment_verifier_agent(self):
        spec = prompt_agent_spec_by_id("agent.alignment_verifier")
        return self._build_crewai_agent(
            role=spec.role,
            goal=spec.goal,
            backstory=spec.backstory,
            inject_date=False,
            prompt_profile=prompt_profile_by_stage(
                "alignment_verifier",
                model_family_from_settings(self.settings),
            ),
        )

    def _build_crewai_agent(
        self,
        *,
        role: str,
        goal: str,
        backstory: str,
        inject_date: bool,
        prompt_profile: PromptProfile | None = None,
        llm_model: str | object = _UNSET,
        llm_provider: str | object = _UNSET,
        llm_base_url: str | object = _UNSET,
        llm_api_key: str | None | object = _UNSET,
    ):
        from crewai import Agent, LLM

        model = self.settings.llm_model if llm_model is _UNSET else llm_model
        provider = self.settings.llm_provider if llm_provider is _UNSET else llm_provider
        base_url = self.settings.llm_base_url if llm_base_url is _UNSET else llm_base_url
        api_key = self.settings.llm_api_key if llm_api_key is _UNSET else llm_api_key
        extra = {}
        if str(provider).lower() in {"gemini", "google"} and base_url:
            extra["client_params"] = {
                "http_options": {
                    "base_url": base_url,
                    "api_version": "v1beta",
                    "timeout": int(self.settings.llm_timeout_seconds * 1000),
                }
            }
            extra["max_output_tokens"] = (
                prompt_profile.max_tokens
                if prompt_profile is not None and prompt_profile.max_tokens is not None
                else self.settings.llm_max_tokens
            )

        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=LLM(
                model=model,
                provider=provider,
                base_url=None if extra else base_url,
                api_key=api_key,
                temperature=(
                    prompt_profile.temperature
                    if prompt_profile is not None and prompt_profile.temperature is not None
                    else self.settings.llm_temperature
                ),
                max_tokens=(
                    None
                    if extra
                    else
                    prompt_profile.max_tokens
                    if prompt_profile is not None and prompt_profile.max_tokens is not None
                    else self.settings.llm_max_tokens
                ),
                timeout=self.settings.llm_timeout_seconds,
                **extra,
            ),
            allow_delegation=False,
            verbose=self.settings.crewai_verbose,
            max_iter=self.settings.crewai_max_iter,
            max_execution_time=self.settings.crewai_max_execution_time,
            max_retry_limit=self.settings.crewai_max_retry_limit,
            planning=False,
            inject_date=inject_date,
            date_format="%Y-%m-%d",
        )
