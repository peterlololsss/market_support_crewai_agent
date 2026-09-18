from __future__ import annotations

# pyright: reportUnnecessaryComparison=false
from typing import Literal, assert_never, final

from pydantic import JsonValue

from market_support_crewai_agent.runtime.hashing import canonical_json_bytes
from market_support_crewai_agent.runtime.integrations.crewai import (
    sdk_adapters,
    sdk_governance,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.prompts.profiles import (
    PromptProfile,
    prompt_profile_by_stage,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    AgentExecutionSpecV1,
    PromptGovernanceError,
    resolve_active_prompt_program_v2,
)
from market_support_crewai_agent.runtime.prompts.provider_target import (
    normalize_provider_id,
)
from market_support_crewai_agent.runtime.prompts.router import (
    model_family_from_settings,
)
from market_support_crewai_agent.settings_model import Settings


@final
class CrewAIAgentFactory:
    def __init__(self, settings: Settings) -> None:
        self.settings: Settings = settings

    def build_planner_agent(self) -> CrewAIAgentAdapterV1:
        _, execution_spec = resolve_active_prompt_program_v2(
            stage="planner_intent",
            scene_key="wecom_group.v1",
        )
        return self._build_crewai_agent(
            execution_spec=execution_spec,
            inject_date=True,
            prompt_profile=prompt_profile_by_stage(
                "planner_intent",
                model_family_from_settings(self.settings, stage="planner_intent"),
            ),
            llm_model=self.settings.planner_llm_model,
            llm_provider=self.settings.planner_llm_provider,
            llm_base_url=self.settings.planner_llm_base_url,
            llm_api_key=self.settings.planner_llm_api_key,
            llm_timeout_seconds=self.settings.llm_timeout_seconds,
        )

    def build_composer_agent(
        self,
        stage: Literal[
            "knowledge_composer", "smalltalk_composer"
        ] = "knowledge_composer",
    ) -> CrewAIAgentAdapterV1:
        _, execution_spec = resolve_active_prompt_program_v2(
            stage=stage,
            scene_key="wecom_group.v1",
        )
        return self._build_crewai_agent(
            execution_spec=execution_spec,
            inject_date=True,
            prompt_profile=prompt_profile_by_stage(
                stage,
                model_family_from_settings(self.settings),
            ),
            llm_model=self.settings.llm_model,
            llm_provider=self.settings.llm_provider,
            llm_base_url=self.settings.llm_base_url,
            llm_api_key=self.settings.llm_api_key,
            llm_timeout_seconds=self.settings.llm_timeout_seconds,
        )

    def build_alignment_verifier_agent(self) -> CrewAIAgentAdapterV1:
        _, execution_spec = resolve_active_prompt_program_v2(
            stage="alignment_verifier",
            scene_key="wecom_group.v1",
        )
        return self._build_crewai_agent(
            execution_spec=execution_spec,
            inject_date=False,
            prompt_profile=prompt_profile_by_stage(
                "alignment_verifier",
                model_family_from_settings(self.settings),
            ),
            llm_model=self.settings.llm_model,
            llm_provider=self.settings.llm_provider,
            llm_base_url=self.settings.llm_base_url,
            llm_api_key=self.settings.llm_api_key,
            llm_timeout_seconds=self.settings.llm_timeout_seconds,
        )

    def build_health_probe_agent(
        self,
        *,
        target_slot: Literal["composer", "planner"],
        prompt_profile: PromptProfile,
    ) -> CrewAIAgentAdapterV1:
        _, execution_spec = resolve_active_prompt_program_v2(
            stage="llm_health_probe",
            scene_key="scene_neutral.v1",
        )
        if execution_spec.spec_id != "agent.llm_health_probe.v1":
            raise PromptGovernanceError("health_probe_execution_spec_mismatch")
        if prompt_profile.id != "llm_health_probe.generic@1":
            raise PromptGovernanceError("health_probe_profile_mismatch")
        match target_slot:
            case "planner":
                model = self.settings.planner_llm_model
                provider = self.settings.planner_llm_provider
                base_url = self.settings.planner_llm_base_url
                api_key = self.settings.planner_llm_api_key
            case "composer":
                model = self.settings.llm_model
                provider = self.settings.llm_provider
                base_url = self.settings.llm_base_url
                api_key = self.settings.llm_api_key
            case unreachable:
                assert_never(unreachable)
        return self._build_crewai_agent(
            execution_spec=execution_spec,
            inject_date=False,
            prompt_profile=prompt_profile,
            llm_model=model,
            llm_provider=provider,
            llm_base_url=base_url,
            llm_api_key=api_key,
            llm_timeout_seconds=self.settings.llm_health_probe_timeout_seconds,
        )

    def _build_crewai_agent(
        self,
        *,
        execution_spec: AgentExecutionSpecV1,
        inject_date: bool,
        prompt_profile: PromptProfile,
        llm_model: str,
        llm_provider: str,
        llm_base_url: str,
        llm_api_key: str | None,
        llm_timeout_seconds: float,
    ) -> CrewAIAgentAdapterV1:
        agent_cls, llm_cls = sdk_governance.load_crewai_types_without_dotenv()
        timeout_seconds = float(llm_timeout_seconds)
        provider_id = normalize_provider_id(llm_provider)
        extra: dict[str, JsonValue] = {}
        if provider_id == "gemini":
            extra["client_params"] = {
                "http_options": {
                    "base_url": llm_base_url,
                    "api_version": "v1beta",
                    "timeout": int(timeout_seconds * 1000),
                    "retry_options": {"attempts": 1},
                }
            }
            extra["max_output_tokens"] = (
                prompt_profile.max_tokens or self.settings.llm_max_tokens
            )

        llm = llm_cls(
            model=llm_model,
            provider=llm_provider,
            base_url=None if extra else llm_base_url,
            api_key=llm_api_key,
            temperature=(
                prompt_profile.temperature
                if prompt_profile.temperature is not None
                else self.settings.llm_temperature
            ),
            max_tokens=None
            if extra
            else prompt_profile.max_tokens or self.settings.llm_max_tokens,
            **extra,
        )
        llm.timeout = timeout_seconds
        llm.max_retries = 0
        try:
            llm_adapter = sdk_adapters.adapt_sdk_llm(llm)
        except AttributeError as exc:
            raise PromptGovernanceError(
                "crewai_provider_retry_configuration_unsupported"
            ) from exc
        sdk_governance.require_provider_limits(
            llm_adapter, provider_id, timeout_seconds
        )
        adapter = sdk_adapters.adapt_sdk_agent(
            agent_cls(
                role=execution_spec.role,
                goal=execution_spec.goal,
                backstory=execution_spec.backstory,
                system_template=(
                    "{{ .System }}\n"
                    + canonical_json_bytes(
                        {"agent_spec_version": execution_spec.agent_spec_version}
                    ).decode("utf-8")
                ),
                prompt_template="{{ .Prompt }}\n" + execution_spec.task_template,
                response_template=(
                    execution_spec.expected_output_template
                    + "\n{{ .Response }}\n"
                    + execution_spec.expected_output_template
                ),
                llm=llm,
                allow_delegation=False,
                verbose=self.settings.crewai_verbose,
                max_iter=self.settings.crewai_max_iter,
                max_execution_time=self.settings.crewai_max_execution_time,
                max_retry_limit=self.settings.crewai_max_retry_limit,
                planning=False,
                inject_date=inject_date,
                date_format="%Y-%m-%d",
            )
        )
        return adapter
