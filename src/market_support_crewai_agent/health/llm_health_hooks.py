from __future__ import annotations

from collections.abc import Callable

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderFailureCodeV1,
)

SuccessHookV1 = Callable[[CrewAIAgentAdapterV1 | None, str], None]
FailureHookV1 = Callable[
    [CrewAIAgentAdapterV1 | None, str, ProviderFailureCodeV1], None
]

_success_hook: SuccessHookV1 | None = None
_failure_hook: FailureHookV1 | None = None


def register_llm_health_hooks(
    *,
    success_hook: SuccessHookV1,
    failure_hook: FailureHookV1,
) -> None:
    global _success_hook, _failure_hook
    _success_hook = success_hook
    _failure_hook = failure_hook


def record_llm_success_for_agent(
    agent: CrewAIAgentAdapterV1 | None, stage: str
) -> None:
    if _success_hook is not None:
        _success_hook(agent, stage)


def record_llm_failure_for_agent(
    agent: CrewAIAgentAdapterV1 | None,
    stage: str,
    reason: ProviderFailureCodeV1,
) -> None:
    if _failure_hook is not None:
        _failure_hook(agent, stage, reason)
