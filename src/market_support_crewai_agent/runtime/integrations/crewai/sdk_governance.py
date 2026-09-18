from __future__ import annotations

# pyright: reportUnnecessaryComparison=false
import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, assert_never

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAILlmAdapterV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.sdk_adapters import (
    CrewAISdkAgentAdapterV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.sdk_payload import (
    mapping_value,
    optional_int_value,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    PromptGovernanceError,
    ProviderIdV1,
)

if TYPE_CHECKING:
    from crewai.llms.base_llm import BaseLLM as CrewAISdkLLM

_CREWAI_IMPORT_LOCK = threading.Lock()
_DOTENV_DISABLED = "PYTHON_DOTENV_DISABLED"


def require_provider_limits(
    llm: CrewAILlmAdapterV1,
    provider_id: ProviderIdV1,
    timeout: float,
) -> None:
    error_code = "crewai_provider_retry_configuration_unsupported"
    match provider_id:
        case "openai_compatible":
            return
        case "gemini":
            options = mapping_value(dict(llm.client_params), "http_options")
            retry_options = mapping_value(options, "retry_options")
            attempts = optional_int_value(retry_options, "attempts")
            configured_ms = optional_int_value(options, "timeout")
            if attempts != 1:
                raise PromptGovernanceError(error_code)
            if configured_ms != int(timeout * 1000):
                raise PromptGovernanceError("crewai_provider_sdk_timeout_unsupported")
        case unreachable:
            assert_never(unreachable)


def load_crewai_types_without_dotenv() -> tuple[
    Callable[..., CrewAISdkAgentAdapterV1],
    Callable[..., CrewAISdkLLM],
]:
    with _CREWAI_IMPORT_LOCK:
        previous = os.environ.get(_DOTENV_DISABLED)
        os.environ[_DOTENV_DISABLED] = "1"
        try:
            from crewai.agent.core import Agent
            from crewai.events.event_listener import event_listener
            from crewai.llm import LLM

            event_listener.formatter.verbose = False
        finally:
            if previous is None:
                _ = os.environ.pop(_DOTENV_DISABLED, None)
            else:
                os.environ[_DOTENV_DISABLED] = previous
    return Agent, LLM
