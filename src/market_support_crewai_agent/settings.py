from __future__ import annotations

import os

from pydantic import BaseModel, Field

from market_support_crewai_agent.schemas import ChannelType

_ALL_CHANNEL_TYPES: tuple[ChannelType, ...] = ("bank", "non_bank")


class Settings(BaseModel):
    api_key: str | None = None
    llm_base_url: str = "https://llm.yanfuinvest.com/v1"
    llm_provider: str = "openai"
    llm_model: str = "deepseek-v4-pro"
    llm_api_key: str | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_temperature: float = Field(default=0.1, ge=0)
    llm_max_tokens: int = Field(default=3000, gt=0)
    crewai_verbose: bool = False
    crewai_max_iter: int = Field(default=5, gt=0)
    crewai_max_execution_time: int = Field(default=60, gt=0)
    crewai_max_retry_limit: int = Field(default=2, ge=0)
    agent_input_max_message_chars: int | None = Field(default=None, gt=0)
    agent_conversation_ttl_seconds: int = Field(default=86400, gt=0)
    agent_conversation_max_messages: int = Field(default=12, gt=0)
    agent_conversation_max_sessions: int = Field(default=5000, gt=0)
    agent_conversation_cleanup_interval_seconds: int = Field(default=300, gt=0)
    adapter_base_url: str = "http://127.0.0.1:8011"
    adapter_api_key: str | None = None
    adapter_timeout_seconds: float = Field(default=5.0, gt=0)
    adapter_preflight_enabled: bool = True
    doc_mcp_base_url: str | None = None
    doc_mcp_timeout_seconds: float = Field(default=5.0, gt=0)
    doc_mcp_enabled: bool = False
    doc_mcp_allowed_channel_types: tuple[ChannelType, ...] = _ALL_CHANNEL_TYPES


def get_settings() -> Settings:
    return Settings(
        api_key=os.getenv("MARKET_AGENT_API_KEY") or None,
        llm_base_url=os.getenv("YANFU_LLM_BASE_URL", "https://llm.yanfuinvest.com/v1"),
        llm_provider=os.getenv("YANFU_LLM_PROVIDER", "openai"),
        llm_model=os.getenv("YANFU_LLM_MODEL", "deepseek-v4-pro"),
        llm_api_key=os.getenv("YANFU_LLM_API_KEY") or None,
        llm_timeout_seconds=_float_env("YANFU_LLM_TIMEOUT_SECONDS", 30.0),
        llm_temperature=_float_env("YANFU_LLM_TEMPERATURE", 0.1),
        llm_max_tokens=_int_env("YANFU_LLM_MAX_TOKENS", 3000),
        crewai_verbose=_bool_env("CREWAI_VERBOSE", False),
        crewai_max_iter=_int_env("CREWAI_MAX_ITER", 5),
        crewai_max_execution_time=_int_env("CREWAI_MAX_EXECUTION_TIME", 60),
        crewai_max_retry_limit=_non_negative_int_env("CREWAI_MAX_RETRY_LIMIT", 2),
        agent_input_max_message_chars=_optional_int_env(
            "AGENT_INPUT_MAX_MESSAGE_CHARS"
        ),
        agent_conversation_ttl_seconds=_int_env(
            "AGENT_CONVERSATION_TTL_SECONDS", 86400
        ),
        agent_conversation_max_messages=_int_env(
            "AGENT_CONVERSATION_MAX_MESSAGES", 12
        ),
        agent_conversation_max_sessions=_int_env(
            "AGENT_CONVERSATION_MAX_SESSIONS", 5000
        ),
        agent_conversation_cleanup_interval_seconds=_int_env(
            "AGENT_CONVERSATION_CLEANUP_INTERVAL_SECONDS", 300
        ),
        adapter_base_url=os.getenv(
            "MARKET_AGENT_ADAPTER_BASE_URL", "http://127.0.0.1:8011"
        ),
        adapter_api_key=os.getenv("MARKET_AGENT_ADAPTER_API_KEY") or None,
        adapter_timeout_seconds=_float_env("MARKET_AGENT_ADAPTER_TIMEOUT_SECONDS", 5.0),
        adapter_preflight_enabled=_bool_env(
            "MARKET_AGENT_ADAPTER_PREFLIGHT_ENABLED", True
        ),
        doc_mcp_base_url=os.getenv("MARKET_AGENT_DOC_MCP_BASE_URL") or None,
        doc_mcp_timeout_seconds=_float_env("MARKET_AGENT_DOC_MCP_TIMEOUT_SECONDS", 5.0),
        doc_mcp_enabled=_bool_env("MARKET_AGENT_DOC_MCP_ENABLED", False),
        doc_mcp_allowed_channel_types=_channel_types_env(
            "MARKET_AGENT_DOC_MCP_ALLOWED_CHANNEL_TYPES",
            _ALL_CHANNEL_TYPES,
        ),
    )


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _non_negative_int_env(name: str, default: int) -> int:
    try:
        parsed = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _channel_types_env(
    name: str,
    default: tuple[ChannelType, ...],
) -> tuple[ChannelType, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    allowed = {"bank", "non_bank"}
    parsed = tuple(
        item
        for item in (part.strip() for part in value.split(","))
        if item in allowed
    )
    return parsed
