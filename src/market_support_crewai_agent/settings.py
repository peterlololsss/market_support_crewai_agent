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
    llm_timeout_seconds: float = Field(default=90.0, gt=0)
    llm_temperature: float = Field(default=0.1, ge=0)
    llm_max_tokens: int = Field(default=6000, gt=0)
    planner_llm_base_url: str = "https://llm.yanfuinvest.com/v1"
    planner_llm_provider: str = "openai"
    planner_llm_model: str = "deepseek-v4-pro"
    planner_llm_api_key: str | None = None
    crewai_verbose: bool = False
    crewai_max_iter: int = Field(default=5, gt=0)
    crewai_max_execution_time: int = Field(default=120, gt=0)
    crewai_max_retry_limit: int = Field(default=2, ge=0)
    planner_transient_retry_attempts: int = Field(default=1, ge=0)
    planner_transient_retry_base_seconds: float = Field(default=0.5, ge=0)
    agent_input_max_message_chars: int | None = Field(default=None, gt=0)
    agent_conversation_ttl_seconds: int = Field(default=86400, gt=0)
    agent_conversation_max_messages: int = Field(default=24, gt=0)
    agent_conversation_max_sessions: int = Field(default=5000, gt=0)
    agent_conversation_cleanup_interval_seconds: int = Field(default=300, gt=0)
    adapter_base_url: str = "http://127.0.0.1:8011"
    adapter_api_key: str | None = None
    adapter_timeout_seconds: float = Field(default=5.0, gt=0)
    doc_mcp_base_url: str | None = None
    doc_mcp_timeout_seconds: float = Field(default=5.0, gt=0)
    doc_mcp_enabled: bool = False
    doc_mcp_allowed_channel_types: tuple[ChannelType, ...] = _ALL_CHANNEL_TYPES
    # Per-document evidence ceiling. Set above the largest real document so a
    # selected document is delivered whole ("locate precisely, then expand to
    # read"); it only fails safe on a pathologically oversized document.
    doc_mcp_max_chars_per_document: int = Field(default=1_000_000, gt=0)
    # Process-wide TTL for the static product manifest and document content.
    # 0 disables caching and re-fetches from the MCP on every query.
    doc_mcp_cache_ttl_seconds: float = Field(default=300.0, ge=0)
    # Preferred document categories to load first when the closed-set selector declines.
    doc_mcp_baseline_categories: tuple[str, ...] = ("常见问答",)
    reply_alignment_verifier_enabled: bool = True
    reply_alignment_max_replans: int = Field(default=1, ge=0)
    reply_alignment_max_evidence_refetches: int = Field(default=1, ge=0)
    reply_alignment_max_recomposes: int = Field(default=1, ge=0)
    reply_alignment_max_total_remediations: int = Field(default=2, ge=0)
    llm_health_enabled: bool = False
    llm_health_check_interval_seconds: float = Field(default=300.0, gt=0)
    llm_health_failure_interval_seconds: float = Field(default=60.0, gt=0)
    llm_health_daily_report_time: str = "09:00"
    llm_health_timezone: str = "Asia/Shanghai"
    llm_health_warning_cooldown_seconds: float = Field(default=900.0, gt=0)
    llm_health_probe_retry_attempts: int = Field(default=1, ge=0)
    llm_health_probe_retry_base_seconds: float = Field(default=1.0, ge=0)
    llm_health_probe_timeout_seconds: float = Field(default=20.0, gt=0)
    feishu_app_id: str | None = None
    feishu_app_secret: str | None = None
    feishu_chat_id: str | None = None
    agent_context_recent_turns_verbatim_count: int = Field(default=12, ge=0)
    agent_context_max_history_message_chars_inline: int = Field(default=2400, gt=0)
    agent_context_max_evidence_chars_inline: int = Field(default=6000, gt=0)
    # Answer-bearing evidence (selected knowledge documents) inline budget. Keep
    # >= doc_mcp_max_chars_per_document so a selected document reaches the
    # composer whole instead of being previewed.
    agent_context_max_answer_evidence_chars_inline: int = Field(default=1_000_000, gt=0)
    agent_context_large_result_preview_chars: int = Field(default=1200, gt=0)
    agent_context_token_budget: int = Field(default=900_000, gt=0)
    agent_context_warning_threshold: float = Field(default=0.75, gt=0)
    agent_context_hard_threshold: float = Field(default=0.92, gt=0)


def get_settings() -> Settings:
    llm_base_url = os.getenv("YANFU_LLM_BASE_URL", "https://llm.yanfuinvest.com/v1")
    llm_provider = os.getenv("YANFU_LLM_PROVIDER", "openai")
    llm_model = os.getenv("YANFU_LLM_MODEL", "deepseek-v4-pro")
    llm_api_key = os.getenv("YANFU_LLM_API_KEY") or None
    planner_llm_api_key = os.getenv("MARKET_AGENT_PLANNER_LLM_API_KEY")
    planner_llm_overridden = any(
        os.getenv(name) is not None
        for name in (
            "MARKET_AGENT_PLANNER_LLM_BASE_URL",
            "MARKET_AGENT_PLANNER_LLM_PROVIDER",
            "MARKET_AGENT_PLANNER_LLM_MODEL",
        )
    )
    if planner_llm_api_key is not None:
        planner_llm_api_key_resolved = planner_llm_api_key or None
    elif planner_llm_overridden:
        planner_llm_api_key_resolved = None
    else:
        planner_llm_api_key_resolved = llm_api_key
    return Settings(
        api_key=os.getenv("MARKET_AGENT_API_KEY") or None,
        llm_base_url=llm_base_url,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_timeout_seconds=_float_env("YANFU_LLM_TIMEOUT_SECONDS", 90.0),
        llm_temperature=_float_env("YANFU_LLM_TEMPERATURE", 0.1),
        llm_max_tokens=_int_env("YANFU_LLM_MAX_TOKENS", 6000),
        planner_llm_base_url=os.getenv("MARKET_AGENT_PLANNER_LLM_BASE_URL")
        or llm_base_url,
        planner_llm_provider=os.getenv("MARKET_AGENT_PLANNER_LLM_PROVIDER")
        or llm_provider,
        planner_llm_model=os.getenv("MARKET_AGENT_PLANNER_LLM_MODEL") or llm_model,
        planner_llm_api_key=planner_llm_api_key_resolved,
        crewai_verbose=_bool_env("CREWAI_VERBOSE", False),
        crewai_max_iter=_int_env("CREWAI_MAX_ITER", 5),
        crewai_max_execution_time=_int_env("CREWAI_MAX_EXECUTION_TIME", 120),
        crewai_max_retry_limit=_non_negative_int_env("CREWAI_MAX_RETRY_LIMIT", 2),
        planner_transient_retry_attempts=_non_negative_int_env(
            "MARKET_AGENT_PLANNER_TRANSIENT_RETRY_ATTEMPTS", 1
        ),
        planner_transient_retry_base_seconds=_non_negative_float_env(
            "MARKET_AGENT_PLANNER_TRANSIENT_RETRY_BASE_SECONDS", 0.5
        ),
        agent_input_max_message_chars=_optional_int_env(
            "AGENT_INPUT_MAX_MESSAGE_CHARS"
        ),
        agent_conversation_ttl_seconds=_int_env(
            "AGENT_CONVERSATION_TTL_SECONDS", 86400
        ),
        agent_conversation_max_messages=_int_env(
            "AGENT_CONVERSATION_MAX_MESSAGES", 24
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
        doc_mcp_base_url=os.getenv("MARKET_AGENT_DOC_MCP_BASE_URL") or None,
        doc_mcp_timeout_seconds=_float_env("MARKET_AGENT_DOC_MCP_TIMEOUT_SECONDS", 5.0),
        doc_mcp_enabled=_bool_env("MARKET_AGENT_DOC_MCP_ENABLED", False),
        doc_mcp_allowed_channel_types=_channel_types_env(
            "MARKET_AGENT_DOC_MCP_ALLOWED_CHANNEL_TYPES",
            _ALL_CHANNEL_TYPES,
        ),
        doc_mcp_max_chars_per_document=_int_env(
            "MARKET_AGENT_DOC_MCP_MAX_CHARS_PER_DOCUMENT", 1_000_000
        ),
        doc_mcp_cache_ttl_seconds=_float_env(
            "MARKET_AGENT_DOC_MCP_CACHE_TTL_SECONDS", 300.0
        ),
        doc_mcp_baseline_categories=_str_tuple_env(
            "MARKET_AGENT_DOC_MCP_BASELINE_CATEGORIES", ("常见问答",)
        ),
        reply_alignment_verifier_enabled=_bool_env(
            "MARKET_AGENT_REPLY_ALIGNMENT_VERIFIER_ENABLED", True
        ),
        reply_alignment_max_replans=_non_negative_int_env(
            "MARKET_AGENT_REPLY_ALIGNMENT_MAX_REPLANS", 1
        ),
        reply_alignment_max_evidence_refetches=_non_negative_int_env(
            "MARKET_AGENT_REPLY_ALIGNMENT_MAX_EVIDENCE_REFETCHES", 1
        ),
        reply_alignment_max_recomposes=_non_negative_int_env(
            "MARKET_AGENT_REPLY_ALIGNMENT_MAX_RECOMPOSES", 1
        ),
        reply_alignment_max_total_remediations=_non_negative_int_env(
            "MARKET_AGENT_REPLY_ALIGNMENT_MAX_TOTAL_REMEDIATIONS", 2
        ),
        llm_health_enabled=_bool_env("MARKET_AGENT_LLM_HEALTH_ENABLED", False),
        llm_health_check_interval_seconds=_float_env(
            "MARKET_AGENT_LLM_HEALTH_CHECK_INTERVAL_SECONDS", 300.0
        ),
        llm_health_failure_interval_seconds=_float_env(
            "MARKET_AGENT_LLM_HEALTH_FAILURE_INTERVAL_SECONDS", 60.0
        ),
        llm_health_daily_report_time=os.getenv(
            "MARKET_AGENT_LLM_HEALTH_DAILY_REPORT_TIME", "09:00"
        ),
        llm_health_timezone=os.getenv(
            "MARKET_AGENT_LLM_HEALTH_TIMEZONE", "Asia/Shanghai"
        ),
        llm_health_warning_cooldown_seconds=_float_env(
            "MARKET_AGENT_LLM_HEALTH_WARNING_COOLDOWN_SECONDS", 900.0
        ),
        llm_health_probe_retry_attempts=_non_negative_int_env(
            "MARKET_AGENT_LLM_HEALTH_PROBE_RETRY_ATTEMPTS", 1
        ),
        llm_health_probe_retry_base_seconds=_non_negative_float_env(
            "MARKET_AGENT_LLM_HEALTH_PROBE_RETRY_BASE_SECONDS", 1.0
        ),
        llm_health_probe_timeout_seconds=_float_env(
            "MARKET_AGENT_LLM_HEALTH_PROBE_TIMEOUT_SECONDS", 20.0
        ),
        feishu_app_id=os.getenv("MARKET_AGENT_FEISHU_APP_ID") or None,
        feishu_app_secret=os.getenv("MARKET_AGENT_FEISHU_APP_SECRET") or None,
        feishu_chat_id=os.getenv("MARKET_AGENT_FEISHU_CHAT_ID") or None,
        agent_context_recent_turns_verbatim_count=_non_negative_int_env(
            "AGENT_CONTEXT_RECENT_TURNS_VERBATIM_COUNT", 12
        ),
        agent_context_max_history_message_chars_inline=_int_env(
            "AGENT_CONTEXT_MAX_HISTORY_MESSAGE_CHARS_INLINE", 2400
        ),
        agent_context_max_evidence_chars_inline=_int_env(
            "AGENT_CONTEXT_MAX_EVIDENCE_CHARS_INLINE", 6000
        ),
        agent_context_max_answer_evidence_chars_inline=_int_env(
            "AGENT_CONTEXT_MAX_ANSWER_EVIDENCE_CHARS_INLINE", 1_000_000
        ),
        agent_context_large_result_preview_chars=_int_env(
            "AGENT_CONTEXT_LARGE_RESULT_PREVIEW_CHARS", 1200
        ),
        agent_context_token_budget=_int_env(
            "AGENT_CONTEXT_TOKEN_BUDGET", 900_000
        ),
        agent_context_warning_threshold=_float_env(
            "AGENT_CONTEXT_WARNING_THRESHOLD", 0.75
        ),
        agent_context_hard_threshold=_float_env(
            "AGENT_CONTEXT_HARD_THRESHOLD", 0.92
        ),
    )


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _non_negative_float_env(name: str, default: float) -> float:
    try:
        parsed = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


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


def _str_tuple_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


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
