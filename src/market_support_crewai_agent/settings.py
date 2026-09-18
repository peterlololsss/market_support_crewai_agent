from __future__ import annotations

import os

from market_support_crewai_agent import settings_model
from market_support_crewai_agent.settings_env import (
    bool_env,
    bounded_float_env,
    channel_types_env,
    float_env,
    governed_float_env,
    governed_int_env,
    group_recall_mode_env,
    int_env,
    non_negative_int_env,
    optional_int_env,
    str_tuple_env,
)


def get_settings() -> settings_model.Settings:
    llm_base_url = os.getenv("MARKET_AGENT_LLM_BASE_URL", "https://llm.example.com/v1")
    llm_provider = os.getenv("MARKET_AGENT_LLM_PROVIDER", "openai")
    llm_model = os.getenv("MARKET_AGENT_LLM_MODEL", "deepseek-v4-pro")
    llm_api_key = os.getenv("MARKET_AGENT_LLM_API_KEY") or None
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
    return settings_model.Settings(
        api_key=os.getenv("MARKET_AGENT_API_KEY") or None,
        deployment_tenant_ref=os.getenv("MARKET_AGENT_DEPLOYMENT_TENANT_REF"),
        internal_dm_enabled=bool_env("MARKET_AGENT_INTERNAL_DM_ENABLED", False),
        llm_base_url=llm_base_url,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_timeout_seconds=float_env("MARKET_AGENT_LLM_TIMEOUT_SECONDS", 90.0),
        llm_temperature=float_env("MARKET_AGENT_LLM_TEMPERATURE", 0.1),
        llm_max_tokens=int_env("MARKET_AGENT_LLM_MAX_TOKENS", 6000),
        planner_llm_base_url=os.getenv("MARKET_AGENT_PLANNER_LLM_BASE_URL")
        or llm_base_url,
        planner_llm_provider=os.getenv("MARKET_AGENT_PLANNER_LLM_PROVIDER")
        or llm_provider,
        planner_llm_model=os.getenv("MARKET_AGENT_PLANNER_LLM_MODEL") or llm_model,
        planner_llm_api_key=planner_llm_api_key_resolved,
        crewai_verbose=bool_env("CREWAI_VERBOSE", False),
        crewai_max_iter=governed_int_env("CREWAI_MAX_ITER", 1),
        crewai_max_execution_time=int_env("CREWAI_MAX_EXECUTION_TIME", 120),
        crewai_max_retry_limit=governed_int_env("CREWAI_MAX_RETRY_LIMIT", 0),
        planner_transient_retry_attempts=governed_int_env(
            "MARKET_AGENT_PLANNER_TRANSIENT_RETRY_ATTEMPTS", 0
        ),
        planner_transient_retry_base_seconds=governed_float_env(
            "MARKET_AGENT_PLANNER_TRANSIENT_RETRY_BASE_SECONDS", 0.0
        ),
        agent_input_max_message_chars=optional_int_env("AGENT_INPUT_MAX_MESSAGE_CHARS"),
        agent_conversation_ttl_seconds=int_env("AGENT_CONVERSATION_TTL_SECONDS", 86400),
        agent_conversation_max_messages=int_env("AGENT_CONVERSATION_MAX_MESSAGES", 12),
        agent_conversation_max_sessions=int_env(
            "AGENT_CONVERSATION_MAX_SESSIONS", 5000
        ),
        agent_conversation_cleanup_interval_seconds=int_env(
            "AGENT_CONVERSATION_CLEANUP_INTERVAL_SECONDS", 300
        ),
        agent_direct_audit_ttl_seconds=int_env("AGENT_DIRECT_AUDIT_TTL_SECONDS", 86400),
        direct_audit_hmac_key=os.getenv("MARKET_AGENT_DIRECT_AUDIT_HMAC_KEY") or None,
        issued_response_ttl_seconds=int_env("AGENT_ISSUED_RESPONSE_TTL_SECONDS", 86400),
        issued_response_pending_ttl_seconds=int_env(
            "AGENT_ISSUED_RESPONSE_PENDING_TTL_SECONDS", 180
        ),
        issued_response_capacity=int_env("AGENT_ISSUED_RESPONSE_CAPACITY", 5000),
        feedback_receipt_capacity=int_env("AGENT_FEEDBACK_RECEIPT_CAPACITY", 20000),
        adapter_base_url=os.getenv(
            "MARKET_AGENT_ADAPTER_BASE_URL", "http://127.0.0.1:8011"
        ),
        adapter_api_key=os.getenv("MARKET_AGENT_ADAPTER_API_KEY") or None,
        adapter_timeout_seconds=float_env("MARKET_AGENT_ADAPTER_TIMEOUT_SECONDS", 5.0),
        doc_mcp_base_url=os.getenv("MARKET_AGENT_DOC_MCP_BASE_URL") or None,
        doc_mcp_timeout_seconds=float_env("MARKET_AGENT_DOC_MCP_TIMEOUT_SECONDS", 5.0),
        doc_mcp_enabled=bool_env("MARKET_AGENT_DOC_MCP_ENABLED", False),
        doc_mcp_allowed_channel_types=channel_types_env(
            "MARKET_AGENT_DOC_MCP_ALLOWED_CHANNEL_TYPES",
            settings_model.ALL_CHANNEL_TYPES,
        ),
        group_recall_mode=group_recall_mode_env(
            "MARKET_AGENT_GROUP_RECALL_MODE", "shortcut"
        ),
        doc_mcp_max_chars_per_document=int_env(
            "MARKET_AGENT_DOC_MCP_MAX_CHARS_PER_DOCUMENT", 1_000_000
        ),
        doc_mcp_cache_ttl_seconds=float_env(
            "MARKET_AGENT_DOC_MCP_CACHE_TTL_SECONDS", 300.0
        ),
        doc_mcp_baseline_categories=str_tuple_env(
            "MARKET_AGENT_DOC_MCP_BASELINE_CATEGORIES", ("常见问答",)
        ),
        reply_alignment_verifier_enabled=bool_env(
            "MARKET_AGENT_REPLY_ALIGNMENT_VERIFIER_ENABLED", True
        ),
        reply_alignment_max_replans=non_negative_int_env(
            "MARKET_AGENT_REPLY_ALIGNMENT_MAX_REPLANS", 1
        ),
        reply_alignment_max_evidence_refetches=non_negative_int_env(
            "MARKET_AGENT_REPLY_ALIGNMENT_MAX_EVIDENCE_REFETCHES", 1
        ),
        reply_alignment_max_recomposes=non_negative_int_env(
            "MARKET_AGENT_REPLY_ALIGNMENT_MAX_RECOMPOSES", 1
        ),
        reply_alignment_max_total_remediations=non_negative_int_env(
            "MARKET_AGENT_REPLY_ALIGNMENT_MAX_TOTAL_REMEDIATIONS", 2
        ),
        llm_health_enabled=bool_env("MARKET_AGENT_LLM_HEALTH_ENABLED", False),
        llm_health_check_interval_seconds=float_env(
            "MARKET_AGENT_LLM_HEALTH_CHECK_INTERVAL_SECONDS", 300.0
        ),
        llm_health_failure_interval_seconds=float_env(
            "MARKET_AGENT_LLM_HEALTH_FAILURE_INTERVAL_SECONDS", 60.0
        ),
        llm_health_daily_report_time=os.getenv(
            "MARKET_AGENT_LLM_HEALTH_DAILY_REPORT_TIME", "09:00"
        ),
        llm_health_timezone=os.getenv(
            "MARKET_AGENT_LLM_HEALTH_TIMEZONE", "Asia/Shanghai"
        ),
        llm_health_warning_cooldown_seconds=float_env(
            "MARKET_AGENT_LLM_HEALTH_WARNING_COOLDOWN_SECONDS", 900.0
        ),
        llm_health_probe_retry_attempts=governed_int_env(
            "MARKET_AGENT_LLM_HEALTH_PROBE_RETRY_ATTEMPTS", 0
        ),
        llm_health_probe_retry_base_seconds=governed_float_env(
            "MARKET_AGENT_LLM_HEALTH_PROBE_RETRY_BASE_SECONDS", 0.0
        ),
        llm_health_probe_timeout_seconds=bounded_float_env(
            "MARKET_AGENT_LLM_HEALTH_PROBE_TIMEOUT_SECONDS",
            20.0,
            minimum=0.1,
            maximum=30.0,
        ),
        feishu_app_id=os.getenv("MARKET_AGENT_FEISHU_APP_ID") or None,
        feishu_app_secret=os.getenv("MARKET_AGENT_FEISHU_APP_SECRET") or None,
        feishu_chat_id=os.getenv("MARKET_AGENT_FEISHU_CHAT_ID") or None,
        adapter_caller_namespace=os.getenv(
            "MARKET_AGENT_ADAPTER_CALLER_NAMESPACE", "assistant-wecom"
        ),
    )
