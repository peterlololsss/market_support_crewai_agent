from __future__ import annotations

from market_support_crewai_agent.settings import get_settings


def test_doc_mcp_configuration_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("MARKET_AGENT_DOC_MCP_BASE_URL", raising=False)
    monkeypatch.delenv("MARKET_AGENT_DOC_MCP_ENABLED", raising=False)
    monkeypatch.delenv("AGENT_INPUT_MAX_MESSAGE_CHARS", raising=False)

    settings = get_settings()

    assert settings.doc_mcp_base_url is None
    assert settings.doc_mcp_enabled is False
    assert settings.agent_input_max_message_chars is None


def test_input_max_message_chars_reads_environment(monkeypatch):
    monkeypatch.setenv("AGENT_INPUT_MAX_MESSAGE_CHARS", "2000")

    settings = get_settings()

    assert settings.agent_input_max_message_chars == 2000


def test_input_max_message_chars_ignores_invalid_environment(monkeypatch):
    monkeypatch.setenv("AGENT_INPUT_MAX_MESSAGE_CHARS", "0")

    settings = get_settings()

    assert settings.agent_input_max_message_chars is None


def test_crewai_max_retry_limit_reads_environment(monkeypatch):
    monkeypatch.setenv("CREWAI_MAX_RETRY_LIMIT", "4")

    settings = get_settings()

    assert settings.crewai_max_retry_limit == 4


def test_crewai_max_retry_limit_ignores_invalid_environment(monkeypatch):
    monkeypatch.setenv("CREWAI_MAX_RETRY_LIMIT", "-1")

    settings = get_settings()

    assert settings.crewai_max_retry_limit == 2


def test_llm_max_tokens_defaults_to_real_structured_output_budget(monkeypatch):
    monkeypatch.delenv("YANFU_LLM_MAX_TOKENS", raising=False)

    settings = get_settings()

    assert settings.llm_max_tokens == 3000


def test_llm_max_tokens_reads_environment(monkeypatch):
    monkeypatch.setenv("YANFU_LLM_MAX_TOKENS", "2500")

    settings = get_settings()

    assert settings.llm_max_tokens == 2500


def test_doc_mcp_configuration_reads_environment(monkeypatch):
    monkeypatch.setenv("MARKET_AGENT_DOC_MCP_BASE_URL", "http://192.168.209.195:23000")
    monkeypatch.setenv("MARKET_AGENT_DOC_MCP_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("MARKET_AGENT_DOC_MCP_ENABLED", "true")

    settings = get_settings()

    assert settings.doc_mcp_base_url == "http://192.168.209.195:23000"
    assert settings.doc_mcp_timeout_seconds == 7
    assert settings.doc_mcp_enabled is True


def test_doc_mcp_allowed_channel_types_default_to_all_supported(monkeypatch):
    monkeypatch.delenv("MARKET_AGENT_DOC_MCP_ALLOWED_CHANNEL_TYPES", raising=False)

    settings = get_settings()

    assert settings.doc_mcp_allowed_channel_types == ("bank", "non_bank")


def test_doc_mcp_allowed_channel_types_reads_environment(monkeypatch):
    monkeypatch.setenv("MARKET_AGENT_DOC_MCP_ALLOWED_CHANNEL_TYPES", "bank,unknown")

    settings = get_settings()

    assert settings.doc_mcp_allowed_channel_types == ("bank",)
