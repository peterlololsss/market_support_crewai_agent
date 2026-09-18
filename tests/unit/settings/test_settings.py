from __future__ import annotations

from typing import Final

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.settings import get_settings
from market_support_crewai_agent.settings_model import Settings


_TENANT_ENV: Final[str] = "MARKET_AGENT_DEPLOYMENT_TENANT_REF"
_INTERNAL_DM_ENV: Final[str] = "MARKET_AGENT_INTERNAL_DM_ENABLED"


def test_doc_mcp_configuration_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("MARKET_AGENT_DOC_MCP_BASE_URL", raising=False)
    monkeypatch.delenv("MARKET_AGENT_DOC_MCP_ENABLED", raising=False)
    monkeypatch.delenv("AGENT_INPUT_MAX_MESSAGE_CHARS", raising=False)

    settings = get_settings()

    assert settings.doc_mcp_base_url is None
    assert settings.doc_mcp_enabled is False
    assert settings.agent_input_max_message_chars is None


def test_deployment_settings_default_to_unconfigured_and_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: deployment identity and internal-DM activation are not configured.
    monkeypatch.delenv(_TENANT_ENV, raising=False)
    monkeypatch.delenv(_INTERNAL_DM_ENV, raising=False)

    # When: settings are loaded from the process environment.
    settings = get_settings()

    # Then: liveness can start without identity, while DM stays fail-closed.
    assert settings.deployment_tenant_ref is None
    assert settings.internal_dm_enabled is False


def test_deployment_settings_read_canonical_tenant_and_existing_boolean_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a canonical tenant and a truthy value accepted by the bool parser.
    monkeypatch.setenv(_TENANT_ENV, "tenant:primary")
    monkeypatch.setenv(_INTERNAL_DM_ENV, "YES")

    # When: settings are loaded from the process environment.
    settings = get_settings()

    # Then: the canonical tenant and enabled flag are preserved as typed values.
    assert settings.deployment_tenant_ref == "tenant:primary"
    assert settings.internal_dm_enabled is True


def test_blank_deployment_tenant_resolves_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an operator leaves the optional deployment tenant blank.
    monkeypatch.setenv(_TENANT_ENV, "   ")

    # When: settings are loaded from the process environment.
    settings = get_settings()

    # Then: blank input is treated as no deployment identity.
    assert settings.deployment_tenant_ref is None


@pytest.mark.parametrize(
    "tenant_ref",
    (
        "corp-raw",
        "tenant:",
        "tenant:unknown",
        "tenant:null",
        "tenant:contract-probe",
        "tenant:primary\n",
        "tenant:" + ("a" * 129),
    ),
)
def test_malformed_deployment_tenant_fails_settings_validation(
    monkeypatch: pytest.MonkeyPatch,
    tenant_ref: str,
) -> None:
    # Given: a tenant value outside the canonical shared wire grammar.
    monkeypatch.setenv(_TENANT_ENV, tenant_ref)

    # When/Then: settings parsing rejects the value at the boundary.
    with pytest.raises(ValidationError, match="deployment_tenant_ref"):
        get_settings()


def test_input_max_message_chars_reads_environment(monkeypatch):
    monkeypatch.setenv("AGENT_INPUT_MAX_MESSAGE_CHARS", "2000")

    settings = get_settings()

    assert settings.agent_input_max_message_chars == 2000


def test_input_max_message_chars_ignores_invalid_environment(monkeypatch):
    monkeypatch.setenv("AGENT_INPUT_MAX_MESSAGE_CHARS", "0")

    settings = get_settings()

    assert settings.agent_input_max_message_chars is None


def test_crewai_max_retry_limit_rejects_nonzero_environment(monkeypatch):
    monkeypatch.setenv("CREWAI_MAX_RETRY_LIMIT", "4")

    with pytest.raises(ValidationError, match="crewai_max_retry_limit"):
        get_settings()


@pytest.mark.parametrize("value", ("-1", "not-an-int"))
def test_crewai_max_retry_limit_rejects_malformed_environment(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("CREWAI_MAX_RETRY_LIMIT", value)

    with pytest.raises(ValueError):
        get_settings()


def test_crewai_max_iter_is_exactly_one_in_defaults_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CREWAI_MAX_ITER", raising=False)
    assert get_settings().crewai_max_iter == 1

    monkeypatch.setenv("CREWAI_MAX_ITER", "2")
    with pytest.raises(ValueError):
        get_settings()


@pytest.mark.parametrize(
    ("env_name", "invalid_value"),
    (
        ("CREWAI_MAX_ITER", "not-an-int"),
        ("MARKET_AGENT_PLANNER_TRANSIENT_RETRY_ATTEMPTS", "not-an-int"),
        ("MARKET_AGENT_PLANNER_TRANSIENT_RETRY_BASE_SECONDS", "not-a-float"),
        ("MARKET_AGENT_LLM_HEALTH_PROBE_RETRY_ATTEMPTS", "-1"),
        ("MARKET_AGENT_LLM_HEALTH_PROBE_RETRY_BASE_SECONDS", "-0.1"),
    ),
)
def test_governed_iteration_and_retry_environment_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    invalid_value: str,
) -> None:
    monkeypatch.setenv(env_name, invalid_value)

    with pytest.raises(ValueError):
        get_settings()


def test_governed_iteration_and_retry_config_accepts_only_exact_values() -> None:
    invalid_rows = (
        {"crewai_max_iter": 2},
        {"crewai_max_retry_limit": 1},
        {"planner_transient_retry_attempts": 1},
        {"planner_transient_retry_base_seconds": 0.1},
        {"llm_health_probe_retry_attempts": 1},
        {"llm_health_probe_retry_base_seconds": 0.1},
    )

    for row in invalid_rows:
        with pytest.raises(ValidationError):
            Settings.model_validate(row)


def test_planner_transient_retry_rejects_nonzero_environment(monkeypatch):
    monkeypatch.setenv("MARKET_AGENT_PLANNER_TRANSIENT_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("MARKET_AGENT_PLANNER_TRANSIENT_RETRY_BASE_SECONDS", "0.25")

    with pytest.raises(ValidationError, match="planner_transient_retry_attempts"):
        get_settings()


def test_llm_max_tokens_defaults_to_real_structured_output_budget(monkeypatch):
    monkeypatch.delenv("MARKET_AGENT_LLM_MAX_TOKENS", raising=False)

    settings = get_settings()

    assert settings.llm_max_tokens == 6000


def test_llm_timeout_defaults_match_live_provider_budget(monkeypatch):
    monkeypatch.delenv("MARKET_AGENT_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("CREWAI_MAX_EXECUTION_TIME", raising=False)

    settings = get_settings()

    assert settings.llm_timeout_seconds == 90
    assert settings.crewai_max_execution_time == 120


def test_llm_max_tokens_reads_environment(monkeypatch):
    monkeypatch.setenv("MARKET_AGENT_LLM_MAX_TOKENS", "2500")

    settings = get_settings()

    assert settings.llm_max_tokens == 2500


def test_planner_llm_override_reads_environment(monkeypatch):
    monkeypatch.setenv(
        "MARKET_AGENT_PLANNER_LLM_BASE_URL", "http://planner.local/gemini"
    )
    monkeypatch.setenv("MARKET_AGENT_PLANNER_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("MARKET_AGENT_PLANNER_LLM_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("MARKET_AGENT_PLANNER_LLM_API_KEY", "planner-key")

    settings = get_settings()

    assert settings.planner_llm_base_url == "http://planner.local/gemini"
    assert settings.planner_llm_provider == "gemini"
    assert settings.planner_llm_model == "gemini-3-flash-preview"
    assert settings.planner_llm_api_key == "planner-key"


def test_blank_planner_llm_api_key_does_not_reuse_default_key(monkeypatch):
    monkeypatch.setenv("MARKET_AGENT_LLM_API_KEY", "default-key")
    monkeypatch.setenv("MARKET_AGENT_PLANNER_LLM_BASE_URL", "http://planner.local")
    monkeypatch.setenv("MARKET_AGENT_PLANNER_LLM_API_KEY", "")

    settings = get_settings()

    assert settings.planner_llm_api_key is None


def test_planner_llm_override_without_key_does_not_reuse_default_key(monkeypatch):
    monkeypatch.setenv("MARKET_AGENT_LLM_API_KEY", "default-key")
    monkeypatch.setenv("MARKET_AGENT_PLANNER_LLM_BASE_URL", "http://planner.local")
    monkeypatch.delenv("MARKET_AGENT_PLANNER_LLM_API_KEY", raising=False)

    settings = get_settings()

    assert settings.planner_llm_api_key is None


def test_planner_llm_uses_default_key_without_override(monkeypatch):
    monkeypatch.setenv("MARKET_AGENT_LLM_API_KEY", "default-key")
    monkeypatch.delenv("MARKET_AGENT_PLANNER_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("MARKET_AGENT_PLANNER_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MARKET_AGENT_PLANNER_LLM_MODEL", raising=False)
    monkeypatch.delenv("MARKET_AGENT_PLANNER_LLM_API_KEY", raising=False)

    settings = get_settings()

    assert settings.planner_llm_api_key == "default-key"


def test_doc_mcp_configuration_reads_environment(monkeypatch):
    monkeypatch.setenv("MARKET_AGENT_DOC_MCP_BASE_URL", "http://10.0.0.12:23000")
    monkeypatch.setenv("MARKET_AGENT_DOC_MCP_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("MARKET_AGENT_DOC_MCP_ENABLED", "true")

    settings = get_settings()

    assert settings.doc_mcp_base_url == "http://10.0.0.12:23000"
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
