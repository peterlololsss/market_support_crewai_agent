from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter

from market_support_crewai_agent.schemas.adapter import (
    AdapterResolveBatchRequest,
    AdapterResolveRequest,
    AvailableArtifact,
)
from market_support_crewai_agent.schemas.adapter_metadata import AdapterCapabilities
from market_support_crewai_agent.schemas.conversation import ReplyRequestV2
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2
from market_support_crewai_agent.settings import get_settings

ROOT = Path(__file__).resolve().parents[2]
REPLY_DOCS = (
    ROOT / "README.md",
    ROOT / "docs/adapter/xiaoyan_adapter_contract.md",
)
ENV_TEMPLATES = (
    ROOT / ".env.example",
    ROOT / "deploy/market-support-crewai-agent.env.example",
)
SETTINGS_ENV_PREFIXES = ("AGENT_", "CREWAI_", "MARKET_AGENT_", "YANFU_")
JSON_FENCE: re.Pattern[str] = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)
JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
AVAILABLE_ARTIFACTS = TypeAdapter(list[AvailableArtifact])


def _json_examples(path: Path) -> tuple[dict[str, JsonValue], ...]:
    text = path.read_text(encoding="utf-8")
    return tuple(
        JSON_OBJECT.validate_json(match.group(1)) for match in JSON_FENCE.finditer(text)
    )


def _reply_examples(path: Path) -> tuple[ReplyRequestV2, ...]:
    return tuple(
        ReplyRequestV2.model_validate(payload)
        for payload in _json_examples(path)
        if payload.get("contract_version") == "reply-request.v2"
    )


def _adapter_capabilities_payload(
    additive_fields: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    return {
        "service": "xiaoyan-wecom-market-agent-adapter",
        "contract_version": "adapter-resolve",
        "batch_contract_version": "adapter-resolve-batch",
        "action_contract_version": "adapter-action",
        "endpoints": {
            "health": "/health",
            "capabilities": "/capabilities",
            "metrics": "/metrics",
            "resolve": "/resolve",
            "batch_resolve": "/resolve/batch",
        },
        "resolve_types": [
            "material_pack",
            "weekly_report",
            "monthly_report",
            "sales_mention",
        ],
        "statuses": [
            "resolved",
            "missing",
            "ambiguous",
            "forbidden",
            "temporarily_unavailable",
        ],
        "max_batch_requests": 16,
        "max_request_body_bytes": 1_048_576,
        **additive_fields,
    }


def _env_template(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        assert separator, f"{path}:{line_number} is not an environment assignment"
        assert name and name == name.strip(), (
            f"{path}:{line_number} has an invalid name"
        )
        assert name not in values, f"{path}:{line_number} duplicates {name}"
        values[name] = value
    return values


def _template_id(path: Path) -> str:
    return path.name


def test_documented_reply_examples_match_the_single_enterprise_scene_contract() -> None:
    # Given: every documented reply-request.v2 JSON example.
    by_document = {path: _reply_examples(path) for path in REPLY_DOCS}

    # When: examples are parsed through the current public DTO.
    examples = tuple(
        example
        for document_examples in by_document.values()
        for example in document_examples
    )
    direct_examples = tuple(
        example for example in examples if example.identity.scene == "direct"
    )

    # Then: each document covers both wire scenes under one deployment tenant.
    assert all(
        {example.identity.scene for example in document_examples} == {"direct", "group"}
        for document_examples in by_document.values()
    )
    assert {example.identity.tenant_ref for example in examples} == {"tenant:primary"}
    assert direct_examples
    assert all(example.business_scope.kind == "unscoped" for example in direct_examples)
    assert all(
        example.grants.read_capabilities == ["query_internal_company_info"]
        for example in direct_examples
    )
    assert all(not example.grants.outbound_actions for example in direct_examples)
    assert all(not example.grants.mention_types for example in direct_examples)


@pytest.mark.parametrize("template", ENV_TEMPLATES, ids=_template_id)
def test_documented_env_templates_load_with_internal_dm_disabled(
    template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: one checked-in deployment environment template and no ambient settings.
    values = _env_template(template)
    for name in tuple(os.environ):
        if name.startswith(SETTINGS_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    # When: the documented assignments cross the production settings boundary.
    settings = get_settings()

    # Then: both templates expose the required placeholders and default DM off.
    assert values["MARKET_AGENT_API_KEY"] == ""
    assert values["MARKET_AGENT_DEPLOYMENT_TENANT_REF"] == ""
    assert values["MARKET_AGENT_INTERNAL_DM_ENABLED"] == "false"
    assert settings.deployment_tenant_ref is None
    assert settings.internal_dm_enabled is False


def test_documented_adapter_json_examples_match_published_adapter_schemas() -> None:
    # Given: every adapter-facing JSON block in the published adapter contract.
    payloads = _json_examples(ROOT / "docs/adapter/xiaoyan_adapter_contract.md")

    # When: each code block is routed through the schema that consumes it.
    request_indexes = {
        index
        for index, payload in enumerate(payloads)
        if payload.get("contract_version") == "reply-request.v2"
    }
    feedback_indexes = {
        index
        for index, payload in enumerate(payloads)
        if payload.get("contract_version") == "action-feedback.v2"
    }
    capability_indexes = {
        index for index, payload in enumerate(payloads) if "supported_scenes" in payload
    }
    resolve_indexes = {
        index for index, payload in enumerate(payloads) if "resolve_type" in payload
    }
    batch_indexes = {
        index for index, payload in enumerate(payloads) if "requests" in payload
    }
    artifact_indexes = {
        index
        for index, payload in enumerate(payloads)
        if "available_artifacts" in payload
    }

    requests = tuple(
        ReplyRequestV2.model_validate(payloads[index]) for index in request_indexes
    )
    feedback = tuple(
        ActionFeedbackRequestV2.model_validate(payloads[index])
        for index in feedback_indexes
    )
    capabilities = tuple(
        AdapterCapabilities.model_validate(
            _adapter_capabilities_payload(payloads[index]),
        )
        for index in capability_indexes
    )
    resolves = tuple(
        AdapterResolveRequest.model_validate(payloads[index])
        for index in resolve_indexes
    )
    batches = tuple(
        AdapterResolveBatchRequest.model_validate(payloads[index])
        for index in batch_indexes
    )
    artifact_lists = tuple(
        AVAILABLE_ARTIFACTS.validate_python(payloads[index]["available_artifacts"])
        for index in artifact_indexes
    )

    # Then: all documented JSON is consumed by current public/adapter DTOs.
    consumed_indexes = (
        request_indexes
        | feedback_indexes
        | capability_indexes
        | resolve_indexes
        | batch_indexes
        | artifact_indexes
    )
    assert consumed_indexes == set(range(len(payloads)))
    assert requests
    assert feedback
    assert capabilities
    assert resolves
    assert batches
    assert artifact_lists
