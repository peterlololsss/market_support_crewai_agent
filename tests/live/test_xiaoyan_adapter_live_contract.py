from __future__ import annotations

import json
import os

import pytest

from market_support_crewai_agent.runtime.integrations.adapter.transport import (
    AdapterClientError,
)
from market_support_crewai_agent.schemas.adapter import AdapterResolveRequest
from tests.live.xiaoyan_adapter_live_support import (
    RAW_SERVER_FIELDS,
    live_adapter_base_url,
    live_adapter_client,
    skip_if_adapter_is_not_running,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("MARKET_AGENT_RUN_LIVE_ADAPTER_TESTS", "").strip() != "1",
        reason="live adapter tests require MARKET_AGENT_RUN_LIVE_ADAPTER_TESTS=1",
    ),
]


def test_live_xiaoyan_adapter_capabilities_contract() -> None:
    base_url = live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None
    skip_if_adapter_is_not_running(base_url, api_key)

    client = live_adapter_client(base_url, api_key)
    capabilities = client.assert_ready()

    assert capabilities.service == "xiaoyan-wecom-market-agent-adapter"
    assert capabilities.contract_version == "adapter-resolve"
    assert capabilities.batch_contract_version == "adapter-resolve-batch"
    assert capabilities.endpoints.metrics == "/adapter/metrics"
    assert capabilities.endpoints.resolve == "/adapter/resolve"
    assert capabilities.endpoints.batch_resolve == "/adapter/resolve/batch"
    assert capabilities.endpoints.report_scope == "/adapter/report-scope"
    assert capabilities.resolve_types == [
        "material_pack",
        "weekly_report",
        "monthly_report",
        "sales_mention",
    ]
    assert capabilities.max_batch_requests >= 4
    assert capabilities.max_request_body_bytes > 0
    assert capabilities.cache_ttl_seconds >= 0
    assert capabilities.cache_max_entries >= 0

    supported_scenes = capabilities.supported_scenes
    reply_versions = capabilities.reply_request_contract_versions
    feedback_versions = capabilities.action_feedback_contract_versions
    identity_versions = capabilities.conversation_identity_contract_versions
    advertised_tenant = capabilities.deployment_tenant_ref
    if (
        supported_scenes is None
        or reply_versions is None
        or feedback_versions is None
        or identity_versions is None
        or advertised_tenant is None
    ):
        pytest.skip(
            "external adapter release blocker: direct capability fields are not advertised"
        )
    deployment_tenant_ref = os.getenv("MARKET_AGENT_DEPLOYMENT_TENANT_REF")
    if api_key is None or deployment_tenant_ref is None:
        pytest.skip(
            "direct capability live check requires adapter auth and deployment tenant"
        )

    compatible = client.assert_scene_compatible("direct", deployment_tenant_ref)

    assert compatible is capabilities
    assert "direct" in supported_scenes
    assert "reply-request.v2" in reply_versions
    assert "action-feedback.v2" in feedback_versions
    assert "conversation-identity.v1" in identity_versions
    assert advertised_tenant == deployment_tenant_ref


def test_live_xiaoyan_adapter_metrics_contract() -> None:
    base_url = live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None
    skip_if_adapter_is_not_running(base_url, api_key)

    metrics = live_adapter_client(base_url, api_key).metrics()

    assert metrics.service == "xiaoyan-wecom-market-agent-adapter"
    assert metrics.uptime_seconds >= 0
    assert metrics.resolver.cache.ttl_seconds >= 0
    assert metrics.resolver.cache.max_entries >= 0
    assert metrics.resolver.cache.entries >= 0
    assert metrics.resolver.cache.hits >= 0
    assert metrics.resolver.cache.misses >= 0
    assert metrics.transport.requests_total >= 0
    assert metrics.transport.inflight_requests >= 0
    assert metrics.transport.errors_total >= 0
    assert metrics.transport.duration_ms.count >= 0
    assert metrics.transport.duration_ms.total >= 0
    assert metrics.transport.duration_ms.max >= 0
    assert set(metrics.transport.routes).issubset(
        {
            "health",
            "capabilities",
            "metrics",
            "resolve",
            "batch_resolve",
            "report_scope",
            "not_found",
        }
    )
    for route_metrics in metrics.transport.routes.values():
        assert route_metrics.requests >= 0
        assert route_metrics.errors >= 0
        assert route_metrics.duration_ms.count >= 0
    payload = metrics.model_dump(mode="json")
    serialized = json.dumps(payload, ensure_ascii=False)
    assert not RAW_SERVER_FIELDS.intersection(payload)
    assert "http://" not in serialized
    assert "https://" not in serialized
    assert "/Users/" not in serialized
    assert "/home/" not in serialized
    assert "portfolio_url_info.csv" not in serialized


def test_live_xiaoyan_adapter_batch_contract() -> None:
    base_url = live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None
    dist_name = os.getenv("MARKET_AGENT_LIVE_ADAPTER_DIST_NAME", "__contract_check__")
    material_pack_option = os.getenv("MARKET_AGENT_LIVE_MATERIAL_PACK_OPTION") or None
    skip_if_adapter_is_not_running(base_url, api_key)

    client = live_adapter_client(base_url, api_key)
    requests = [
        AdapterResolveRequest(
            resolve_type="material_pack",
            dist_name=dist_name,
            material_pack_option=material_pack_option,
        ),
        AdapterResolveRequest(resolve_type="weekly_report", dist_name=dist_name),
        AdapterResolveRequest(resolve_type="monthly_report", dist_name=dist_name),
        AdapterResolveRequest(resolve_type="sales_mention", dist_name=dist_name),
    ]
    try:
        results = client.resolve_many(requests)
    except AdapterClientError as exc:
        pytest.fail(f"live xiaoyan adapter returned invalid batch contract: {exc}")

    assert [result.resolve_type for result in results] == [
        request.resolve_type for request in requests
    ]
    assert len(results) == len(requests)
    for result in results:
        assert result.contract_version == "adapter-resolve"
        assert result.reason_code
        assert isinstance(result.resolved_at, int)
        payload = result.model_dump(mode="json", exclude_none=True)
        assert not RAW_SERVER_FIELDS.intersection(payload)
        serialized = json.dumps(payload, ensure_ascii=False)
        assert "http://" not in serialized
        assert "https://" not in serialized
        assert "/Users/" not in serialized
        assert "/home/" not in serialized
        assert "portfolio_url_info.csv" not in serialized
