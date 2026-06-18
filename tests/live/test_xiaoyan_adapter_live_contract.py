from __future__ import annotations

import asyncio
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from market_support_crewai_agent.runtime.evidence.adapter_preflight import AdapterPreflightService
from market_support_crewai_agent.runtime.evidence.adapter_client import (
    AdapterClientError,
    AdapterResolveClient,
)
from market_support_crewai_agent.schemas import (
    AdapterReportScopeRequest,
    AdapterResolveRequest,
    ReplyRequest,
)
from market_support_crewai_agent.settings import Settings


RAW_SERVER_FIELDS = {
    "receiver",
    "group_id",
    "room_id",
    "conversation_id",
    "actual_user_id",
    "user_id",
    "to_user_id",
    "from_user_id",
    "at_list",
    "mention_ids",
    "url",
    "link",
    "file_path",
    "path",
}


def test_live_xiaoyan_adapter_capabilities_contract():
    base_url = _live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None

    _skip_if_adapter_is_not_running(base_url, api_key)

    capabilities = _live_adapter_client(base_url, api_key).assert_ready()

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


def test_live_xiaoyan_adapter_metrics_contract():
    base_url = _live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None

    _skip_if_adapter_is_not_running(base_url, api_key)

    metrics = _live_adapter_client(base_url, api_key).metrics()

    assert metrics.service == "xiaoyan-wecom-market-agent-adapter"
    assert metrics.uptime_seconds >= 0
    cache = metrics.resolver.cache
    assert cache.ttl_seconds >= 0
    assert cache.max_entries >= 0
    assert cache.entries >= 0
    assert cache.hits >= 0
    assert cache.misses >= 0
    transport = metrics.transport
    assert transport.requests_total >= 0
    assert transport.inflight_requests >= 0
    assert transport.errors_total >= 0
    assert transport.duration_ms.count >= 0
    assert transport.duration_ms.total >= 0
    assert transport.duration_ms.max >= 0
    assert set(transport.routes).issubset(
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
    for route_metrics in transport.routes.values():
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


def test_live_xiaoyan_adapter_batch_contract():
    base_url = _live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None
    dist_name = os.getenv("MARKET_AGENT_LIVE_ADAPTER_DIST_NAME", "__contract_check__")
    material_pack_option = os.getenv("MARKET_AGENT_LIVE_MATERIAL_PACK_OPTION") or None

    _skip_if_adapter_is_not_running(base_url, api_key)

    client = _live_adapter_client(base_url, api_key)
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

def test_live_xiaoyan_adapter_preflight_service_material_pack_option_contract():
    material_pack_option = os.getenv("MARKET_AGENT_LIVE_MATERIAL_PACK_OPTION")
    if not material_pack_option:
        pytest.skip("material-pack option live eval requires MARKET_AGENT_LIVE_MATERIAL_PACK_OPTION")

    base_url = _live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None
    dist_name = os.getenv("MARKET_AGENT_LIVE_ADAPTER_DIST_NAME", "MaterialPackOptionTest")

    _skip_if_adapter_is_not_running(base_url, api_key)

    client = _live_adapter_client(base_url, api_key)
    service = AdapterPreflightService(adapter_client=client)
    request = ReplyRequest.model_validate(
        {
            "context_id": "live-scope-msg-1",
            "conversation_key": "wecom:live-scope:sender-1",
            "group_id": "live-scope-group",
            "sender_id": "sender-1",
            "message": "请确认{}材料包".format(dist_name),
            "is_group": True,
            "group_name": "{}-群".format(dist_name),
            "dist_channel_name": dist_name,
            "sender_nickname": "live tester",
            "available_materials": ["material", "weekly", "monthly"],
            "material_pack_options": [material_pack_option],
            "channel_type": "bank",
        }
    )

    snapshot = asyncio.run(
        service.collect(
            request,
            resolve_material_pack_options={"material_pack": material_pack_option},
        )
    )

    assert snapshot.available is True
    assert [item.resolve_type for item in snapshot.items] == [
        "material_pack",
        "weekly_report",
        "monthly_report",
        "sales_mention",
    ]
    weekly = next(item.result for item in snapshot.items if item.resolve_type == "weekly_report")
    material = next(item.result for item in snapshot.items if item.resolve_type == "material_pack")
    assert material is not None
    assert material.material_pack_option == material_pack_option
    assert weekly is not None
    assert weekly.material_pack_option is None


def test_live_xiaoyan_adapter_report_scope_contract():
    if os.getenv("MARKET_AGENT_LIVE_ADAPTER_EXPECT_REPORT_SCOPE", "").strip() != "1":
        pytest.skip("report-scope live eval requires fixture-backed adapter")

    base_url = _live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None
    dist_name = os.getenv("MARKET_AGENT_LIVE_ADAPTER_DIST_NAME", "ReportScopeTest")

    _skip_if_adapter_is_not_running(base_url, api_key)

    client = _live_adapter_client(base_url, api_key)
    result = client.report_scope(
        AdapterReportScopeRequest(
            material_type="weekly",
            dist_name=dist_name,
            command="summary",
        )
    )

    assert result.contract_version == "adapter-report-scope"
    assert result.material_type == "weekly"
    assert result.dist_name == dist_name
    assert result.status == "resolved"
    assert result.period
    assert result.report_sections or result.expected_product_count is not None


def test_live_xiaoyan_adapter_rejects_short_resolve_endpoint():
    base_url = _live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None

    _skip_if_adapter_is_not_running(base_url, api_key)

    body = json.dumps(
        {"resolve_type": "weekly_report", "dist_name": "__contract_check__"},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(
        f"{base_url.rstrip('/')}/resolve",
        data=body,
        headers=headers,
        method="POST",
    )
    with pytest.raises(HTTPError) as caught:
        urlopen(request, timeout=3)

    assert caught.value.code == 404


def test_live_xiaoyan_adapter_rejects_raw_list_batch_payload():
    base_url = _live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None

    _skip_if_adapter_is_not_running(base_url, api_key)

    body = json.dumps(
        [{"resolve_type": "weekly_report", "dist_name": "__contract_check__"}],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(
        f"{base_url.rstrip('/')}/adapter/resolve/batch",
        data=body,
        headers=headers,
        method="POST",
    )
    with pytest.raises(HTTPError) as caught:
        urlopen(request, timeout=3)

    assert caught.value.code == 400
    payload = json.loads(caught.value.read().decode("utf-8"))
    assert payload == {
        "error": "bad_request",
        "detail": "batch resolve request must be a JSON object",
    }


def _live_adapter_base_url() -> str:
    return os.getenv(
        "MARKET_AGENT_LIVE_ADAPTER_BASE_URL",
        os.getenv("MARKET_AGENT_ADAPTER_BASE_URL", "http://127.0.0.1:8011"),
    ).rstrip("/")


def _live_adapter_client(base_url: str, api_key: str | None) -> AdapterResolveClient:
    return AdapterResolveClient(
        Settings(
            llm_api_key="test-key",
            adapter_base_url=base_url,
            adapter_api_key=api_key,
            adapter_timeout_seconds=float(
                os.getenv("MARKET_AGENT_LIVE_ADAPTER_TIMEOUT_SECONDS", "3")
            ),
        )
    )


def _skip_if_adapter_is_not_running(base_url: str, api_key: str | None) -> None:
    try:
        body = _get_json(f"{base_url.rstrip('/')}/health", api_key, timeout=1.5)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"live xiaoyan adapter is not running at {base_url}: {exc}")

    if body.get("service") != "xiaoyan-wecom-market-agent-adapter":
        pytest.skip(f"{base_url} is not the xiaoyan market-agent adapter")


def _get_json(url: str, api_key: str | None, timeout: float = 3.0) -> dict:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
