from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from market_support_crewai_agent.runtime.evidence.adapter_client import (
    AdapterClientError,
    AdapterResolveClient,
)
from market_support_crewai_agent.schemas import (
    AdapterCapabilities,
    AdapterMetrics,
    AdapterResolveRequest,
    AdapterResolveResult,
)
from market_support_crewai_agent.settings import Settings


class ResolveHandler(BaseHTTPRequestHandler):
    payloads: list[dict] = []
    capabilities_response: dict | None = None

    def do_GET(self):
        ResolveHandler.payloads.append(
            {
                "payload": {},
                "authorization": self.headers.get("Authorization", ""),
                "path": self.path,
            }
        )
        if self.path == "/adapter/capabilities":
            response = ResolveHandler.capabilities_response or _capabilities_response()
        elif self.path == "/adapter/metrics":
            response = _metrics_response()
        else:
            response = {
                "status": "ok",
                "service": "xiaoyan-wecom-market-agent-adapter",
            }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        ResolveHandler.payloads.append(
            {
                "payload": payload,
                "authorization": self.headers.get("Authorization", ""),
                "path": self.path,
            }
        )
        if self.path.endswith("/batch"):
            response = {
                "contract_version": "adapter-resolve-batch",
                "results": [_resolve_response(item) for item in payload["requests"]],
            }
        else:
            response = _resolve_response(payload)
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def _resolve_response(payload):
    return {
        "contract_version": "adapter-resolve",
        "resolve_type": payload["resolve_type"],
        "status": "resolved",
        "display_name": payload["dist_name"],
        "reason_code": "ok",
        "candidates": [],
        "channel_type": "bank",
        "available_materials": ["material", "weekly"],
        "material_pack_options": ["指增"],
        "resolved_at": 1,
        "resolve_ref": "wecom-adapter:test",
        "material_pack_option": payload.get("material_pack_option"),
        "period": "20260529",
    }


def _capabilities_response():
    return {
        "service": "xiaoyan-wecom-market-agent-adapter",
        "contract_version": "adapter-resolve",
        "batch_contract_version": "adapter-resolve-batch",
        "action_contract_version": "adapter-action",
        "endpoints": {
            "health": "/health",
            "capabilities": "/adapter/capabilities",
            "metrics": "/adapter/metrics",
            "resolve": "/adapter/resolve",
            "batch_resolve": "/adapter/resolve/batch",
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
        "max_request_body_bytes": 65536,
        "cache_ttl_seconds": 30,
        "cache_max_entries": 512,
        "auth": {
            "header_schemes": [
                "Authorization: Bearer <key>",
                "X-API-Key: <key>",
            ],
            "protected_endpoints": [
                "/adapter/resolve",
                "/adapter/resolve/batch",
                "/adapter/metrics",
            ],
        },
    }


def _metrics_response():
    return {
        "service": "xiaoyan-wecom-market-agent-adapter",
        "uptime_seconds": 12,
        "resolver": {
            "cache": {
                "ttl_seconds": 30,
                "max_entries": 512,
                "entries": 4,
                "hits": 7,
                "misses": 3,
                "sets": 4,
                "expired": 1,
                "evictions": 0,
            }
        },
        "transport": {
            "requests_total": 3,
            "inflight_requests": 1,
            "errors_total": 1,
            "status_codes": {"200": 2, "404": 1},
            "duration_ms": {"count": 3, "total": 14.5, "max": 9.0},
            "routes": {
                "batch_resolve": {
                    "requests": 2,
                    "errors": 0,
                    "status_codes": {"200": 2},
                    "duration_ms": {"count": 2, "total": 12.0, "max": 9.0},
                },
                "not_found": {
                    "requests": 1,
                    "errors": 1,
                    "status_codes": {"404": 1},
                    "duration_ms": {"count": 1, "total": 2.5, "max": 2.5},
                },
            },
        },
    }


def test_adapter_capabilities_schema_accepts_adapter_contract():
    capabilities = AdapterCapabilities.model_validate(_capabilities_response())

    assert capabilities.service == "xiaoyan-wecom-market-agent-adapter"
    assert capabilities.endpoints.resolve == "/adapter/resolve"
    assert capabilities.endpoints.metrics == "/adapter/metrics"
    assert capabilities.resolve_types == [
        "material_pack",
        "weekly_report",
        "monthly_report",
        "sales_mention",
    ]
    assert capabilities.max_batch_requests == 16
    assert capabilities.action_contract_version == "adapter-action"
    assert capabilities.cache_ttl_seconds == 30
    assert capabilities.cache_max_entries == 512
    assert capabilities.auth is not None
    assert "/adapter/metrics" in capabilities.auth.protected_endpoints


def test_adapter_metrics_schema_accepts_adapter_contract():
    metrics = AdapterMetrics.model_validate(_metrics_response())

    assert metrics.service == "xiaoyan-wecom-market-agent-adapter"
    assert metrics.uptime_seconds == 12
    assert metrics.resolver.cache.hits == 7
    assert metrics.resolver.cache.evictions == 0
    assert metrics.transport.requests_total == 3
    assert metrics.transport.errors_total == 1
    assert metrics.transport.status_codes["404"] == 1
    assert metrics.transport.routes["batch_resolve"].requests == 2
    assert metrics.transport.routes["not_found"].errors == 1


def test_adapter_resolve_schema_accepts_adapter_contract():
    result = AdapterResolveResult.model_validate(
        {
            "contract_version": "adapter-resolve",
            "resolve_type": "weekly_report",
            "status": "resolved",
            "display_name": "测试渠道",
            "reason_code": "ok",
            "candidates": [],
            "channel_type": "non_bank",
            "available_materials": ["weekly"],
            "material_pack_options": [],
            "resolved_at": 1,
            "resolve_ref": "wecom-adapter:test",
            "period": "20260529",
            "report_date": "2026-05-29",
            "detail": "weekly report artifact unavailable",
        }
    )

    assert result.status == "resolved"
    assert result.period == "20260529"
    assert result.report_date == "2026-05-29"
    assert result.detail == "weekly report artifact unavailable"


def test_adapter_resolve_schema_rejects_raw_locator_detail():
    with pytest.raises(ValueError, match="detail contains raw locator values"):
        AdapterResolveResult.model_validate(
            {
                "contract_version": "adapter-resolve",
                "resolve_type": "weekly_report",
                "status": "missing",
                "display_name": "测试渠道",
                "reason_code": "weekly_report_unavailable",
                "candidates": [],
                "channel_type": "non_bank",
                "available_materials": [],
                "material_pack_options": [],
                "resolved_at": 1,
                "detail": "CSV file not found: /Users/ivan/private/portfolio_url_info.csv",
            }
        )


def test_adapter_resolve_schema_rejects_raw_locator_resolve_ref():
    with pytest.raises(ValueError, match="resolve_ref contains raw locator values"):
        AdapterResolveResult.model_validate(
            {
                "contract_version": "adapter-resolve",
                "resolve_type": "weekly_report",
                "status": "resolved",
                "display_name": "测试渠道",
                "reason_code": "ok",
                "candidates": [],
                "channel_type": "non_bank",
                "available_materials": ["weekly"],
                "material_pack_options": [],
                "resolved_at": 1,
                "resolve_ref": "https://drive.weixin.qq.com/s?k=secret",
            }
        )


def test_adapter_resolve_schema_rejects_resolved_without_resolve_ref():
    with pytest.raises(ValueError, match="resolved adapter results must include resolve_ref"):
        AdapterResolveResult.model_validate(
            {
                "contract_version": "adapter-resolve",
                "resolve_type": "weekly_report",
                "status": "resolved",
                "display_name": "测试渠道",
                "reason_code": "ok",
                "candidates": [],
                "channel_type": "non_bank",
                "available_materials": ["weekly"],
                "material_pack_options": [],
                "resolved_at": 1,
            }
        )


def test_adapter_resolve_schema_rejects_removed_locator_field():
    removed_locator_field = "card" + "_ref"
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AdapterResolveResult.model_validate(
            {
                "contract_version": "adapter-resolve",
                "resolve_type": "weekly_report",
                "status": "resolved",
                "display_name": "测试渠道",
                "reason_code": "ok",
                "candidates": [],
                "channel_type": "non_bank",
                "available_materials": ["weekly"],
                "material_pack_options": [],
                "resolved_at": 1,
                "resolve_ref": "wecom-adapter:test",
                removed_locator_field: "removed-locator",
            }
        )


def test_adapter_resolve_schema_rejects_unowned_metadata_bucket():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AdapterResolveResult.model_validate(
            {
                "contract_version": "adapter-resolve",
                "resolve_type": "weekly_report",
                "status": "resolved",
                "display_name": "测试渠道",
                "reason_code": "ok",
                "candidates": [],
                "channel_type": "non_bank",
                "available_materials": ["weekly"],
                "material_pack_options": [],
                "resolved_at": 1,
                "resolve_ref": "wecom-adapter:test",
                "metadata": {"report_url": "https://drive.weixin.qq.com/s?k=secret"},
            }
        )


def test_adapter_client_gets_capabilities_with_bearer_auth():
    ResolveHandler.payloads = []
    ResolveHandler.capabilities_response = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), ResolveHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    try:
        settings = Settings(
            llm_api_key="test-key",
            adapter_base_url=f"http://127.0.0.1:{server.server_address[1]}",
            adapter_api_key="secret",
        )
        client = AdapterResolveClient(settings)
        capabilities = client.capabilities()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert capabilities.contract_version == "adapter-resolve"
    assert capabilities.endpoints.capabilities == "/adapter/capabilities"
    assert capabilities.max_request_body_bytes == 65536
    assert ResolveHandler.payloads[0]["path"] == "/adapter/capabilities"
    assert ResolveHandler.payloads[0]["authorization"] == "Bearer secret"


def test_adapter_client_gets_metrics_with_bearer_auth():
    ResolveHandler.payloads = []
    ResolveHandler.capabilities_response = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), ResolveHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    try:
        settings = Settings(
            llm_api_key="test-key",
            adapter_base_url=f"http://127.0.0.1:{server.server_address[1]}",
            adapter_api_key="secret",
        )
        client = AdapterResolveClient(settings)
        metrics = client.metrics()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert metrics.resolver.cache.entries == 4
    assert ResolveHandler.payloads[0]["path"] == "/adapter/metrics"
    assert ResolveHandler.payloads[0]["authorization"] == "Bearer secret"


def test_adapter_client_assert_ready_validates_and_caches_capabilities():
    ResolveHandler.payloads = []
    ResolveHandler.capabilities_response = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), ResolveHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    try:
        settings = Settings(
            llm_api_key="test-key",
            adapter_base_url=f"http://127.0.0.1:{server.server_address[1]}",
            adapter_api_key="secret",
        )
        client = AdapterResolveClient(settings)
        first = client.assert_ready()
        second = client.assert_ready()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert first.service == "xiaoyan-wecom-market-agent-adapter"
    assert second is first
    assert [request["path"] for request in ResolveHandler.payloads] == [
        "/adapter/capabilities"
    ]


def test_adapter_client_assert_ready_rejects_contract_mismatch():
    capabilities = _capabilities_response()
    capabilities["endpoints"]["batch_resolve"] = "/resolve/batch"
    ResolveHandler.payloads = []
    ResolveHandler.capabilities_response = capabilities
    server = ThreadingHTTPServer(("127.0.0.1", 0), ResolveHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    try:
        settings = Settings(
            llm_api_key="test-key",
            adapter_base_url=f"http://127.0.0.1:{server.server_address[1]}",
        )
        client = AdapterResolveClient(settings)
        with pytest.raises(AdapterClientError, match="endpoints.batch_resolve"):
            client.assert_ready()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        ResolveHandler.capabilities_response = None


def test_adapter_client_posts_bearer_auth_and_validates_result():
    ResolveHandler.payloads = []
    ResolveHandler.capabilities_response = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), ResolveHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    try:
        settings = Settings(
            llm_api_key="test-key",
            adapter_base_url=f"http://127.0.0.1:{server.server_address[1]}",
            adapter_api_key="secret",
        )
        client = AdapterResolveClient(settings)
        result = client.resolve(
            AdapterResolveRequest(
                resolve_type="material_pack",
                dist_name="测试渠道",
                material_pack_option="指增",
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert result.status == "resolved"
    assert result.resolve_ref == "wecom-adapter:test"
    assert ResolveHandler.payloads[0]["authorization"] == "Bearer secret"
    assert ResolveHandler.payloads[0]["payload"]["resolve_type"] == "material_pack"
    assert ResolveHandler.payloads[0]["payload"]["material_pack_option"] == "指增"


def test_adapter_client_posts_batch_resolve_request():
    ResolveHandler.payloads = []
    ResolveHandler.capabilities_response = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), ResolveHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    try:
        settings = Settings(
            llm_api_key="test-key",
            adapter_base_url=f"http://127.0.0.1:{server.server_address[1]}",
            adapter_api_key="secret",
        )
        client = AdapterResolveClient(settings)
        results = client.resolve_many(
            [
                AdapterResolveRequest(
                    resolve_type="weekly_report",
                    dist_name="测试渠道",
                ),
                AdapterResolveRequest(
                    resolve_type="sales_mention",
                    dist_name="测试渠道",
                ),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert [result.resolve_type for result in results] == ["weekly_report", "sales_mention"]
    assert ResolveHandler.payloads[0]["authorization"] == "Bearer secret"
    assert [item["resolve_type"] for item in ResolveHandler.payloads[0]["payload"]["requests"]] == [
        "weekly_report",
        "sales_mention",
    ]
