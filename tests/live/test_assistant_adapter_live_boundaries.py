from __future__ import annotations

import json
import os
from functools import partial
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import anyio
import pytest
from pydantic import BaseModel, ConfigDict

from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightService,
)
from market_support_crewai_agent.schemas.adapter import AdapterReportScopeRequest
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.live.assistant_adapter_live_support import (
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


class ErrorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    error: str
    detail: str


def test_live_assistant_adapter_preflight_service_material_pack_option_contract() -> (
    None
):
    material_pack_option = os.getenv("MARKET_AGENT_LIVE_MATERIAL_PACK_OPTION")
    if not material_pack_option:
        pytest.skip(
            "material-pack option live eval requires MARKET_AGENT_LIVE_MATERIAL_PACK_OPTION"
        )
    base_url = live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None
    dist_name = os.getenv(
        "MARKET_AGENT_LIVE_ADAPTER_DIST_NAME",
        "MaterialPackOptionTest",
    )
    skip_if_adapter_is_not_running(base_url, api_key)

    request = make_v2_envelope(
        f"请确认{dist_name}材料包",
        context_id="ctx:live-scope-msg-1",
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "group",
            "tenant_ref": "tenant:live",
            "group_ref": "group:live-scope-group",
            "principal_ref": "principal:sender-1",
        },
        presentation={
            "contract_version": "group-presentation.v1",
            "conversation_name": f"{dist_name}-群",
            "principal_name": "live tester",
        },
        business_scope={
            "kind": "distribution",
            "dist_channel_name": dist_name,
            "channel_type": "bank",
            "available_artifacts": [
                {"type": "material_pack", "options": [material_pack_option]},
                {"type": "weekly_report"},
                {"type": "monthly_report"},
            ],
        },
    ).request
    collect = partial(
        AdapterPreflightService(live_adapter_client(base_url, api_key)).collect,
        request,
        resolve_material_pack_options={"material_pack": material_pack_option},
    )

    snapshot = anyio.run(collect)

    assert snapshot.available is True
    assert [item.resolve_type for item in snapshot.items] == [
        "material_pack",
        "weekly_report",
        "monthly_report",
        "sales_mention",
    ]
    weekly = next(
        item.result for item in snapshot.items if item.resolve_type == "weekly_report"
    )
    material = next(
        item.result for item in snapshot.items if item.resolve_type == "material_pack"
    )
    assert material is not None
    assert material.material_pack_option == material_pack_option
    assert weekly is not None
    assert weekly.material_pack_option is None


def test_live_assistant_adapter_report_scope_contract() -> None:
    if os.getenv("MARKET_AGENT_LIVE_ADAPTER_EXPECT_REPORT_SCOPE", "").strip() != "1":
        pytest.skip("report-scope live eval requires fixture-backed adapter")
    base_url = live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None
    dist_name = os.getenv("MARKET_AGENT_LIVE_ADAPTER_DIST_NAME", "ReportScopeTest")
    skip_if_adapter_is_not_running(base_url, api_key)

    result = live_adapter_client(base_url, api_key).report_scope(
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


def test_live_assistant_adapter_rejects_short_resolve_endpoint() -> None:
    base_url = live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None
    skip_if_adapter_is_not_running(base_url, api_key)

    request = _post_request(
        f"{base_url.rstrip('/')}/resolve",
        api_key,
        {"resolve_type": "weekly_report", "dist_name": "__contract_check__"},
    )
    with pytest.raises(HTTPError) as caught:
        _ = urlopen(request, timeout=3)

    assert caught.value.code == 404


def test_live_assistant_adapter_rejects_raw_list_batch_payload() -> None:
    base_url = live_adapter_base_url()
    api_key = os.getenv("MARKET_AGENT_LIVE_ADAPTER_API_KEY") or None
    skip_if_adapter_is_not_running(base_url, api_key)
    body = json.dumps(
        [{"resolve_type": "weekly_report", "dist_name": "__contract_check__"}],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
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
        _ = urlopen(request, timeout=3)

    assert caught.value.code == 400
    payload = ErrorPayload.model_validate_json(caught.value.read())
    assert payload == ErrorPayload(
        error="bad_request",
        detail="batch resolve request must be a JSON object",
    )


def _post_request(
    url: str,
    api_key: str | None,
    payload: dict[str, str],
) -> Request:
    body = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return Request(url, data=body, headers=headers, method="POST")
