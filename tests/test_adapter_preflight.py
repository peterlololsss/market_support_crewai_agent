from __future__ import annotations

import asyncio

from market_support_crewai_agent.runtime.adapter_client import AdapterClientError
from market_support_crewai_agent.runtime.adapter_preflight import (
    AdapterPreflightService,
    NoopAdapterPreflightService,
)
from market_support_crewai_agent.schemas import AdapterResolveResult


def make_payload(**overrides):
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": "hello",
        "is_group": True,
        "group_name": "test group",
        "dist_channel_name": "test channel",
        "sender_nickname": "test user",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": [],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return payload


class FakeAdapterClient:
    def __init__(
            self,
            failures: set[str] | None = None,
            omissions: set[str] | None = None,
            readiness_error: str = "",
    ):
        self.failures = failures or set()
        self.omissions = omissions or set()
        self.readiness_error = readiness_error
        self.ready_calls = 0
        self.requests = []

    async def assert_ready_async(self):
        self.ready_calls += 1
        if self.readiness_error:
            raise AdapterClientError(self.readiness_error)
        return None

    async def resolve_many_async(self, requests):
        self.requests.extend(requests)
        failures = [request.resolve_type for request in requests if request.resolve_type in self.failures]
        if failures:
            raise AdapterClientError(f"{failures[0]} unavailable")
        return [
            self._resolve(request)
            for request in requests
            if request.resolve_type not in self.omissions
        ]

    def _resolve(self, request):
        return AdapterResolveResult.model_validate(
            {
                "contract_version": "adapter-resolve.v1",
                "resolve_type": request.resolve_type,
                "status": "resolved",
                "display_name": request.dist_name,
                "reason_code": "ok",
                "candidates": [],
                "channel_type": "bank",
                "available_materials": ["material", "weekly", "monthly"],
                "available_strategies": ["指增"],
                "resolved_at": 1,
                "strategy": request.strategy,
                "period": "20260529" if request.resolve_type == "weekly_report" else None,
                "scope_status": "unknown",
            }
        )


def test_preflight_collects_all_adapter_resolve_types_with_single_strategy():
    from market_support_crewai_agent.schemas import ReplyRequest

    request = ReplyRequest.model_validate(
        make_payload(
            available_strategies=["指增"],
            dist_channel_name="测试渠道",
        )
    )
    fake_client = FakeAdapterClient()
    service = AdapterPreflightService(adapter_client=fake_client)

    snapshot = asyncio.run(service.collect(request))

    assert snapshot.available is True
    assert [item.resolve_type for item in snapshot.items] == [
        "material_pack",
        "weekly_report",
        "monthly_report",
        "sales_mention",
    ]
    assert fake_client.requests[0].strategy == "指增"
    assert fake_client.requests[1].strategy == "指增"
    assert fake_client.requests[2].strategy == "指增"
    assert all(resolve_request.dist_name == "测试渠道" for resolve_request in fake_client.requests)


def test_preflight_omits_strategy_when_multiple_candidates_exist():
    from market_support_crewai_agent.schemas import ReplyRequest

    request = ReplyRequest.model_validate(
        make_payload(available_strategies=["指增", "量化"])
    )
    fake_client = FakeAdapterClient()
    service = AdapterPreflightService(adapter_client=fake_client)

    asyncio.run(service.collect(request))

    assert fake_client.requests[0].resolve_type == "material_pack"
    assert all(resolve_request.strategy is None for resolve_request in fake_client.requests)


def test_preflight_uses_canonical_strategy_from_user_message():
    from market_support_crewai_agent.schemas import ReplyRequest

    request = ReplyRequest.model_validate(
        make_payload(
            message="1000所有号的周报我想看看",
            available_strategies=["中证500", "中证1000"],
        )
    )
    fake_client = FakeAdapterClient()
    service = AdapterPreflightService(adapter_client=fake_client)

    asyncio.run(service.collect(request))

    assert fake_client.requests[0].resolve_type == "material_pack"
    assert fake_client.requests[0].strategy == "中证1000"
    assert fake_client.requests[1].strategy == "中证1000"
    assert fake_client.requests[2].strategy == "中证1000"
    assert fake_client.requests[3].strategy is None


def test_preflight_request_projection_keeps_conversation_identity_out_of_adapter_contract():
    from market_support_crewai_agent.schemas import ReplyRequest

    request = ReplyRequest.model_validate(
        make_payload(
            context_id="trace-1",
            conversation_key="wecom:group:sender",
            available_strategies=["指增"],
        )
    )
    fake_client = FakeAdapterClient()
    service = AdapterPreflightService(adapter_client=fake_client)

    asyncio.run(service.collect(request))

    payload = fake_client.requests[0].model_dump(mode="json", exclude_none=True)
    assert payload == {
        "resolve_type": "material_pack",
        "dist_name": "test channel",
        "strategy": "指增",
    }


def test_preflight_can_limit_adapter_resolve_types():
    from market_support_crewai_agent.schemas import ReplyRequest

    request = ReplyRequest.model_validate(
        make_payload(
            message="1000所有号的周报我想看看",
            available_strategies=["中证500", "中证1000"],
        )
    )
    fake_client = FakeAdapterClient()
    service = AdapterPreflightService(adapter_client=fake_client)

    snapshot = asyncio.run(
        service.collect(
            request,
            resolve_types=["weekly_report", "sales_mention"],
        )
    )

    assert [item.resolve_type for item in snapshot.items] == [
        "weekly_report",
        "sales_mention",
    ]
    assert [request.resolve_type for request in fake_client.requests] == [
        "weekly_report",
        "sales_mention",
    ]
    assert fake_client.requests[0].strategy == "中证1000"
    assert fake_client.requests[1].strategy is None


def test_preflight_uses_plan_strategy_override_for_report_resolve():
    from market_support_crewai_agent.schemas import ReplyRequest

    request = ReplyRequest.model_validate(
        make_payload(
            message="这个周报发一下",
            available_strategies=["中证500", "中证1000"],
        )
    )
    fake_client = FakeAdapterClient()
    service = AdapterPreflightService(adapter_client=fake_client)

    asyncio.run(
        service.collect(
            request,
            resolve_types=["weekly_report", "sales_mention"],
            resolve_strategies={"weekly_report": "中证1000"},
        )
    )

    assert fake_client.requests[0].resolve_type == "weekly_report"
    assert fake_client.requests[0].strategy == "中证1000"
    assert fake_client.requests[1].resolve_type == "sales_mention"
    assert fake_client.requests[1].strategy is None


def test_preflight_records_adapter_errors_without_raising():
    from market_support_crewai_agent.schemas import ReplyRequest

    request = ReplyRequest.model_validate(make_payload())
    service = AdapterPreflightService(
        adapter_client=FakeAdapterClient(failures={"weekly_report"}),
    )

    snapshot = asyncio.run(service.collect(request))

    weekly = next(item for item in snapshot.items if item.resolve_type == "weekly_report")
    assert snapshot.available is False
    assert weekly.status == "adapter_unavailable"
    assert "weekly_report unavailable" in weekly.error
    assert all(item.status == "adapter_unavailable" for item in snapshot.items)


def test_preflight_records_adapter_readiness_error_without_batch_request():
    from market_support_crewai_agent.schemas import ReplyRequest

    request = ReplyRequest.model_validate(make_payload())
    fake_client = FakeAdapterClient(readiness_error="adapter capabilities mismatch")
    service = AdapterPreflightService(adapter_client=fake_client)

    snapshot = asyncio.run(service.collect(request))

    assert snapshot.available is False
    assert fake_client.ready_calls == 1
    assert fake_client.requests == []
    assert all(item.status == "adapter_unavailable" for item in snapshot.items)
    assert all("adapter capabilities mismatch" in item.error for item in snapshot.items)


def test_preflight_records_missing_batch_result_without_dropping_item():
    from market_support_crewai_agent.schemas import ReplyRequest

    request = ReplyRequest.model_validate(make_payload())
    service = AdapterPreflightService(
        adapter_client=FakeAdapterClient(omissions={"weekly_report"}),
    )

    snapshot = asyncio.run(service.collect(request))

    assert [item.resolve_type for item in snapshot.items] == [
        "material_pack",
        "weekly_report",
        "monthly_report",
        "sales_mention",
    ]
    weekly = next(item for item in snapshot.items if item.resolve_type == "weekly_report")
    assert snapshot.available is False
    assert weekly.status == "adapter_unavailable"
    assert weekly.error == "adapter batch result missing"


def test_noop_preflight_returns_empty_snapshot():
    from market_support_crewai_agent.schemas import ReplyRequest

    request = ReplyRequest.model_validate(make_payload())
    snapshot = asyncio.run(NoopAdapterPreflightService().collect(request))

    assert snapshot.items == []
    assert snapshot.available is True
