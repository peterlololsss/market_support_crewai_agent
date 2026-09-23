from __future__ import annotations

from functools import partial

import anyio
import pytest

from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightService,
)
from market_support_crewai_agent.schemas.adapter import AdapterResolveRequest
from tests.contract.adapter_preflight_support import (
    FakeAdapterClient,
    make_preflight_request,
)


def test_preflight_records_adapter_errors_without_raising() -> None:
    service = AdapterPreflightService(FakeAdapterClient(failures={"weekly_report"}))

    snapshot = anyio.run(service.collect, make_preflight_request())

    weekly = next(
        item for item in snapshot.items if item.resolve_type == "weekly_report"
    )
    assert snapshot.available is False
    assert weekly.status == "adapter_unavailable"
    assert "weekly_report unavailable" in weekly.error
    assert all(item.status == "adapter_unavailable" for item in snapshot.items)


def test_preflight_records_adapter_readiness_error_without_batch_request() -> None:
    fake_client = FakeAdapterClient(readiness_error="adapter capabilities mismatch")

    snapshot = anyio.run(
        AdapterPreflightService(fake_client).collect,
        make_preflight_request(),
    )

    assert snapshot.available is False
    assert fake_client.ready_calls == 1
    assert fake_client.requests == []
    assert all(item.status == "adapter_unavailable" for item in snapshot.items)
    assert all("adapter capabilities mismatch" in item.error for item in snapshot.items)


@pytest.mark.parametrize("advertised_tenant", [None, "tenant:test"])
def test_preflight_accepts_adapter_without_tenant_or_with_matching_tenant(
    advertised_tenant: str | None,
) -> None:
    fake_client = FakeAdapterClient(deployment_tenant_ref=advertised_tenant)

    snapshot = anyio.run(
        AdapterPreflightService(fake_client).collect,
        make_preflight_request(),
    )

    assert snapshot.available is True
    assert fake_client.requests != []


def test_preflight_rejects_adapter_advertising_another_tenant() -> None:
    fake_client = FakeAdapterClient(deployment_tenant_ref="tenant:other")

    snapshot = anyio.run(
        AdapterPreflightService(fake_client).collect,
        make_preflight_request(),
    )

    assert snapshot.available is False
    assert fake_client.ready_calls == 1
    assert fake_client.requests == []
    assert all(item.status == "adapter_unavailable" for item in snapshot.items)
    assert all("deployment tenant mismatch" in item.error for item in snapshot.items)
    assert all("tenant:other" not in item.error for item in snapshot.items)


def test_preflight_records_missing_batch_result_without_dropping_item() -> None:
    service = AdapterPreflightService(FakeAdapterClient(omissions={"weekly_report"}))

    snapshot = anyio.run(service.collect, make_preflight_request())

    assert [item.resolve_type for item in snapshot.items] == [
        "material_pack",
        "weekly_report",
        "monthly_report",
        "sales_mention",
    ]
    weekly = next(
        item for item in snapshot.items if item.resolve_type == "weekly_report"
    )
    assert snapshot.available is False
    assert weekly.status == "adapter_unavailable"
    assert weekly.error == "adapter batch result missing"


def test_preflight_rejects_resolve_type_not_in_registry() -> None:
    invalid_request = AdapterResolveRequest.model_construct(
        resolve_type="unknown",
        dist_name="test channel",
    )
    collect = partial(
        AdapterPreflightService(FakeAdapterClient()).collect,
        make_preflight_request(),
        resolve_types=[invalid_request.resolve_type],
    )

    with pytest.raises(ValueError, match="Unknown adapter resolve type"):
        _ = anyio.run(collect)
