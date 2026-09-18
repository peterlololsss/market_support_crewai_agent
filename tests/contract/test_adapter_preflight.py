from __future__ import annotations

from functools import partial

import anyio

from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightService,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.schemas.adapter import AvailableArtifact
from tests.contract.adapter_preflight_support import (
    FakeAdapterClient,
    make_preflight_request,
)


def test_group_capabilities_without_scene_fields_preserve_existing_preflight_results() -> (
    None
):
    request = make_preflight_request(
        dist_channel_name="测试渠道",
        available_artifacts=[
            AvailableArtifact(type="material_pack", options=["指增"]),
            AvailableArtifact(type="weekly_report"),
            AvailableArtifact(type="monthly_report"),
        ],
    )
    fake_client = FakeAdapterClient()

    snapshot = anyio.run(AdapterPreflightService(fake_client).collect, request)

    assert snapshot.available is True
    assert fake_client.ready_calls == 1
    assert [item.resolve_type for item in snapshot.items] == [
        "material_pack",
        "weekly_report",
        "monthly_report",
        "sales_mention",
    ]
    assert all(item.material_pack_option is None for item in fake_client.requests)
    assert all(item.dist_name == "测试渠道" for item in fake_client.requests)


def test_preflight_omits_material_pack_option_when_multiple_candidates_exist() -> None:
    request = make_preflight_request(
        available_artifacts=[
            AvailableArtifact(type="material_pack", options=["指增", "量化"]),
            AvailableArtifact(type="weekly_report"),
            AvailableArtifact(type="monthly_report"),
        ]
    )
    fake_client = FakeAdapterClient()

    _ = anyio.run(AdapterPreflightService(fake_client).collect, request)

    assert fake_client.requests[0].resolve_type == "material_pack"
    assert all(item.material_pack_option is None for item in fake_client.requests)


def test_preflight_ignores_query_without_material_pack_option_selector() -> None:
    request = make_preflight_request(
        message="1000所有号的周报我想看看",
        available_artifacts=[
            AvailableArtifact(
                type="material_pack",
                options=["中证500", "中证1000"],
            ),
            AvailableArtifact(type="weekly_report"),
            AvailableArtifact(type="monthly_report"),
        ],
    )
    fake_client = FakeAdapterClient()

    _ = anyio.run(AdapterPreflightService(fake_client).collect, request)

    assert fake_client.requests[0].resolve_type == "material_pack"
    assert all(item.material_pack_option is None for item in fake_client.requests)


def test_preflight_request_projection_keeps_conversation_identity_out_of_adapter_contract() -> (
    None
):
    fake_client = FakeAdapterClient()

    _ = anyio.run(
        AdapterPreflightService(fake_client).collect,
        make_preflight_request(),
    )

    assert fake_client.requests[0].model_dump(mode="json", exclude_none=True) == {
        "resolve_type": "material_pack",
        "dist_name": "test channel",
    }


def test_preflight_can_limit_adapter_resolve_types() -> None:
    fake_client = FakeAdapterClient()
    collect = partial(
        AdapterPreflightService(fake_client).collect,
        make_preflight_request(message="1000所有号的周报我想看看"),
        resolve_types=["weekly_report", "sales_mention"],
    )

    snapshot = anyio.run(collect)

    assert [item.resolve_type for item in snapshot.items] == [
        "weekly_report",
        "sales_mention",
    ]
    assert [item.resolve_type for item in fake_client.requests] == [
        "weekly_report",
        "sales_mention",
    ]
    assert all(item.material_pack_option is None for item in fake_client.requests)


def test_preflight_returns_empty_snapshot_when_plan_needs_no_adapter_resolves() -> None:
    fake_client = FakeAdapterClient()
    collect = partial(
        AdapterPreflightService(fake_client).collect,
        make_preflight_request(message="hi"),
        resolve_types=[],
    )

    snapshot = anyio.run(collect)

    assert snapshot == AdapterPreflightSnapshot.empty()
    assert snapshot.available is True
    assert fake_client.ready_calls == 0
    assert fake_client.requests == []


def test_preflight_uses_material_pack_option_only_for_material_pack_resolve() -> None:
    fake_client = FakeAdapterClient()
    collect = partial(
        AdapterPreflightService(fake_client).collect,
        make_preflight_request(message="这个周报发一下"),
        resolve_types=["material_pack", "weekly_report", "sales_mention"],
        resolve_material_pack_options={"material_pack": "中证1000"},
    )

    _ = anyio.run(collect)

    assert [item.resolve_type for item in fake_client.requests] == [
        "material_pack",
        "weekly_report",
        "sales_mention",
    ]
    assert fake_client.requests[0].material_pack_option == "中证1000"
    assert all(item.material_pack_option is None for item in fake_client.requests[1:])
