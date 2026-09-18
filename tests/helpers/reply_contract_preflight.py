from __future__ import annotations

from typing import override

from pydantic import JsonValue

from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightItem,
    AdapterPreflightService,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.schemas.adapter import AdapterResolveResult
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from tests.helpers.reply_contract_json import JsonInput, json_mapping


def resolved_item(
    resolve_type: AdapterResolveType,
    **overrides: JsonInput,
) -> AdapterPreflightItem:
    payload: dict[str, JsonValue] = {
        "contract_version": "adapter-resolve",
        "resolve_type": resolve_type,
        "status": "resolved",
        "display_name": "测试渠道",
        "reason_code": "ok",
        "candidates": [],
        "channel_type": "bank",
        "available_artifacts": [
            {"type": "material_pack", "options": ["指增"]},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "resolved_at": 1,
        "resolve_ref": f"{resolve_type}:ref",
    }
    payload.update(json_mapping(overrides))
    return AdapterPreflightItem(
        resolve_type=resolve_type,
        result=AdapterResolveResult.model_validate(payload),
    )


class ResolvedWeeklyPreflight(AdapterPreflightService):
    @override
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot(
            items=[
                resolved_item(
                    "weekly_report",
                    resolve_ref="weekly:ref",
                    period="20260529",
                    report_date="2026-05-29",
                )
            ]
        )


class ResolvedMonthlyPreflight(AdapterPreflightService):
    @override
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot(
            items=[
                resolved_item(
                    "monthly_report",
                    resolve_ref="monthly:ref",
                    period="202605",
                    report_date="2026-05-31",
                )
            ]
        )


class ResolvedWeeklyMonthlyPreflight(AdapterPreflightService):
    @override
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot(
            items=[
                resolved_item(
                    "weekly_report",
                    resolve_ref="weekly:ref",
                    period="20260529",
                    report_date="2026-05-29",
                ),
                resolved_item(
                    "monthly_report",
                    resolve_ref="monthly:ref",
                    period="202605",
                    report_date="2026-05-31",
                ),
            ]
        )


class CapturingResolvedWeeklyPreflight(AdapterPreflightService):
    def __init__(self) -> None:
        super().__init__()
        self.resolve_material_pack_options: dict[AdapterResolveType, str] = {}

    @override
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, resolve_types
        self.resolve_material_pack_options = resolve_material_pack_options or {}
        return AdapterPreflightSnapshot(
            items=[
                resolved_item(
                    "weekly_report",
                    resolve_ref="weekly:ref",
                    period="20260529",
                    report_date="2026-05-29",
                )
            ]
        )


class CapturingResolvedMaterialPreflight(AdapterPreflightService):
    def __init__(self) -> None:
        super().__init__()
        self.resolve_material_pack_options: dict[AdapterResolveType, str] = {}

    @override
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, resolve_types
        self.resolve_material_pack_options = resolve_material_pack_options or {}
        material_pack_option = self.resolve_material_pack_options.get("material_pack")
        return AdapterPreflightSnapshot(
            items=[
                resolved_item(
                    "material_pack",
                    resolve_ref="material:ref",
                    material_pack_option=material_pack_option,
                )
            ]
        )


class MissingWeeklyWithSalesPreflight(AdapterPreflightService):
    @override
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, resolve_types, resolve_material_pack_options
        missing: dict[str, JsonValue] = {
            "contract_version": "adapter-resolve",
            "resolve_type": "weekly_report",
            "status": "missing",
            "display_name": "测试渠道",
            "reason_code": "weekly_report_unavailable",
            "candidates": [],
            "channel_type": "bank",
            "available_artifacts": [],
            "resolved_at": 1,
        }
        return AdapterPreflightSnapshot(
            items=[
                AdapterPreflightItem(
                    resolve_type="weekly_report",
                    result=AdapterResolveResult.model_validate(missing),
                ),
                resolved_item("sales_mention", resolve_ref="sales:ref"),
            ]
        )


class EmptyPreflightService(AdapterPreflightService):
    @override
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot.empty()


class CapturingEmptyPreflightService(AdapterPreflightService):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, list[str] | dict[AdapterResolveType, str]]] = []

    @override
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        self.calls.append(
            {
                "resolve_types": list(resolve_types or []),
                "resolve_material_pack_options": dict(
                    resolve_material_pack_options or {}
                ),
            }
        )
        del request
        return AdapterPreflightSnapshot.empty()
