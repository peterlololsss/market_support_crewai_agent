from __future__ import annotations

from typing import Literal, assert_never

from market_support_crewai_agent.runtime.context.models import PreflightFactViewV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType


def project_preflight_fact_views_v1(
    snapshot: AdapterPreflightSnapshot,
) -> tuple[PreflightFactViewV1, ...]:
    return tuple(_project_preflight_item(item) for item in snapshot.items)


def _project_preflight_item(item: AdapterPreflightItem) -> PreflightFactViewV1:
    result = item.result
    return PreflightFactViewV1(
        resolve_type=item.resolve_type,
        status="adapter_unavailable" if result is None else result.status,
        display_name=None if result is None else result.display_name,
        reason_code="adapter_unavailable" if result is None else result.reason_code,
        resolve_ref_available=result is not None and result.resolve_ref is not None,
        artifact_type=_artifact_type(item.resolve_type),
        material_pack_option=None if result is None else result.material_pack_option,
        period=None if result is None else result.period,
        report_date=None if result is None else result.report_date,
    )


def _artifact_type(
    resolve_type: AdapterResolveType,
) -> Literal["material_pack", "weekly_report", "monthly_report"] | None:
    match resolve_type:
        case "material_pack" | "weekly_report" | "monthly_report":
            return resolve_type
        case "sales_mention":
            return None
        case unreachable:
            assert_never(unreachable)
