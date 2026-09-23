from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.integrations.adapter.transport import (
    AdapterClientError,
)
from market_support_crewai_agent.runtime.planning import (
    ExecutionPlanV2,
    PlanSpec,
    finalize_execution_plan_v2,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.schemas.adapter import (
    AdapterReportScopeRequest,
    AdapterReportScopeResult,
    AdapterResolveResult,
)
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.unit.planning._execution_plan_v2_fixtures import policy_for


@dataclass(frozen=True, slots=True)
class ReportInputs:
    request: KernelReplyRequestV1
    policy: PolicyManifestV2
    scope: BusinessScopeAuthorityV1
    plan: ExecutionPlanV2
    preflight: AdapterPreflightSnapshot


def report_inputs(
    *,
    capability_id: str = "weekly_report.product_list",
    read_capability: str = "query_weekly_report_product_list",
) -> ReportInputs:
    request = make_v2_envelope(
        "这份报告有哪些产品？",
        business_scope={
            "kind": "distribution",
            "dist_channel_name": "TestDist",
            "channel_type": "bank",
            "available_artifacts": [
                {"type": "weekly_report"},
                {"type": "monthly_report"},
            ],
        },
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": [read_capability],
            "outbound_actions": [],
            "mention_types": [],
        },
    ).request
    policy, scope = policy_for(request)
    plan = finalize_execution_plan_v2(
        _report_plan_spec(capability_id, scope),
        policy,
        scope,
        origin="planner",
        confidence=0.9,
    )
    resolve_type = _resolve_type_for_capability(capability_id)
    return ReportInputs(
        request=request,
        policy=policy,
        scope=scope,
        plan=plan,
        preflight=_preflight(resolve_type),
    )


class FakeReportScopeClient:
    def __init__(
        self,
        *,
        product_pages: dict[int, tuple[str, ...]] | None = None,
        product_total_count: int | None = None,
        response_overrides: dict[str, str] | None = None,
        readiness_error: str = "",
    ) -> None:
        self.calls: list[AdapterReportScopeRequest] = []
        self.ready_tenants: list[str] = []
        self.readiness_error: str = readiness_error
        self.product_pages: dict[int, tuple[str, ...]] = product_pages or {
            1: ("Product1", "Product2")
        }
        self.product_total_count: int = (
            product_total_count
            if product_total_count is not None
            else sum(len(page) for page in self.product_pages.values())
        )
        self.response_overrides: dict[str, str] = response_overrides or {}

    async def assert_ready_for_tenant_async(self, tenant_ref: str) -> None:
        self.ready_tenants.append(tenant_ref)
        if self.readiness_error:
            raise AdapterClientError(self.readiness_error)

    async def report_scope_async(
        self,
        request: AdapterReportScopeRequest,
    ) -> AdapterReportScopeResult:
        self.calls.append(request)
        common = {
            "contract_version": "adapter-report-scope",
            "material_type": request.material_type,
            "dist_name": request.dist_name,
            "period": request.period or "20260612",
            "status": "resolved",
            "reason_code": "ok",
            "report_date": "2026-06-12",
            "period_start": "2026-06-08",
            "period_end": "2026-06-12",
            "period_label": "2026-06-08 to 2026-06-12",
            "scope_complete": True,
            **self.response_overrides,
        }
        match request.command:
            case "summary":
                return AdapterReportScopeResult.model_validate(
                    {
                        **common,
                        "expected_product_count": 2,
                        "generated_product_count": 2,
                        "missing_product_count": 0,
                        "report_sections": [
                            {
                                "name": "IndexPlus",
                                "expected_product_count": 2,
                                "generated_product_count": 2,
                                "missing_product_count": 0,
                            }
                        ],
                    }
                )
            case "match" | "list_products":
                return AdapterReportScopeResult.model_validate(
                    {
                        **common,
                        "products": [
                            {
                                "product_name": product,
                                "portfolio_type": "IndexPlus",
                                "report_section": "IndexPlus",
                                "source_pdf_status": "found",
                                "final_report_status": "generated",
                            }
                            for product in self.product_pages.get(request.page or 1, ())
                        ],
                        "product_page": request.page or 1,
                        "product_page_size": 50,
                        "product_total_count": self.product_total_count,
                    }
                )
            case _:
                assert_never(request.command)


def _report_plan_spec(
    capability_id: str,
    scope: BusinessScopeAuthorityV1,
) -> PlanSpec:
    return PlanSpec.model_validate(
        {
            "plan_id": "report-products",
            "user_intent_summary": "answer report product list",
            "plan_units": [
                {
                    "unit_id": "report-products",
                    "selected_capability_id": capability_id,
                    "domain_scope": {
                        "kind": "distribution",
                        "business_scope_ref": scope.business_scope_ref,
                        "channel_kind": "bank",
                    },
                    "answerability_policy": "answer",
                    "output_schema_ref": f"{capability_id}:output_schema",
                }
            ],
        }
    )


def _resolve_type_for_capability(capability_id: str) -> AdapterResolveType:
    return "weekly_report" if capability_id.startswith("weekly") else "monthly_report"


def _preflight(resolve_type: AdapterResolveType) -> AdapterPreflightSnapshot:
    period = "20260612" if resolve_type == "weekly_report" else "202606"
    report_date = "2026-06-12" if resolve_type == "weekly_report" else "2026-06-30"
    return AdapterPreflightSnapshot(
        items=[
            AdapterPreflightItem(
                resolve_type=resolve_type,
                result=AdapterResolveResult.model_validate(
                    {
                        "contract_version": "adapter-resolve",
                        "resolve_type": resolve_type,
                        "status": "resolved",
                        "display_name": "TestDist",
                        "reason_code": "ok",
                        "available_artifacts": [{"type": resolve_type}],
                        "resolved_at": 1_750_000_000,
                        "resolve_ref": f"adapter:{resolve_type}:current",
                        "period": period,
                        "report_date": report_date,
                    }
                ),
            )
        ]
    )
