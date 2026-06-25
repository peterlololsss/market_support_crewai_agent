from __future__ import annotations

from math import ceil
from typing import Literal, Sequence

from market_support_crewai_agent.runtime.domain.capabilities import capability_by_name
from market_support_crewai_agent.runtime.domain.ontology import artifact_scope_for_evidence
from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.evidence.adapter_client import (
    AdapterClientError,
    AdapterResolveClient,
)
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.schemas import (
    AdapterReportScopeRequest,
    AdapterReportScopeResult,
    AdapterResolveType,
    ReplyRequest,
    ReportScopeProduct,
)
from market_support_crewai_agent.settings import Settings, get_settings

_REPORT_SCOPE_SUMMARY_QUERY = "report_scope_summary"
_REPORT_SCOPE_PRODUCTS_QUERY = "report_scope_products"
_PAGE_SIZE = 50
_MAX_PRODUCTS_IN_PROMPT = 200


class ReportScopeEvidenceService:
    def __init__(
        self,
        settings: Settings | None = None,
        adapter_client: AdapterResolveClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.adapter_client = adapter_client or AdapterResolveClient(self.settings)

    async def collect(
        self,
        request: ReplyRequest,
        plan: ExecutionPlan,
        policy: PolicyManifest,
        preflight: AdapterPreflightSnapshot,
    ) -> list[EvidenceFact]:
        if plan.compliance.is_compliant is not True:
            return []
        if plan.response_mode != "knowledge_answer" and not plan.answer_capabilities:
            return []

        answer_resolve_types = {
            capability.resolve_type
            for capability_name in plan.answer_capabilities
            if (capability := capability_by_name(capability_name)) is not None
            and capability.resolve_type in {"weekly_report", "monthly_report"}
        }
        report_resolves = [
            resolve_spec
            for resolve_spec in plan.adapter_resolves
            if resolve_spec.resolve_type in {"weekly_report", "monthly_report"}
            and resolve_spec.resolve_type in policy.allowed_adapter_resolves
            and (
                plan.response_mode == "knowledge_answer"
                or resolve_spec.resolve_type in answer_resolve_types
            )
        ]
        if not report_resolves:
            return []

        facts: list[EvidenceFact] = []
        for resolve_spec in report_resolves:
            material_type = _material_type(resolve_spec.resolve_type)
            period = _period_from_preflight(preflight, resolve_spec.resolve_type)
            query = _scope_query(plan)
            summary_only = query == _REPORT_SCOPE_SUMMARY_QUERY
            products_requested = query == _REPORT_SCOPE_PRODUCTS_QUERY
            if not query:
                continue
            try:
                summary = await self.adapter_client.report_scope_async(
                    AdapterReportScopeRequest(
                        material_type=material_type,
                        dist_name=request.dist_channel_name,
                        command="summary",
                        period=period,
                    )
                )
            except AdapterClientError as exc:
                facts.append(
                    _unavailable_fact(
                        resolve_spec.resolve_type,
                        reason_code="adapter_report_scope_error",
                        error_type=type(exc).__name__,
                    )
                )
                continue

            facts.append(_summary_fact(resolve_spec.resolve_type, summary))
            if products_requested:
                facts.extend(
                    await self._product_facts(
                        request=request,
                        resolve_type=resolve_spec.resolve_type,
                        material_type=material_type,
                        period=period,
                    )
                )
                continue
            if query and not summary_only:
                facts.extend(
                    await self._match_facts(
                        request=request,
                        resolve_type=resolve_spec.resolve_type,
                        material_type=material_type,
                        period=period,
                        query=query,
                    )
                )
        return facts

    async def _match_facts(
        self,
        *,
        request: ReplyRequest,
        resolve_type: AdapterResolveType,
        material_type: str,
        period: str | None,
        query: str,
    ) -> list[EvidenceFact]:
        try:
            match_result = await self.adapter_client.report_scope_async(
                AdapterReportScopeRequest(
                    material_type=material_type,  # type: ignore[arg-type]
                    dist_name=request.dist_channel_name,
                    command="match",
                    period=period,
                    query=query,
                    page=1,
                    page_size=10,
                )
            )
        except AdapterClientError as exc:
            return [
                _unavailable_fact(
                    resolve_type,
                    reason_code="adapter_report_scope_match_error",
                    error_type=type(exc).__name__,
                )
            ]

        return [_match_fact(resolve_type, match_result)]

    async def _product_facts(
        self,
        *,
        request: ReplyRequest,
        resolve_type: AdapterResolveType,
        material_type: str,
        period: str | None,
    ) -> list[EvidenceFact]:
        try:
            result = await self._product_page(
                request=request,
                material_type=material_type,
                period=period,
                page=1,
            )
            products = list(result.products)
            total_count = result.product_total_count
            if (
                result.status == "resolved"
                and total_count is not None
                and len(products) < min(total_count, _MAX_PRODUCTS_IN_PROMPT)
            ):
                target_count = min(total_count, _MAX_PRODUCTS_IN_PROMPT)
                for page in range(2, ceil(target_count / _PAGE_SIZE) + 1):
                    page_result = await self._product_page(
                        request=request,
                        material_type=material_type,
                        period=period,
                        page=page,
                    )
                    products.extend(page_result.products)
        except AdapterClientError as exc:
            return [
                _unavailable_fact(
                    resolve_type,
                    reason_code="adapter_report_scope_products_error",
                    error_type=type(exc).__name__,
                )
            ]
        return [_products_fact(resolve_type, result, products=products)]

    async def _product_page(
        self,
        *,
        request: ReplyRequest,
        material_type: str,
        period: str | None,
        page: int,
    ) -> AdapterReportScopeResult:
        return await self.adapter_client.report_scope_async(
            AdapterReportScopeRequest(
                material_type=material_type,  # type: ignore[arg-type]
                dist_name=request.dist_channel_name,
                command="list_products",
                period=period,
                page=page,
                page_size=_PAGE_SIZE,
            )
        )


class NoopReportScopeEvidenceService:
    async def collect(
        self,
        request: ReplyRequest,
        plan: ExecutionPlan,
        policy: PolicyManifest,
        preflight: AdapterPreflightSnapshot,
    ) -> list[EvidenceFact]:
        del request, plan, policy, preflight
        return []


def _summary_fact(
    resolve_type: AdapterResolveType,
    result: AdapterReportScopeResult,
) -> EvidenceFact:
    metadata = _result_metadata(result)
    return EvidenceFact(
        fact_type="report_scope_summary",
        value=result.status == "resolved",
        source_type="adapter_report_scope",
        source_id=resolve_type,
        resolve_type=resolve_type,
        metadata=metadata,
        artifact_type=resolve_type,
        scope=_scope(resolve_type, metadata),
    )


def _match_fact(
    resolve_type: AdapterResolveType,
    result: AdapterReportScopeResult,
) -> EvidenceFact:
    metadata = _result_metadata(result)
    if result.match is not None:
        match_payload = result.match.model_dump(mode="json", exclude_none=True)
        metadata["match"] = match_payload
    return EvidenceFact(
        fact_type="report_scope_match",
        value=(
            result.match.status
            if result.match is not None
            else "not_found"
        ),
        source_type="adapter_report_scope",
        source_id=resolve_type,
        resolve_type=resolve_type,
        metadata=metadata,
        artifact_type=resolve_type,
        scope=_scope(resolve_type, metadata),
    )


def _products_fact(
    resolve_type: AdapterResolveType,
    result: AdapterReportScopeResult,
    *,
    products: Sequence[ReportScopeProduct] | None = None,
) -> EvidenceFact:
    metadata = _result_metadata(result)
    product_items = list(result.products if products is None else products)
    metadata.update(
        {
            "products": [
                product.model_dump(mode="json", exclude_none=True)
                for product in product_items
            ],
            "product_page": result.product_page,
            "product_page_size": result.product_page_size,
            "product_total_count": result.product_total_count,
            "product_returned_count": len(product_items),
        }
    )
    returned_count = len(product_items)
    total_count = result.product_total_count
    metadata["full_product_list_in_prompt"] = (
        total_count is not None and total_count <= returned_count
    )
    return EvidenceFact(
        fact_type="report_scope_products",
        value=result.status == "resolved",
        source_type="adapter_report_scope",
        source_id=resolve_type,
        resolve_type=resolve_type,
        metadata=metadata,
        artifact_type=resolve_type,
        scope=_scope(resolve_type, metadata),
    )


def _unavailable_fact(
    resolve_type: AdapterResolveType,
    *,
    reason_code: str,
    error_type: str = "",
) -> EvidenceFact:
    metadata = {"reason_code": reason_code}
    if error_type:
        metadata["error_type"] = error_type
    return EvidenceFact(
        fact_type="report_scope_unavailable",
        value=False,
        source_type="adapter_report_scope",
        source_id=resolve_type,
        resolve_type=resolve_type,
        metadata=metadata,
        artifact_type=resolve_type,
        scope=_scope(resolve_type, metadata),
    )


def _result_metadata(result: AdapterReportScopeResult) -> dict:
    return {
        "status": result.status,
        "reason_code": result.reason_code,
        "material_type": result.material_type,
        "period": result.period,
        "report_date": result.report_date,
        "period_start": result.period_start,
        "period_end": result.period_end,
        "period_label": result.period_label,
        "scope_complete": result.scope_complete,
        "expected_product_count": result.expected_product_count,
        "generated_product_count": result.generated_product_count,
        "missing_product_count": result.missing_product_count,
        "report_sections": [
            section.model_dump(mode="json", exclude_none=True)
            for section in result.report_sections
        ],
        "products_are_paginated": True,
        "full_product_list_in_prompt": False,
    }


def _scope(resolve_type: AdapterResolveType, metadata: dict) -> object:
    return artifact_scope_for_evidence(
        channel_id="unknown",
        artifact_type=resolve_type,
        resolve_type=resolve_type,
        source_id=resolve_type,
        source_type="adapter_report_scope",
        metadata=metadata,
    )


def _scope_query(plan: ExecutionPlan) -> str:
    return str(plan.evidence_query or "").strip()


def _period_from_preflight(
    preflight: AdapterPreflightSnapshot,
    resolve_type: AdapterResolveType,
) -> str | None:
    for item in preflight.items:
        if item.resolve_type != resolve_type or item.result is None:
            continue
        if item.result.period:
            return item.result.period
    return None


def _material_type(resolve_type: AdapterResolveType) -> Literal["weekly", "monthly"]:
    return "weekly" if resolve_type == "weekly_report" else "monthly"
