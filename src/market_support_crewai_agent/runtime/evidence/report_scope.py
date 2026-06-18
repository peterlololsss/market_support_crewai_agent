from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.domain.capabilities import capability_by_name
from market_support_crewai_agent.runtime.domain.ontology import artifact_scope_for_evidence
from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.llm.prompting.assembler import (
    assembleCanonicalizationPrompt,
)
from market_support_crewai_agent.runtime.llm.prompting.registry import prompt_agent_spec_by_id
from market_support_crewai_agent.runtime.evidence.adapter_client import (
    AdapterClientError,
    AdapterResolveClient,
)
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.state.runtime_trace import trace_span
from market_support_crewai_agent.schemas import (
    AdapterReportScopeRequest,
    AdapterReportScopeResult,
    AdapterResolveType,
    ReplyRequest,
    ReportScopeProduct,
    ReportScopeSection,
    StrictModel,
)
from market_support_crewai_agent.settings import Settings, get_settings

_REPORT_CAPABILITIES = {"weekly_report", "monthly_report"}
_REPORT_SCOPE_SUMMARY_QUERY = "report_scope_summary"
_REPORT_SCOPE_PRODUCTS_QUERY = "report_scope_products"
_MAX_SELECTOR_PRODUCTS = 250
_PAGE_SIZE = 50


class ReportScopeSelection(StrictModel):
    selected_name: str = Field(default="", max_length=240)
    selected_type: Literal["section", "product", "none"] = "none"
    confidence: Literal["none", "low", "medium", "high"] = "none"
    rationale: str = Field(default="", max_length=400)


class ReportScopeSelector:
    """Closed-set report-scope selector that runs outside the main reply prompt."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def select(
        self,
        *,
        request: ReplyRequest,
        query: str,
        material_type: str,
        sections: list[ReportScopeSection],
        products: list[ReportScopeProduct],
    ) -> ReportScopeSelection:
        if not self.settings.llm_api_key or not query.strip():
            return ReportScopeSelection()
        prompt = _selector_prompt(
            request=request,
            query=query,
            material_type=material_type,
            sections=sections,
            products=products,
        )
        return await _run_selector_llm(
            prompt,
            self.settings,
            timeout_seconds=self.settings.llm_timeout_seconds,
        )


class ReportScopeEvidenceService:
    def __init__(
        self,
        settings: Settings | None = None,
        adapter_client: AdapterResolveClient | None = None,
        selector: ReportScopeSelector | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.adapter_client = adapter_client or AdapterResolveClient(self.settings)
        self.selector = selector or ReportScopeSelector(self.settings)

    async def collect(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ExecutionPlan,
        policy: PolicyManifest,
        preflight: AdapterPreflightSnapshot,
    ) -> list[EvidenceFact]:
        del canonical_context
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
                        summary=summary,
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
        summary: AdapterReportScopeResult,
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

        facts = [_match_fact(resolve_type, match_result, selector_used=False)]
        if match_result.match is not None and match_result.match.status == "matched":
            return facts

        selected_name = await self._select_name_from_products(
            request=request,
            material_type=material_type,
            period=period,
            query=query,
            summary=summary,
        )
        if not selected_name:
            return facts

        try:
            selected_match = await self.adapter_client.report_scope_async(
                AdapterReportScopeRequest(
                    material_type=material_type,  # type: ignore[arg-type]
                    dist_name=request.dist_channel_name,
                    command="match",
                    period=period,
                    query=selected_name,
                    page=1,
                    page_size=10,
                )
            )
        except AdapterClientError:
            return facts
        facts.append(_match_fact(resolve_type, selected_match, selector_used=True))
        return facts

    async def _select_name_from_products(
        self,
        *,
        request: ReplyRequest,
        material_type: str,
        period: str | None,
        query: str,
        summary: AdapterReportScopeResult,
    ) -> str:
        total_count = int(summary.expected_product_count or 0)
        if total_count <= 0 or total_count > _MAX_SELECTOR_PRODUCTS:
            return ""
        products = await self._fetch_products(
            request.dist_channel_name,
            material_type,
            period,
            total_count,
        )
        if not products:
            return ""
        selection = await self.selector.select(
            request=request,
            query=query,
            material_type=material_type,
            sections=summary.report_sections,
            products=products,
        )
        if selection.confidence == "none" or selection.selected_type == "none":
            return ""
        return selection.selected_name.strip()

    async def _fetch_products(
        self,
        dist_name: str,
        material_type: str,
        period: str | None,
        total_count: int,
    ) -> list[ReportScopeProduct]:
        products: list[ReportScopeProduct] = []
        page_count = min((total_count + _PAGE_SIZE - 1) // _PAGE_SIZE, 5)
        for page in range(1, page_count + 1):
            result = await self.adapter_client.report_scope_async(
                AdapterReportScopeRequest(
                    material_type=material_type,  # type: ignore[arg-type]
                    dist_name=dist_name,
                    command="list_products",
                    period=period,
                    page=page,
                    page_size=_PAGE_SIZE,
                )
            )
            products.extend(result.products)
        return products[:_MAX_SELECTOR_PRODUCTS]

    async def _product_facts(
        self,
        *,
        request: ReplyRequest,
        resolve_type: AdapterResolveType,
        material_type: str,
        period: str | None,
    ) -> list[EvidenceFact]:
        try:
            result = await self.adapter_client.report_scope_async(
                AdapterReportScopeRequest(
                    material_type=material_type,  # type: ignore[arg-type]
                    dist_name=request.dist_channel_name,
                    command="list_products",
                    period=period,
                    page=1,
                    page_size=_PAGE_SIZE,
                )
            )
        except AdapterClientError as exc:
            return [
                _unavailable_fact(
                    resolve_type,
                    reason_code="adapter_report_scope_products_error",
                    error_type=type(exc).__name__,
                )
            ]
        return [_products_fact(resolve_type, result)]


class NoopReportScopeEvidenceService:
    async def collect(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ExecutionPlan,
        policy: PolicyManifest,
        preflight: AdapterPreflightSnapshot,
    ) -> list[EvidenceFact]:
        del request, canonical_context, plan, policy, preflight
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
    *,
    selector_used: bool,
) -> EvidenceFact:
    metadata = _result_metadata(result)
    if result.match is not None:
        match_payload = result.match.model_dump(mode="json", exclude_none=True)
        metadata["match"] = match_payload
        metadata["selector_used"] = selector_used
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
) -> EvidenceFact:
    metadata = _result_metadata(result)
    metadata.update(
        {
            "products": [
                product.model_dump(mode="json", exclude_none=True)
                for product in result.products
            ],
            "product_page": result.product_page,
            "product_page_size": result.product_page_size,
            "product_total_count": result.product_total_count,
        }
    )
    returned_count = len(result.products)
    total_count = int(result.product_total_count or returned_count)
    metadata["full_product_list_in_prompt"] = total_count <= returned_count
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


def _selector_prompt(
    *,
    request: ReplyRequest,
    query: str,
    material_type: str,
    sections: list[ReportScopeSection],
    products: list[ReportScopeProduct],
) -> str:
    payload = {
        "user_message": request.message,
        "query": query,
        "material_type": material_type,
        "candidate_sections": [
            section.model_dump(mode="json", exclude_none=True)
            for section in sections
        ],
        "candidate_products": [
            product.model_dump(mode="json", exclude_none=True)
            for product in products
        ],
    }
    return assembleCanonicalizationPrompt(
        "canonicalization.report_scope_selector",
        stage="report_scope_selector",
        selector_input_json=json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    )


async def _run_selector_llm(
    prompt: str,
    settings: Settings,
    *,
    timeout_seconds: float,
) -> ReportScopeSelection:
    from crewai import Agent, LLM

    spec = prompt_agent_spec_by_id("agent.report_scope_selector")
    agent = Agent(
        role=spec.role,
        goal=spec.goal,
        backstory=spec.backstory,
        llm=LLM(
            model=settings.llm_model,
            provider=settings.llm_provider,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=0,
            max_tokens=min(settings.llm_max_tokens, 1000),
            timeout=settings.llm_timeout_seconds,
        ),
        allow_delegation=False,
        verbose=settings.crewai_verbose,
        max_iter=1,
        max_execution_time=settings.crewai_max_execution_time,
        max_retry_limit=settings.crewai_max_retry_limit,
        planning=False,
    )
    with trace_span(
        "llm.selector",
        stage="report_scope_selector",
        prompt_chars=len(prompt),
        response_format="ReportScopeSelection",
    ):
        result = await asyncio.wait_for(
            agent.kickoff_async(prompt, response_format=ReportScopeSelection),
            timeout=timeout_seconds,
        )
    if result.pydantic is not None:
        return ReportScopeSelection.model_validate(result.pydantic)
    return ReportScopeSelection.model_validate_json(result.raw)
