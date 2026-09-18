from __future__ import annotations

from math import ceil
from typing import Protocol

from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.client import (
    AdapterResolveClient,
)
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.integrations.adapter.report_scope_facts import (
    match_fact,
    products_fact,
    summary_fact,
    unavailable_fact,
)
from market_support_crewai_agent.runtime.integrations.adapter.report_scope_targets import (
    MAX_PRODUCTS_IN_PROJECTION,
    PAGE_SIZE,
    ReportTarget,
    report_request,
    report_scope_binding_error,
    report_targets,
)
from market_support_crewai_agent.runtime.integrations.adapter.transport import (
    AdapterClientError,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    AlignmentRefetchRequestV1,
)
from market_support_crewai_agent.schemas.adapter import (
    AdapterReportScopeRequest,
    AdapterReportScopeResult,
)
from market_support_crewai_agent.settings_model import Settings


class ReportScopeClient(Protocol):
    async def report_scope_async(
        self,
        request: AdapterReportScopeRequest,
    ) -> AdapterReportScopeResult: ...


class ReportScopeEvidenceProvider(Protocol):
    async def collect(
        self,
        request: KernelReplyRequestV1,
        plan: ExecutionPlanV2,
        policy: PolicyManifestV2,
        preflight: AdapterPreflightSnapshot,
        *,
        alignment_refetch_request: AlignmentRefetchRequestV1 | None = None,
    ) -> tuple[CanonicalEvidenceFactV1, ...]: ...


class ReportScopeEvidenceService:
    def __init__(
        self,
        settings: Settings | None = None,
        adapter_client: ReportScopeClient | None = None,
    ) -> None:
        self.adapter_client: ReportScopeClient = adapter_client or AdapterResolveClient(
            settings
        )

    async def collect(
        self,
        request: KernelReplyRequestV1,
        plan: ExecutionPlanV2,
        policy: PolicyManifestV2,
        preflight: AdapterPreflightSnapshot,
        *,
        alignment_refetch_request: AlignmentRefetchRequestV1 | None = None,
    ) -> tuple[CanonicalEvidenceFactV1, ...]:
        if plan.compliance.is_compliant is not True:
            return ()
        targets = report_targets(
            plan,
            policy,
            preflight,
            alignment_refetch_request,
        )
        facts: list[CanonicalEvidenceFactV1] = []
        for target in targets:
            try:
                summary_request = report_request(request, target, command="summary")
                summary = await self.adapter_client.report_scope_async(summary_request)
            except AdapterClientError:
                facts.append(unavailable_fact(target, "adapter_report_scope_error"))
                continue
            binding_error = report_scope_binding_error(summary_request, summary)
            if binding_error is not None:
                facts.append(unavailable_fact(target, binding_error))
                continue
            facts.append(summary_fact(target, summary))
            match target.command:
                case "summary":
                    continue
                case "match":
                    try:
                        match_request = report_request(
                            request,
                            target,
                            command="match",
                        )
                        result = await self.adapter_client.report_scope_async(
                            match_request
                        )
                    except AdapterClientError:
                        facts.append(
                            unavailable_fact(
                                target,
                                "adapter_report_scope_match_error",
                            )
                        )
                    else:
                        binding_error = report_scope_binding_error(
                            match_request,
                            result,
                        )
                        if binding_error is not None:
                            facts.append(unavailable_fact(target, binding_error))
                        else:
                            facts.append(match_fact(target, result))
                case "list_products":
                    facts.append(await self._collect_products_fact(request, target))
        return tuple({fact.evidence_id: fact for fact in facts}.values())

    async def _collect_products_fact(
        self,
        request: KernelReplyRequestV1,
        target: ReportTarget,
    ) -> CanonicalEvidenceFactV1:
        try:
            first_request = report_request(
                request,
                target,
                command="list_products",
                page=1,
            )
            result = await self.adapter_client.report_scope_async(first_request)
            binding_error = report_scope_binding_error(first_request, result)
            if binding_error is not None:
                return unavailable_fact(target, binding_error)
            products = list(result.products)
            total_count = result.product_total_count
            if result.status != "resolved" or total_count is None:
                return unavailable_fact(
                    target,
                    "adapter_report_scope_products_invalid",
                )
            target_count = min(total_count, MAX_PRODUCTS_IN_PROJECTION)
            for page in range(2, ceil(target_count / PAGE_SIZE) + 1):
                page_request = report_request(
                    request,
                    target,
                    command="list_products",
                    page=page,
                )
                page_result = await self.adapter_client.report_scope_async(page_request)
                binding_error = report_scope_binding_error(
                    page_request,
                    page_result,
                )
                if binding_error is not None:
                    return unavailable_fact(target, binding_error)
                products.extend(page_result.products)
        except AdapterClientError:
            return unavailable_fact(
                target,
                "adapter_report_scope_products_error",
            )
        return products_fact(target, result, products, total_count)
