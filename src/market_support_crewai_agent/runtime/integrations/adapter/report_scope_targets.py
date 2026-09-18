from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    kernel_distribution_name,
)
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.planning.models import (
    ExecutionPlanUnitV2,
    ExecutionPlanV2,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    AlignmentRefetchRequestV1,
)
from market_support_crewai_agent.schemas.adapter import (
    AdapterReportScopeRequest,
    AdapterReportScopeResult,
    AdapterResolveResult,
)

ReportCommand = Literal["summary", "match", "list_products"]
ReportMaterialType = Literal["weekly", "monthly"]
ReportArtifactType = Literal["weekly_report", "monthly_report"]
ReportScopeBindingError = Literal[
    "adapter_report_scope_material_type_mismatch",
    "adapter_report_scope_distribution_mismatch",
    "adapter_report_scope_period_mismatch",
]
REPORT_SCOPE_CONTRACT_VERSION: Final = "adapter-report-scope"
PAGE_SIZE: Final = 50
MAX_PRODUCTS_IN_PROJECTION: Final = 200
_REPORT_SCOPE_SUMMARY_QUERY: Final = "report_scope_summary"
_REPORT_SCOPE_PRODUCTS_QUERY: Final = "report_scope_products"


@dataclass(frozen=True, slots=True)
class ReportTarget:
    unit: ExecutionPlanUnitV2
    resolve_type: ReportArtifactType
    material_type: ReportMaterialType
    command: ReportCommand
    query: str | None
    period: str | None
    preflight_result: AdapterResolveResult | None


def report_targets(
    plan: ExecutionPlanV2,
    policy: PolicyManifestV2,
    preflight: AdapterPreflightSnapshot,
    alignment_refetch_request: AlignmentRefetchRequestV1 | None,
) -> tuple[ReportTarget, ...]:
    preflight_by_type = {
        item.resolve_type: item.result
        for item in preflight.items
        if item.result is not None
    }
    targets: list[ReportTarget] = []
    for unit in plan.units:
        if (
            unit.answerability_policy != "answer"
            or unit.manifest_ref not in policy.eligible_capabilities
        ):
            continue
        manifest = CAPABILITY_MANIFEST_REGISTRY.find(unit.manifest_ref.manifest_id)
        if (
            manifest is None
            or manifest.manifest_version != unit.manifest_ref.manifest_version
        ):
            continue
        if (
            "adapter_report_scope"
            not in manifest.evidence_contract.allowed_source_types
        ):
            continue
        resolve_type = _report_artifact_type(unit)
        if resolve_type is None or resolve_type not in policy.allowed_adapter_resolves:
            continue
        query = _target_query(unit, alignment_refetch_request)
        command = _target_command(query, manifest.evidence_contract.allowed_fact_types)
        if command is None:
            continue
        result = preflight_by_type.get(resolve_type)
        targets.append(
            ReportTarget(
                unit=unit,
                resolve_type=resolve_type,
                material_type=_material_type(resolve_type),
                command=command,
                query=query if command == "match" else None,
                period=result.period if result is not None else None,
                preflight_result=result,
            )
        )
    return tuple(targets)


def report_request(
    request: KernelReplyRequestV1,
    target: ReportTarget,
    *,
    command: ReportCommand,
    page: int | None = None,
) -> AdapterReportScopeRequest:
    return AdapterReportScopeRequest(
        material_type=target.material_type,
        dist_name=kernel_distribution_name(request),
        command=command,
        period=target.period,
        query=target.query if command == "match" else None,
        page=page if command == "list_products" else None,
        page_size=PAGE_SIZE if command == "list_products" else None,
    )


def report_scope_binding_error(
    request: AdapterReportScopeRequest,
    result: AdapterReportScopeResult,
) -> ReportScopeBindingError | None:
    if result.material_type != request.material_type:
        return "adapter_report_scope_material_type_mismatch"
    if result.dist_name != request.dist_name:
        return "adapter_report_scope_distribution_mismatch"
    if request.period is not None and result.period != request.period:
        return "adapter_report_scope_period_mismatch"
    return None


def _target_query(
    unit: ExecutionPlanUnitV2,
    refetch: AlignmentRefetchRequestV1 | None,
) -> str | None:
    if (
        refetch is not None
        and refetch.unit_id == unit.unit_id
        and refetch.manifest_ref == unit.manifest_ref
    ):
        return refetch.refined_evidence_query.strip()
    return unit.evidence_query.strip() if unit.evidence_query is not None else None


def _target_command(
    query: str | None,
    allowed_fact_types: tuple[str, ...],
) -> ReportCommand | None:
    allowed = frozenset(allowed_fact_types)
    if query == _REPORT_SCOPE_PRODUCTS_QUERY and "report_scope_products" in allowed:
        return "list_products"
    if query == _REPORT_SCOPE_SUMMARY_QUERY and "report_scope_summary" in allowed:
        return "summary"
    if query and "report_scope_match" in allowed:
        return "match"
    if "report_scope_products" in allowed:
        return "list_products"
    if "report_scope_summary" in allowed:
        return "summary"
    return None


def _report_artifact_type(unit: ExecutionPlanUnitV2) -> ReportArtifactType | None:
    match unit.artifact_kind:
        case "weekly_report":
            return "weekly_report"
        case "monthly_report":
            return "monthly_report"
        case _:
            return None


def _material_type(resolve_type: ReportArtifactType) -> ReportMaterialType:
    match resolve_type:
        case "weekly_report":
            return "weekly"
        case "monthly_report":
            return "monthly"
