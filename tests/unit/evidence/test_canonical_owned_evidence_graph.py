from __future__ import annotations

from dataclasses import dataclass

import anyio

from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.integrations.adapter.report_scope import (
    ReportScopeEvidenceService,
)
from market_support_crewai_agent.runtime.planning import ExecutionPlanV2, PlanSpec
from market_support_crewai_agent.runtime.planning.compiler import (
    finalize_execution_plan_v2,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.schemas.adapter import AdapterResolveResult
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.unit.evidence._report_scope_evidence_fixtures import FakeReportScopeClient
from tests.unit.planning._execution_plan_v2_fixtures import policy_for


@dataclass(frozen=True, slots=True)
class _ReportInputs:
    request: KernelReplyRequestV1
    policy: PolicyManifestV2
    scope: BusinessScopeAuthorityV1
    plan: ExecutionPlanV2
    preflight: AdapterPreflightSnapshot


class FakePreflight:
    def __init__(self, snapshot: AdapterPreflightSnapshot) -> None:
        self.snapshot: AdapterPreflightSnapshot = snapshot

    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, resolve_types, resolve_material_pack_options
        return self.snapshot


def _report_inputs() -> _ReportInputs:
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
            "read_capabilities": ["query_weekly_report_product_list"],
            "outbound_actions": [],
            "mention_types": [],
        },
    ).request
    policy, scope = policy_for(request)
    capability_id = "weekly_report.product_list"
    plan_spec = PlanSpec.model_validate(
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
    plan = finalize_execution_plan_v2(
        plan_spec,
        policy,
        scope,
        origin="planner",
        confidence=0.9,
    )
    preflight = AdapterPreflightSnapshot(
        items=[
            AdapterPreflightItem(
                resolve_type="weekly_report",
                result=AdapterResolveResult.model_validate(
                    {
                        "contract_version": "adapter-resolve",
                        "resolve_type": "weekly_report",
                        "status": "resolved",
                        "display_name": "TestDist",
                        "reason_code": "ok",
                        "available_artifacts": [{"type": "weekly_report"}],
                        "resolved_at": 1_750_000_000,
                        "resolve_ref": "adapter:weekly_report:current",
                        "period": "20260612",
                        "report_date": "2026-06-12",
                    }
                ),
            )
        ]
    )
    return _ReportInputs(request, policy, scope, plan, preflight)


def test_owned_evidence_executor_returns_canonical_grounded_report_scope_facts() -> (
    None
):
    # Given: a V2 product-list plan and fake adapter seams at the preflight/report boundary.
    source = _report_inputs()
    report_client = FakeReportScopeClient()
    executor = EvidenceExecutor(
        FakePreflight(source.preflight),
        report_scope_service=ReportScopeEvidenceService(adapter_client=report_client),
    )

    # When: the owned evidence executor fulfills the report product-list unit.
    async def execute() -> CanonicalEvidenceExecutionResultV1:
        return await executor.execute_v2(
            source.request,
            source.plan,
            source.policy,
            scope_authority=source.scope,
        )

    result = anyio.run(execute)

    # Then: callers receive typed canonical facts and unit groundings, not file inventory proof.
    assert isinstance(result, CanonicalEvidenceExecutionResultV1)
    assert tuple(fact.fact_type for fact in result.canonical_facts) == (
        "weekly_report_resolvable",
        "report_scope_summary",
        "report_scope_products",
    )
    assert [call.command for call in report_client.calls] == [
        "summary",
        "list_products",
    ]
    assert len(result.groundings) == 1
    grounding = result.groundings[0]
    assert grounding.unit_id == source.plan.units[0].unit_id
    assert tuple(fact.fact_type for fact in grounding.allowed_evidence) == (
        "report_scope_products",
    )
    assert all(
        fact.contract_version == "canonical-evidence-fact.v1"
        for fact in result.canonical_facts
    )
