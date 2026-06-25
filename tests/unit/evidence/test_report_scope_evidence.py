from __future__ import annotations

import asyncio

from market_support_crewai_agent.runtime.domain.planning import (
    AdapterResolveSpec,
    ComplianceDecision,
    ExecutionPlan,
)
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.evidence.report_scope import (
    ReportScopeEvidenceService,
)
from market_support_crewai_agent.schemas import (
    AdapterReportScopeResult,
    AdapterResolveResult,
    ReplyRequest,
)


def test_report_scope_evidence_collects_summary_and_match_without_listing_products():
    request = make_request()
    plan = ExecutionPlan(
        user_need="answer weekly report scope question",
        artifact_kind="knowledge_answer",
        response_mode="knowledge_answer",
        compliance=ComplianceDecision(
            is_compliant=True,
            reason_code="compliant_product_request",
            reason="normal report question",
        ),
        evidence_query="A500",
        capabilities=["weekly_report"],
        adapter_resolves=[AdapterResolveSpec(resolve_type="weekly_report")],
        action_intents=[],
        ambiguity_slots=[],
        confidence=0.9,
    )
    fake_client = FakeReportScopeClient()
    service = ReportScopeEvidenceService(adapter_client=fake_client)

    facts = asyncio.run(
        service.collect(
            request,
            plan,
            compile_policy(request),
            AdapterPreflightSnapshot(
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
                                "available_artifacts": [
                                    {"type": "weekly_report"},
                                    {"type": "monthly_report"},
                                ],
                                "resolved_at": 1,
                                "resolve_ref": "wecom-adapter:test",
                                "period": "20260612",
                                "report_date": "2026-06-12",
                            }
                        ),
                    )
                ]
            ),
        )
    )

    assert [call.command for call in fake_client.calls] == ["summary", "match"]
    assert [fact.fact_type for fact in facts] == [
        "report_scope_summary",
        "report_scope_match",
    ]
    assert facts[0].metadata["full_product_list_in_prompt"] is False
    assert facts[1].metadata["match"]["status"] == "matched"


def test_report_scope_match_not_found_does_not_list_products_or_fallback():
    request = make_request()
    plan = ExecutionPlan(
        user_need="answer weekly report scope question",
        artifact_kind="knowledge_answer",
        response_mode="knowledge_answer",
        compliance=ComplianceDecision(
            is_compliant=True,
            reason_code="compliant_product_request",
            reason="normal report question",
        ),
        evidence_query="A500",
        capabilities=["weekly_report"],
        adapter_resolves=[AdapterResolveSpec(resolve_type="weekly_report")],
        action_intents=[],
        ambiguity_slots=[],
        confidence=0.9,
    )
    fake_client = FakeReportScopeClient(match_status="not_found")
    service = ReportScopeEvidenceService(adapter_client=fake_client)

    facts = asyncio.run(
        service.collect(
            request,
            plan,
            compile_policy(request),
            AdapterPreflightSnapshot(
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
                                "resolved_at": 1,
                                "resolve_ref": "wecom-adapter:test",
                                "period": "20260612",
                                "report_date": "2026-06-12",
                            }
                        ),
                    )
                ]
            ),
        )
    )

    assert [call.command for call in fake_client.calls] == ["summary", "match"]
    assert facts[1].value == "not_found"
    assert facts[1].metadata["match"]["status"] == "not_found"
    assert "selector_used" not in facts[1].metadata


def test_report_period_answer_does_not_call_report_scope_endpoint():
    request = make_request(message="这个周报是什么时间段")
    plan = ExecutionPlan(
        user_need="answer weekly report period question",
        artifact_kind="knowledge_answer",
        response_mode="knowledge_answer",
        compliance=ComplianceDecision(
            is_compliant=True,
            reason_code="compliant_product_request",
            reason="normal report period question",
        ),
        evidence_query=None,
        capabilities=["weekly_report"],
        adapter_resolves=[AdapterResolveSpec(resolve_type="weekly_report")],
        action_intents=[],
        ambiguity_slots=[],
        confidence=0.9,
    )
    fake_client = FakeReportScopeClient()
    service = ReportScopeEvidenceService(adapter_client=fake_client)

    facts = asyncio.run(
        service.collect(
            request,
            plan,
            compile_policy(request),
            AdapterPreflightSnapshot(
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
                                "available_artifacts": [
                                    {"type": "weekly_report"},
                                    {"type": "monthly_report"},
                                ],
                                "resolved_at": 1,
                                "resolve_ref": "wecom-adapter:test",
                                "period": "20260612",
                                "report_date": "2026-06-12",
                                "period_start": "2026-06-08",
                                "period_end": "2026-06-12",
                            }
                        ),
                    )
                ]
            ),
        )
    )

    assert fake_client.calls == []
    assert facts == []


def test_report_scope_summary_sentinel_collects_summary_only():
    request = make_request(message="这个周报用了哪些产品生成")
    plan = ExecutionPlan(
        user_need="answer weekly report scope summary question",
        artifact_kind="knowledge_answer",
        response_mode="knowledge_answer",
        compliance=ComplianceDecision(
            is_compliant=True,
            reason_code="compliant_product_request",
            reason="normal report scope question",
        ),
        evidence_query="report_scope_summary",
        capabilities=["weekly_report"],
        adapter_resolves=[AdapterResolveSpec(resolve_type="weekly_report")],
        action_intents=[],
        ambiguity_slots=[],
        confidence=0.9,
    )
    fake_client = FakeReportScopeClient()
    service = ReportScopeEvidenceService(adapter_client=fake_client)

    facts = asyncio.run(
        service.collect(
            request,
            plan,
            compile_policy(request),
            AdapterPreflightSnapshot(
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
                                "available_artifacts": [
                                    {"type": "weekly_report"},
                                    {"type": "monthly_report"},
                                ],
                                "resolved_at": 1,
                                "resolve_ref": "wecom-adapter:test",
                                "period": "20260612",
                                "report_date": "2026-06-12",
                            }
                        ),
                    )
                ]
            ),
        )
    )

    assert [call.command for call in fake_client.calls] == ["summary"]
    assert [fact.fact_type for fact in facts] == ["report_scope_summary"]


def test_report_scope_products_sentinel_collects_bounded_product_page():
    request = make_request(message="刚发的周报有哪些产品")
    plan = ExecutionPlan(
        user_need="answer weekly report product list question",
        artifact_kind="knowledge_answer",
        response_mode="knowledge_answer",
        compliance=ComplianceDecision(
            is_compliant=True,
            reason_code="compliant_product_request",
            reason="normal report product list question",
        ),
        evidence_query="report_scope_products",
        capabilities=["weekly_report"],
        adapter_resolves=[AdapterResolveSpec(resolve_type="weekly_report")],
        action_intents=[],
        ambiguity_slots=[],
        confidence=0.9,
    )
    fake_client = FakeReportScopeClient()
    service = ReportScopeEvidenceService(adapter_client=fake_client)

    facts = asyncio.run(
        service.collect(
            request,
            plan,
            compile_policy(request),
            AdapterPreflightSnapshot(
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
                                "available_artifacts": [
                                    {"type": "weekly_report"},
                                    {"type": "monthly_report"},
                                ],
                                "resolved_at": 1,
                                "resolve_ref": "wecom-adapter:test",
                                "period": "20260612",
                                "report_date": "2026-06-12",
                            }
                        ),
                    )
                ]
            ),
        )
    )

    assert [call.command for call in fake_client.calls] == ["summary", "list_products"]
    assert [fact.fact_type for fact in facts] == [
        "report_scope_summary",
        "report_scope_products",
    ]
    assert facts[1].metadata["product_total_count"] == 2
    assert [product["product_name"] for product in facts[1].metadata["products"]] == [
        "Product1",
        "Product2",
    ]
    assert facts[1].metadata["full_product_list_in_prompt"] is True


def test_report_scope_products_fetches_all_pages_when_bounded():
    request = make_request(message="刚发的周报有A500指增吗")
    plan = ExecutionPlan(
        user_need="answer weekly report shorthand product question",
        artifact_kind="knowledge_answer",
        response_mode="knowledge_answer",
        compliance=ComplianceDecision(
            is_compliant=True,
            reason_code="compliant_product_request",
            reason="normal report product question",
        ),
        evidence_query="report_scope_products",
        capabilities=["weekly_report"],
        adapter_resolves=[AdapterResolveSpec(resolve_type="weekly_report")],
        action_intents=[],
        ambiguity_slots=[],
        confidence=0.9,
    )
    names = [f"Product{i:02d}" for i in range(53)]
    fake_client = FakeReportScopeClient(
        product_pages={1: names[:50], 2: names[50:]},
        product_total_count=53,
    )
    service = ReportScopeEvidenceService(adapter_client=fake_client)

    facts = asyncio.run(
        service.collect(
            request,
            plan,
            compile_policy(request),
            AdapterPreflightSnapshot(
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
                                "resolved_at": 1,
                                "resolve_ref": "wecom-adapter:test",
                                "period": "20260612",
                                "report_date": "2026-06-12",
                            }
                        ),
                    )
                ]
            ),
        )
    )

    product_fact = facts[1]
    assert [(call.command, call.page) for call in fake_client.calls] == [
        ("summary", None),
        ("list_products", 1),
        ("list_products", 2),
    ]
    assert product_fact.metadata["product_returned_count"] == 53
    assert product_fact.metadata["full_product_list_in_prompt"] is True
    assert product_fact.metadata["products"][-1]["product_name"] == "Product52"


def test_report_scope_products_fetches_until_prompt_cap_when_list_is_too_large():
    request = make_request(message="周报里有A500指增吗")
    plan = ExecutionPlan(
        user_need="answer weekly report shorthand product question",
        artifact_kind="knowledge_answer",
        response_mode="knowledge_answer",
        compliance=ComplianceDecision(
            is_compliant=True,
            reason_code="compliant_product_request",
            reason="normal report product question",
        ),
        evidence_query="report_scope_products",
        capabilities=["weekly_report"],
        adapter_resolves=[AdapterResolveSpec(resolve_type="weekly_report")],
        action_intents=[],
        ambiguity_slots=[],
        confidence=0.9,
    )
    fake_client = FakeReportScopeClient(
        product_pages={
            1: [f"Product{i:03d}" for i in range(50)],
            2: [f"Product{i:03d}" for i in range(50, 100)],
            3: [f"Product{i:03d}" for i in range(100, 150)],
            4: [f"Product{i:03d}" for i in range(150, 200)],
        },
        product_total_count=201,
    )
    service = ReportScopeEvidenceService(adapter_client=fake_client)

    facts = asyncio.run(
        service.collect(
            request,
            plan,
            compile_policy(request),
            AdapterPreflightSnapshot(
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
                                "resolved_at": 1,
                                "resolve_ref": "wecom-adapter:test",
                                "period": "20260612",
                                "report_date": "2026-06-12",
                            }
                        ),
                    )
                ]
            ),
        )
    )

    assert [(call.command, call.page) for call in fake_client.calls] == [
        ("summary", None),
        ("list_products", 1),
        ("list_products", 2),
        ("list_products", 3),
        ("list_products", 4),
    ]
    assert facts[1].metadata["product_returned_count"] == 200
    assert facts[1].metadata["full_product_list_in_prompt"] is False


def test_report_scope_evidence_for_mixed_action_uses_answer_capabilities_only():
    request = make_request(message="weekly products, then send monthly")
    plan = ExecutionPlan(
        user_need="answer weekly products and send monthly",
        artifact_kind="multi_action",
        response_mode="action",
        compliance=ComplianceDecision(
            is_compliant=True,
            reason_code="compliant_product_request",
            reason="normal mixed report request",
        ),
        evidence_query="report_scope_products",
        capabilities=["monthly_report", "weekly_report"],
        answer_capabilities=["weekly_report"],
        adapter_resolves=[
            AdapterResolveSpec(resolve_type="monthly_report"),
            AdapterResolveSpec(resolve_type="weekly_report"),
        ],
        action_intents=[],
        ambiguity_slots=[],
        confidence=0.9,
    )
    fake_client = FakeReportScopeClient()
    service = ReportScopeEvidenceService(adapter_client=fake_client)

    facts = asyncio.run(
        service.collect(
            request,
            plan,
            compile_policy(request),
            AdapterPreflightSnapshot(
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
                                "available_artifacts": [
                                    {"type": "weekly_report"},
                                    {"type": "monthly_report"},
                                ],
                                "resolved_at": 1,
                                "resolve_ref": "wecom-adapter:weekly",
                                "period": "20260612",
                                "report_date": "2026-06-12",
                            }
                        ),
                    ),
                    AdapterPreflightItem(
                        resolve_type="monthly_report",
                        result=AdapterResolveResult.model_validate(
                            {
                                "contract_version": "adapter-resolve",
                                "resolve_type": "monthly_report",
                                "status": "resolved",
                                "display_name": "TestDist",
                                "reason_code": "ok",
                                "available_artifacts": [
                                    {"type": "weekly_report"},
                                    {"type": "monthly_report"},
                                ],
                                "resolved_at": 1,
                                "resolve_ref": "wecom-adapter:monthly",
                                "period": "2026-05",
                                "report_date": "2026-05-31",
                            }
                        ),
                    ),
                ]
            ),
        )
    )

    assert [call.material_type for call in fake_client.calls] == ["weekly", "weekly"]
    assert [fact.resolve_type for fact in facts] == ["weekly_report", "weekly_report"]


class FakeReportScopeClient:
    def __init__(
        self,
        match_status="matched",
        *,
        product_pages: dict[int, list[str]] | None = None,
        product_total_count: int | None = None,
    ) -> None:
        self.calls = []
        self.match_status = match_status
        self.product_pages = product_pages or {1: ["Product1", "Product2"]}
        self.product_total_count = (
            product_total_count
            if product_total_count is not None
            else sum(len(products) for products in self.product_pages.values())
        )

    async def report_scope_async(self, request):
        self.calls.append(request)
        if request.command == "summary":
            return AdapterReportScopeResult.model_validate(
                {
                    "contract_version": "adapter-report-scope",
                    "material_type": "weekly",
                    "dist_name": request.dist_name,
                    "period": request.period or "20260612",
                    "status": "resolved",
                    "reason_code": "ok",
                    "report_date": "2026-06-12",
                    "period_start": "2026-06-08",
                    "period_end": "2026-06-12",
                    "period_label": "2026-06-08至2026-06-12周报",
                    "scope_complete": True,
                    "expected_product_count": 100,
                    "generated_product_count": 100,
                    "missing_product_count": 0,
                    "report_sections": [
                        {
                            "name": "A500",
                            "expected_product_count": 12,
                            "generated_product_count": 12,
                            "missing_product_count": 0,
                        }
                    ],
                }
            )
        if request.command == "list_products":
            products = self.product_pages.get(request.page or 1, [])
            return AdapterReportScopeResult.model_validate(
                {
                    "contract_version": "adapter-report-scope",
                    "material_type": "weekly",
                    "dist_name": request.dist_name,
                    "period": request.period or "20260612",
                    "status": "resolved",
                    "reason_code": "ok",
                    "report_date": "2026-06-12",
                    "period_start": "2026-06-08",
                    "period_end": "2026-06-12",
                    "period_label": "2026-06-08至2026-06-12周报",
                    "scope_complete": True,
                    "expected_product_count": 2,
                    "generated_product_count": 2,
                    "missing_product_count": 0,
                    "report_sections": [],
                    "products": [
                        {
                            "product_name": product,
                            "portfolio_type": "IndexPlus",
                            "report_section": "IndexPlus",
                            "source_pdf_status": "found",
                            "final_report_status": "generated",
                        }
                        for product in products
                    ],
                    "product_page": request.page or 1,
                    "product_page_size": 50,
                    "product_total_count": self.product_total_count,
                }
            )
        return AdapterReportScopeResult.model_validate(
            {
                "contract_version": "adapter-report-scope",
                "material_type": "weekly",
                "dist_name": request.dist_name,
                "period": request.period or "20260612",
                "status": "resolved",
                "reason_code": "ok",
                "report_date": "2026-06-12",
                "period_start": "2026-06-08",
                "period_end": "2026-06-12",
                "period_label": "2026-06-08至2026-06-12周报",
                "scope_complete": True,
                "expected_product_count": 100,
                "generated_product_count": 100,
                "missing_product_count": 0,
                "report_sections": [],
                "match": {
                    "status": self.match_status,
                    "query": request.query,
                    "match_type": "section" if self.match_status == "matched" else None,
                    "matched_section": "A500" if self.match_status == "matched" else None,
                    "candidate_count": 12 if self.match_status == "matched" else 0,
                    "products": [],
                    "product_page": 1,
                    "product_page_size": 10,
                },
            }
        )


def make_request(**overrides) -> ReplyRequest:
    payload = {
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "context_id": "msg-1",
        "message": "why is A500 missing from this weekly report",
        "is_group": True,
        "group_name": "test group",
        "dist_channel_name": "TestDist",
        "sender_nickname": "tester",
        "available_artifacts": [
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "bank",
        "allowed_read_capabilities": [
            "resolve_weekly_report",
            "resolve_monthly_report",
            "resolve_sales_mention",
        ],
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)
