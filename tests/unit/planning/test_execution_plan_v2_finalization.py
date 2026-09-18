from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.planning import (
    PlanSpec,
    finalize_execution_plan_v2,
    validate_execution_plan_v2,
)
from tests.unit.planning._execution_plan_v2_fixtures import (
    knowledge_with_abstention_fixture,
    material_and_weekly_fixture,
    policy_for,
    weekly_with_material_clarification_fixture,
)
from tests.helpers.reply_contract_requests import make_v2_envelope


def test_finalizes_unscoped_company_plan_for_direct_knowledge_gateway() -> None:
    request = make_v2_envelope(
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:finalizer",
            "direct_thread_ref": "direct:finalizer",
            "principal_ref": "principal:finalizer",
        },
        presentation={
            "contract_version": "direct-presentation.v1",
            "principal_name": "finalizer",
        },
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"],
            "outbound_actions": [],
            "mention_types": [],
        },
    ).request
    policy, scope = policy_for(request)
    spec = PlanSpec.model_validate(
        {
            "plan_id": "direct-company",
            "user_intent_summary": "company public fact",
            "plan_units": [
                {
                    "unit_id": "company",
                    "selected_capability_id": "answer_internal_company_knowledge",
                    "domain_scope": {"kind": "unscoped"},
                    "answerability_policy": "answer",
                    "output_schema_ref": (
                        "answer_internal_company_knowledge:output_schema"
                    ),
                }
            ],
        }
    )

    plan = finalize_execution_plan_v2(spec, policy, scope, origin="planner")

    assert policy.internal_company_knowledge_enabled is True
    assert plan.units[0].scope.kind == "unscoped"
    assert plan.units[0].manifest_ref.manifest_id == "answer_internal_company_knowledge"
    assert plan.plan_spec is spec


@pytest.mark.parametrize(
    ("capability_id", "read_capability", "period_label"),
    [
        (
            "weekly_report.product_list",
            "query_weekly_report_product_list",
            "weekly",
        ),
        (
            "monthly_report.product_list",
            "query_monthly_report_product_list",
            "monthly",
        ),
    ],
)
def test_product_list_plan_rejects_model_selected_evidence_query(
    capability_id: str,
    read_capability: str,
    period_label: str,
) -> None:
    # Given: a product-list plan with a model-proposed report command.
    request = make_v2_envelope(
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": [read_capability],
            "outbound_actions": [],
            "mention_types": [],
        },
    ).request
    policy, scope = policy_for(request)
    spec = PlanSpec.model_validate(
        {
            "plan_id": f"{period_label}-products",
            "user_intent_summary": f"{period_label} product list",
            "plan_units": [
                {
                    "unit_id": f"{period_label}-products",
                    "selected_capability_id": capability_id,
                    "domain_scope": {
                        "kind": "distribution",
                        "business_scope_ref": scope.business_scope_ref,
                        "channel_kind": "bank",
                    },
                    "answerability_policy": "answer",
                    "output_schema_ref": f"{capability_id}:output_schema",
                    "steps": [
                        {
                            "step_id": "query",
                            "description": "query report scope",
                            "evidence_query": "summary",
                        }
                    ],
                }
            ],
        }
    )

    # When/Then: product-list commands are not model-selectable.
    with pytest.raises(ValueError, match="product_list_evidence_query_forbidden"):
        _ = finalize_execution_plan_v2(spec, policy, scope, origin="planner")


def test_retains_distribution_material_pack_option() -> None:
    request = make_v2_envelope(
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["resolve_material_pack"],
            "outbound_actions": ["send_material_pack"],
            "mention_types": [],
        },
    ).request
    policy, scope = policy_for(request)
    spec = PlanSpec.model_validate(
        {
            "plan_id": "group-material",
            "user_intent_summary": "send material pack",
            "plan_units": [
                {
                    "unit_id": "material",
                    "selected_capability_id": "material_pack.send",
                    "domain_scope": {
                        "kind": "distribution",
                        "business_scope_ref": scope.business_scope_ref,
                        "channel_kind": "bank",
                        "material_pack_option": "growth",
                        "product_ids": [],
                    },
                    "answerability_policy": "send",
                    "output_schema_ref": "material_pack.send:output_schema",
                }
            ],
        }
    )

    plan = finalize_execution_plan_v2(spec, policy, scope, origin="planner")

    assert plan.units[0].scope.kind == "distribution"
    assert plan.units[0].scope.material_pack_option == "growth"
    assert plan.units[0].action_intents[0].material_pack_option == "growth"


def test_v2_finalization_preserves_material_and_weekly_multi_send() -> None:
    fixture = material_and_weekly_fixture()

    plan = finalize_execution_plan_v2(
        fixture.spec,
        fixture.policy,
        fixture.scope,
        origin="planner",
    )

    assert (plan.response_mode, plan.artifact_kind) == ("action", "multi_action")
    assert [intent.action_type for intent in plan.action_intents] == [
        "send_material_pack",
        "send_weekly_report",
    ]
    assert plan.action_intents[0].material_pack_option == "指增"
    assert validate_execution_plan_v2(plan, fixture.policy).valid


def test_v2_finalization_keeps_weekly_with_material_clarification() -> None:
    fixture = weekly_with_material_clarification_fixture()

    plan = finalize_execution_plan_v2(
        fixture.spec,
        fixture.policy,
        fixture.scope,
        origin="planner",
    )

    assert plan.response_mode == "action"
    assert plan.units[1].answerability_policy == "clarify"
    assert plan.units[1].ambiguity_slots == ("material_pack_option",)
    assert [intent.action_type for intent in plan.action_intents] == [
        "send_weekly_report"
    ]
    assert validate_execution_plan_v2(plan, fixture.policy).valid


def test_v2_finalization_keeps_knowledge_answer_beside_sibling_abstention() -> None:
    fixture = knowledge_with_abstention_fixture()

    plan = finalize_execution_plan_v2(
        fixture.spec,
        fixture.policy,
        fixture.scope,
        origin="planner",
    )

    assert (plan.response_mode, plan.artifact_kind) == (
        "knowledge_answer",
        "knowledge_answer",
    )
    assert [unit.manifest_ref.manifest_id for unit in plan.units] == [
        "answer_internal_company_knowledge",
        "general.abstention",
    ]
    assert [unit.answerability_policy for unit in plan.units] == [
        "answer",
        "abstain",
    ]
    assert plan.action_intents == ()
    assert validate_execution_plan_v2(plan, fixture.policy).valid
