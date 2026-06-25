from __future__ import annotations

from market_support_crewai_agent.runtime.domain.capabilities.adapters import (
    planner_capability_cards,
)
from market_support_crewai_agent.runtime.domain.capabilities.registry import (
    CapabilityManifest,
    CapabilityRegistry,
    EvidenceContract,
)
from market_support_crewai_agent.runtime.domain.ontology import ArtifactScope
from market_support_crewai_agent.runtime.domain.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.validation.plan_spec_verifier import (
    verify_plan_spec,
)


def reply_payload(kind: str = "answer", text: str = "grounded answer") -> dict:
    return {"reply": {"kind": kind, "text": text}, "actions": []}


def output_schema() -> dict:
    return {
        "type": "object",
        "required": ["reply"],
        "properties": {
            "reply": {
                "type": "object",
                "required": ["kind", "text"],
                "properties": {
                    "kind": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
            "actions": {"type": "array"},
        },
    }


def dummy_manifest(**overrides) -> CapabilityManifest:
    payload = {
        "id": "dummy.echo",
        "version": "test.1",
        "display_name": "Dummy echo",
        "description": "Dummy capability for generic PlanSpec verification.",
        "capability_type": "answer",
        "domain_entities": ["channel", "artifact"],
        "required_inputs": [],
        "optional_inputs": [],
        "required_artifacts": ["adapter_context"],
        "allowed_artifacts": ["adapter_context"],
        "forbidden_artifacts": ["weekly_report"],
        "required_tools": ["dummy.echo"],
        "output_schema": output_schema(),
        "evidence_contract": {
            "required_evidence_types": ["dummy_fact"],
            "allowed_source_types": ["adapter_resolve"],
            "allowed_artifact_types": ["adapter_context"],
            "minimum_evidence_count": 1,
            "required_scope_match": ["channel_id"],
        },
        "abstention_policy": {
            "requires_abstention_when_evidence_missing": True,
            "abstention_reply_kinds": ["unable_to_answer", "clarification"],
        },
        "planner_guidance": "Use only when explicitly selected.",
        "agent_guidance": "Answer only from dummy evidence.",
        "verifier_checks": [
            "output_schema",
            "required_evidence_present",
            "evidence_artifact_type_allowed",
            "forbidden_source_not_used",
            "abstention_correctness",
        ],
        "examples_positive": ["dummy"],
        "examples_negative": ["weekly report"],
    }
    payload.update(overrides)
    return CapabilityManifest.model_validate(payload)


def plan_spec(**overrides) -> PlanSpec:
    unit_payload = {
        "unit_id": "unit-1",
        "selected_capability_id": "dummy.echo",
        "domain_scope": {
            "channel_id": "channel-1",
            "channel_kind": "bank",
            "product_ids": [],
        },
        "required_artifacts": ["adapter_context"],
        "allowed_artifacts": ["adapter_context"],
        "forbidden_artifacts": ["weekly_report"],
        "required_tools": ["dummy.echo"],
        "answerability_policy": "answer",
        "output_schema_ref": "dummy.echo:output_schema",
        "output_schema": None,
        "evidence_contract_ref": "dummy.echo:evidence_contract",
        "evidence_contract": None,
        "steps": [
            {
                "step_id": "step-1",
                "description": "answer from dummy evidence",
                "uses_artifacts": ["adapter_context"],
                "required_artifacts": ["adapter_context"],
                "allowed_artifacts": ["adapter_context"],
                "forbidden_artifacts": ["weekly_report"],
                "required_tools": ["dummy.echo"],
            }
        ],
        "acceptance_criteria": ["dummy evidence is present"],
        "abstention_cases": ["missing dummy evidence"],
        "risk_flags": [],
    }
    unit_keys = set(unit_payload)
    for key in list(overrides):
        if key in unit_keys:
            unit_payload[key] = overrides.pop(key)
    payload = {
        "plan_id": "plan-test",
        "user_intent_summary": "answer from dummy evidence",
        "plan_units": [unit_payload],
        "risk_flags": [],
    }
    payload.update(overrides)
    return PlanSpec.model_validate(payload)


def fact(
    fact_type: str = "dummy_fact",
    *,
    source_type: str = "adapter_resolve",
    source_id: str = "source-1",
    artifact_type: str = "adapter_context",
    channel_id: str = "channel-1",
    material_pack_option: str | None = None,
    value=True,
    metadata: dict | None = None,
) -> EvidenceFact:
    fact_metadata = dict(metadata or {})
    if material_pack_option is not None:
        fact_metadata["material_pack_option"] = material_pack_option
    return EvidenceFact(
        fact_type=fact_type,  # type: ignore[arg-type]
        value=value,
        source_type=source_type,  # type: ignore[arg-type]
        source_id=source_id,
        metadata=fact_metadata,
        artifact_type=artifact_type,  # type: ignore[arg-type]
        scope=ArtifactScope(
            channel_id=channel_id,
            provenance=source_type,
            source_id=source_id,
        ),
    )


def test_dummy_manifest_plan_spec_is_generic_planner_and_verifier_contract():
    registry = CapabilityRegistry([dummy_manifest()])

    cards = planner_capability_cards(
        None,
        {"allowed_capability_ids": ["dummy.echo"]},
        registry=registry,
    )
    result = verify_plan_spec(
        plan_spec(),
        registry=registry,
        output_payload=reply_payload(),
        evidence_facts=[fact()],
    )

    assert [card["id"] for card in cards] == ["dummy.echo"]
    assert result.valid is True


def test_legacy_single_capability_plan_spec_shape_is_invalid():
    result = verify_plan_spec(
        {
            "plan_id": "legacy-plan",
            "selected_capability_id": "dummy.echo",
            "user_intent_summary": "legacy single capability shape",
            "domain_scope": {
                "channel_id": "channel-1",
                "channel_kind": "bank",
                "product_ids": [],
            },
            "required_artifacts": ["adapter_context"],
            "allowed_artifacts": ["adapter_context"],
            "forbidden_artifacts": [],
            "required_tools": ["dummy.echo"],
            "answerability_policy": "answer",
            "output_schema_ref": "dummy.echo:output_schema",
            "evidence_contract_ref": "dummy.echo:evidence_contract",
            "steps": [],
            "acceptance_criteria": [],
            "abstention_cases": [],
            "risk_flags": [],
        },
        registry=CapabilityRegistry([dummy_manifest()]),
        output_payload=reply_payload(),
        evidence_facts=[fact()],
    )

    assert result.valid is False
    assert result.issues[0].code == "plan_spec_invalid_schema"


def test_plan_spec_requiring_weekly_report_fails_when_only_material_pack_exists():
    spec = plan_spec(
        selected_capability_id="weekly_report.product_performance",
        required_artifacts=["weekly_report"],
        allowed_artifacts=["weekly_report"],
        forbidden_artifacts=["material_pack"],
        required_tools=["adapter_resolve.weekly_report", "adapter_report_scope"],
        output_schema_ref="weekly_report.product_performance:output_schema",
        evidence_contract_ref="weekly_report.product_performance:evidence_contract",
        steps=[
            {
                "step_id": "step-1",
                "description": "answer weekly report question",
                "uses_artifacts": ["weekly_report"],
                "required_artifacts": ["weekly_report"],
                "allowed_artifacts": ["weekly_report"],
                "forbidden_artifacts": ["material_pack"],
                "required_tools": ["adapter_resolve.weekly_report", "adapter_report_scope"],
            }
        ],
    )

    result = verify_plan_spec(
        spec,
        output_payload=reply_payload(),
        evidence_facts=[
            fact(
                "material_pack_resolvable",
                artifact_type="material_pack",
                source_id="material",
            )
        ],
    )

    assert result.valid is False
    assert "required_artifact_missing" in {issue.code for issue in result.issues}


def test_plan_spec_allow_history_false_fails_when_using_history_evidence():
    registry = CapabilityRegistry(
        [
            dummy_manifest(
                evidence_contract=EvidenceContract(
                    required_evidence_types=["recent_executed_action"],
                    allowed_source_types=["action_ledger"],
                    allowed_artifact_types=["history"],
                    minimum_evidence_count=1,
                    allow_history=False,
                ),
                required_artifacts=["history"],
                allowed_artifacts=["history"],
                forbidden_artifacts=[],
            )
        ]
    )
    spec = plan_spec(
        required_artifacts=["history"],
        allowed_artifacts=["history"],
        forbidden_artifacts=[],
        steps=[
            {
                "step_id": "step-1",
                "description": "answer from history",
                "uses_artifacts": ["history"],
                "required_artifacts": ["history"],
                "allowed_artifacts": ["history"],
                "forbidden_artifacts": [],
                "required_tools": ["dummy.echo"],
            }
        ],
    )

    result = verify_plan_spec(
        spec,
        registry=registry,
        output_payload=reply_payload(),
        evidence_facts=[
            fact(
                "recent_executed_action",
                source_type="action_ledger",
                artifact_type="history",
                source_id="ledger-1",
            )
        ],
        cited_evidence_ids=["ledger-1"],
    )

    assert result.valid is False
    assert "history_evidence_not_allowed" in {issue.code for issue in result.issues}


def test_plan_spec_rejects_output_with_evidence_from_wrong_material_pack_option():
    contract = EvidenceContract(
        required_evidence_types=["dummy_fact"],
        allowed_source_types=["adapter_resolve"],
        allowed_artifact_types=["adapter_context"],
        required_scope_match=["material_pack_option"],
        minimum_evidence_count=1,
    )
    spec = plan_spec(
        domain_scope={
            "channel_id": "channel-1",
            "channel_kind": "bank",
            "material_pack_option": "option-a",
            "product_ids": [],
        },
    )

    result = verify_plan_spec(
        spec,
        registry=CapabilityRegistry([dummy_manifest(evidence_contract=contract)]),
        output_payload=reply_payload(),
        evidence_facts=[fact(material_pack_option="option-b")],
        cited_evidence_ids=["source-1"],
    )

    assert result.valid is False
    assert "evidence_scope_mismatch" in {issue.code for issue in result.issues}


def test_plan_spec_accepts_missing_mechanical_contract_ref():
    spec = plan_spec(evidence_contract_ref=None, evidence_contract=None)

    result = verify_plan_spec(
        spec,
        registry=CapabilityRegistry([dummy_manifest()]),
        output_payload=reply_payload(),
        evidence_facts=[fact()],
        cited_evidence_ids=["source-1"],
    )

    assert result.valid is True


def test_plan_spec_ignores_mismatched_mechanical_contract_ref():
    spec = plan_spec(evidence_contract_ref="other.capability:evidence_contract")

    result = verify_plan_spec(
        spec,
        registry=CapabilityRegistry([dummy_manifest()]),
        output_payload=reply_payload(),
        evidence_facts=[fact()],
        cited_evidence_ids=["source-1"],
    )

    assert result.valid is True


def test_plan_spec_inline_contract_cannot_loosen_registry_contract():
    spec = plan_spec(evidence_contract_ref=None, evidence_contract=EvidenceContract())

    result = verify_plan_spec(
        spec,
        registry=CapabilityRegistry([dummy_manifest()]),
        output_payload=reply_payload(),
        evidence_facts=[],
    )

    assert result.valid is False
    assert "required_evidence_missing" in {issue.code for issue in result.issues}


def test_plan_spec_accepts_abstention_when_required_artifacts_are_missing():
    spec = plan_spec(
        selected_capability_id="weekly_report.product_performance",
        required_artifacts=["weekly_report"],
        allowed_artifacts=["weekly_report"],
        forbidden_artifacts=["material_pack"],
        required_tools=["adapter_resolve.weekly_report", "adapter_report_scope"],
        answerability_policy="abstain",
        output_schema_ref="weekly_report.product_performance:output_schema",
        evidence_contract_ref="weekly_report.product_performance:evidence_contract",
        steps=[
            {
                "step_id": "step-1",
                "description": "abstain on missing weekly report evidence",
                "uses_artifacts": [],
                "required_artifacts": ["weekly_report"],
                "allowed_artifacts": ["weekly_report"],
                "forbidden_artifacts": ["material_pack"],
                "required_tools": ["adapter_resolve.weekly_report", "adapter_report_scope"],
            }
        ],
    )

    result = verify_plan_spec(
        spec,
        output_payload=reply_payload(
            kind="unable_to_answer",
            text="老师，这个信息我这边暂时无法确认，先不回答避免信息不准确。",
        ),
        evidence_facts=[],
        abstained=True,
    )

    assert result.valid is True


def test_plan_spec_accepts_handoff_abstention_when_sales_mention_is_missing():
    spec = plan_spec(
        selected_capability_id="sales.handoff",
        user_intent_summary="route to sales support",
        required_artifacts=["adapter_context"],
        allowed_artifacts=["adapter_context"],
        forbidden_artifacts=["material_pack", "weekly_report", "monthly_report"],
        required_tools=["adapter_resolve.sales_mention"],
        answerability_policy="handoff",
        output_schema_ref="sales.handoff:output_schema",
        evidence_contract_ref="sales.handoff:evidence_contract",
        steps=[
            {
                "step_id": "step-1",
                "description": "handoff when sales mention is available",
                "uses_artifacts": [],
                "required_artifacts": ["adapter_context"],
                "allowed_artifacts": ["adapter_context"],
                "forbidden_artifacts": ["material_pack", "weekly_report", "monthly_report"],
                "required_tools": ["adapter_resolve.sales_mention"],
            }
        ],
    )

    result = verify_plan_spec(
        spec,
        output_payload=reply_payload(
            kind="unable_to_answer",
            text="当前渠道暂未配置可用负责人。",
        ),
        evidence_facts=[],
        abstained=True,
    )

    assert result.valid is True
