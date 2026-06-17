from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.domain.capabilities.adapters import (
    planner_capability_cards,
    verifier_manifest_contracts,
)
from market_support_crewai_agent.runtime.domain.capabilities.registry import (
    CapabilityManifest,
    CapabilityRegistry,
    EvidenceContract,
)
from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
)
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.validation.capability_manifest_verifier import (
    verify_capability_contracts,
)


def dummy_manifest(**overrides) -> CapabilityManifest:
    payload = {
        "id": "dummy.echo",
        "version": "test.1",
        "display_name": "Dummy echo",
        "description": "Dummy manifest used to prove registry-driven extension.",
        "capability_type": "answer",
        "domain_entities": ["channel", "artifact"],
        "required_inputs": ["request.dist_channel_name"],
        "optional_inputs": [],
        "required_artifacts": ["dummy_artifact"],
        "allowed_artifacts": ["dummy_artifact"],
        "forbidden_artifacts": ["weekly_report"],
        "required_tools": ["dummy_tool.echo"],
        "output_schema": {
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
                }
            },
        },
        "evidence_contract": {
            "required_fact_types": ["dummy_fact"],
            "allowed_source_types": ["adapter_resolve"],
            "forbidden_source_types": ["document_mcp"],
            "required_artifact_types": ["dummy_artifact"],
            "allowed_artifact_types": ["dummy_artifact"],
            "min_facts": 1,
        },
        "abstention_policy": {
            "requires_abstention_when_evidence_missing": True,
            "abstention_reply_kinds": ["unable_to_answer"],
            "guidance": "Abstain if dummy evidence is absent.",
        },
        "planner_guidance": "Planner may use this only when explicitly selected.",
        "agent_guidance": "Answer only from dummy evidence.",
        "verifier_checks": [
            "output_schema",
            "required_runtime_input_present",
            "required_evidence_present",
            "evidence_artifact_type_allowed",
            "forbidden_source_not_used",
            "abstention_correctness",
        ],
        "examples_positive": ["Use dummy evidence."],
        "examples_negative": ["Do not use reports."],
    }
    payload.update(overrides)
    return CapabilityManifest.model_validate(payload)


def test_builtin_registry_contains_required_capability_manifests():
    ids = {manifest.id for manifest in CAPABILITY_MANIFEST_REGISTRY.list()}

    assert {
        "material_pack.product_list",
        "material_pack.open_calendar",
        "weekly_report.product_performance",
        "monthly_report.product_performance",
        "channel.strategy_summary",
        "channel.product_summary",
    } <= ids
    for manifest in CAPABILITY_MANIFEST_REGISTRY.list():
        CAPABILITY_MANIFEST_REGISTRY.validateManifest(manifest)
        assert manifest.verifier_checks
        assert "output_schema" in manifest.verifier_checks
        assert manifest.planner_guidance
        assert manifest.agent_guidance


def test_registry_rejects_invalid_manifest_shape():
    with pytest.raises(ValueError, match="required_artifacts cannot also be forbidden"):
        dummy_manifest(forbidden_artifacts=["dummy_artifact"])

    with pytest.raises(ValueError, match="required_evidence_present requires"):
        dummy_manifest(
            evidence_contract=EvidenceContract(),
        )

    registry = CapabilityRegistry([dummy_manifest()])
    with pytest.raises(ValueError, match="duplicate capability manifest id"):
        registry.register(dummy_manifest())


def test_dummy_manifest_is_enough_for_planner_and_verifier_adapters():
    registry = CapabilityRegistry([dummy_manifest()])

    cards = planner_capability_cards(
        userRequest=None,
        runtimeContext={"allowed_capability_ids": ["dummy.echo"]},
        registry=registry,
    )

    assert [card["id"] for card in cards] == ["dummy.echo"]
    assert "required_evidence_present" in cards[0]["verifier_checks"]
    assert verifier_manifest_contracts(["dummy.echo"], registry) == [
        registry.get("dummy.echo").to_verifier_contract()
    ]

    facts = [
        EvidenceFact(
            fact_type="dummy_fact",  # type: ignore[arg-type]
            value=True,
            source_type="adapter_resolve",
            metadata={"artifact_type": "dummy_artifact"},
        )
    ]
    result = verify_capability_contracts(
        ["dummy.echo"],
        registry=registry,
        output_payload={"reply": {"kind": "answer", "text": "dummy answer"}},
        runtime_inputs={"request": {"dist_channel_name": "test channel"}},
        evidence_facts=facts,
    )

    assert result.valid is True


def test_manifest_verifier_uses_generic_primitives_for_missing_dummy_evidence():
    registry = CapabilityRegistry([dummy_manifest()])

    result = verify_capability_contracts(
        ["dummy.echo"],
        registry=registry,
        output_payload={"reply": {"kind": "answer", "text": "ungrounded"}},
        runtime_inputs={"request": {"dist_channel_name": "test channel"}},
        evidence_facts=[],
    )

    assert result.valid is False
    assert {issue.check for issue in result.issues} == {
        "required_evidence_present",
        "abstention_correctness",
    }
    assert all(issue.capability_id == "dummy.echo" for issue in result.issues)


def test_candidate_resolver_uses_explicit_runtime_context_not_message_keywords():
    registry = CapabilityRegistry([dummy_manifest()])

    cards = planner_capability_cards(
        userRequest={"message": "weekly monthly material dummy"},
        runtimeContext={"allowed_capability_ids": []},
        registry=registry,
    )

    assert [card["id"] for card in cards] == ["dummy.echo"]
