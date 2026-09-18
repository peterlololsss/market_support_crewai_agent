from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.runtime.context.stage_inputs import (
    AlignmentVerifierPromptInputSourceV1,
    AlignmentVerifierPromptInputV1,
    KnowledgeComposerPromptInputSourceV1,
    KnowledgeComposerPromptInputV1,
    PlannerPromptInputSourceV1,
    PlannerPromptInputV1,
    SanitizedAlignmentVerifierInputV1,
    SmalltalkComposerPromptInputSourceV1,
    SmalltalkComposerPromptInputV1,
    build_alignment_verifier_prompt_input_v1,
    build_knowledge_composer_prompt_input_v1,
    build_planner_prompt_input_v1,
    build_smalltalk_composer_prompt_input_v1,
)
from scripts.check_request_consumer_migration import PlannedSymbolInventoryV1
from tests.unit.llm._stage_input_fixtures import stage_sources


ROOT = Path(__file__).resolve().parents[3]
PLANNED_SYMBOLS = ROOT / "tests/fixtures/planned_symbol_ownership.v1.json"
STAGE_INPUT_PATH = "src/market_support_crewai_agent/runtime/context/stage_inputs.py"


def test_verifier_input_ownership_names_canonical_class_and_identity_alias() -> None:
    # Given: the sealed inventory rows owned by the strict stage-input slice.
    inventory = PlannedSymbolInventoryV1.model_validate_json(
        PLANNED_SYMBOLS.read_bytes()
    )

    # When: the verifier boundary rows are indexed by symbol.
    rows = {
        row.symbol: row
        for row in inventory.rows
        if row.path == STAGE_INPUT_PATH
        and row.symbol
        in {
            "AlignmentVerifierPromptInputV1",
            "SanitizedAlignmentVerifierInputV1",
        }
    }

    # Then: the public model is canonical and the sanitized name is its alias.
    assert set(rows) == {
        "AlignmentVerifierPromptInputV1",
        "SanitizedAlignmentVerifierInputV1",
    }
    assert rows["AlignmentVerifierPromptInputV1"].kind == "class"
    assert rows["SanitizedAlignmentVerifierInputV1"].kind == "alias"


def test_user_facing_stage_roles_have_exact_field_contracts() -> None:
    # Given: every user-facing prompt-input model.
    expected = {
        PlannerPromptInputV1: {
            "contract_version",
            "scene",
            "message",
            "history",
            "runtime_clock",
            "pending_clarification",
            "recent_executed_actions",
            "material_pack_options",
            "presentation",
            "business_scope",
            "effective_policy",
            "intent_gate",
            "eligible_capabilities",
            "recall",
            "retry_overlay",
        },
        KnowledgeComposerPromptInputV1: {
            "contract_version",
            "scene",
            "message",
            "history",
            "runtime_clock",
            "presentation",
            "validated_plan",
            "directive",
            "output_ceilings",
            "selected_capabilities",
            "preflight",
            "unit_groundings",
            "retry_overlay",
        },
        SmalltalkComposerPromptInputV1: {
            "contract_version",
            "scene",
            "message",
            "history",
            "presentation",
            "directive",
            "output_ceilings",
            "selected_capabilities",
            "guardrails",
            "retry_overlay",
        },
        AlignmentVerifierPromptInputV1: {
            "contract_version",
            "scene",
            "message",
            "history",
            "presentation",
            "validated_plan",
            "directive",
            "selected_capabilities",
            "unit_groundings",
            "candidate",
            "attempt",
        },
    }

    # When: each machine-consumed field set is inspected.
    actual = {model: set(model.model_fields) for model in expected}

    # Then: no role gains a fallback bag or another role's authority.
    assert actual == expected
    assert SanitizedAlignmentVerifierInputV1 is AlignmentVerifierPromptInputV1


def test_stage_builders_return_frozen_extra_forbid_models() -> None:
    # Given: valid typed projected sources for all four stages.
    planner_source, knowledge_source, smalltalk_source, verifier_source = (
        stage_sources()
    )

    # When: each role-specific builder constructs its strict DTO.
    inputs = (
        build_planner_prompt_input_v1(planner_source),
        build_knowledge_composer_prompt_input_v1(knowledge_source),
        build_smalltalk_composer_prompt_input_v1(smalltalk_source),
        build_alignment_verifier_prompt_input_v1(verifier_source),
    )

    # Then: all outputs are immutable and reject undeclared authority fields.
    for input_value in inputs:
        assert input_value.model_config.get("frozen") is True
        assert input_value.model_config.get("extra") == "forbid"
        payload = input_value.model_dump(mode="python")
        with pytest.raises(ValidationError):
            _ = type(input_value).model_validate(
                payload | {"identity": "principal:private"}
            )


def test_builder_sources_contain_only_projected_role_fields() -> None:
    # Given: the four concrete internal build-source value objects.
    source_types = (
        PlannerPromptInputSourceV1,
        KnowledgeComposerPromptInputSourceV1,
        SmalltalkComposerPromptInputSourceV1,
        AlignmentVerifierPromptInputSourceV1,
    )
    forbidden_type_fragments = {
        "KernelReplyRequest",
        "PolicyManifest",
        "DomainContext",
        "AdapterPreflightSnapshot",
        "ActionLedgerRecord",
        "dict",
    }
    source_values = stage_sources()

    # When: their declared field types are resolved without constructing data bags.
    declared_types = {
        source_type: " ".join(source_type.__annotations__.values())
        for source_type in source_types
    }

    # Then: every source is frozen/slot-backed and names only projected DTOs.
    for source_type, source_value in zip(
        source_types,
        source_values,
        strict=True,
    ):
        annotation_text = declared_types[source_type]
        assert is_dataclass(source_type)
        assert "__dict__" not in vars(source_type)
        assert not any(
            fragment in annotation_text for fragment in forbidden_type_fragments
        )
        with pytest.raises(FrozenInstanceError):
            setattr(source_value, "scene", "direct")


def test_smalltalk_and_verifier_have_no_flat_grounding_fallbacks() -> None:
    # Given: fields that would dissolve unit association or leak planner state.
    forbidden = {
        "runtime_clock",
        "validated_plan",
        "recall",
        "preflight",
        "unit_groundings",
        "allowed_evidence",
        "allowed_evidence_ids",
        "business_facts",
        "answerability",
        "effective_policy",
    }
    verifier_forbidden = {
        "allowed_evidence",
        "allowed_evidence_ids",
        "business_facts",
        "answerability",
        "prior_verdict",
        "runtime",
    }

    # When/Then: smalltalk has no knowledge inputs and verifier has no flat fallback.
    assert forbidden.isdisjoint(SmalltalkComposerPromptInputV1.model_fields)
    assert verifier_forbidden.isdisjoint(AlignmentVerifierPromptInputV1.model_fields)
