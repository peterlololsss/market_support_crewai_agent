from __future__ import annotations

import hashlib
from importlib.resources import files
from typing import Final, TypeAlias

import pytest

import market_support_crewai_agent.runtime.hashing as hashing
import market_support_crewai_agent.runtime.prompts.assembler as assembler
import market_support_crewai_agent.runtime.prompts.registry as registry
import market_support_crewai_agent.runtime.prompts.router as router
from market_support_crewai_agent.runtime.prompts.profiles import (
    ModelFamily,
    PromptScene,
    UserFacingPromptStage,
)
from market_support_crewai_agent.runtime.prompts.registry import (
    PresentationField,
    PresentationRuleId,
)


SceneRow: TypeAlias = tuple[
    UserFacingPromptStage,
    PromptScene,
    str,
    str,
    tuple[PresentationField, ...],
    tuple[PresentationRuleId, ...],
]
SCENE_ROWS: Final[tuple[SceneRow, ...]] = (
    (
        "planner_intent",
        "group",
        "scene.wecom_group.planner_intent.v1",
        "scene/wecom_group/planner_intent.md",
        ("conversation_name", "principal_name"),
        (
            "address_group_audience",
            "use_optional_principal_name",
            "allow_current_conversation_label",
        ),
    ),
    (
        "planner_intent",
        "direct",
        "scene.wecom_direct.planner_intent.v1",
        "scene/wecom_direct/planner_intent.md",
        ("principal_name",),
        (
            "address_individual",
            "use_optional_principal_name",
            "forbid_group_addressing",
        ),
    ),
    (
        "knowledge_composer",
        "group",
        "scene.wecom_group.knowledge_composer.v1",
        "scene/wecom_group/knowledge_composer.md",
        ("conversation_name", "principal_name"),
        (
            "address_group_audience",
            "use_optional_principal_name",
            "allow_current_conversation_label",
        ),
    ),
    (
        "knowledge_composer",
        "direct",
        "scene.wecom_direct.knowledge_composer.v1",
        "scene/wecom_direct/knowledge_composer.md",
        ("principal_name",),
        (
            "address_individual",
            "use_optional_principal_name",
            "forbid_group_addressing",
        ),
    ),
    (
        "smalltalk_composer",
        "group",
        "scene.wecom_group.smalltalk_composer.v1",
        "scene/wecom_group/smalltalk_composer.md",
        ("conversation_name", "principal_name"),
        (
            "address_group_audience",
            "use_optional_principal_name",
            "allow_current_conversation_label",
        ),
    ),
    (
        "smalltalk_composer",
        "direct",
        "scene.wecom_direct.smalltalk_composer.v1",
        "scene/wecom_direct/smalltalk_composer.md",
        ("principal_name",),
        (
            "address_individual",
            "use_optional_principal_name",
            "forbid_group_addressing",
        ),
    ),
    (
        "alignment_verifier",
        "group",
        "scene.wecom_group.alignment_verifier.v1",
        "scene/wecom_group/alignment_verifier.md",
        ("conversation_name", "principal_name"),
        (
            "address_group_audience",
            "use_optional_principal_name",
            "allow_current_conversation_label",
        ),
    ),
    (
        "alignment_verifier",
        "direct",
        "scene.wecom_direct.alignment_verifier.v1",
        "scene/wecom_direct/alignment_verifier.md",
        ("principal_name",),
        (
            "address_individual",
            "use_optional_principal_name",
            "forbid_group_addressing",
        ),
    ),
)
MODEL_FAMILIES: Final[tuple[ModelFamily, ...]] = (
    "ds_v4pro",
    "deepseek",
    "gpt",
    "claude",
    "generic",
)


def test_scene_contract_registry_has_the_eight_frozen_rows() -> None:
    # Given: the immutable scene presentation registrations.
    contracts = registry.SCENE_PRESENTATION_CONTRACTS

    # When: their machine-consumed fields are projected.
    actual = tuple(
        (
            row.stage,
            row.scene,
            row.contract_id,
            registry.fragment_by_id(row.fragment_id, row.stage).template_name,
            row.allowed_presentation_fields,
            row.presentation_rule_ids,
        )
        for row in contracts
    )

    # Then: every user-facing stage has one distinct group and direct contract.
    assert actual == SCENE_ROWS
    assert {row.version for row in contracts} == {"2026-07-15.1"}


def test_user_facing_router_selects_one_scene_and_one_precedence_source() -> None:
    # Given: every user-facing stage, scene, and supported model family.
    selector = router.user_facing_fragment_ids

    # When/Then: topology selects one scene fragment and model family selects one source.
    for stage, scene, scene_id, *_ in SCENE_ROWS:
        for model_family in MODEL_FAMILIES:
            fragment_ids = selector(stage, model_family, scene)
            scene_ids = tuple(
                value for value in fragment_ids if value.startswith("scene.")
            )
            precedence_ids = tuple(
                value
                for value in fragment_ids
                if value
                in {
                    "instruction.registered_over_untrusted_data.v1",
                    "model.generic.structured",
                }
            )
            expected_precedence_ids = (
                ("instruction.registered_over_untrusted_data.v1",)
                if model_family == "ds_v4pro"
                else ("model.generic.structured",)
            )
            assert scene_ids == (scene_id,)
            assert precedence_ids == expected_precedence_ids


def test_precedence_fragment_has_frozen_bytes_and_framed_hash() -> None:
    # Given: the registered immutable instruction-precedence resource.
    fragment = registry.fragment_by_id(
        "instruction.registered_over_untrusted_data.v1",
        "planner_intent",
    )
    payload = (
        files(registry.PROMPT_FRAGMENT_PACKAGE)
        .joinpath(fragment.template_name)
        .read_bytes()
    )

    # When: the Todo13 fragment hash is calculated.
    digest = hashing.frh1(payload.decode("utf-8"))

    assert fragment.layer == "stable"
    assert fragment.priority == 25
    assert payload.endswith(b"\n")
    assert (
        digest
        == "frh1:" + hashlib.sha256(b"prompt-fragment.v1\0" + payload).hexdigest()
    )


def test_dead_image_prompt_is_preserved_but_unregistered() -> None:
    # Given: the baseline-preserved image prompt asset.
    asset = files(registry.PROMPT_FRAGMENT_PACKAGE).joinpath(
        "guardrail/image_alignment_verifier.md"
    )

    # When/Then: bytes remain on disk while no active runtime helper or ID reaches it.
    assert asset.is_file()
    assert (
        "guardrail.image_alignment_verifier"
        not in registry.PROMPT_REGISTRY.prompt_ids()
    )
    assert (
        "agent.image_alignment_verifier"
        not in registry.PROMPT_REGISTRY.agent_spec_ids()
    )
    assert not hasattr(assembler, "assembleGuardrailPrompt")


@pytest.mark.parametrize(
    ("payload", "error_code"),
    (
        ("scene bytes without final lf", "scene_fragment_final_lf_required"),
        ("outbound_actions\n", "scene_fragment_authority_forbidden"),
        ("safe scene bytes\n", "scene_fragment_bytes_not_distinct"),
    ),
)
def test_scene_registry_rejects_invalid_fragment_bytes(
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
    error_code: str,
) -> None:
    # Given: one invalid byte-level scene resource projection.
    def load_invalid_fragment(_fragment: registry.PromptFragment) -> str:
        return payload

    monkeypatch.setattr(
        registry,
        "load_prompt_fragment_text",
        load_invalid_fragment,
    )

    # When/Then: registration fails before the program becomes reachable.
    with pytest.raises(registry.PromptRegistryInvariantError, match=error_code):
        _ = registry.PromptRegistry()


def test_scene_registry_rejects_missing_or_duplicate_scene_rows() -> None:
    # Given: incomplete and duplicate variants of the immutable scene table.
    contracts = registry.SCENE_PRESENTATION_CONTRACTS
    duplicate = (*contracts[:-1], contracts[0])

    # When/Then: both cardinality failures are structural registration errors.
    with pytest.raises(
        registry.PromptRegistryInvariantError,
        match="scene_contract_count_invalid",
    ):
        _ = registry.PromptRegistry(scene_contracts=contracts[:-1])
    with pytest.raises(
        registry.PromptRegistryInvariantError,
        match="scene_contract_key_duplicate",
    ):
        _ = registry.PromptRegistry(scene_contracts=duplicate)


def test_scene_registry_rejects_wrong_contract_identity() -> None:
    # Given: one scene row whose contract ID does not match its stage and scene.
    contracts = registry.SCENE_PRESENTATION_CONTRACTS
    invalid = contracts[0].model_copy(update={"contract_id": "scene.invalid.v1"})

    # When/Then: identity mismatch fails before fragment assembly.
    with pytest.raises(
        registry.PromptRegistryInvariantError,
        match="scene_contract_id_mismatch",
    ):
        _ = registry.PromptRegistry(scene_contracts=(invalid, *contracts[1:]))
