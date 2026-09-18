from __future__ import annotations

# pyright: reportAny=false

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src/market_support_crewai_agent/runtime/prompts/resources"
FIXTURES = ROOT / "tests/fixtures"


def test_program_and_agent_resources_are_byte_identical() -> None:
    # Given: the two versioned prompt governance resources.
    names = (
        "prompt_program_registry.v2.json",
        "prompt_agent_execution_specs.v1.json",
    )

    # When/Then: runtime and evidence bytes are identical.
    for name in names:
        assert (PACKAGE / name).read_bytes() == (FIXTURES / name).read_bytes()


def test_registry_freezes_eleven_scene_and_neutral_rows() -> None:
    # Given: the checked-in registry and execution specifications.
    registry = json.loads(
        (PACKAGE / "prompt_program_registry.v2.json").read_text(encoding="utf-8")
    )
    specs = json.loads(
        (PACKAGE / "prompt_agent_execution_specs.v1.json").read_text(encoding="utf-8")
    )

    # When/Then: all eight user-facing and three neutral rows are predeclared.
    assert len(registry["programs"]) == 11
    assert len(specs["specs"]) == 11
    assert len({row["program_id"] for row in registry["programs"]}) == 11
    assert {row["scene_key"] for row in registry["programs"]} == {
        "scene_neutral.v1",
        "wecom_direct.v1",
        "wecom_group.v1",
    }
    assert {row["status"] for row in registry["programs"]} == {"active"}
    assert {row["status"] for row in specs["specs"]} == {"active"}
    for row in registry["programs"]:
        assert row["sources"]
        precedence = [
            source
            for source in row["sources"]
            if source["purpose"] == "instruction_precedence"
        ]
        assert precedence
        if row["scene_key"] == "scene_neutral.v1":
            assert row["scene_contract_id"] is None
            assert row["scene_contract_version"] is None
            assert not any(
                source["purpose"] == "scene_presentation" for source in row["sources"]
            )
        else:
            assert row["scene_contract_id"].startswith("scene.wecom_")
            assert row["scene_contract_version"] == "2026-07-15.1"
            assert (
                sum(
                    source["purpose"] == "scene_presentation"
                    for source in row["sources"]
                )
                == 1
            )


def test_dead_image_asset_is_preserved_but_not_invoked() -> None:
    # Given: the preserved legacy image fragment and invocation inventory.
    asset = (
        ROOT
        / "src/market_support_crewai_agent/runtime/prompts/fragments/guardrail"
        / "image_alignment_verifier.md"
    )
    inventory = json.loads(
        (FIXTURES / "real_llm_invocations.v1.json").read_text(encoding="utf-8")
    )
    assets = json.loads(
        (FIXTURES / "baseline_preserved_prompt_assets.v1.json").read_text(
            encoding="utf-8"
        )
    )

    # When/Then: bytes remain present while no production invocation claims that stage.
    asset_row = next(
        row
        for row in assets["assets"]
        if row["path"].endswith("image_alignment_verifier.md")
    )
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == asset_row["sha256"]
    stage_kinds = {row["stage_kind"] for row in inventory["invocations"]}
    assert stage_kinds == {
        "alignment_verifier",
        "approved_knowledge_selector",
        "document_product_selector",
        "knowledge_composer",
        "llm_health_probe",
        "planner_intent",
        "smalltalk_composer",
    }
    assert "image_alignment_verifier" not in stage_kinds
