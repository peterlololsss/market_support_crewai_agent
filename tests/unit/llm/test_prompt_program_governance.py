from __future__ import annotations

from pathlib import Path

import pytest

from market_support_crewai_agent.runtime.context.stage_inputs import (
    build_planner_prompt_input_v1,
)
from market_support_crewai_agent.runtime.prompts import program_models
from market_support_crewai_agent.runtime.prompts.program_models import (
    PromptGovernanceError,
    PromptProgramV2,
    load_agent_execution_specs_v1,
    load_prompt_program_registry_v2,
)
from market_support_crewai_agent.runtime.prompts.router import (
    select_stage_input_prompt_program,
)
from tests.unit.llm._stage_input_fixtures import stage_sources


ROOT = Path(__file__).resolve().parents[3]


def test_packaged_program_registry_has_eleven_governed_rows_and_mirror_bytes() -> None:
    # Given: Todo13 sealed runtime resources and evidence mirrors.
    package = ROOT / "src/market_support_crewai_agent/runtime/prompts/resources"
    fixtures = ROOT / "tests/fixtures"

    # When: Todo14 parses the packaged program/spec registries.
    programs = load_prompt_program_registry_v2()
    specs = load_agent_execution_specs_v1()

    # Then: all seven stage kinds are covered by the exact eleven active rows.
    assert len(programs) == 11
    assert len(specs) == 11
    assert {program.stage for program in programs} == {
        "alignment_verifier",
        "approved_knowledge_selector",
        "document_product_selector",
        "knowledge_composer",
        "llm_health_probe",
        "planner_intent",
        "smalltalk_composer",
    }
    assert (package / "prompt_program_registry.v2.json").read_bytes() == (
        fixtures / "prompt_program_registry.v2.json"
    ).read_bytes()
    assert (package / "prompt_agent_execution_specs.v1.json").read_bytes() == (
        fixtures / "prompt_agent_execution_specs.v1.json"
    ).read_bytes()


def test_prompt_program_rejects_missing_precedence_before_activation() -> None:
    # Given: a valid packaged row with its instruction-precedence source removed.
    program = load_prompt_program_registry_v2()[0]
    sources = tuple(
        source
        for source in program.sources
        if source.purpose != "instruction_precedence"
    )

    # When/Then: the typed governance model fails before model I/O.
    with pytest.raises(ValueError, match="prompt_program_precedence_missing"):
        _ = PromptProgramV2.model_validate(
            program.model_dump(mode="json") | {"sources": sources}
        )


def test_prompt_program_requires_one_universal_precedence_source() -> None:
    # Given: each governed program carries the universal precedence source once.
    programs = load_prompt_program_registry_v2()

    # When: the packaged rows are inspected structurally, without prompt prose.
    counts = {
        program.program_id: sum(
            1
            for source in program.sources
            if source.fragment_id == "instruction.registered_over_untrusted_data.v1"
            and source.purpose == "instruction_precedence"
        )
        for program in programs
    }

    # Then: every provider-bound program has exactly one universal precedence source.
    assert counts
    assert set(counts.values()) == {1}


def test_select_stage_input_prompt_program_requires_packaged_v2_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: production prompt routing for a group planner but no packaged active V2 row.
    planner_source, _, _, _ = stage_sources()
    planner_input = build_planner_prompt_input_v1(planner_source)
    programs = tuple(
        program
        for program in load_prompt_program_registry_v2()
        if program.program_id != "planner_intent.wecom_group.v1@1"
    )
    monkeypatch.setattr(
        program_models,
        "load_prompt_program_registry_v2",
        lambda: programs,
    )

    # When/Then: the production selector fails before legacy topology can authorize it.
    with pytest.raises(
        PromptGovernanceError,
        match="prompt_program_v2_authority_missing",
    ):
        _ = select_stage_input_prompt_program(
            planner_input,
            "ds_v4pro",
        )
