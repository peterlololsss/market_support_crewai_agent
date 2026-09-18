from __future__ import annotations

# pyright: reportAny=false

import hashlib
import json
from pathlib import Path

import pytest

from market_support_crewai_agent.runtime.prompts.budgets import (
    PromptStaticBudgetRegistryV1,
    load_prompt_static_budgets,
)
from scripts.check_request_consumer_migration import PlannedSymbolInventoryV1


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src/market_support_crewai_agent/runtime/prompts/resources"
FIXTURES = ROOT / "tests/fixtures"


def test_packaged_budget_resource_matches_immutable_fixture() -> None:
    # Given: package and evidence copies of the versioned budget registry.
    package_bytes = (PACKAGE / "prompt_static_budgets.v1.json").read_bytes()
    fixture_bytes = (FIXTURES / "prompt_static_budgets.v1.json").read_bytes()

    # When: the production loader parses its package-owned source.
    registry = load_prompt_static_budgets()

    # Then: both copies are byte-identical and cover all scene/neutral programs.
    assert package_bytes == fixture_bytes
    assert isinstance(registry, PromptStaticBudgetRegistryV1)
    assert {row.scene_key for row in registry.rows} == {
        "scene_neutral.v1",
        "wecom_direct.v1",
        "wecom_group.v1",
    }
    assert len(registry.rows) == 43


def test_budget_rejects_oversize_without_ratcheting() -> None:
    # Given: an immutable parsed budget row and its source digest.
    source = PACKAGE / "prompt_static_budgets.v1.json"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    row = load_prompt_static_budgets().rows[0]

    # When/Then: exact maximum passes and one extra byte fails.
    row.assert_within_budget(row.allowed_max_bytes)
    with pytest.raises(ValueError, match="prompt_static_budget_exceeded"):
        row.assert_within_budget(row.allowed_max_bytes + 1)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_budget_rows_use_frozen_formula() -> None:
    # Given: every current static budget row.
    payload = json.loads((PACKAGE / "prompt_static_budgets.v1.json").read_text())

    # When/Then: allowance is the exact baseline plus fixed formula, never observed drift.
    for row in payload["rows"]:
        baseline = row["baseline_bytes"]
        assert row["allowed_max_bytes"] == baseline + max(
            (baseline + 9) // 10,
            1024,
        )


def test_direct_stage_input_has_no_raw_request_todo_9() -> None:
    planned = PlannedSymbolInventoryV1.model_validate_json(
        (FIXTURES / "planned_symbol_ownership.v1.json").read_bytes()
    )
    row = next(row for row in planned.rows if row.symbol == "PlannerPromptInputV1")
    assert row.status == "migrated"
