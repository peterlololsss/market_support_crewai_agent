from __future__ import annotations

import hashlib
from importlib.resources import files

import pytest

from market_support_crewai_agent.runtime.prompts.budgets import (
    PromptStaticBudgetExceededError,
    load_prompt_static_budgets,
)


def test_rejects_oversize_without_ratcheting() -> None:
    # Given: the first immutable package-owned prompt budget and its source hash.
    resource = files("market_support_crewai_agent.runtime.prompts.resources").joinpath(
        "prompt_static_budgets.v1.json"
    )
    before = hashlib.sha256(resource.read_bytes()).hexdigest()
    row = load_prompt_static_budgets().rows[0]

    # When/Then: the exact negative boundary rejects and source bytes never change.
    with pytest.raises(
        PromptStaticBudgetExceededError,
        match="prompt_static_budget_exceeded",
    ):
        row.assert_within_budget(row.allowed_max_bytes + 1)
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == before


def test_registration_rejects_missing_immutable_budget_row() -> None:
    # Given: an otherwise valid registry with one direct program row removed.
    registry = load_prompt_static_budgets()
    rows = tuple(
        row
        for row in registry.rows
        if not (
            row.program_id == "planner_intent.wecom_direct.v1@1"
            and row.model_family == "ds_v4pro"
        )
    )
    incomplete = registry.model_copy(update={"rows": rows})

    # When/Then: activation lookup fails instead of creating or ratcheting a row.
    with pytest.raises(
        PromptStaticBudgetExceededError,
        match="prompt_static_budget_row_missing",
    ):
        _ = incomplete.row_for(
            "planner_intent.wecom_direct.v1@1",
            "ds_v4pro",
            "wecom_direct.v1",
        )
