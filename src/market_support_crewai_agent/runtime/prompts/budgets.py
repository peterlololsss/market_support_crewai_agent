from __future__ import annotations

from importlib.resources import files
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


SceneKeyV1 = Literal[
    "wecom_group.v1",
    "wecom_direct.v1",
    "scene_neutral.v1",
]


class PromptStaticBudgetExceededError(ValueError):
    pass


class PromptStaticBudgetV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    program_id: str = Field(min_length=1, max_length=160)
    stage: str = Field(min_length=1, max_length=80)
    model_family: str = Field(min_length=1, max_length=80)
    scene_key: SceneKeyV1
    baseline_bytes: int = Field(ge=0)
    allowed_max_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_frozen_allowance(self) -> Self:
        expected = self.baseline_bytes + max(
            (self.baseline_bytes + 9) // 10,
            1024,
        )
        if self.allowed_max_bytes != expected:
            raise PromptStaticBudgetExceededError(
                "prompt_static_budget_formula_mismatch"
            )
        return self

    def assert_within_budget(self, actual_bytes: int) -> None:
        if actual_bytes > self.allowed_max_bytes:
            raise PromptStaticBudgetExceededError("prompt_static_budget_exceeded")


class PromptStaticBudgetRegistryV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["prompt-static-budgets.v1"]
    rows: tuple[PromptStaticBudgetV1, ...]

    @model_validator(mode="after")
    def validate_unique_keys(self) -> Self:
        keys = [(row.program_id, row.model_family, row.scene_key) for row in self.rows]
        if len(keys) != len(set(keys)):
            raise PromptStaticBudgetExceededError("duplicate_prompt_static_budget_key")
        return self

    def row_for(
        self,
        program_id: str,
        model_family: str,
        scene_key: SceneKeyV1,
    ) -> PromptStaticBudgetV1:
        matches = tuple(
            row
            for row in self.rows
            if row.program_id == program_id
            and row.model_family == model_family
            and row.scene_key == scene_key
        )
        if len(matches) != 1:
            raise PromptStaticBudgetExceededError("prompt_static_budget_row_missing")
        return matches[0]


def load_prompt_static_budgets() -> PromptStaticBudgetRegistryV1:
    resource = files("market_support_crewai_agent.runtime.prompts.resources").joinpath(
        "prompt_static_budgets.v1.json"
    )
    return PromptStaticBudgetRegistryV1.model_validate_json(resource.read_bytes())
