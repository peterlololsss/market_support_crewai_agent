from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from market_support_crewai_agent.schemas.base import StrictModel


class CapabilityRuntimeInputsV1(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["capability-runtime-inputs.v1"] = (
        "capability-runtime-inputs.v1"
    )
    distribution_name: str | None = None
    material_pack_options: tuple[str, ...] = ()
    evidence_query: str | None = Field(default=None, max_length=200)
    strategy_ids: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def _validate_canonical_values(self) -> CapabilityRuntimeInputsV1:
        if len(set(self.material_pack_options)) != len(self.material_pack_options):
            raise ValueError("duplicate_material_pack_option")
        if len(set(self.strategy_ids)) != len(self.strategy_ids):
            raise ValueError("duplicate_strategy_id")
        if any(not option or len(option) > 80 for option in self.material_pack_options):
            raise ValueError("invalid_material_pack_option")
        if any(
            not strategy_id or len(strategy_id) > 120
            for strategy_id in self.strategy_ids
        ):
            raise ValueError("invalid_strategy_id")
        return self


ManifestInputPathV1 = Literal[
    "runtime_inputs.distribution_name",
    "runtime_inputs.material_pack_options",
    "runtime_inputs.evidence_query",
    "runtime_inputs.strategy_ids",
]
