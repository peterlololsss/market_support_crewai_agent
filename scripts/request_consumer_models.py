from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, NamedTuple, Self, override

from pydantic import BaseModel, ConfigDict, Field, model_validator


Status = Literal["legacy", "planned_boundary", "migrated", "retained_boundary"]
ConsumerCountKey = tuple[str, str, str, int, str, Status]


class InventoryContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceScanError(InventoryContractError):
    path: str
    line: int | None
    reason: str

    @override
    def __str__(self) -> str:
        line = "unknown" if self.line is None else str(self.line)
        return f"scan_source_parse_error:{self.path}:{line}:{self.reason}"


class ConsumerRowV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    path: str
    symbol: str
    match_kind: str
    expected_count: int = Field(ge=1)
    target_todo: int = Field(ge=1, le=16)
    target_type: str
    status: Status


class ConsumerInventoryV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["request-consumer-migration.v1"]
    rows: tuple[ConsumerRowV1, ...]

    @model_validator(mode="after")
    def validate_unique_rows(self) -> Self:
        keys = [(row.path, row.symbol, row.match_kind) for row in self.rows]
        if len(keys) != len(set(keys)):
            raise InventoryContractError("duplicate_consumer_inventory_row")
        return self


class PlannedSymbolRowV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    path: str
    target_todo: int = Field(ge=1, le=16)
    kind: Literal["class", "alias", "function", "method", "export"]
    expected_count: int = Field(ge=1)
    status: Status


class PlannedSymbolInventoryV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["planned-symbol-ownership.v1"]
    rows: tuple[PlannedSymbolRowV1, ...]

    @model_validator(mode="after")
    def validate_unique_symbols(self) -> Self:
        keys = [(row.path, row.symbol) for row in self.rows]
        if len(keys) != len(set(keys)):
            raise InventoryContractError("duplicate_planned_symbol_row")
        return self


class GitStatusEntryV1(NamedTuple):
    record_kind: str
    index_status: str
    worktree_status: str
    path: str
    original_path: str | None


class GitBaselineV1(NamedTuple):
    head: str | None
    status: bytes
    rows: tuple[GitStatusEntryV1, ...]
