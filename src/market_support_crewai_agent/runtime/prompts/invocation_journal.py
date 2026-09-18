from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar, Token
from typing import ClassVar, Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator

from market_support_crewai_agent.runtime.hashing import hash_canonical_model
from market_support_crewai_agent.runtime.prompts.profiles import SceneKeyV1
from market_support_crewai_agent.runtime.prompts.program_models import (
    ProviderTargetIdentityV1,
    StageKindV1,
    TargetSlotV1,
)
from market_support_crewai_agent.schemas.base import StrictModel

JournalStatusV1 = Literal[
    "reserved",
    "success",
    "output_contract_error",
    "transport_error",
    "uncaptured_network_error",
]


class InvocationJournalError(ValueError):
    pass


class _FrozenJournalModel(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class TurnLlmInvocationRowV1(_FrozenJournalModel):
    contract_version: Literal["turn-llm-invocation-row.v1"] = (
        "turn-llm-invocation-row.v1"
    )
    invocation_ordinal: int = Field(ge=1, le=18)
    logical_attempt: int = Field(ge=1, le=18)
    transport_attempt: Literal[1] = 1
    stage_kind: StageKindV1
    program_id: str = Field(min_length=1, max_length=180)
    program_version: str | None = Field(default=None, min_length=1, max_length=40)
    target_slot: TargetSlotV1
    scene_key: SceneKeyV1 | None = None
    scene_contract_ref: str | None = Field(default=None, max_length=240)
    purpose: str = Field(min_length=1, max_length=80)
    status: JournalStatusV1
    osh1: str | None = Field(default=None, pattern=r"^osh1:[0-9a-f]{64}$")
    poh1: str | None = Field(default=None, pattern=r"^poh1:[0-9a-f]{64}$")
    hph1: str | None = Field(default=None, pattern=r"^hph1:[0-9a-f]{64}$")
    prh1: str | None = Field(default=None, pattern=r"^prh1:[0-9a-f]{64}$")
    input_digest: str | None = Field(default=None, pattern=r"^mch1:[0-9a-f]{64}$")
    output_digest: str | None = Field(default=None, pattern=r"^out1:[0-9a-f]{64}$")
    error_code: str | None = Field(default=None, max_length=80)
    model_family: str | None = Field(default=None, max_length=80)
    provider_id: str | None = Field(default=None, max_length=80)
    transport_id: str | None = Field(default=None, max_length=80)
    input_schema_version: str | None = Field(default=None, max_length=160)
    output_schema_version: str | None = Field(default=None, max_length=160)
    latency_ms: int = Field(default=0, ge=0)
    input_bytes: int = Field(default=0, ge=0)
    output_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_status_fields(self) -> Self:
        match self.status:
            case "success":
                if (
                    self.output_digest is None
                    or self.output_bytes is None
                    or self.error_code is not None
                ):
                    raise InvocationJournalError("journal_success_capture_invalid")
            case (
                "reserved"
                | "output_contract_error"
                | "transport_error"
                | "uncaptured_network_error"
            ):
                if self.status != "reserved" and self.error_code is None:
                    raise InvocationJournalError("journal_error_code_required")
            case unreachable:
                raise AssertionError(f"unreachable journal status: {unreachable}")
        if self.status != "reserved" and any(
            value is None
            for value in (
                self.program_version,
                self.scene_key,
                self.hph1,
                self.input_digest,
                self.model_family,
                self.provider_id,
                self.transport_id,
                self.input_schema_version,
                self.output_schema_version,
            )
        ):
            raise InvocationJournalError("journal_transport_metadata_required")
        if self.status not in {"reserved", "success"} and (
            self.output_digest is not None or self.output_bytes is not None
        ):
            raise InvocationJournalError("journal_error_capture_forbidden")
        if self.scene_key == "scene_neutral.v1":
            if self.scene_contract_ref is not None:
                raise InvocationJournalError("journal_neutral_scene_contract_forbidden")
        elif self.status != "reserved" and self.scene_contract_ref is None:
            raise InvocationJournalError("journal_scene_contract_required")
        return self


class TurnLlmInvocationJournalV1:
    def __init__(self, *, max_rows: int = 18) -> None:
        if max_rows < 1 or max_rows > 18:
            raise InvocationJournalError("llm_invocation_journal_capacity_invalid")
        self.max_rows = max_rows
        self._rows: list[TurnLlmInvocationRowV1] = []

    @property
    def rows(self) -> tuple[TurnLlmInvocationRowV1, ...]:
        return tuple(self._rows)

    def reserve(
        self,
        *,
        stage_kind: StageKindV1,
        program_id: str,
        target_slot: TargetSlotV1,
        purpose: str,
        logical_attempt: int | None = None,
    ) -> TurnLlmInvocationRowV1:
        ordinal = len(self._rows) + 1
        attempt = ordinal if logical_attempt is None else logical_attempt
        if ordinal > self.max_rows:
            raise InvocationJournalError("llm_invocation_journal_capacity_exceeded")
        row = TurnLlmInvocationRowV1(
            invocation_ordinal=ordinal,
            logical_attempt=attempt,
            stage_kind=stage_kind,
            program_id=program_id,
            target_slot=target_slot,
            purpose=purpose,
            status="reserved",
        )
        self._rows.append(row)
        return row

    def replace_reserved(
        self,
        ordinal: int,
        replacement: TurnLlmInvocationRowV1,
    ) -> None:
        if ordinal < 1 or ordinal > len(self._rows):
            raise InvocationJournalError("llm_invocation_journal_ordinal_invalid")
        current = self._rows[ordinal - 1]
        if current.status != "reserved":
            raise InvocationJournalError("llm_invocation_journal_row_closed")
        if replacement.invocation_ordinal != ordinal:
            raise InvocationJournalError("llm_invocation_journal_ordinal_mismatch")
        self._rows[ordinal - 1] = replacement


_CURRENT_TURN_JOURNAL: ContextVar[TurnLlmInvocationJournalV1 | None] = ContextVar(
    "market_agent_turn_llm_invocation_journal",
    default=None,
)


def use_turn_llm_invocation_journal(
    journal: TurnLlmInvocationJournalV1,
) -> _TurnLlmInvocationJournalScopeV1:
    return _TurnLlmInvocationJournalScopeV1(journal)


class _TurnLlmInvocationJournalScopeV1:
    def __init__(self, journal: TurnLlmInvocationJournalV1) -> None:
        self._journal = journal
        self._token: Token[TurnLlmInvocationJournalV1 | None] | None = None

    def __enter__(self) -> TurnLlmInvocationJournalV1:
        active = _CURRENT_TURN_JOURNAL.get()
        if active is not None and active is not self._journal:
            raise InvocationJournalError("turn_llm_invocation_journal_conflict")
        self._token = _CURRENT_TURN_JOURNAL.set(self._journal)
        return self._journal

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        if self._token is None:
            raise InvocationJournalError(
                "turn_llm_invocation_journal_scope_not_entered"
            )
        _CURRENT_TURN_JOURNAL.reset(self._token)
        self._token = None
        return False


def current_turn_llm_invocation_journal() -> TurnLlmInvocationJournalV1 | None:
    return _CURRENT_TURN_JOURNAL.get()


def resolve_turn_llm_invocation_journal(
    explicit: TurnLlmInvocationJournalV1 | None,
) -> TurnLlmInvocationJournalV1 | None:
    active = current_turn_llm_invocation_journal()
    if explicit is not None and active is not None and explicit is not active:
        raise InvocationJournalError("turn_llm_invocation_journal_conflict")
    return explicit or active


class TurnSelectorOutcomeCacheV1:
    max_entries: int = 4

    def __init__(self) -> None:
        self._entries: OrderedDict[str, JsonValue] = OrderedDict()

    def get(self, sih1: str) -> JsonValue | None:
        value = self._entries.get(sih1)
        if value is None:
            return None
        self._entries.move_to_end(sih1)
        return value

    def put(self, sih1: str, outcome: JsonValue) -> None:
        self._entries[sih1] = outcome
        self._entries.move_to_end(sih1)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._entries)


class SelectorCacheIdentityV1(_FrozenJournalModel):
    contract_version: Literal["selector-cache-identity.v1"] = (
        "selector-cache-identity.v1"
    )
    selector_input_json: str = Field(min_length=2, max_length=2_000_000)
    program_id: str = Field(min_length=1, max_length=180)
    program_version: str = Field(min_length=1, max_length=40)
    provider_target: ProviderTargetIdentityV1


_CURRENT_TURN_SELECTOR_CACHE: ContextVar[TurnSelectorOutcomeCacheV1 | None] = (
    ContextVar(
        "market_agent_turn_selector_outcome_cache",
        default=None,
    )
)


class _TurnSelectorOutcomeCacheScopeV1:
    def __init__(self, cache: TurnSelectorOutcomeCacheV1) -> None:
        self._cache = cache
        self._token: Token[TurnSelectorOutcomeCacheV1 | None] | None = None

    def __enter__(self) -> TurnSelectorOutcomeCacheV1:
        active = _CURRENT_TURN_SELECTOR_CACHE.get()
        if active is not None and active is not self._cache:
            raise InvocationJournalError("turn_selector_outcome_cache_conflict")
        self._token = _CURRENT_TURN_SELECTOR_CACHE.set(self._cache)
        return self._cache

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        if self._token is None:
            raise InvocationJournalError(
                "turn_selector_outcome_cache_scope_not_entered"
            )
        _CURRENT_TURN_SELECTOR_CACHE.reset(self._token)
        self._token = None
        return False


def use_turn_selector_outcome_cache(
    cache: TurnSelectorOutcomeCacheV1,
) -> _TurnSelectorOutcomeCacheScopeV1:
    return _TurnSelectorOutcomeCacheScopeV1(cache)


def resolve_turn_selector_outcome_cache(
    explicit: TurnSelectorOutcomeCacheV1 | None,
) -> TurnSelectorOutcomeCacheV1 | None:
    active = _CURRENT_TURN_SELECTOR_CACHE.get()
    if explicit is not None and active is not None and explicit is not active:
        raise InvocationJournalError("turn_selector_outcome_cache_conflict")
    return explicit or active


def selector_input_hash(identity: SelectorCacheIdentityV1) -> str:
    return hash_canonical_model("turn-selector-input.v1", identity, prefix="sih1")


def journal_row_hash(row: TurnLlmInvocationRowV1) -> str:
    return hash_canonical_model("turn-llm-invocation-row.v1", row, prefix="tlj1")
