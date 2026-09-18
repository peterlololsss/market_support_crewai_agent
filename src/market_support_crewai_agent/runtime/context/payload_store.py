from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from secrets import token_hex
from threading import RLock
from time import monotonic
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.schemas.base import StrictModel


class PayloadStoreMigrationError(RuntimeError):
    pass


class PayloadHandleV1(StrictModel):
    """Opaque, random payload reference valid only with its state key."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: Literal["payload-handle.v1"] = "payload-handle.v1"
    value: str = Field(pattern=r"^cpv1:[0-9a-f]{32}$")

    @classmethod
    def create(cls) -> "PayloadHandleV1":
        return cls(value=f"cpv1:{token_hex(16)}")


@dataclass(frozen=True, slots=True)
class ScopedPayloadV1:
    payload: JsonValue
    metadata: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _StoredPayloadV1:
    payload: JsonValue
    metadata: Mapping[str, JsonValue]
    expires_at_monotonic: float


class ScopedContextPayloadStoreV1:
    """Bounded in-process payload store keyed by full conversation identity."""

    def __init__(
        self,
        *,
        conversation_ttl_seconds: float,
        direct_audit_ttl_seconds: float,
        max_payloads: int = 128,
        monotonic_clock: Callable[[], float] = monotonic,
    ) -> None:
        if conversation_ttl_seconds <= 0:
            raise PayloadStoreMigrationError(
                "conversation_ttl_seconds_must_be_positive"
            )
        if direct_audit_ttl_seconds <= 0:
            raise PayloadStoreMigrationError(
                "direct_audit_ttl_seconds_must_be_positive"
            )
        if max_payloads <= 0:
            raise PayloadStoreMigrationError("max_payloads_must_be_positive")
        self._ttl_seconds = min(conversation_ttl_seconds, direct_audit_ttl_seconds)
        self._max_payloads = max_payloads
        self._monotonic_clock = monotonic_clock
        self._payloads: OrderedDict[
            tuple[ConversationStateKey, PayloadHandleV1], _StoredPayloadV1
        ] = OrderedDict()
        self._lock = RLock()

    def put(
        self,
        state_key: ConversationStateKey,
        payload: JsonValue,
        *,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> PayloadHandleV1:
        now = self._monotonic_clock()
        with self._lock:
            self._remove_expired(now)
            handle = PayloadHandleV1.create()
            storage_key = (state_key, handle)
            while storage_key in self._payloads:
                handle = PayloadHandleV1.create()
                storage_key = (state_key, handle)
            self._payloads[storage_key] = _StoredPayloadV1(
                payload=deepcopy(payload),
                metadata=deepcopy(dict(metadata or {})),
                expires_at_monotonic=now + self._ttl_seconds,
            )
            self._trim_to_capacity()
            return handle

    def get(
        self,
        state_key: ConversationStateKey,
        handle: PayloadHandleV1,
    ) -> ScopedPayloadV1 | None:
        now = self._monotonic_clock()
        with self._lock:
            self._remove_expired(now)
            stored = self._payloads.get((state_key, handle))
            if stored is None:
                return None
            return ScopedPayloadV1(
                payload=deepcopy(stored.payload),
                metadata=deepcopy(dict(stored.metadata)),
            )

    def delete(self, state_key: ConversationStateKey, handle: PayloadHandleV1) -> bool:
        with self._lock:
            self._remove_expired(self._monotonic_clock())
            return self._payloads.pop((state_key, handle), None) is not None

    def cleanup(self) -> int:
        with self._lock:
            return self._remove_expired(self._monotonic_clock())

    def count(self) -> int:
        with self._lock:
            self._remove_expired(self._monotonic_clock())
            return len(self._payloads)

    def clear(self) -> None:
        with self._lock:
            self._payloads.clear()

    def _remove_expired(self, now: float) -> int:
        expired_keys = [
            storage_key
            for storage_key, stored in self._payloads.items()
            if stored.expires_at_monotonic <= now
        ]
        for storage_key in expired_keys:
            del self._payloads[storage_key]
        return len(expired_keys)

    def _trim_to_capacity(self) -> None:
        while len(self._payloads) > self._max_payloads:
            self._payloads.popitem(last=False)


class ContextPayloadStore:
    def __init__(self, *args: JsonValue, **kwargs: JsonValue) -> None:
        del args, kwargs
        raise PayloadStoreMigrationError("unscoped_context_payload_store_removed")
