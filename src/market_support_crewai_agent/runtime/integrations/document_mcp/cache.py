from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Final

from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentMcpCacheConfigV1,
    DocumentQueryEvidenceCommandV1,
    EvidenceCommandCacheKeyV1,
)
from market_support_crewai_agent.runtime.hashing import (
    evidence_cache_key_hash,
    evidence_command_hash,
)

_CACHE_KEY_REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"^ck1:[0-9a-f]{64}$")


class DocumentCacheKeyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DocumentMcpCacheAuthorityV1:
    """Final-policy inputs required to derive a Document MCP cache key."""

    state_key_ref: str
    policy_id: str
    manifest_ref: str
    business_scope_hash: str
    source_cache_config: DocumentMcpCacheConfigV1

    def key_for_document_query(
        self,
        *,
        query: str,
        document_ids: tuple[str, ...],
        max_results: int,
    ) -> str:
        command = DocumentQueryEvidenceCommandV1(
            source_version=self.source_cache_config.client_contract_version,
            query=query,
            document_ids=document_ids,
            page=1,
            page_size=min(max_results, 50),
            max_results=max_results,
        )
        cache_key = EvidenceCommandCacheKeyV1(
            state_key_ref=self.state_key_ref,
            policy_id=self.policy_id,
            manifest_ref=self.manifest_ref,
            business_scope_hash=self.business_scope_hash,
            evidence_command_hash=evidence_command_hash(command),
            source_cache_config=self.source_cache_config,
        )
        return evidence_cache_key_hash(cache_key)


class TtlCache:
    """TTL storage addressed exclusively by a canonical `ck1` reference."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, cache_key_ref: str) -> Any | None:
        _require_cache_key_ref(cache_key_ref)
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(cache_key_ref)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= now:
                self._store.pop(cache_key_ref, None)
                return None
            return value

    def set(self, cache_key_ref: str, value: Any, ttl_seconds: float) -> None:
        _require_cache_key_ref(cache_key_ref)
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._store[cache_key_ref] = (time.monotonic() + ttl_seconds, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


DOCUMENT_CACHE = TtlCache()


def document_cache_get(cache_key_ref: str) -> Any | None:
    return DOCUMENT_CACHE.get(cache_key_ref)


def document_cache_set(cache_key_ref: str, value: Any, ttl_seconds: float) -> None:
    DOCUMENT_CACHE.set(cache_key_ref, value, ttl_seconds)


def _require_cache_key_ref(cache_key_ref: str) -> None:
    if _CACHE_KEY_REF_PATTERN.fullmatch(cache_key_ref) is None:
        raise DocumentCacheKeyError("document_cache_key_must_be_canonical_ck1")
