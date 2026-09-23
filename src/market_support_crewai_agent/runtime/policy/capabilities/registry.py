from __future__ import annotations

from collections.abc import Iterable

from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityManifestIdV2,
    CapabilityManifestV2,
)
from market_support_crewai_agent.runtime.policy.capabilities.manifests import (
    BUILTIN_CAPABILITY_MANIFESTS,
    SEALED_MANIFEST_CONTENT_HASHES,
    SEALED_MANIFEST_ORDER,
    capability_manifest_content_hash_v2,
)


class CapabilityRegistryV2:
    def __init__(self, manifests: Iterable[CapabilityManifestV2]) -> None:
        ordered = tuple(manifests)
        by_id = {manifest.manifest_id: manifest for manifest in ordered}
        if len(ordered) != 13 or len(by_id) != len(ordered):
            raise ValueError("capability registry requires thirteen unique manifests")
        if tuple(manifest.manifest_id for manifest in ordered) != SEALED_MANIFEST_ORDER:
            raise ValueError("capability registry manifest order mismatch")
        if set(by_id) != set(SEALED_MANIFEST_CONTENT_HASHES):
            raise ValueError("capability registry manifest set mismatch")
        for manifest in ordered:
            content_hash = capability_manifest_content_hash_v2(manifest)
            if content_hash != manifest.expected_content_hash:
                raise ValueError("capability manifest semantic hash mismatch")
            if content_hash != SEALED_MANIFEST_CONTENT_HASHES[manifest.manifest_id]:
                raise ValueError("capability manifest same-version content drift")
        self._ordered = ordered
        self._by_id = by_id

    def get(self, manifest_id: CapabilityManifestIdV2) -> CapabilityManifestV2:
        try:
            return self._by_id[manifest_id]
        except KeyError as exc:
            raise KeyError(f"unknown capability manifest id: {manifest_id}") from exc

    def find(self, manifest_id: str) -> CapabilityManifestV2 | None:
        for manifest in self._ordered:
            if manifest.manifest_id == manifest_id:
                return manifest
        return None

    def list(self) -> tuple[CapabilityManifestV2, ...]:
        return self._ordered


CAPABILITY_MANIFEST_REGISTRY = CapabilityRegistryV2(BUILTIN_CAPABILITY_MANIFESTS)
