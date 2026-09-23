from __future__ import annotations

from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityManifestV2,
)
from market_support_crewai_agent.runtime.policy.capabilities.registry import (
    CAPABILITY_MANIFEST_REGISTRY,
)


def capability_manifest_by_id(manifest_id: str) -> CapabilityManifestV2 | None:
    return CAPABILITY_MANIFEST_REGISTRY.find(manifest_id)


def capability_manifests() -> tuple[CapabilityManifestV2, ...]:
    return CAPABILITY_MANIFEST_REGISTRY.list()
