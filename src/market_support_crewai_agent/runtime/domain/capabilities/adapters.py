from __future__ import annotations

from collections.abc import Mapping

from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    CapabilityManifest,
    CapabilityRegistry,
)


def planner_capability_cards(
    userRequest: object | None,
    runtimeContext: Mapping[str, object] | object | None = None,
    registry: CapabilityRegistry | None = None,
) -> list[dict[str, object]]:
    """Return compact registry cards for planner prompt/context assembly."""
    selected_registry = registry or CAPABILITY_MANIFEST_REGISTRY
    return [
        manifest.to_planner_card()
        for manifest in selected_registry.resolveCandidateCapabilities(
            userRequest,
            runtimeContext,
        )
    ]


def verifier_manifest_contracts(
    manifest_ids: list[str] | tuple[str, ...],
    registry: CapabilityRegistry | None = None,
) -> list[dict[str, object]]:
    return [
        manifest.to_verifier_contract()
        for manifest in manifests_for_manifest_ids(manifest_ids, registry)
    ]


def manifests_for_manifest_ids(
    manifest_ids: list[str] | tuple[str, ...],
    registry: CapabilityRegistry | None = None,
) -> tuple[CapabilityManifest, ...]:
    selected_registry = registry or CAPABILITY_MANIFEST_REGISTRY
    manifests: list[CapabilityManifest] = []
    seen: set[str] = set()
    for manifest_id in manifest_ids:
        manifest = selected_registry.find(manifest_id)
        if manifest is None or manifest.id in seen:
            continue
        seen.add(manifest.id)
        manifests.append(manifest)
    return tuple(manifests)
