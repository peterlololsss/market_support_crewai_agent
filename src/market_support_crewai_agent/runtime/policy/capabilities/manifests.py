from __future__ import annotations

import json
from hashlib import sha256
from importlib.resources import files
from typing import Final

from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityManifestV2,
)

_CAPABILITY_REGISTRY_CONTRACT_VERSION = "capability-registry-target.2026-07-18.1"
_CAPABILITY_REGISTRY_RESOURCE_SHA256: Final = (
    "23b1aab84833a7d3ebe7e8d2074e32041f9173ac6572f640691350195c2dca3a"
)


def capability_manifest_content_hash_v2(manifest: CapabilityManifestV2) -> str:
    payload = manifest.model_dump(mode="json")
    del payload["expected_content_hash"]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = sha256(b"capability-manifest.v2\0" + encoded).hexdigest()
    return f"cmh1:{digest}"


def _load_builtin_capability_manifests() -> tuple[CapabilityManifestV2, ...]:
    resource = files(__package__).joinpath("resources/capability_registry.v2.json")
    resource_bytes = resource.read_bytes()
    if sha256(resource_bytes).hexdigest() != _CAPABILITY_REGISTRY_RESOURCE_SHA256:
        raise ValueError("built-in capability registry resource hash mismatch")
    payload = json.loads(resource_bytes)
    if (
        not isinstance(payload, dict)
        or payload.get("contract_version") != _CAPABILITY_REGISTRY_CONTRACT_VERSION
    ):
        raise ValueError("invalid built-in capability registry resource")
    rows = payload.get("manifests")
    if not isinstance(rows, list):
        raise ValueError("built-in capability registry manifests must be a list")
    manifests = tuple(CapabilityManifestV2.model_validate(row) for row in rows)
    for manifest in manifests:
        if (
            capability_manifest_content_hash_v2(manifest)
            != manifest.expected_content_hash
        ):
            raise ValueError("built-in capability manifest semantic hash mismatch")
    return manifests


BUILTIN_CAPABILITY_MANIFESTS = _load_builtin_capability_manifests()
SEALED_MANIFEST_CONTENT_HASHES: Final = {
    manifest.manifest_id: manifest.expected_content_hash
    for manifest in BUILTIN_CAPABILITY_MANIFESTS
}
SEALED_MANIFEST_ORDER: Final = tuple(
    manifest.manifest_id for manifest in BUILTIN_CAPABILITY_MANIFESTS
)
