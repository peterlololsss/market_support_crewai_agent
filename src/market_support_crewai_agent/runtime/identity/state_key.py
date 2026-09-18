from __future__ import annotations

import hashlib
import json
from typing import Any

from market_support_crewai_agent.runtime.identity.models import ConversationStateKey


def canonical_state_key_payload(key: ConversationStateKey) -> dict[str, str | int]:
    return {
        "version": 1,
        "surface": key.surface,
        "adapter_namespace": key.adapter_namespace,
        "tenant_ref": key.tenant_ref,
        "scene": key.scene,
        "subject_ref": key.subject_ref,
        "principal_ref": key.principal_ref,
    }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def state_key_ref(key: ConversationStateKey) -> str:
    digest = hashlib.sha256(
        b"conversation-state-key.v1\0"
        + canonical_json_bytes(canonical_state_key_payload(key))
    ).hexdigest()
    return f"csk1:{digest}"
