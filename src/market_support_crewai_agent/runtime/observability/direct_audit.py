from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
from typing import Final, Literal

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.identity.state_key import (
    canonical_json_bytes,
    canonical_state_key_payload,
)

DirectAuditPurposeV1 = Literal[
    "message",
    "request_id",
    "request",
    "reply",
    "evidence",
    "directive",
    "plan",
    "planner_input",
    "planner_output",
    "composer_input",
    "composer_output",
    "verifier_input",
    "verifier_output",
    "approved_selector_input",
    "approved_selector_output",
    "document_selector_input",
    "document_selector_output",
]

_DIRECT_AUDIT_DOMAIN: Final = b"direct-audit.v1"
_DIRECT_AUDIT_PURPOSES: Final[frozenset[str]] = frozenset(
    {
        "message",
        "request_id",
        "request",
        "reply",
        "evidence",
        "directive",
        "plan",
        "planner_input",
        "planner_output",
        "composer_input",
        "composer_output",
        "verifier_input",
        "verifier_output",
        "approved_selector_input",
        "approved_selector_output",
        "document_selector_input",
        "document_selector_output",
    }
)


class DirectAuditKeyError(ValueError):
    pass


def decode_direct_audit_hmac_key(encoded_key: str) -> bytes:
    if not encoded_key or encoded_key.strip() != encoded_key or "=" in encoded_key:
        raise DirectAuditKeyError("direct_audit_hmac_key_invalid")
    padding = "=" * (-len(encoded_key) % 4)
    try:
        decoded = base64.urlsafe_b64decode((encoded_key + padding).encode("ascii"))
    except (UnicodeEncodeError, binascii.Error) as error:
        raise DirectAuditKeyError("direct_audit_hmac_key_invalid") from error
    if not 32 <= len(decoded) <= 64:
        raise DirectAuditKeyError("direct_audit_hmac_key_invalid")
    canonical = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
    if not hmac.compare_digest(canonical, encoded_key):
        raise DirectAuditKeyError("direct_audit_hmac_key_invalid")
    return decoded


def direct_audit_digest(
    *,
    key: bytes,
    purpose: DirectAuditPurposeV1,
    state_key: ConversationStateKey,
    content: str | bytes,
) -> str:
    if purpose not in _DIRECT_AUDIT_PURPOSES:
        raise DirectAuditKeyError("direct_audit_purpose_invalid")
    payload = content.encode("utf-8") if isinstance(content, str) else content
    message = b"\0".join(
        (
            _DIRECT_AUDIT_DOMAIN,
            purpose.encode("ascii"),
            canonical_json_bytes(canonical_state_key_payload(state_key)),
            payload,
        )
    )
    return "dah1:" + hmac.new(key, message, hashlib.sha256).hexdigest()
