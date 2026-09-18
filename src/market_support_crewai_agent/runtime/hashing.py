from __future__ import annotations

import base64
import hashlib
import hmac
import json

from pydantic import BaseModel

from market_support_crewai_agent.runtime.identity import (
    ConversationStateKey,
    state_key_ref,
)
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2

type CanonicalScalar = str | int | bool | None
type CanonicalValue = (
    CanonicalScalar | list["CanonicalValue"] | dict[str, "CanonicalValue"]
)


class HashingError(ValueError):
    def __init__(self, code: str) -> None:
        self.code: str = code
        super().__init__(code)


def canonical_json_bytes(value: CanonicalValue) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_frame(domain_tag: str, value: CanonicalValue, *, prefix: str) -> str:
    digest = hashlib.sha256(
        domain_tag.encode("ascii") + b"\0" + canonical_json_bytes(value)
    ).hexdigest()
    return f"{prefix}:{digest}"


def hash_canonical_model(domain_tag: str, model: BaseModel, *, prefix: str) -> str:
    return sha256_frame(
        domain_tag,
        model.model_dump(mode="json", exclude_none=False, exclude_defaults=False),
        prefix=prefix,
    )


def health_target_hmac(process_health_key: bytes, target: CanonicalValue) -> str:
    if len(process_health_key) != 32:
        raise HashingError("health_process_key_length")
    digest = hmac.new(
        process_health_key,
        b"llm-health-target.v1\0" + canonical_json_bytes(target),
        hashlib.sha256,
    ).digest()[:16]
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"htk1:{encoded}"


def evidence_command_hash(command: BaseModel) -> str:
    return hash_canonical_model("evidence-command.v1", command, prefix="ech1")


def evidence_cache_key_hash(cache_key: BaseModel) -> str:
    return hash_canonical_model("evidence-cache-key.v1", cache_key, prefix="ck1")


def evidence_value_hash(payload: BaseModel) -> str:
    return hash_canonical_model("evidence-value.v1", payload, prefix="evh1")


def evidence_scope_ref(scope: BaseModel) -> str:
    return hash_canonical_model("evidence-scope.v1", scope, prefix="escope1")


def evidence_source_record_ref(source_record: BaseModel) -> str:
    return hash_canonical_model(
        "evidence-source-record.v1", source_record, prefix="esr1"
    )


def evidence_provenance_hash(provenance: BaseModel) -> str:
    return hash_canonical_model("evidence-provenance.v1", provenance, prefix="eph1")


def evidence_fact_id(identity: BaseModel) -> str:
    return hash_canonical_model("evidence-fact-id.v1", identity, prefix="eid1")


def public_evidence_url_hash(url: str) -> str:
    return sha256_frame("public-evidence-url.v1", {"url": url}, prefix="puh1")


def frh1(rendered_fragment_text: str) -> str:
    digest = hashlib.sha256(
        b"prompt-fragment.v1\0" + rendered_fragment_text.encode("utf-8")
    ).hexdigest()
    return f"frh1:{digest}"


def hph1(rendered_prompt_text: str, agent_execution_spec: BaseModel) -> str:
    return sha256_frame(
        "harness-prompt.v1",
        {
            "agent_execution_spec": agent_execution_spec.model_dump(
                mode="json",
                exclude_none=False,
                exclude_defaults=False,
            ),
            "prompt_text": rendered_prompt_text,
        },
        prefix="hph1",
    )


def _canonical_bytes(value: CanonicalValue) -> bytes:
    return canonical_json_bytes(value)


def feedback_hash(feedback: ActionFeedbackRequestV2) -> str:
    payload = feedback.model_dump(mode="json")
    return (
        "fbh1:"
        + hashlib.sha256(
            b"action-feedback.v2\0" + _canonical_bytes(payload)
        ).hexdigest()
    )


def clarification_ref_hash(
    state_key: ConversationStateKey, ordinal: int, proposal: str
) -> str:
    payload: CanonicalValue = {
        "state_key_ref": state_key_ref(state_key),
        "ordinal": ordinal,
        "proposal": proposal,
    }
    return (
        "clr1:"
        + hashlib.sha256(
            b"clarification-record.v1\0" + _canonical_bytes(payload)
        ).hexdigest()
    )


def feedback_receipt_id(
    state_key: ConversationStateKey,
    feedback_id: str,
    feedback_digest: str,
    response_id: str,
    record_revision: int,
) -> str:
    payload: CanonicalValue = {
        "state_key_ref": state_key_ref(state_key),
        "feedback_id": feedback_id,
        "feedback_hash": feedback_digest,
        "response_id": response_id,
        "record_revision": record_revision,
    }
    return (
        "fbr1:"
        + hashlib.sha256(
            b"feedback-receipt.v1\0" + _canonical_bytes(payload)
        ).hexdigest()
    )
