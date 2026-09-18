from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Mapping, Sequence

from pydantic import Field, JsonValue, RootModel, ValidationInfo, field_validator

from market_support_crewai_agent.schemas.adapter import reject_raw_locator_text
from market_support_crewai_agent.schemas.base import JsonObject


class SanitizedAdapterResultV1(RootModel[dict[str, JsonValue]]):
    root: dict[str, JsonValue] = Field(default_factory=dict, max_length=16)

    @field_validator("root")
    @classmethod
    def validate_values(cls, values: JsonObject, info: ValidationInfo) -> JsonObject:
        configured_secrets = _configured_secret_values(info.context)
        normalized = _sanitize_adapter_result_object(
            values,
            depth=1,
            configured_secrets=configured_secrets,
        )
        try:
            canonical_bytes = json.dumps(
                normalized,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("adapter_result must be strict finite JSON") from exc
        if len(canonical_bytes) > 4_096:
            raise ValueError("adapter_result canonical JSON exceeds 4096 bytes")
        return normalized


_SANITIZED_ADAPTER_RESULT_DENIED_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "access_token",
        "token",
        "secret",
        "signature",
        "sig",
        "key",
        "cookie",
        "set_cookie",
        "url",
        "uri",
        "path",
        "file_path",
        "group_id",
        "sender_id",
        "tenant_ref",
        "principal_ref",
        "direct_thread_ref",
        "target_id",
        "receiver",
        "webhook",
        "resolve_ref",
    }
)


def _configured_secret_values(
    context: Mapping[str, Sequence[str]] | None,
) -> frozenset[str]:
    if context is None:
        return frozenset()
    return frozenset(
        value for value in context.get("configured_secret_values", ()) if value
    )


def _sanitize_adapter_result_object(
    value: JsonObject,
    *,
    depth: int,
    configured_secrets: frozenset[str],
) -> JsonObject:
    if depth > 4 or len(value) > 16:
        raise ValueError("adapter_result exceeds object depth or key bounds")
    normalized: JsonObject = {}
    for raw_key, nested in value.items():
        canonical_key = _canonical_adapter_result_key(raw_key)
        if canonical_key in normalized:
            raise ValueError("adapter_result contains duplicate normalized keys")
        normalized[canonical_key] = _sanitize_adapter_result_value(
            nested,
            depth=depth + 1,
            configured_secrets=configured_secrets,
        )
    return normalized


def _canonical_adapter_result_key(raw_key: str) -> str:
    canonical_key = unicodedata.normalize("NFC", raw_key).casefold()
    canonical_key = canonical_key.replace("-", "_").replace(" ", "_")
    if not canonical_key or canonical_key in _SANITIZED_ADAPTER_RESULT_DENIED_KEYS:
        raise ValueError("adapter_result contains a forbidden key")
    return canonical_key


def _sanitize_adapter_result_value(
    value: JsonValue,
    *,
    depth: int,
    configured_secrets: frozenset[str],
) -> JsonValue:
    if isinstance(value, dict):
        return _sanitize_adapter_result_object(
            value,
            depth=depth,
            configured_secrets=configured_secrets,
        )
    if isinstance(value, list):
        if depth > 4 or len(value) > 16:
            raise ValueError("adapter_result exceeds array depth or item bounds")
        return [
            _sanitize_adapter_result_value(
                item,
                depth=depth + 1,
                configured_secrets=configured_secrets,
            )
            for item in value
        ]
    if isinstance(value, str):
        if len(value) > 256:
            raise ValueError("adapter_result string values must be at most 256 chars")
        if any(secret in value for secret in configured_secrets):
            raise ValueError("adapter_result contains a configured secret")
        reject_raw_locator_text(value, "adapter_result")
        return value
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if not math.isfinite(value):
        raise ValueError("adapter_result numbers must be finite")
    return value
