from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from pydantic import JsonValue

_LOCAL_LOCATOR_RE = re.compile(
    r"(?i)(file://\S+|(?<![\w:/\\])/(?!/)(?:[a-z0-9._-]+/)+\S+|"
    + r"[A-Za-z]:\\[^\s]+|wecom-adapter:[^\s]+|mcp://\S+)"
)
_AUTHORIZATION_SECRET_RE = re.compile(
    r"(?i)\bauthorization[ \t]*[:=][ \t]*(?:bearer|basic|digest|token)?[ \t]*\S+"
)
_AUTH_SCHEME_SECRET_RE = re.compile(r"(?i)\b(?:bearer|basic|digest)[ \t]+\S+")
_SECRET_RE = re.compile(
    r"(?i)(?<![\w-])(?:api[ _-]?key|apikey|authorization|token|password|secret)\b"
    + r"[ \t]*[:=][ \t]*\S+"
)
_BARE_SECRET_RE = re.compile(
    r"(?i)(?<![\w-])(?:api[ _-]?key|apikey|authorization|token|password|secret)\b[ \t]+"
    + r"\S+"
)
_SECRET_TOKEN_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|apikey|authorization|token|password|secret)"
    + r"[-_][A-Za-z0-9][\w.-]*\b"
)
_SECRET_KEY_RE = re.compile(
    r"(?i)(api[ _-]?key|apikey|authorization|bearer|token|password|secret)"
)
_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"(?i)\bignore (all )?(previous|above|prior) instructions\b"),
    re.compile(r"(?i)\b(system|developer) (prompt|message|instruction)s?\b"),
    re.compile(r"(?i)\b(call|use|execute) (the )?(tool|function)\b"),
    re.compile(r"(?i)\byou are now\b"),
    re.compile(r"(?i)\bdisregard (all )?(previous|above|prior) instructions\b"),
    re.compile(r"忽略(以上|之前|所有).*指令"),
    re.compile(r"不要遵守(以上|之前|所有).*指令"),
    re.compile(r"(调用|执行).*工具"),
    re.compile(r"你现在是"),
)
_REDACTED_LOCATOR = "[REDACTED_INTERNAL_LOCATOR]"
_REDACTED_SECRET = "[REDACTED_SECRET]"
_REMOVED_DOCUMENT_INSTRUCTION = "[REMOVED_DOCUMENT_INSTRUCTION]"

JsonMap = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SanitizedDocumentText:
    text: str
    metadata: JsonMap


def sanitize_document_text_for_evidence(text: str) -> SanitizedDocumentText:
    metadata: JsonMap = {
        "sanitized": False,
        "internal_locator_redacted": False,
        "secret_redacted": False,
        "document_instruction_removed": False,
        "char_count": 0,
    }
    sanitized = _LOCAL_LOCATOR_RE.sub(_REDACTED_LOCATOR, str(text or ""))
    if sanitized != text:
        metadata["sanitized"] = True
        metadata["internal_locator_redacted"] = True

    without_authorization = _AUTHORIZATION_SECRET_RE.sub(_REDACTED_SECRET, sanitized)
    without_auth_scheme = _AUTH_SCHEME_SECRET_RE.sub(
        _REDACTED_SECRET,
        without_authorization,
    )
    without_key_value = _SECRET_RE.sub(_REDACTED_SECRET, without_auth_scheme)
    without_bare_secrets = _BARE_SECRET_RE.sub(_REDACTED_SECRET, without_key_value)
    without_secrets = _SECRET_TOKEN_RE.sub(_REDACTED_SECRET, without_bare_secrets)
    if without_secrets != sanitized:
        metadata["sanitized"] = True
        metadata["secret_redacted"] = True
    sanitized = without_secrets

    cleaned_lines: list[str] = []
    for line in sanitized.splitlines():
        if any(pattern.search(line) for pattern in _PROMPT_INJECTION_PATTERNS):
            cleaned_lines.append(_REMOVED_DOCUMENT_INSTRUCTION)
            metadata["sanitized"] = True
            metadata["document_instruction_removed"] = True
            continue
        cleaned_lines.append(line)

    sanitized = "\n".join(cleaned_lines).strip()
    metadata["char_count"] = len(sanitized)
    return SanitizedDocumentText(text=sanitized, metadata=metadata)


def bound_sanitized_document_text(
    sanitized: SanitizedDocumentText,
    *,
    max_chars: int,
) -> tuple[str, JsonMap]:
    text = sanitized.text
    metadata = dict(sanitized.metadata)
    truncated = len(text) > max_chars
    if truncated:
        metadata["original_char_count"] = len(text)
        text = text[:max_chars].rstrip()
    metadata["truncated"] = truncated
    metadata["char_count"] = len(text)
    return text, metadata


def safe_document_source_id(value: str) -> str:
    sanitized = sanitize_document_text_for_evidence(value)
    if sanitized_label_was_redacted(sanitized):
        digest = hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:12]
        return f"document:{digest}"
    return sanitized.text[:160].strip() or "document"


def safe_product_identifier(value: str) -> str:
    return safe_document_source_id(value)


def safe_document_label(value: str) -> str:
    sanitized = sanitize_document_text_for_evidence(value)
    if sanitized_label_was_redacted(sanitized):
        return "document"
    return sanitized.text[:160].strip() or "document"


def safe_document_metadata(metadata: JsonMap) -> JsonMap:
    return {
        safe_document_metadata_key(str(key)): safe_document_metadata_value(
            str(key),
            value,
        )
        for key, value in metadata.items()
    }


def safe_document_metadata_key(key: str) -> str:
    sanitized = sanitize_document_text_for_evidence(key)
    if sanitized_label_was_redacted(sanitized) or metadata_key_is_secret(key):
        digest = hashlib.sha256(str(key or "").encode("utf-8")).hexdigest()[:12]
        return f"metadata:{digest}"
    return sanitized.text[:120].strip() or "metadata"


def safe_document_metadata_value(key: str, value: JsonValue) -> JsonValue:
    if metadata_key_is_secret(key):
        return _REDACTED_SECRET
    if isinstance(value, str):
        sanitized = sanitize_document_text_for_evidence(value)
        if bool(sanitized.metadata.get("internal_locator_redacted")):
            return "document"
        if bool(sanitized.metadata.get("secret_redacted")):
            return _REDACTED_SECRET
        return sanitized.text
    if isinstance(value, dict):
        return safe_document_metadata(value)
    if isinstance(value, list):
        return [safe_document_metadata_value(key, item) for item in value]
    return value


def metadata_key_is_secret(key: str) -> bool:
    return _SECRET_KEY_RE.search(key) is not None


def sanitized_label_was_redacted(sanitized: SanitizedDocumentText) -> bool:
    return bool(sanitized.metadata.get("internal_locator_redacted")) or bool(
        sanitized.metadata.get("secret_redacted")
    )
