from __future__ import annotations

from collections.abc import Mapping
import re

from market_support_crewai_agent.runtime.domain.ontology import ArtifactType, DomainContext
from market_support_crewai_agent.runtime.domain.sources.metadata import (
    SourceMetadata,
    is_history_source,
)
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.validation.guardrail_types import RequestedScope

_IMAGE_MARKER_RE = re.compile(r"%%([\w\d_.-]+\.png)%%")


def requested_scope(plan_or_frame: object) -> RequestedScope | None:
    value = getattr(plan_or_frame, "requested_scope", None)
    if value is None:
        return None
    if isinstance(value, RequestedScope):
        return value
    return RequestedScope.model_validate(value)


def requested_scope_dict(scope: RequestedScope) -> dict:
    return scope.model_dump(mode="json", exclude_none=True)


def adapter_channel_id(domain_context: DomainContext) -> str:
    return f"adapter_channel:{domain_context.channel.kind}:{domain_context.channel.name}"


def artifact_type_for_capability(capability_name: str) -> ArtifactType:
    if capability_name in {"material_pack", "weekly_report", "monthly_report"}:
        return capability_name  # type: ignore[return-value]
    if capability_name == "document_context":
        return "document_context"
    return "adapter_context"


def ordered_unique(values) -> list:
    seen = set()
    output = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def evidence_id(fact: EvidenceFact) -> str:
    return ":".join(
        item
        for item in (
            fact.source_type,
            fact.source_id,
            fact.fact_type,
        )
        if item
    )


def evidence_artifact_type(fact: EvidenceFact) -> str:
    artifact = str(fact.metadata.get("artifact_type") or "").strip()
    if artifact:
        return artifact
    source_metadata = source_metadata_for_fact(fact)
    if source_metadata is not None and source_metadata.artifact_type:
        return source_metadata.artifact_type
    if fact.fact_type in {"document_context", "document_context_unavailable"}:
        return "document_context"
    return str(fact.artifact_type or "")


def image_marker_filenames(text: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for match in _IMAGE_MARKER_RE.finditer(str(text or "")):
        filename = match.group(1)
        if filename in seen:
            continue
        seen.add(filename)
        output.append(filename)
    return output


def trusted_document_context(fact: EvidenceFact) -> bool:
    if fact.fact_type != "document_context" or not fact.value:
        return False
    if fact.source_type == "document_mcp":
        return True
    return (
        fact.source_type == "approved_static_knowledge"
        and fact.metadata.get("approved_static_knowledge") is True
        and fact.metadata.get("content_is_data_only") is True
    )


def marker_in_trusted_document_context(
    marker: str,
    evidence_facts: list[EvidenceFact],
) -> bool:
    return any(
        trusted_document_context(fact) and marker in str(fact.value or "")
        for fact in evidence_facts
    )


def payload_dict(payload: object) -> dict[str, object]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json", exclude_none=True)  # type: ignore[no-any-return]
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def reply_kind(payload: Mapping[str, object]) -> str:
    reply = payload.get("reply")
    if not isinstance(reply, Mapping):
        return ""
    return str(reply.get("kind") or "")


def infer_abstained(payload: Mapping[str, object]) -> bool:
    return reply_kind(payload) in {
        "unable_to_answer",
        "clarification",
        "human_handoff",
        "no_reply",
    }


def lookup_path(value: object, path: str) -> object | None:
    current: object | None = value
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


def is_missing(value: object | None) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list | tuple | set | frozenset | dict):
        return not bool(value)
    return False


def string_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        for key in ("source_id", "evidence_id", "id"):
            text = str(value.get(key) or "").strip()
            if text:
                return [text]
        return []
    if isinstance(value, list | tuple | set | frozenset):
        output: list[str] = []
        for item in value:
            output.extend(string_values(item))
        return output
    return []


def schema_errors(schema: Mapping[str, object], value: object, path: str = "$") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_json_type(value, str(expected_type)):
        return [f"{path} must be {expected_type}"]

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path} must equal {schema['const']!r}")
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path} must be one of {enum!r}")

    if expected_type == "object":
        if not isinstance(value, Mapping):
            return [f"{path} must be object"]
        for field_name in schema.get("required") or []:
            if str(field_name) not in value:
                errors.append(f"{path}.{field_name} is required")
        properties = schema.get("properties") or {}
        if isinstance(properties, Mapping):
            for field_name, nested_schema in properties.items():
                if field_name not in value or not isinstance(nested_schema, Mapping):
                    continue
                errors.extend(
                    schema_errors(
                        nested_schema,
                        value[field_name],
                        f"{path}.{field_name}",
                    )
                )
        if schema.get("additionalProperties") is False and isinstance(properties, Mapping):
            extra = sorted(set(value) - {str(key) for key in properties})
            errors.extend(f"{path}.{field_name} is not allowed" for field_name in extra)
    elif expected_type == "array":
        if not isinstance(value, list):
            return [f"{path} must be array"]
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"{path} must contain at least {min_items} items")
        if isinstance(max_items, int) and len(value) > max_items:
            errors.append(f"{path} must contain at most {max_items} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                errors.extend(schema_errors(item_schema, item, f"{path}[{index}]"))
    elif expected_type == "string":
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        if isinstance(min_length, int) and isinstance(value, str) and len(value) < min_length:
            errors.append(f"{path} must be at least {min_length} characters")
        if isinstance(max_length, int) and isinstance(value, str) and len(value) > max_length:
            errors.append(f"{path} must be at most {max_length} characters")
    return errors


def _matches_json_type(value: object, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def source_metadata_for_fact(fact: EvidenceFact) -> SourceMetadata | None:
    return fact.source_metadata if isinstance(fact.source_metadata, SourceMetadata) else None


def is_history_fact(fact: EvidenceFact) -> bool:
    return (
        fact.source_type in {"conversation_history", "action_ledger"}
        or fact.artifact_type == "history"
        or is_history_source(source_metadata_for_fact(fact))
    )


def source_provenance_missing(fact: EvidenceFact) -> bool:
    source_metadata = source_metadata_for_fact(fact)
    provenance = (
        str(source_metadata.provenance or "").strip()
        if source_metadata is not None
        else ""
    )
    if not provenance:
        provenance = str(fact.scope.provenance or "").strip()
    return (not provenance or provenance == "unknown") and not str(fact.source_id or "").strip()
