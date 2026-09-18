from __future__ import annotations

from typing import Final

from pydantic import JsonValue


CANONICAL_JSON_SCHEMA_TRANSFORM_VERSION: Final = "canonical-json-schema-strip-docs.v1"
_DOCUMENTATION_KEYS: Final = frozenset({"title", "description", "examples", "$comment"})
_SCHEMA_MAP_KEYS: Final = frozenset(
    {"$defs", "definitions", "properties", "patternProperties", "dependentSchemas"}
)
_DATA_VALUE_KEYS: Final = frozenset({"const", "default", "enum"})


def canonicalize_json_schema(schema: JsonValue) -> JsonValue:
    return _canonicalize_schema_value(schema)


def _canonicalize_schema_value(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return [_canonicalize_schema_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    canonical: dict[str, JsonValue] = {}
    for key, item in value.items():
        if key in _DOCUMENTATION_KEYS:
            continue
        if key in _SCHEMA_MAP_KEYS and isinstance(item, dict):
            canonical[key] = {
                name: _canonicalize_schema_value(child) for name, child in item.items()
            }
            continue
        canonical[key] = (
            item if key in _DATA_VALUE_KEYS else _canonicalize_schema_value(item)
        )
    return canonical
