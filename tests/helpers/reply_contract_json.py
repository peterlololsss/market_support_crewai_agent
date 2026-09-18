from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import JsonValue, TypeAdapter

type JsonInput = (
    str | int | float | bool | None | Mapping[str, JsonInput] | Sequence[JsonInput]
)
_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def json_value(value: JsonInput) -> JsonValue:
    return _JSON_VALUE_ADAPTER.validate_python(value)


def json_mapping(mapping: Mapping[str, JsonInput]) -> dict[str, JsonValue]:
    return {key: json_value(value) for key, value in mapping.items()}


def json_object(value: JsonValue) -> dict[str, JsonValue]:
    if isinstance(value, dict):
        return value
    raise AssertionError("json_object_required")


def str_value(payload: Mapping[str, JsonValue], key: str, default: str) -> str:
    value = payload.get(key, default)
    if isinstance(value, str):
        return value
    raise AssertionError(f"{key}_must_be_string")


def str_list_value(payload: Mapping[str, JsonValue], key: str) -> list[str] | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item for item in value if isinstance(item, str)]
    raise AssertionError(f"{key}_must_be_string_list")


def optional_str_value(payload: Mapping[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    if value is None or isinstance(value, str):
        return value
    raise AssertionError(f"{key}_must_be_optional_string")
