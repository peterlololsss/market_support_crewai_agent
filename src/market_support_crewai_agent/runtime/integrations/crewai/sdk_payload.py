from __future__ import annotations

from pydantic import JsonValue


def str_value(payload: dict[str, JsonValue], key: str, default: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) and value else default


def optional_str_value(payload: dict[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def optional_int_value(payload: dict[str, JsonValue], key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def optional_float_value(payload: dict[str, JsonValue], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def mapping_value(payload: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def string_sequence_value(
    payload: dict[str, JsonValue],
    key: str,
) -> tuple[str, ...] | None:
    value = payload.get(key)
    if not isinstance(value, list):
        return None
    return tuple(item for item in value if isinstance(item, str))
