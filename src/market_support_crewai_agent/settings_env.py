from __future__ import annotations

import os
from typing import Final

from pydantic import TypeAdapter

from market_support_crewai_agent.schemas.type_ids import ChannelType
from market_support_crewai_agent.settings_model import (
    GroupRecallMode,
    SettingsEnvironmentError,
)

_GROUP_RECALL_MODE_ADAPTER: Final[TypeAdapter[GroupRecallMode]] = TypeAdapter(
    GroupRecallMode
)


def float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def governed_float_env(name: str, expected: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return expected
    try:
        return float(raw)
    except ValueError as exc:
        raise SettingsEnvironmentError(name, raw) from exc


def bounded_float_env(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    del minimum, maximum
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def non_negative_int_env(name: str, default: int) -> int:
    try:
        parsed = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def governed_int_env(name: str, expected: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return expected
    try:
        return int(raw)
    except ValueError as exc:
        raise SettingsEnvironmentError(name, raw) from exc


def optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def str_tuple_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def group_recall_mode_env(name: str, default: GroupRecallMode) -> GroupRecallMode:
    raw = os.getenv(name, default)
    return _GROUP_RECALL_MODE_ADAPTER.validate_python(raw)


def channel_types_env(
    name: str,
    default: tuple[ChannelType, ...],
) -> tuple[ChannelType, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    parsed: list[ChannelType] = []
    for item in (part.strip() for part in value.split(",")):
        match item:
            case "bank" | "non_bank":
                parsed.append(item)
            case _:
                continue
    return tuple(parsed)
