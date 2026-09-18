from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Literal, TypeAlias, TypeGuard

from market_support_crewai_agent.runtime.policy.ontology_models import TimeRange

SourceContextType = Literal[
    "current_artifact",
    "adapter_context",
    "user_message",
    "assistant_message",
    "history_summary",
    "retrieved_doc",
    "tool_result",
]
SourceMetadataPrimitive: TypeAlias = (
    datetime | TimeRange | str | int | float | bool | None
)
SourceMetadataValue: TypeAlias = (
    SourceMetadataPrimitive
    | Sequence[SourceMetadataPrimitive]
    | Mapping[str, SourceMetadataPrimitive]
)
SourcePromptValue: TypeAlias = str | bool | list[str] | dict[str, str]
SourcePromptDict: TypeAlias = dict[str, SourcePromptValue]


def coerce_source_context_type(value: SourceMetadataValue) -> SourceContextType:
    text = clean_metadata_value(value)
    if is_source_context_type(text):
        return text
    return "tool_result"


def is_source_context_type(value: str) -> TypeGuard[SourceContextType]:
    match value:
        case (
            "current_artifact"
            | "adapter_context"
            | "user_message"
            | "assistant_message"
            | "history_summary"
            | "retrieved_doc"
            | "tool_result"
        ):
            return True
        case _:
            return False


def time_range_from_metadata(value: SourceMetadataValue) -> TimeRange | None:
    if isinstance(value, TimeRange):
        return value
    if not isinstance(value, Mapping):
        return None
    return TimeRange(
        period=optional_metadata_string(value.get("period")),
        start=optional_metadata_string(value.get("start")),
        end=optional_metadata_string(value.get("end")),
        label=optional_metadata_string(value.get("label")),
    )


def source_product_ids(value: SourceMetadataValue) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item))
    return ()


def first_non_unknown(*values: SourceMetadataValue) -> str | None:
    for value in values:
        text = optional_metadata_string(value)
        if text and text != "unknown":
            return text
    return None


def optional_metadata_string(value: SourceMetadataValue) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def clean_metadata_value(value: SourceMetadataValue) -> str:
    return str(value or "").strip()


def isoformat_metadata_value(value: datetime | str | None) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return optional_metadata_string(value)


def time_range_prompt_dict(time_range: TimeRange | None) -> dict[str, str]:
    if time_range is None:
        return {}
    result: dict[str, str] = {}
    if time_range.period:
        result["period"] = time_range.period
    if time_range.start:
        result["start"] = time_range.start
    if time_range.end:
        result["end"] = time_range.end
    if time_range.label:
        result["label"] = time_range.label
    return result
