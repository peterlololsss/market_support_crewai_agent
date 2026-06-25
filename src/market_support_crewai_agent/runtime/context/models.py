from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from typing import Any, Literal

ContextBlockType = Literal[
    "recent_verbatim",
    "compacted_summary",
    "large_result_preview",
    "allowed_evidence",
    "context_only",
    "disallowed_evidence",
    "app_state",
    "current_task",
    "output_schema",
    "ephemeral",
]
ProjectionAction = Literal["include", "exclude", "summarize", "preview", "redact"]


@dataclass(frozen=True)
class LargeResultPreview:
    result_id: str
    source_id: str
    source_type: str
    preview: str
    original_char_count: int
    preview_char_count: int
    truncation_reason: str
    reload_handle: str
    artifact_type: str | None = None
    fact_type: str | None = None
    status: str | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_safe(self))


@dataclass(frozen=True)
class CompactedSpanSummary:
    span_id: str
    original_message_count: int
    role_counts: dict[str, int]
    summary: str
    unresolved_items: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_safe(self))


@dataclass(frozen=True)
class ContextBlock:
    block_id: str
    block_type: ContextBlockType
    title: str
    payload: Any
    source_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    token_estimate: int = 0
    included_reason: str = ""
    redacted: bool = False
    reload_handle: str | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_safe(self))


@dataclass(frozen=True)
class ProjectionDecision:
    source_id: str
    decision: ProjectionAction
    block_type: ContextBlockType
    reason: str
    original_char_count: int | None = None
    projected_char_count: int | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_safe(self))


@dataclass(frozen=True)
class ContextPressureEstimate:
    token_budget: int
    estimated_tokens_before: int
    estimated_tokens_after: int
    pressure_ratio: float
    warning_threshold: float
    hard_threshold: float
    warning: bool
    hard_blocked: bool

    def to_prompt_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class ContextProjectionPolicy:
    recent_turns_verbatim_count: int = 4
    max_history_message_chars_inline: int = 1200
    max_evidence_chars_inline: int = 6000
    # Answer evidence is small today and models have room; inline it unless it
    # is truly pathological.
    max_answer_evidence_chars_inline: int = 1_000_000
    max_metadata_chars_inline: int = 2000
    large_result_preview_chars: int = 1200
    token_budget: int = 900_000
    warning_threshold: float = 0.75
    hard_threshold: float = 0.92
    allow_history_as_evidence: bool = False
    preserve_current_user_message: bool = True
    preserve_runtime_state: bool = True
    preserve_answerability: bool = True

    @classmethod
    def from_settings(cls, settings: object | None = None) -> "ContextProjectionPolicy":
        if settings is None:
            return cls()
        return cls(
            recent_turns_verbatim_count=int(
                getattr(settings, "agent_context_recent_turns_verbatim_count", 4)
            ),
            max_history_message_chars_inline=int(
                getattr(settings, "agent_context_max_history_message_chars_inline", 1200)
            ),
            max_evidence_chars_inline=int(
                getattr(settings, "agent_context_max_evidence_chars_inline", 6000)
            ),
            max_answer_evidence_chars_inline=int(
                getattr(settings, "agent_context_max_answer_evidence_chars_inline", 1_000_000)
            ),
            large_result_preview_chars=int(
                getattr(settings, "agent_context_large_result_preview_chars", 1200)
            ),
            token_budget=int(
                getattr(settings, "agent_context_token_budget", None)
                or 900_000
            ),
            warning_threshold=float(
                getattr(settings, "agent_context_warning_threshold", 0.75)
            ),
            hard_threshold=float(
                getattr(settings, "agent_context_hard_threshold", 0.92)
            ),
        )


@dataclass(frozen=True)
class RuntimeAppState:
    stage: str
    request_metadata: dict[str, Any]
    current_user_message: str
    domain_context: dict[str, Any]
    policy: dict[str, Any]
    current_goal: str | None = None
    intent_gate: dict[str, Any] | None = None
    execution_plan: dict[str, Any] | None = None
    plan_validation: dict[str, Any] | None = None
    preflight: Any | None = None
    business_facts: dict[str, Any] | None = None
    answerability_assessment: dict[str, Any] | None = None
    guardrail_decisions: list[dict[str, Any]] = field(default_factory=list)
    candidate_response: dict[str, Any] | None = None
    alignment_attempt: int = 0
    alignment_verdict: dict[str, Any] | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_safe(self))


@dataclass(frozen=True)
class ModelVisibleContext:
    projection_id: str
    stage: str
    blocks: list[ContextBlock]
    allowed_evidence_ids: list[str] = field(default_factory=list)
    disallowed_evidence_ids: list[str] = field(default_factory=list)
    context_only_source_ids: list[str] = field(default_factory=list)
    decisions: list[ProjectionDecision] = field(default_factory=list)
    pressure: ContextPressureEstimate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = "model-visible-context"

    def to_prompt_runtime_payload(self) -> dict[str, Any]:
        return _drop_empty(
            {
                "contract_version": self.contract_version,
                "projection_id": self.projection_id,
                "stage": self.stage,
                "allowed_evidence_ids": list(self.allowed_evidence_ids),
                "disallowed_evidence_ids": list(self.disallowed_evidence_ids),
                "context_only_source_ids": list(self.context_only_source_ids),
                "blocks": [block.to_prompt_dict() for block in self.blocks],
            }
        )


def stable_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def prompt_json(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json", exclude_none=True))
    if hasattr(value, "to_prompt_dict") and not is_dataclass(value):
        return _json_safe(value.to_prompt_dict())
    if is_dataclass(value):
        return {
            item.name: _json_safe(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            lowered = text_key.lower()
            if lowered.endswith(("api_key", "secret", "token")):
                continue
            if text_key == "resolve_ref":
                output["resolve_ref_available"] = bool(item)
                continue
            output[text_key] = _json_safe(item)
        return output
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_empty(item)
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [_drop_empty(item) for item in value]
    return value
