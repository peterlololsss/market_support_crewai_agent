from __future__ import annotations

from typing import Final, Literal, override

from pydantic import BaseModel, Field, field_validator

from market_support_crewai_agent.schemas.conversation import (
    validate_canonical_tenant_ref,
)
from market_support_crewai_agent.schemas.type_ids import ChannelType

ALL_CHANNEL_TYPES: Final[tuple[ChannelType, ...]] = ("bank", "non_bank")
GroupRecallMode = Literal["shortcut", "advisory", "off"]


class SettingsEnvironmentError(ValueError):
    def __init__(self, name: str, value: str) -> None:
        self.name: str = name
        self.value: str = value
        super().__init__(str(self))

    @override
    def __str__(self) -> str:
        return f"{self.name} has invalid governed value {self.value!r}"


class SettingsContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code: str = code
        super().__init__(code)


class Settings(BaseModel):
    api_key: str | None = None
    deployment_tenant_ref: str | None = None
    internal_dm_enabled: bool = False
    llm_base_url: str = "https://llm.example.com/v1"
    llm_provider: str = "openai"
    llm_model: str = "deepseek-v4-pro"
    llm_api_key: str | None = None
    llm_timeout_seconds: float = Field(default=90.0, gt=0)
    llm_temperature: float = Field(default=0.1, ge=0)
    llm_max_tokens: int = Field(default=6000, gt=0)
    planner_llm_base_url: str = "https://llm.example.com/v1"
    planner_llm_provider: str = "openai"
    planner_llm_model: str = "deepseek-v4-pro"
    planner_llm_api_key: str | None = None
    crewai_verbose: bool = False
    crewai_max_iter: int = Field(default=1, ge=1, le=1)
    crewai_max_execution_time: int = Field(default=120, gt=0)
    crewai_max_retry_limit: int = Field(default=0, ge=0, le=0)
    planner_transient_retry_attempts: int = Field(default=0, ge=0, le=0)
    planner_transient_retry_base_seconds: float = Field(default=0.0, ge=0, le=0)
    agent_input_max_message_chars: int | None = Field(default=None, gt=0)
    agent_conversation_ttl_seconds: int = Field(default=86400, ge=1, le=604800)
    agent_conversation_max_messages: int = Field(default=12, ge=1, le=100)
    agent_conversation_max_sessions: int = Field(default=5000, ge=1, le=100000)
    agent_conversation_cleanup_interval_seconds: int = Field(default=300, ge=1, le=3600)
    agent_direct_audit_ttl_seconds: int = Field(default=86400, ge=1, le=604800)
    direct_audit_hmac_key: str | None = None
    issued_response_ttl_seconds: int = Field(default=86400, ge=60, le=604800)
    issued_response_pending_ttl_seconds: int = Field(default=180, ge=5, le=3600)
    issued_response_capacity: int = Field(default=5000, ge=1, le=100000)
    feedback_receipt_capacity: int = Field(default=20000, ge=1, le=500000)
    adapter_base_url: str = "http://127.0.0.1:8011"
    adapter_api_key: str | None = None
    adapter_timeout_seconds: float = Field(default=5.0, gt=0)
    doc_mcp_base_url: str | None = None
    doc_mcp_timeout_seconds: float = Field(default=5.0, gt=0)
    doc_mcp_enabled: bool = False
    doc_mcp_allowed_channel_types: tuple[ChannelType, ...] = ALL_CHANNEL_TYPES
    group_recall_mode: GroupRecallMode = "shortcut"
    # Per-document evidence ceiling. Set above the largest real document so a
    # selected document is delivered whole ("locate precisely, then expand to
    # read"); it only fails safe on a pathologically oversized document.
    doc_mcp_max_chars_per_document: int = Field(default=1_000_000, gt=0)
    # Process-wide TTL for the static product manifest and document content.
    # 0 disables caching and re-fetches from the MCP on every query.
    doc_mcp_cache_ttl_seconds: float = Field(default=300.0, ge=0)
    # Preferred document categories to load first when the closed-set selector declines.
    doc_mcp_baseline_categories: tuple[str, ...] = ("常见问答",)
    reply_alignment_verifier_enabled: bool = True
    reply_alignment_max_replans: int = Field(default=1, ge=0, le=2)
    reply_alignment_max_evidence_refetches: int = Field(default=1, ge=0, le=2)
    reply_alignment_max_recomposes: int = Field(default=1, ge=0, le=2)
    reply_alignment_max_total_remediations: int = Field(default=2, ge=0, le=2)
    llm_health_enabled: bool = False
    llm_health_check_interval_seconds: float = Field(default=300.0, gt=0)
    llm_health_failure_interval_seconds: float = Field(default=60.0, gt=0)
    llm_health_daily_report_time: str = "09:00"
    llm_health_timezone: str = "Asia/Shanghai"
    llm_health_warning_cooldown_seconds: float = Field(default=900.0, gt=0)
    llm_health_probe_retry_attempts: int = Field(default=0, ge=0, le=0)
    llm_health_probe_retry_base_seconds: float = Field(default=0.0, ge=0, le=0)
    llm_health_probe_timeout_seconds: float = Field(default=20.0, ge=0.1, le=30.0)
    feishu_app_id: str | None = None
    feishu_app_secret: str | None = None
    feishu_chat_id: str | None = None
    adapter_caller_namespace: str = Field(
        default="xiaoyan-wecom",
        pattern=r"^[a-z][a-z0-9._-]{0,63}$",
    )

    @field_validator("deployment_tenant_ref")
    @classmethod
    def validate_deployment_tenant_ref(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return validate_canonical_tenant_ref(
            value,
            field_name="deployment_tenant_ref",
        )

    @override
    def model_post_init(self, __context: object, /) -> None:
        if self.issued_response_pending_ttl_seconds > self.issued_response_ttl_seconds:
            raise SettingsContractError("issued_response_pending_ttl_exceeds_ttl")
        if self.feedback_receipt_capacity < self.issued_response_capacity:
            raise SettingsContractError("feedback_receipt_capacity_insufficient")
        if any(
            value > self.reply_alignment_max_total_remediations
            for value in (
                self.reply_alignment_max_replans,
                self.reply_alignment_max_evidence_refetches,
                self.reply_alignment_max_recomposes,
            )
        ):
            raise SettingsContractError("alignment_action_cap_exceeds_total")
        if self.maximum_llm_invocation_rows > 18:
            raise SettingsContractError("maximum_llm_invocation_rows_exceeds_18")

    @property
    def maximum_llm_invocation_rows(self) -> int:
        remediation_costs = sorted(
            (
                *((5,) * self.reply_alignment_max_replans),
                *((3,) * self.reply_alignment_max_evidence_refetches),
                *((2,) * self.reply_alignment_max_recomposes),
            ),
            reverse=True,
        )
        return 8 + sum(remediation_costs[: self.reply_alignment_max_total_remediations])
