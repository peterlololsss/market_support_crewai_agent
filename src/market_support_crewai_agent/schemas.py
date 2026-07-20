from __future__ import annotations

import base64
import binascii
from typing import Annotated, Any, Literal, Union
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MaterialType = Literal["material", "weekly", "monthly"]
AvailableArtifactType = Literal["material_pack", "weekly_report", "monthly_report"]
ChannelType = Literal["bank", "non_bank", "unknown"]
AdapterResolveType = Literal[
    "material_pack",
    "weekly_report",
    "monthly_report",
    "sales_mention",
    "outbound_message_target",
]
OutboundTargetKind = Literal["channel", "group"]
ReadCapability = Literal[
    "resolve_material_pack",
    "resolve_weekly_report",
    "resolve_monthly_report",
    "resolve_sales_mention",
    "query_internal_company_info",
]
AdapterResolveStatus = Literal[
    "resolved",
    "missing",
    "ambiguous",
    "forbidden",
    "temporarily_unavailable",
]
AdapterReportScopeCommand = Literal["summary", "match", "list_products"]
AdapterReportScopeMaterialType = Literal["weekly", "monthly"]
ActionExecutionStatus = Literal["executed", "failed", "skipped"]
ActionExecutionType = Literal[
    "send_material_pack",
    "send_weekly_report",
    "send_monthly_report",
    "mention_sales",
    "send_text",
    "prepare_outbound_message",
    "execute_prepared_outbound_message",
]
ReplyKind = Literal[
    "answer",
    "clarification",
    "human_handoff",
    "unable_to_answer",
    "no_reply",
]
ReplyMentionType = Literal["sales"]
OutboundActionType = Literal[
    "send_material_pack",
    "send_weekly_report",
    "send_monthly_report",
    "prepare_outbound_message",
    "execute_prepared_outbound_message",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


MetricCount = Annotated[int, Field(ge=0)]


class AvailableArtifact(StrictModel):
    type: AvailableArtifactType
    options: list[str] = Field(default_factory=list)

    @field_validator("options")
    @classmethod
    def clean_options(cls, values: list[str]) -> list[str]:
        seen = set()
        output: list[str] = []
        for value in values:
            option = str(value).strip()
            if not option or option in seen:
                continue
            seen.add(option)
            output.append(option)
        return output

    @model_validator(mode="after")
    def validate_options_only_for_material_pack(self):
        if self.type != "material_pack" and self.options:
            raise ValueError("available_artifacts options are only valid for material_pack")
        return self


class ReplyRequest(StrictModel):
    conversation_key: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    sender_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    is_group: bool
    context_id: str | None = None
    group_name: str = Field(min_length=1)
    dist_channel_name: str = Field(min_length=1)
    sender_nickname: str = Field(min_length=1)
    available_artifacts: list[AvailableArtifact]
    channel_type: ChannelType
    allowed_read_capabilities: list[ReadCapability] = Field(default_factory=list)

    @field_validator("available_artifacts")
    @classmethod
    def validate_available_artifacts_unique(cls, values: list[AvailableArtifact]) -> list[AvailableArtifact]:
        _reject_duplicate_available_artifacts(values)
        return values


class AdapterResolveRequest(StrictModel):
    resolve_type: AdapterResolveType
    dist_name: str = Field(min_length=1)
    material_pack_option: str | None = None


class ReportScopeSection(StrictModel):
    name: str = Field(min_length=1)
    expected_product_count: int = Field(default=0, ge=0)
    generated_product_count: int = Field(default=0, ge=0)
    missing_product_count: int = Field(default=0, ge=0)


class ReportScopeProduct(StrictModel):
    product_name: str = Field(min_length=1)
    portfolio_type: str | None = None
    report_section: str = Field(default="unknown", min_length=1)
    source_pdf_status: Literal["found", "missing"]
    final_report_status: Literal["generated", "not_generated"]
    file_name: str | None = None


class AdapterResolveResult(StrictModel):
    contract_version: Literal["adapter-resolve"]
    resolve_type: AdapterResolveType
    status: AdapterResolveStatus
    display_name: str
    reason_code: str
    candidates: list[str] = Field(default_factory=list)
    channel_type: ChannelType = "unknown"
    available_artifacts: list[AvailableArtifact]
    resolved_at: int
    detail: str | None = None
    resolve_ref: str | None = None
    material_pack_option: str | None = None
    period: str | None = None
    report_date: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    period_label: str | None = None
    source_trade_date: str | None = None
    scope_complete: bool | None = None
    expected_product_count: int | None = Field(default=None, ge=0)
    generated_product_count: int | None = Field(default=None, ge=0)
    missing_product_count: int | None = Field(default=None, ge=0)
    report_sections: list[ReportScopeSection] = Field(default_factory=list, max_length=64)

    @field_validator("available_artifacts")
    @classmethod
    def validate_available_artifacts_unique(cls, values: list[AvailableArtifact]) -> list[AvailableArtifact]:
        _reject_duplicate_available_artifacts(values)
        return values

    @field_validator("detail")
    @classmethod
    def validate_detail_is_sanitized(cls, value: str | None) -> str | None:
        _reject_raw_locator_text(value, "detail")
        return value

    @field_validator("resolve_ref")
    @classmethod
    def validate_resolve_ref_is_opaque(cls, value: str | None) -> str | None:
        _reject_raw_locator_text(value, "resolve_ref")
        return value

    @model_validator(mode="after")
    def validate_resolved_has_ref(self):
        if self.status == "resolved" and not (self.resolve_ref or "").strip():
            raise ValueError("resolved adapter results must include resolve_ref")
        return self


class OutboundTargetResolveResult(StrictModel):
    status: Literal["resolved", "missing", "ambiguous", "temporarily_unavailable"]
    reason_code: str = Field(min_length=1)
    display_name: str = Field(min_length=1, max_length=128)
    target_kind: OutboundTargetKind
    target_count: int = Field(ge=0)
    resolved_count: int = Field(ge=0)
    resolve_ref: str = ""

    @field_validator("resolve_ref")
    @classmethod
    def validate_resolve_ref(cls, value: str) -> str:
        if value and (
            not value.startswith("outbound-target:")
            or len(value) != len("outbound-target:") + 64
            or any(char not in "0123456789abcdef" for char in value[16:])
        ):
            raise ValueError("outbound target resolve_ref is invalid")
        return value

    @model_validator(mode="after")
    def validate_resolved_shape(self):
        if self.resolved_count > self.target_count:
            raise ValueError("resolved_count cannot exceed target_count")
        if self.status == "resolved" and (
            not self.resolve_ref or self.resolved_count != self.target_count
        ):
            raise ValueError("resolved outbound target must be complete")
        if self.status != "resolved" and self.resolve_ref:
            raise ValueError("unresolved outbound target must not include resolve_ref")
        return self


class AdapterResolveBatchRequest(StrictModel):
    requests: list[AdapterResolveRequest] = Field(min_length=1, max_length=16)


class AdapterResolveBatchResult(StrictModel):
    contract_version: Literal["adapter-resolve-batch"]
    results: list[AdapterResolveResult]


class AdapterReportScopeRequest(StrictModel):
    material_type: AdapterReportScopeMaterialType
    dist_name: str = Field(min_length=1)
    command: AdapterReportScopeCommand
    period: str | None = None
    query: str | None = None
    page: int | None = Field(default=None, ge=1)
    page_size: int | None = Field(default=None, ge=1, le=50)
    section_name: str | None = None


class AdapterReportScopeMatch(StrictModel):
    status: Literal["matched", "not_found", "ambiguous"]
    query: str = ""
    match_type: Literal["section", "product"] | None = None
    matched_section: str | None = None
    candidate_count: int = Field(default=0, ge=0)
    products: list[ReportScopeProduct] = Field(default_factory=list, max_length=50)
    product_page: int = Field(default=1, ge=1)
    product_page_size: int = Field(default=20, ge=1, le=50)


class AdapterReportScopeResult(StrictModel):
    contract_version: Literal["adapter-report-scope"]
    material_type: AdapterReportScopeMaterialType
    dist_name: str
    period: str
    status: AdapterResolveStatus
    reason_code: str
    report_date: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    period_label: str | None = None
    detail: str | None = None
    schema_version: str | None = None
    source_trade_date: str | None = None
    scope_complete: bool | None = None
    expected_product_count: int | None = Field(default=None, ge=0)
    generated_product_count: int | None = Field(default=None, ge=0)
    missing_product_count: int | None = Field(default=None, ge=0)
    report_sections: list[ReportScopeSection] = Field(default_factory=list, max_length=64)
    match: AdapterReportScopeMatch | None = None
    products: list[ReportScopeProduct] = Field(default_factory=list, max_length=50)
    product_page: int | None = Field(default=None, ge=1)
    product_page_size: int | None = Field(default=None, ge=1, le=50)
    product_total_count: int | None = Field(default=None, ge=0)

    @field_validator("detail")
    @classmethod
    def validate_detail_is_sanitized(cls, value: str | None) -> str | None:
        _reject_raw_locator_text(value, "detail")
        return value


class AdapterCapabilityEndpoints(StrictModel):
    health: str
    capabilities: str
    metrics: str
    resolve: str
    batch_resolve: str
    report_scope: str | None = None


class AdapterCapabilityAuth(StrictModel):
    header_schemes: list[str] = Field(default_factory=list)
    protected_endpoints: list[str] = Field(default_factory=list)


class AdapterOutboundReadiness(StrictModel):
    ready: bool
    reason_codes: list[str] = Field(default_factory=list, max_length=16)


class AdapterOutboundConstraints(StrictModel):
    dm_only: bool
    later_trusted_message_required: bool
    prepare_feedback_ack_required: bool
    single_use: bool
    sdk_result_semantics: Literal["accepted_not_delivered"]


class AdapterOutboundMessagingCapability(StrictModel):
    enabled: bool
    readiness: AdapterOutboundReadiness
    action_types: list[Literal[
        "prepare_outbound_message",
        "execute_prepared_outbound_message",
    ]]
    content_types: list[Literal["link", "link_card", "report_card", "text"]]
    report_types: list[Literal["monthly_report", "weekly_report"]]
    constraints: AdapterOutboundConstraints
    result_fields: list[str]

    @property
    def ready(self) -> bool:
        required_actions = {
            "prepare_outbound_message",
            "execute_prepared_outbound_message",
        }
        return (
            self.enabled
            and self.readiness.ready
            and required_actions.issubset(self.action_types)
            and self.constraints.dm_only
            and self.constraints.later_trusted_message_required
            and self.constraints.prepare_feedback_ack_required
            and self.constraints.single_use
        )


class AdapterCapabilities(StrictModel):
    service: Literal["xiaoyan-wecom-market-agent-adapter"]
    contract_version: Literal["adapter-resolve"]
    batch_contract_version: Literal["adapter-resolve-batch"]
    action_contract_version: Literal["adapter-action"]
    endpoints: AdapterCapabilityEndpoints
    resolve_types: list[AdapterResolveType]
    statuses: list[AdapterResolveStatus]
    max_batch_requests: int = Field(gt=0)
    max_request_body_bytes: int = Field(gt=0)
    cache_ttl_seconds: float = Field(default=0, ge=0)
    cache_max_entries: int = Field(default=0, ge=0)
    auth: AdapterCapabilityAuth | None = None
    outbound_target_kinds: list[OutboundTargetKind] = Field(default_factory=list)
    outbound_target_response_fields: list[str] = Field(default_factory=list)
    outbound_messaging: AdapterOutboundMessagingCapability | None = None


class AdapterCacheMetrics(StrictModel):
    ttl_seconds: float = Field(ge=0)
    max_entries: int = Field(ge=0)
    entries: int = Field(ge=0)
    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    sets: int = Field(ge=0)
    expired: int = Field(ge=0)
    evictions: int = Field(ge=0)


class AdapterResolverMetrics(StrictModel):
    cache: AdapterCacheMetrics


class AdapterDurationMetrics(StrictModel):
    count: MetricCount
    total: float = Field(ge=0)
    max: float = Field(ge=0)


class AdapterTransportRouteMetrics(StrictModel):
    requests: MetricCount
    errors: MetricCount
    status_codes: dict[str, MetricCount] = Field(default_factory=dict)
    duration_ms: AdapterDurationMetrics


class AdapterTransportMetrics(StrictModel):
    requests_total: MetricCount
    inflight_requests: MetricCount
    errors_total: MetricCount
    status_codes: dict[str, MetricCount] = Field(default_factory=dict)
    duration_ms: AdapterDurationMetrics
    routes: dict[str, AdapterTransportRouteMetrics] = Field(default_factory=dict)


class AdapterMetrics(StrictModel):
    service: Literal["xiaoyan-wecom-market-agent-adapter"]
    uptime_seconds: int = Field(ge=0)
    resolver: AdapterResolverMetrics
    transport: AdapterTransportMetrics


class ReplyMention(StrictModel):
    type: ReplyMentionType
    reason: str | None = None


class PrimaryReply(StrictModel):
    kind: ReplyKind
    text: str = ""
    mentions: list[ReplyMention] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_no_reply_shape(self):
        if self.kind == "no_reply" and (self.text.strip() or self.mentions):
            raise ValueError("no_reply must not include text or mentions")
        return self


class OutboundActionBase(StrictModel):
    action_id: str = ""


class SendActionBase(OutboundActionBase):
    resolve_ref: str = Field(min_length=1)

    @field_validator("resolve_ref")
    @classmethod
    def validate_resolve_ref_is_opaque(cls, value: str) -> str:
        _reject_raw_locator_text(value, "resolve_ref")
        return value


class SendMaterialPackAction(SendActionBase):
    type: Literal["send_material_pack"]
    resolve_type: Literal["material_pack"]
    material_pack_option: str | None = None


class SendWeeklyReportAction(SendActionBase):
    type: Literal["send_weekly_report"]
    resolve_type: Literal["weekly_report"]
    period: str = Field(min_length=1)
    report_date: str = Field(min_length=1)


class SendMonthlyReportAction(SendActionBase):
    type: Literal["send_monthly_report"]
    resolve_type: Literal["monthly_report"]
    period: str = Field(min_length=1)
    report_date: str = Field(min_length=1)


class OutboundMessageTarget(StrictModel):
    kind: OutboundTargetKind
    name: str = Field(min_length=1, max_length=128)
    resolve_ref: str = Field(pattern=r"^outbound-target:[0-9a-f]{64}$")

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("outbound target name must be trimmed")
        return value


class OutboundTextContent(StrictModel):
    kind: Literal["text"]
    text: str = Field(min_length=1, max_length=4000)


class OutboundLinkContent(StrictModel):
    kind: Literal["link"]
    url: str = Field(min_length=1, max_length=2048)
    label: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _validate_outbound_https_url(value)


class OutboundLinkCardContent(StrictModel):
    kind: Literal["link_card"]
    title: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=512)
    url: str = Field(min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _validate_outbound_https_url(value)


class OutboundReportCardContent(StrictModel):
    kind: Literal["report_card"]
    report_kind: Literal["weekly_report", "monthly_report"]
    resolve_ref: str = Field(min_length=1)
    source_channel: str = Field(min_length=1, max_length=128)
    period: str = Field(min_length=1, max_length=64)
    report_date: str = Field(min_length=1, max_length=64)

    @field_validator("resolve_ref")
    @classmethod
    def validate_resolve_ref(cls, value: str) -> str:
        _reject_raw_locator_text(value, "resolve_ref")
        return value


OutboundMessageContent = Annotated[
    Union[
        OutboundTextContent,
        OutboundLinkContent,
        OutboundLinkCardContent,
        OutboundReportCardContent,
    ],
    Field(discriminator="kind"),
]


class PrepareOutboundMessageAction(OutboundActionBase):
    type: Literal["prepare_outbound_message"]
    target: OutboundMessageTarget
    content: OutboundMessageContent


class ExecutePreparedOutboundMessageAction(OutboundActionBase):
    type: Literal["execute_prepared_outbound_message"]
    confirmation_ref: str

    @field_validator("confirmation_ref")
    @classmethod
    def validate_confirmation_ref(cls, value: str) -> str:
        prefix = "wecom-adapter-confirmation:"
        token = value.removeprefix(prefix)
        if (
            not value.startswith(prefix)
            or len(token) != 43
            or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for char in token)
        ):
            raise ValueError("confirmation_ref is invalid")
        try:
            decoded = base64.urlsafe_b64decode(token + "=")
        except (binascii.Error, ValueError) as exc:
            raise ValueError("confirmation_ref is invalid") from exc
        canonical = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
        if len(decoded) != 32 or canonical != token:
            raise ValueError("confirmation_ref is invalid")
        return value


OutboundAction = Annotated[
    Union[
        SendMaterialPackAction,
        SendWeeklyReportAction,
        SendMonthlyReportAction,
        PrepareOutboundMessageAction,
        ExecutePreparedOutboundMessageAction,
    ],
    Field(discriminator="type"),
]


def _validate_outbound_https_url(value: str) -> str:
    if value != value.strip():
        raise ValueError("outbound URL must be trimmed")
    parsed = urlsplit(value)
    if parsed.scheme.casefold() != "https" or not parsed.hostname or parsed.username:
        raise ValueError("outbound URL must be HTTPS without credentials")
    return value


class ReplyResponse(StrictModel):
    contract_version: Literal["reply"] = "reply"
    response_id: str = ""
    reply: PrimaryReply
    actions: list[OutboundAction] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_no_reply_has_no_actions(self):
        if self.reply.kind == "no_reply" and self.actions:
            raise ValueError("no_reply must not include outbound actions")
        return self


_SENSITIVE_ADAPTER_RESULT_KEYS = {
    "dist_name",
    "file_path",
    "group_id",
    "material_url",
    "path",
    "receiver",
    "report_url",
    "sender_id",
    "target_id",
    "url",
}


def _reject_sensitive_adapter_result(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in _SENSITIVE_ADAPTER_RESULT_KEYS:
                raise ValueError("adapter_result contains adapter-owned fields")
            _reject_sensitive_adapter_result(nested)
        return
    if isinstance(value, list):
        for nested in value:
            _reject_sensitive_adapter_result(nested)
        return
    if isinstance(value, str):
        _reject_raw_locator_text(value, "adapter_result")


def _reject_duplicate_available_artifacts(values: list[AvailableArtifact]) -> None:
    seen: set[str] = set()
    for artifact in values:
        if artifact.type in seen:
            raise ValueError("available_artifacts must not contain duplicate artifact types")
        seen.add(artifact.type)


def _reject_raw_locator_text(value: str | None, field_name: str) -> None:
    if not value:
        return
    if (
        "://" in value
        or "/" in value
        or "\\" in value
        or "file:" in value
        or value.startswith("~")
    ):
        raise ValueError(f"{field_name} contains raw locator values")


class ActionFeedbackArtifactBase(StrictModel):
    resolve_ref: str | None = None
    artifact_ref: str | None = None

    @field_validator("artifact_ref")
    @classmethod
    def validate_opaque_artifact_ref(cls, value: str | None) -> str | None:
        _reject_raw_locator_text(value, "artifact_ref")
        return value

    @field_validator("resolve_ref")
    @classmethod
    def validate_opaque_resolve_ref(cls, value: str | None) -> str | None:
        _reject_raw_locator_text(value, "resolve_ref")
        return value


class MaterialPackFeedbackArtifact(ActionFeedbackArtifactBase):
    type: Literal["material_pack"]
    option: str | None = None


class WeeklyReportFeedbackArtifact(ActionFeedbackArtifactBase):
    type: Literal["weekly_report"]
    period: str | None = None
    report_date: str | None = None


class MonthlyReportFeedbackArtifact(ActionFeedbackArtifactBase):
    type: Literal["monthly_report"]
    period: str | None = None
    report_date: str | None = None


ActionFeedbackArtifact = Annotated[
    Union[
        MaterialPackFeedbackArtifact,
        WeeklyReportFeedbackArtifact,
        MonthlyReportFeedbackArtifact,
    ],
    Field(discriminator="type"),
]


class ActionExecutionFeedback(StrictModel):
    action_type: ActionExecutionType
    status: ActionExecutionStatus
    action_id: str | None = None
    artifact: ActionFeedbackArtifact | None = None
    adapter_result: dict[str, Any] = Field(default_factory=dict)

    @field_validator("adapter_result")
    @classmethod
    def validate_adapter_result_is_sanitized(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_sensitive_adapter_result(value)
        return value

    @model_validator(mode="after")
    def validate_artifact_matches_action(self):
        expected = {
            "send_material_pack": "material_pack",
            "send_weekly_report": "weekly_report",
            "send_monthly_report": "monthly_report",
        }.get(self.action_type)
        if expected is None:
            if self.artifact is not None:
                raise ValueError("non-artifact execution must not include artifact")
            return self
        if self.artifact is None:
            raise ValueError(f"{self.action_type} feedback requires artifact")
        if self.artifact.type != expected:
            raise ValueError(f"{self.action_type} feedback artifact must be {expected}")
        return self


class ActionFeedbackRequest(StrictModel):
    conversation_key: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    sender_id: str = Field(min_length=1)
    context_id: str | None = None
    response_id: str | None = None
    executions: list[ActionExecutionFeedback] = Field(default_factory=list)


class ActionFeedbackResponse(StrictModel):
    status: Literal["accepted"]
    stored: int


class HealthResponse(StrictModel):
    status: Literal["ok"]
    service: str
