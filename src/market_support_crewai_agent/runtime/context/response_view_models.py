from __future__ import annotations

from datetime import date
from typing import ClassVar, Final, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.policy.capabilities import ResponseMode
from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.schemas.type_ids import (
    AdapterResolveType,
    AvailableArtifactType,
    OutboundActionType,
    ReplyKind,
)


class _FrozenResponseView(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class ResponseDirectiveViewV1(_FrozenResponseView):
    contract_version: Literal["response-directive-view.v1"] = (
        "response-directive-view.v1"
    )
    mode: ResponseMode
    reply_kind: ReplyKind
    reason_code: str = Field(
        min_length=1,
        max_length=120,
        pattern=r"^[a-z][a-z0-9_]{0,119}$",
    )
    requires_composer: bool
    composer_stage: Literal["knowledge_composer", "smalltalk_composer"] | None = None
    deterministic_text: str | None = Field(default=None, max_length=4_000)
    mentions_requested: bool
    action_intent_count: int = Field(ge=0, le=3)

    @model_validator(mode="after")
    def validate_composer_binding(self) -> ResponseDirectiveViewV1:
        if self.requires_composer != (self.composer_stage is not None):
            raise ContextViewInvariantError(
                "response_directive_composer_binding_mismatch"
            )
        if self.requires_composer and self.deterministic_text is not None:
            raise ContextViewInvariantError("response_directive_composer_text_conflict")
        return self


class PreflightFactViewV1(_FrozenResponseView):
    contract_version: Literal["preflight-fact-view.v1"] = "preflight-fact-view.v1"
    resolve_type: AdapterResolveType
    status: Literal[
        "resolved",
        "missing",
        "ambiguous",
        "forbidden",
        "temporarily_unavailable",
        "adapter_unavailable",
    ]
    display_name: str | None = Field(default=None, max_length=120)
    reason_code: str = Field(min_length=1, max_length=120)
    resolve_ref_available: bool
    artifact_type: AvailableArtifactType | None = None
    material_pack_option: str | None = Field(default=None, max_length=80)
    period: str | None = Field(default=None, max_length=40)
    report_date: str | None = Field(default=None, max_length=10)

    @field_validator("report_date")
    @classmethod
    def validate_iso_report_date(cls, value: str | None) -> str | None:
        if value is not None:
            _ = date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def validate_artifact_binding(self) -> PreflightFactViewV1:
        expected_artifact = _ARTIFACT_BY_RESOLVE[self.resolve_type]
        if self.artifact_type != expected_artifact:
            raise ContextViewInvariantError("preflight_view_artifact_type_mismatch")
        return self


class EffectiveOutputCeilingsViewV1(_FrozenResponseView):
    contract_version: Literal["effective-output-ceilings.v1"] = (
        "effective-output-ceilings.v1"
    )
    allowed_reply_kinds: tuple[ReplyKind, ...] = Field(min_length=1, max_length=5)
    actions_allowed: Literal[False] = False
    mentions_allowed: bool
    max_actions: Literal[0] = 0
    max_mentions: int = Field(ge=0, le=1)
    max_reply_chars: int = Field(ge=1, le=4_000)

    @model_validator(mode="after")
    def validate_canonical_ceilings(self) -> EffectiveOutputCeilingsViewV1:
        if self.allowed_reply_kinds != tuple(sorted(set(self.allowed_reply_kinds))):
            raise ContextViewInvariantError("output_ceilings_reply_kinds_not_canonical")
        if not self.mentions_allowed and self.max_mentions != 0:
            raise ContextViewInvariantError("output_ceilings_mentions_mismatch")
        return self


class CandidateMentionViewV1(_FrozenResponseView):
    contract_version: Literal["candidate-mention-view.v1"] = "candidate-mention-view.v1"
    type: Literal["sales"] = "sales"
    reason: str | None = Field(default=None, max_length=240)


class CandidateActionViewV1(_FrozenResponseView):
    type: OutboundActionType
    resolve_type: Literal["material_pack", "weekly_report", "monthly_report"]
    resolve_ref_available: bool
    material_pack_option: str | None = Field(default=None, max_length=80)
    period: str | None = Field(default=None, max_length=40)
    report_date: str | None = Field(default=None, max_length=10)

    @field_validator("report_date")
    @classmethod
    def validate_iso_report_date(cls, value: str | None) -> str | None:
        if value is not None:
            _ = date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def validate_action_binding(self) -> CandidateActionViewV1:
        expected_resolve_type = _RESOLVE_BY_ACTION[self.type]
        if self.resolve_type != expected_resolve_type:
            raise ContextViewInvariantError("candidate_action_resolve_type_mismatch")
        if self.resolve_type == "material_pack":
            if self.period is not None or self.report_date is not None:
                raise ContextViewInvariantError(
                    "candidate_action_report_metadata_forbidden"
                )
        elif self.material_pack_option is not None:
            raise ContextViewInvariantError(
                "candidate_action_material_option_forbidden"
            )
        return self


_ARTIFACT_BY_RESOLVE: Final[dict[AdapterResolveType, AvailableArtifactType | None]] = {
    "material_pack": "material_pack",
    "weekly_report": "weekly_report",
    "monthly_report": "monthly_report",
    "sales_mention": None,
}
_RESOLVE_BY_ACTION: Final[
    dict[
        OutboundActionType,
        Literal["material_pack", "weekly_report", "monthly_report"],
    ]
] = {
    "send_material_pack": "material_pack",
    "send_weekly_report": "weekly_report",
    "send_monthly_report": "monthly_report",
}


class CandidateReplyViewV1(_FrozenResponseView):
    contract_version: Literal["candidate-reply-view.v1"] = "candidate-reply-view.v1"
    reply_kind: ReplyKind
    text: str = Field(max_length=4_000)
    mentions: tuple[CandidateMentionViewV1, ...] = Field(default=(), max_length=1)
    actions: tuple[CandidateActionViewV1, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def validate_no_reply_shape(self) -> CandidateReplyViewV1:
        if self.reply_kind == "no_reply" and (
            self.text or self.mentions or self.actions
        ):
            raise ContextViewInvariantError("candidate_no_reply_payload_forbidden")
        return self
