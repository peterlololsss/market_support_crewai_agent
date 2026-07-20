from __future__ import annotations

from typing import Annotated, Literal, assert_never

from pydantic import Field, model_validator

from market_support_crewai_agent.schemas import (
    OutboundTargetKind,
    PrimaryReply,
    StrictModel,
)


class DirectComposerContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DirectTargetDraft(StrictModel):
    kind: OutboundTargetKind
    name: str = Field(min_length=1, max_length=128)


class DirectTextDraft(StrictModel):
    kind: Literal["text"]
    text: str = Field(min_length=1, max_length=4000)


class DirectLinkDraft(StrictModel):
    kind: Literal["link"]
    url: str = Field(min_length=1, max_length=2048)
    label: str | None = Field(default=None, min_length=1, max_length=128)


class DirectLinkCardDraft(StrictModel):
    kind: Literal["link_card"]
    title: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=512)
    url: str = Field(min_length=1, max_length=2048)


class DirectReportCardDraft(StrictModel):
    kind: Literal["report_card"]
    report_kind: Literal["weekly_report", "monthly_report"]
    source_channel: str = Field(min_length=1, max_length=128)


DirectContentDraft = Annotated[
    DirectTextDraft | DirectLinkDraft | DirectLinkCardDraft | DirectReportCardDraft,
    Field(discriminator="kind"),
]


class DirectComposerOutput(StrictModel):
    contract_version: Literal["direct-composer"] = "direct-composer"
    response_mode: Literal[
        "answer_company_info",
        "prepare_outbound_message",
        "execute_prepared_outbound_message",
        "clarify",
        "abstain",
    ]
    claims: list[str] = Field(default_factory=list, max_length=20)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    reply: PrimaryReply
    target: DirectTargetDraft | None = None
    content: DirectContentDraft | None = None
    confirmation_ref: str | None = None

    @model_validator(mode="after")
    def validate_mode_shape(self):
        if self.reply.mentions:
            raise DirectComposerContractError("direct_composer_mentions_forbidden")
        match self.response_mode:
            case "answer_company_info":
                if self.reply.kind != "answer" or not self.reply.text.strip():
                    raise DirectComposerContractError("direct_company_answer_required")
                if (
                    self.target is not None
                    or self.content is not None
                    or self.confirmation_ref
                ):
                    raise DirectComposerContractError(
                        "direct_company_outbound_forbidden"
                    )
            case "prepare_outbound_message":
                if self.reply.kind != "clarification" or not self.reply.text.strip():
                    raise DirectComposerContractError(
                        "direct_prepare_confirmation_required"
                    )
                if self.target is None or self.content is None or self.confirmation_ref:
                    raise DirectComposerContractError("direct_prepare_shape_invalid")
                if self.claims or self.evidence_ids:
                    raise DirectComposerContractError(
                        "direct_prepare_evidence_forbidden"
                    )
            case "execute_prepared_outbound_message":
                if self.reply.kind != "answer" or self.reply.text.strip():
                    raise DirectComposerContractError("direct_execute_reply_invalid")
                if self.target is not None or self.content is not None:
                    raise DirectComposerContractError(
                        "direct_execute_mutation_forbidden"
                    )
                if not self.confirmation_ref:
                    raise DirectComposerContractError(
                        "direct_execute_confirmation_required"
                    )
                if self.claims or self.evidence_ids:
                    raise DirectComposerContractError(
                        "direct_execute_evidence_forbidden"
                    )
            case "clarify":
                if self.reply.kind != "clarification" or not self.reply.text.strip():
                    raise DirectComposerContractError("direct_clarification_required")
                self._validate_no_action_fields()
            case "abstain":
                if self.reply.kind != "unable_to_answer" or not self.reply.text.strip():
                    raise DirectComposerContractError("direct_abstention_required")
                self._validate_no_action_fields()
            case unreachable:
                assert_never(unreachable)
        return self

    def _validate_no_action_fields(self) -> None:
        if self.target is not None or self.content is not None or self.confirmation_ref:
            raise DirectComposerContractError("direct_non_action_outbound_forbidden")
        if self.claims or self.evidence_ids:
            raise DirectComposerContractError("direct_non_answer_evidence_forbidden")
