from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from market_support_crewai_agent.runtime.domain.planning.models import (
    ActionIntentSpec,
    AdapterResolveSpec,
    ComplianceDecision,
    ExecutionPlan,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.validation.guardrail_types import make_decision
from market_support_crewai_agent.schemas import OutboundActionType, ReplyRequest

DirectSendStatus = Literal[
    "no_match",
    "direct_action",
    "needs_material_pack_option",
]


@dataclass(frozen=True)
class DirectSendCommandResult:
    status: DirectSendStatus
    plan: ExecutionPlan | None = None
    reason_code: str = ""
    pattern_id: str = ""

    @property
    def matched(self) -> bool:
        return self.status != "no_match" and self.plan is not None

    @property
    def requires_confirmation(self) -> bool:
        return self.status == "needs_material_pack_option"


@dataclass(frozen=True)
class _ArtifactCommand:
    artifact_kind: Literal["material_pack", "weekly_report", "monthly_report"]
    capability: Literal["material_pack", "weekly_report", "monthly_report"]
    action_type: OutboundActionType
    resolve_type: Literal["material_pack", "weekly_report", "monthly_report"]
    user_need: str


# Closed-set action grammar only; never add product, strategy, or report-scope selectors here.
_COMMAND_PREFIX = (
    r"(?:(?:老师|请|麻烦|烦请|辛苦|"
    r"劳驾|帮忙|帮我)[,，:：]?){0,2}"
)
_SEND_VERB = (
    r"(?:给我发|发给我|发我|"
    r"发送|转发|发|来(?:个|一份|一下)?)"
)
_FILLER = (
    r"(?:(?:一下|下|一份|一个|个|"
    r"最新|本期|当前|这份|这个|"
    r"一版|版))*"
)
_SUFFIX = r"(?:给我)?"
_TRAILING_PUNCTUATION = " ,，.。!！?？~～;；"


def _pattern_for_aliases(aliases: tuple[str, ...]) -> str:
    artifact = "(?:" + "|".join(aliases) + ")"
    return (
        r"^"
        + _COMMAND_PREFIX
        + _SEND_VERB
        + _FILLER
        + artifact
        + _FILLER
        + _SUFFIX
        + r"$"
    )


_ARTIFACTS: tuple[tuple[_ArtifactCommand, re.Pattern[str]], ...] = (
    (
        _ArtifactCommand(
            artifact_kind="weekly_report",
            capability="weekly_report",
            action_type="send_weekly_report",
            resolve_type="weekly_report",
            user_need="direct send weekly report command",
        ),
        re.compile(
            _pattern_for_aliases(
                (
                    r"周报",
                    r"周度报告",
                )
            ),
            flags=re.IGNORECASE,
        ),
    ),
    (
        _ArtifactCommand(
            artifact_kind="monthly_report",
            capability="monthly_report",
            action_type="send_monthly_report",
            resolve_type="monthly_report",
            user_need="direct send monthly report command",
        ),
        re.compile(
            _pattern_for_aliases(
                (
                    r"月报",
                    r"月度报告",
                )
            ),
            flags=re.IGNORECASE,
        ),
    ),
    (
        _ArtifactCommand(
            artifact_kind="material_pack",
            capability="material_pack",
            action_type="send_material_pack",
            resolve_type="material_pack",
            user_need="direct send material pack command",
        ),
        re.compile(
            _pattern_for_aliases(
                (
                    r"材料包",
                    r"推介材料",
                    r"产品材料",
                    r"路演材料",
                    r"一页通",
                    r"开放日历",
                    r"ppt",
                )
            ),
            flags=re.IGNORECASE,
        ),
    ),
)


def match_direct_send_command(
    request: ReplyRequest,
    policy: PolicyManifest,
) -> DirectSendCommandResult:
    """Match only narrow, closed-set send commands before the LLM planner."""

    normalized = _normalize_message(request.message)
    if not normalized:
        return DirectSendCommandResult(status="no_match", reason_code="empty_message")

    command = _match_direct_artifact(normalized)
    if command is None:
        return DirectSendCommandResult(status="no_match", reason_code="no_match")

    pattern_id = f"direct_send.{command.resolve_type}"
    if command.action_type not in policy.allowed_outbound_actions:
        return DirectSendCommandResult(
            status="no_match",
            reason_code="action_not_allowed_by_policy",
            pattern_id=pattern_id,
        )

    if command.action_type == "send_material_pack" and policy.material_pack_options:
        return DirectSendCommandResult(
            status="needs_material_pack_option",
            plan=_confirmation_plan(command, policy),
            reason_code="material_pack_option_confirmation_required",
            pattern_id=pattern_id,
        )

    return DirectSendCommandResult(
        status="direct_action",
        plan=_action_plan(command),
        reason_code="direct_send_command_matched",
        pattern_id=pattern_id,
    )


def _normalize_message(message: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(message or ""))
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.strip(_TRAILING_PUNCTUATION)


def _match_direct_artifact(normalized_message: str) -> _ArtifactCommand | None:
    for command, pattern in _ARTIFACTS:
        if pattern.fullmatch(normalized_message):
            return command
    return None


def _action_plan(command: _ArtifactCommand) -> ExecutionPlan:
    return ExecutionPlan(
        user_need=command.user_need,
        artifact_kind=command.artifact_kind,
        response_mode="action",
        compliance=_compliant_decision(),
        capabilities=[command.capability],
        adapter_resolves=[
            AdapterResolveSpec(resolve_type=command.resolve_type),
            AdapterResolveSpec(resolve_type="sales_mention"),
        ],
        action_intents=[
            ActionIntentSpec(
                action_type=command.action_type,
                capability=command.capability,
            )
        ],
        guardrail_decisions=[
            make_decision(
                "allow",
                "input",
                "direct_send_command_matched",
                metadata={
                    "action_type": command.action_type,
                    "artifact_kind": command.artifact_kind,
                    "pattern_id": f"direct_send.{command.resolve_type}",
                },
            )
        ],
        confidence=1.0,
    )


def _confirmation_plan(
    command: _ArtifactCommand,
    policy: PolicyManifest,
) -> ExecutionPlan:
    return ExecutionPlan(
        user_need="confirm material pack option before direct send",
        artifact_kind=command.artifact_kind,
        response_mode="clarification",
        compliance=_compliant_decision(),
        guardrail_decisions=[
            make_decision(
                "require_confirmation",
                "input",
                "material_pack_option_confirmation_required",
                metadata={
                    "action_type": command.action_type,
                    "artifact_kind": command.artifact_kind,
                    "material_pack_option_count": len(policy.material_pack_options),
                    "pattern_id": f"direct_send.{command.resolve_type}",
                },
            )
        ],
        ambiguity_slots=["material_pack_option"],
        confidence=1.0,
    )


def _compliant_decision() -> ComplianceDecision:
    return ComplianceDecision(
        is_compliant=True,
        reason_code="compliant_product_request",
        reason="deterministic direct send command",
    )
