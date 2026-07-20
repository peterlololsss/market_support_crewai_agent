from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, assert_never

from pydantic import ValidationError

from market_support_crewai_agent.runtime.evidence.adapter_client import (
    AdapterClientError,
)
from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectComposerOutput,
    DirectLinkCardDraft,
    DirectLinkDraft,
    DirectReportCardDraft,
    DirectTextDraft,
)
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.schemas import (
    AdapterResolveRequest,
    AdapterResolveResult,
    ExecutePreparedOutboundMessageAction,
    OutboundLinkCardContent,
    OutboundLinkContent,
    OutboundMessageContent,
    OutboundMessageTarget,
    OutboundReportCardContent,
    OutboundTargetKind,
    OutboundTargetResolveResult,
    OutboundTextContent,
    PrepareOutboundMessageAction,
    PrimaryReply,
    ReplyResponse,
)


class DirectAdapterClient(Protocol):
    async def resolve_outbound_target_async(
        self,
        target_kind: OutboundTargetKind,
        target_name: str,
    ) -> OutboundTargetResolveResult: ...

    async def resolve_async(
        self,
        request: AdapterResolveRequest,
    ) -> AdapterResolveResult: ...


@dataclass(frozen=True, slots=True)
class DirectMaterialization:
    mode: Literal["knowledge_answer", "action", "clarification", "unable"]
    response: ReplyResponse
    capability: Literal["document_context", "outbound_message"] | None = None


async def materialize_direct_output(
    output: DirectComposerOutput,
    *,
    adapter_client: DirectAdapterClient,
    action_history: list[ActionLedgerRecord],
) -> DirectMaterialization:
    match output.response_mode:
        case "answer_company_info":
            return DirectMaterialization(
                mode="knowledge_answer",
                response=ReplyResponse(reply=output.reply, actions=[]),
                capability="document_context",
            )
        case "prepare_outbound_message":
            return await _materialize_prepare(output, adapter_client)
        case "execute_prepared_outbound_message":
            return _materialize_execute(output, action_history)
        case "clarify":
            return DirectMaterialization(
                mode="clarification",
                response=ReplyResponse(reply=output.reply, actions=[]),
            )
        case "abstain":
            return DirectMaterialization(
                mode="unable",
                response=ReplyResponse(reply=output.reply, actions=[]),
            )
        case unreachable:
            assert_never(unreachable)


async def _materialize_prepare(
    output: DirectComposerOutput,
    adapter_client: DirectAdapterClient,
) -> DirectMaterialization:
    target = output.target
    content = output.content
    if target is None or content is None:
        return _clarification("请补充要发送到的群或渠道，以及具体发送内容。")
    try:
        resolved_target = await adapter_client.resolve_outbound_target_async(
            target.kind,
            target.name,
        )
    except AdapterClientError:
        return _unable("暂时无法确认发送目标，请稍后再试。")
    if resolved_target.status != "resolved":
        return _clarification("请确认要发送到的群或渠道的准确名称。")
    try:
        outbound_content = await _materialize_content(content, adapter_client)
    except (AdapterClientError, ValidationError):
        return _unable("暂时无法确认要发送的内容，请稍后再试。")
    if outbound_content is None:
        return _unable("暂时无法确认要发送的报告，请稍后再试。")
    try:
        action = PrepareOutboundMessageAction(
            type="prepare_outbound_message",
            target=OutboundMessageTarget(
                kind=resolved_target.target_kind,
                name=resolved_target.display_name,
                resolve_ref=resolved_target.resolve_ref,
            ),
            content=outbound_content,
        )
    except ValidationError:
        return _unable("暂时无法准备这条发送请求，请稍后再试。")
    return DirectMaterialization(
        mode="action",
        response=ReplyResponse(reply=output.reply, actions=[action]),
        capability="outbound_message",
    )


async def _materialize_content(
    content: DirectTextDraft
    | DirectLinkDraft
    | DirectLinkCardDraft
    | DirectReportCardDraft,
    adapter_client: DirectAdapterClient,
) -> OutboundMessageContent | None:
    match content:
        case DirectTextDraft(text=text):
            return OutboundTextContent(kind="text", text=text)
        case DirectLinkDraft(url=url, label=label):
            return OutboundLinkContent(kind="link", url=url, label=label)
        case DirectLinkCardDraft(title=title, description=description, url=url):
            return OutboundLinkCardContent(
                kind="link_card",
                title=title,
                description=description,
                url=url,
            )
        case DirectReportCardDraft(
            report_kind=report_kind,
            source_channel=source_channel,
        ):
            result = await adapter_client.resolve_async(
                AdapterResolveRequest(
                    resolve_type=(
                        "weekly_report"
                        if report_kind == "weekly_report"
                        else "monthly_report"
                    ),
                    dist_name=source_channel,
                )
            )
            if (
                result.status != "resolved"
                or not result.resolve_ref
                or not result.period
                or not result.report_date
            ):
                return None
            return OutboundReportCardContent(
                kind="report_card",
                report_kind=report_kind,
                resolve_ref=result.resolve_ref,
                source_channel=result.display_name,
                period=result.period,
                report_date=result.report_date,
            )
        case unreachable:
            assert_never(unreachable)


def _materialize_execute(
    output: DirectComposerOutput,
    action_history: list[ActionLedgerRecord],
) -> DirectMaterialization:
    confirmation_ref = output.confirmation_ref or ""
    if confirmation_ref not in _prepared_confirmation_refs(action_history):
        return _clarification("请先说明发送目标和内容，确认后我再发送。")
    try:
        action = ExecutePreparedOutboundMessageAction(
            type="execute_prepared_outbound_message",
            confirmation_ref=confirmation_ref,
        )
    except ValidationError:
        return _unable("这条确认已失效，请重新发起发送请求。")
    return DirectMaterialization(
        mode="action",
        response=ReplyResponse(reply=output.reply, actions=[action]),
        capability="outbound_message",
    )


def _prepared_confirmation_refs(
    action_history: list[ActionLedgerRecord],
) -> frozenset[str]:
    refs: set[str] = set()
    for record in action_history:
        execution = record.execution
        if (
            execution.status == "executed"
            and execution.action_type == "prepare_outbound_message"
        ):
            value = execution.adapter_result.get("confirmation_ref")
            if isinstance(value, str) and value:
                refs.add(value)
    return frozenset(refs)


def _clarification(text: str) -> DirectMaterialization:
    return DirectMaterialization(
        mode="clarification",
        response=ReplyResponse(
            reply=PrimaryReply(kind="clarification", text=text, mentions=[]),
            actions=[],
        ),
    )


def _unable(text: str) -> DirectMaterialization:
    return DirectMaterialization(
        mode="unable",
        response=ReplyResponse(
            reply=PrimaryReply(kind="unable_to_answer", text=text, mentions=[]),
            actions=[],
        ),
    )
