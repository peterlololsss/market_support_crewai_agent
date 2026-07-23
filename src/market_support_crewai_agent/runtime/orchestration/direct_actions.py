from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, assert_never

from pydantic import ValidationError

from market_support_crewai_agent.runtime.context.pending import (
    PendingOutboundConfirmation,
)
from market_support_crewai_agent.runtime.evidence.adapter_client import (
    AdapterClientError,
)
from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectComposerOutput,
    DirectOutboundDraft,
)
from market_support_crewai_agent.runtime.orchestration.direct_content import (
    materialize_direct_content,
    outbound_confirmation_content,
)
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.orchestration.direct_target_resolution import (
    resolve_direct_target,
)
from market_support_crewai_agent.schemas import (
    AdapterResolveRequest,
    AdapterResolveResult,
    ExecutePreparedOutboundMessageAction,
    OutboundMessageTarget,
    OutboundTargetKind,
    OutboundTargetResolveResult,
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
    mode: Literal[
        "knowledge_answer",
        "smalltalk",
        "action",
        "clarification",
        "unable",
    ]
    response: ReplyResponse
    capability: Literal["document_context", "outbound_message"] | None = None
    pending_outbound_draft: DirectOutboundDraft | None = None


async def materialize_direct_output(
    output: DirectComposerOutput,
    *,
    adapter_client: DirectAdapterClient,
    action_history: list[ActionLedgerRecord],
    pending_outbound_draft: DirectOutboundDraft | None = None,
    pending_confirmation: PendingOutboundConfirmation | None = None,
) -> DirectMaterialization:
    merged_draft = _merge_outbound_draft(output, pending_outbound_draft)
    match output.response_mode:
        case "request_company_info":
            return DirectMaterialization(
                mode="unable",
                response=ReplyResponse(reply=output.reply, actions=[]),
            )
        case "answer_company_info":
            return DirectMaterialization(
                mode="knowledge_answer",
                response=ReplyResponse(reply=output.reply, actions=[]),
                capability="document_context",
            )
        case "smalltalk":
            return DirectMaterialization(
                mode="smalltalk",
                response=ReplyResponse(reply=output.reply, actions=[]),
            )
        case "prepare_outbound_message":
            return await _materialize_prepare(output, merged_draft, adapter_client)
        case "execute_prepared_outbound_message":
            return _materialize_execute(output, action_history, pending_confirmation)
        case "clarify":
            return DirectMaterialization(
                mode="clarification",
                response=ReplyResponse(reply=output.reply, actions=[]),
                pending_outbound_draft=merged_draft,
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
    draft: DirectOutboundDraft | None,
    adapter_client: DirectAdapterClient,
) -> DirectMaterialization:
    target = draft.target if draft is not None else None
    content = draft.content if draft is not None else None
    if target is None or content is None:
        return _clarification(output.reply.text, draft)
    resolution_status, resolved_target = await resolve_direct_target(
        adapter_client,
        target.kind,
        target.name,
    )
    if resolution_status == "ambiguous":
        return _clarification(
            "同时找到了同名群和渠道，请确认是发送到单个群，还是该渠道下当前可达的所有群。",
            draft,
        )
    if resolution_status == "unavailable":
        return _unable("暂时无法确认发送目标，请稍后再试。", draft)
    if resolution_status == "unresolved":
        retry_draft = DirectOutboundDraft(content=content)
        if resolved_target is not None:
            return _unable(
                (
                    f"已识别渠道「{resolved_target.display_name}」，但当前 "
                    f"0/{resolved_target.target_count} 个配置群可达；同名群也不可发送，"
                    "因此本次没有准备发送。请确认群在线状态、名称或稍后重试。"
                ),
                retry_draft,
            )
        return _unable(
            "同名渠道和群目前都不可发送，因此本次没有准备发送。请确认名称或稍后重试。",
            retry_draft,
        )
    assert resolved_target is not None
    used_fallback_kind = (
        target.kind is not None and target.kind != resolved_target.target_kind
    )
    try:
        outbound_content = await materialize_direct_content(content, adapter_client)
    except (AdapterClientError, ValidationError):
        return _unable("暂时无法确认要发送的内容，请稍后再试。")
    if outbound_content is None:
        return _unable("暂时无法确认要发送的报告，请稍后再试。")
    match resolved_target.target_kind:
        case "channel":
            target_label = "渠道"
        case "group":
            target_label = "群"
        case unreachable:
            assert_never(unreachable)
    sections: list[str] = []
    if target.kind is None:
        sections.append(f"已找到可发送的{target_label}。")
    if used_fallback_kind:
        sections.append(f"原请求目标类型当前不可发送，已改用可发送的同名{target_label}。")
    if resolved_target.resolved_count < resolved_target.target_count:
        sections.append(
            f"当前可发送到 {resolved_target.resolved_count}/{resolved_target.target_count} 个群。"
        )
    sections.append(f"发送目标：{target_label}「{resolved_target.display_name}」")
    sections.append(outbound_confirmation_content(outbound_content))
    sections.append("请确认是否发送？")
    reply = PrimaryReply(
        kind="clarification",
        text="\n\n".join(sections),
        mentions=[],
    )
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
        response=ReplyResponse(reply=reply, actions=[action]),
        capability="outbound_message",
    )


def _materialize_execute(
    output: DirectComposerOutput,
    action_history: list[ActionLedgerRecord],
    pending_confirmation: PendingOutboundConfirmation | None,
) -> DirectMaterialization:
    confirmation_ref = _confirmation_ref_for_pending(
        output,
        action_history,
        pending_confirmation,
    )
    if confirmation_ref is None:
        if pending_confirmation is not None:
            return _clarification("发送准备尚未完成，请稍后再确认。")
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


def _confirmation_ref_for_pending(
    output: DirectComposerOutput,
    action_history: list[ActionLedgerRecord],
    pending: PendingOutboundConfirmation | None,
) -> str | None:
    for record in reversed(action_history):
        execution = record.execution
        if execution.action_type != "prepare_outbound_message":
            continue
        if pending is not None and (
            record.response_id != pending.response_id
            or execution.action_id != pending.action.action_id
        ):
            continue
        value = execution.adapter_result.get("confirmation_ref")
        if not isinstance(value, str) or not value:
            return None
        if pending is not None or value == output.confirmation_ref:
            return value
    return None


def _clarification(
    text: str,
    pending_outbound_draft: DirectOutboundDraft | None = None,
) -> DirectMaterialization:
    return DirectMaterialization(
        mode="clarification",
        response=ReplyResponse(
            reply=PrimaryReply(kind="clarification", text=text, mentions=[]),
            actions=[],
        ),
        pending_outbound_draft=pending_outbound_draft,
    )


def _merge_outbound_draft(
    output: DirectComposerOutput,
    pending: DirectOutboundDraft | None,
) -> DirectOutboundDraft | None:
    if output.response_mode not in {"clarify", "prepare_outbound_message"}:
        return None
    target = output.target or (pending.target if pending is not None else None)
    content = output.content or (pending.content if pending is not None else None)
    if target is None and content is None:
        return None
    return DirectOutboundDraft(target=target, content=content)


def _unable(
    text: str,
    pending_outbound_draft: DirectOutboundDraft | None = None,
) -> DirectMaterialization:
    return DirectMaterialization(
        mode="unable",
        response=ReplyResponse(
            reply=PrimaryReply(kind="unable_to_answer", text=text, mentions=[]),
            actions=[],
        ),
        pending_outbound_draft=pending_outbound_draft,
    )
