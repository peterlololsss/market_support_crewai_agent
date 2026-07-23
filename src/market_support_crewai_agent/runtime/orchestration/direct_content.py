from __future__ import annotations

from typing import Protocol, assert_never

from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectContentDraft,
    DirectLinkCardDraft,
    DirectLinkDraft,
    DirectReportCardDraft,
    DirectTextDraft,
)
from market_support_crewai_agent.schemas import (
    AdapterResolveRequest,
    AdapterResolveResult,
    OutboundLinkCardContent,
    OutboundLinkContent,
    OutboundMessageContent,
    OutboundReportCardContent,
    OutboundTextContent,
)


class DirectContentAdapter(Protocol):
    async def resolve_async(
        self,
        request: AdapterResolveRequest,
    ) -> AdapterResolveResult: ...


def outbound_confirmation_content(content: OutboundMessageContent) -> str:
    match content:
        case OutboundTextContent(text=text):
            return f"待发送原文：\n{text}"
        case OutboundLinkContent(url=url, label=label):
            label_prefix = f"{label}\n" if label else ""
            return f"待发送链接：\n{label_prefix}{url}"
        case OutboundLinkCardContent(title=title, description=description, url=url):
            return f"待发送链接卡片：\n标题：{title}\n说明：{description}\n链接：{url}"
        case OutboundReportCardContent(
            report_kind=report_kind,
            source_channel=source_channel,
            period=period,
            report_date=report_date,
        ):
            return (
                "待发送报告卡片：\n"
                f"类型：{report_kind}\n来源：{source_channel}\n"
                f"期间：{period}\n日期：{report_date}"
            )
        case unreachable:
            assert_never(unreachable)


async def materialize_direct_content(
    content: DirectContentDraft,
    adapter_client: DirectContentAdapter,
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
