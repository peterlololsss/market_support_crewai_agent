from typing import Literal

MaterialType = Literal["material", "weekly", "monthly"]
AvailableArtifactType = Literal["material_pack", "weekly_report", "monthly_report"]
ChannelType = Literal["bank", "non_bank", "unknown"]
AdapterResolveType = Literal[
    "material_pack",
    "weekly_report",
    "monthly_report",
    "sales_mention",
]
ReadCapability = Literal[
    "resolve_material_pack",
    "resolve_weekly_report",
    "resolve_monthly_report",
    "resolve_sales_mention",
    "query_internal_company_info",
    "query_weekly_report_product_list",
    "query_monthly_report_product_list",
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
]
