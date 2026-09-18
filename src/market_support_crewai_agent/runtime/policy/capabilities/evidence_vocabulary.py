from __future__ import annotations

from typing import Literal

EvidenceFactTypeV2 = Literal[
    "material_pack_resolvable",
    "weekly_report_resolvable",
    "monthly_report_resolvable",
    "sales_mention_resolvable",
    "report_period",
    "report_scope_summary",
    "report_scope_match",
    "report_scope_products",
    "report_scope_unavailable",
    "recent_executed_action",
    "document_context",
    "document_context_unavailable",
]
EvidenceSourceTypeV2 = Literal[
    "adapter_resolve",
    "adapter_report_scope",
    "action_ledger",
    "document_mcp",
    "approved_static_knowledge",
    "conversation_history",
    "user_upload",
    "current_artifact",
    "adapter_context",
    "user_message",
    "assistant_message",
    "history_summary",
    "retrieved_doc",
    "tool_result",
]
EvidenceArtifactTypeV2 = Literal[
    "material_pack",
    "weekly_report",
    "monthly_report",
    "document_context",
    "adapter_context",
    "history",
    "user_upload",
    "unknown",
]
EvidenceScopeMatchFieldV2 = Literal[
    "channel_id",
    "channel_kind",
    "material_pack_option",
    "time_range",
    "product_id",
    "product_ids",
    "artifact_type",
]
HistoryRoleV1 = Literal["user", "assistant"]
FallbackPolicyV2 = Literal["clarify", "abstain", "ignore"]
StaleDataActionV1 = Literal["allow", "abstain", "refresh"]
