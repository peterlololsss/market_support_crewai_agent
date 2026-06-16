IntentFrame compact schema:
{
  "contract_version": "intent-frame",
  "user_need": "short semantic summary",
  "artifact_kind": "material_pack|weekly_report|monthly_report|multi_action|knowledge_answer|human_support|refusal|unclear|smalltalk",
  "action_intent": "send|answer|handoff|refuse|none",
  "compliance": {
    "is_compliant": true|false|null,
    "reason_code": "one listed compliance reason code",
    "reason": "short reason"
  },
  "evidence_query": null or "short normalized query for knowledge_answer only",
  "strategy_mentions": [],
  "selected_strategy": null or "one canonical strategy",
  "report_scope": "channel_all|strategy|ambiguous|none",
  "ambiguity_slots": [],
  "requested_capabilities": [],
  "work_items": [
    {
      "intent": "answer|send|handoff",
      "capability": "material_pack|weekly_report|monthly_report|sales_mention|document_context",
      "evidence_query": null or "short normalized query for this answer item",
      "selected_strategy": null or "one canonical strategy for this item",
      "report_scope": "channel_all|strategy|ambiguous|none"
    }
  ],
  "confidence": 0.0
}

For refusal: artifact_kind="refusal", action_intent="refuse", requested_capabilities=[], ambiguity_slots=[], compliance.is_compliant=false.
For side-effect sends: compliance.is_compliant must be true. Unknown compliance cannot send.
For multiple explicit side-effect sends in one message: artifact_kind="multi_action", action_intent="send", requested_capabilities is the ordered list of concrete send capabilities, ambiguity_slots=[] unless a strategy/scope is genuinely unclear.
For mixed requests that contain both a factual answer and one or more sends: artifact_kind="multi_action", action_intent="send", and fill work_items in user-request order. Use intent="answer" for the factual part and intent="send" only for artifacts the user explicitly asks to send. Do not convert an answer item into a send because the same artifact is mentioned.
For knowledge_answer: action_intent="answer", and requested_capabilities must include the needed evidence source: ["document_context"] for internal company/product documents, ["weekly_report"] for weekly report period/date/scope/content questions, or ["monthly_report"] for monthly report period/date/scope/content questions. evidence_query may be null for report period/date questions because adapter resolve provides period metadata by default. Use evidence_query="report_scope_products" for "which products are in/generated this report" questions. Use evidence_query="report_scope_summary" for general report-scope summary questions without one named product/section.
For smalltalk: action_intent="none", requested_capabilities=[], report_scope="none".
