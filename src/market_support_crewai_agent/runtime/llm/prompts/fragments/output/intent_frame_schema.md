IntentFrame compact schema:
{
  "contract_version": "intent-frame",
  "user_need": "short semantic summary",
  "artifact_kind": "material_pack|weekly_report|monthly_report|knowledge_answer|human_support|refusal|unclear|smalltalk",
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
  "confidence": 0.0
}

For refusal: artifact_kind="refusal", action_intent="refuse", requested_capabilities=[], ambiguity_slots=[], compliance.is_compliant=false.
For side-effect sends: compliance.is_compliant must be true. Unknown compliance cannot send.
For knowledge_answer: action_intent="answer", requested_capabilities=["document_context"], evidence_query must be non-empty.
For smalltalk: action_intent="none", requested_capabilities=[], report_scope="none".
