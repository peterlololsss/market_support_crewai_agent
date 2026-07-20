DirectComposerOutput compact schema:
{
  "contract_version": "direct-composer",
  "response_mode": "answer_company_info|prepare_outbound_message|execute_prepared_outbound_message|clarify|abstain",
  "claims": [],
  "evidence_ids": [],
  "reply": {
    "kind": "answer|clarification|unable_to_answer",
    "text": "customer-visible text; empty only for execute mode",
    "mentions": []
  },
  "target": null or {"kind": "channel|group", "name": "exact logical display name"},
  "content": null or one of:
    {"kind": "text", "text": "exact outbound text"}
    {"kind": "link", "url": "exact HTTPS URL", "label": null or "label"}
    {"kind": "link_card", "title": "title", "description": "description", "url": "exact HTTPS URL"}
    {"kind": "report_card", "report_kind": "weekly_report|monthly_report", "source_channel": "exact source channel"},
  "confirmation_ref": null or "exact ref from executed prepare feedback"
}

Mode shapes are exclusive:
- answer_company_info: answer reply plus grounded claims/evidence_ids; no outbound fields.
- prepare_outbound_message: clarification reply plus target/content; no claims, evidence_ids, or confirmation_ref.
- execute_prepared_outbound_message: empty answer reply plus confirmation_ref; no target/content, claims, or evidence_ids.
- clarify: clarification reply only.
- abstain: unable_to_answer reply only.
