DirectComposerOutput compact schema:
{
  "contract_version": "direct-composer",
  "response_mode": "request_company_info|answer_company_info|smalltalk|prepare_outbound_message|execute_prepared_outbound_message|clarify|abstain",
  "pending_confirmation_resolution": "not_applicable|confirm|correct|cancel|topic_switch|ambiguous",
  "claims": [],
  "evidence_ids": [],
  "reply": {
    "kind": "answer|clarification|unable_to_answer",
    "text": "customer-visible text; empty only for execute mode",
    "mentions": []
  },
  "target": null or {"kind": "channel|group|null", "name": "exact logical display name"},
  "content": null or one of:
    {"kind": "text", "text": "exact outbound text"}
    {"kind": "link", "url": "exact HTTPS URL", "label": null or "label"}
    {"kind": "link_card", "title": "title", "description": "description", "url": "exact HTTPS URL"}
    {"kind": "report_card", "report_kind": "weekly_report|monthly_report", "source_channel": "exact source channel"},
  "confirmation_ref": null or "exact ref from executed prepare feedback"
}

Mode shapes are exclusive:
- request_company_info: unable_to_answer reply used only to request first-pass bounded retrieval; no claims, evidence_ids, or outbound fields.
- answer_company_info: answer reply plus grounded claims/evidence_ids; no outbound fields.
- smalltalk: answer reply only; no claims, evidence_ids, or outbound fields.
- prepare_outbound_message: clarification reply plus target/content; a partial draft is accepted only so deterministic runtime can safely continue slot collection; no claims, evidence_ids, or confirmation_ref.
- execute_prepared_outbound_message: empty answer reply plus confirmation_ref; no target/content, claims, or evidence_ids. Actual execution feedback is composed later.
- clarify: clarification reply plus any already-known target/content fields; no confirmation_ref, claims, or evidence_ids.
- abstain: unable_to_answer reply only.

Pending confirmation resolution is mandatory semantic control data:
- Use not_applicable only when Pending clarification context JSON has no active pending_confirmation.
- With an active pending_confirmation: confirm requires execute mode; correct requires a new prepare mode; cancel requires a non-action answer; topic_switch uses the new intent's mode but never executes the pending action; ambiguous requires clarify mode.
