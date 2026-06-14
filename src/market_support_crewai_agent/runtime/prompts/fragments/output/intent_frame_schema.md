IntentFrame schema summary:
- contract_version must be "intent-frame".
- user_need is a concise semantic summary.
- artifact_kind is one of material_pack, weekly_report, monthly_report, knowledge_answer, human_support, refusal, unclear, smalltalk.
- action_intent is one of send, answer, handoff, refuse, none.
- compliance contains is_compliant, reason_code, and a short reason.
- strategy_mentions lists strategy names found in the current message or canonical context.
- selected_strategy is a single intended strategy or null.
- report_scope is channel_all, strategy, ambiguous, or none.
- ambiguity_slots can include artifact, strategy, report_scope, request_meaning.
- requested_capabilities must use allowed capability names only.
- confidence is 0.0 to 1.0.

Canonical JSON schema:
$intent_frame_schema_json
