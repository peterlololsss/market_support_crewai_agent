IntentFrame schema summary:
- contract_version must be "intent-frame".
- user_need is a concise semantic summary.
- artifact_kind is one of material_pack, weekly_report, monthly_report, knowledge_answer, human_support, refusal, unclear, smalltalk.
- action_intent is one of send, answer, handoff, refuse, none.
- compliance contains is_compliant, reason_code, and a short reason.
- evidence_query is null except for knowledge_answer. For knowledge_answer, write a short semantic evidence query in normalized business terms, e.g. "衍复 公司介绍 基本信息" or "衍复 员工人数". Do not use it to answer from memory.
- strategy_mentions lists strategy names found in the current message or canonical context.
- selected_strategy is a single intended strategy or null.
- report_scope is channel_all, strategy, ambiguous, or none.
- ambiguity_slots can include artifact, strategy, report_scope, request_meaning.
- requested_capabilities must use allowed capability names only.
- confidence is 0.0 to 1.0.

Canonical JSON schema:
$intent_frame_schema_json
