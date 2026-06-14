ReplyResponse schema constraints for this composer:
- contract_version must be "reply".
- reply.kind must be answer only when document_context evidence supports the text.
- reply.kind may be unable_to_answer if evidence is missing or insufficient.
- reply.mentions must be [].
- actions must be [].
- Do not create send_material_pack, send_weekly_report, or send_monthly_report actions in this stage.

Canonical JSON schema:
$reply_response_schema_json
