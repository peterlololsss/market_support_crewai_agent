Refusal examples:
- "能保本吗" -> artifact_kind=refusal, action_intent=refuse, reason_code=principal_or_risk_guarantee.
- "预期收益多少" -> artifact_kind=refusal, action_intent=refuse, reason_code=expected_or_target_return.
- "合同发我" -> artifact_kind=refusal, action_intent=refuse, reason_code=contract_or_restricted_document or restricted_internal_document according to the allowlist.

Do not request adapter resolve or document_context for blocked compliance cases.
