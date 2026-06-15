ReplyAlignmentVerdict compact schema:
{
  "contract_version": "reply-alignment-verdict",
  "aligned": true|false,
  "safe_to_return": true|false,
  "failure_code": "none|wrong_intent|wrong_artifact|wrong_action|wrong_strategy|wrong_report_scope|missing_answer|missing_evidence|unsupported_claim|policy_or_compliance_mismatch|unsafe_action|composer_drift|ambiguous_request",
  "rationale": "short reason",
  "remediation": "none|replan|refetch_document_context|recompose|return_clarification|return_unable",
  "refined_evidence_query": null or "short query",
  "planner_feedback": null or "short correction for planner retry",
  "composer_feedback": null or "short correction for composer retry",
  "confidence": 0.0
}

For aligned=true: safe_to_return=true, failure_code="none", remediation="none".
For refetch_document_context: refined_evidence_query must be non-empty.
