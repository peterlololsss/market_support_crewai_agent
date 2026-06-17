ComposerReplyOutput compact no-action schema:
{
  "contract_version": "composer-reply",
  "response_id": "",
  "response_mode": "answer|abstain|clarify",
  "claims": ["short factual claims; empty for abstain or clarify"],
  "evidence_ids": ["only IDs from Runtime Capability & Evidence Boundary allowed_evidence_ids"],
  "missing_inputs": ["required runtime inputs or artifacts that are missing"],
  "reply": {
    "kind": "answer|unable_to_answer|clarification",
    "text": "customer-visible text",
    "mentions": []
  },
  "actions": []
}

This composer must never output actions or mentions.
