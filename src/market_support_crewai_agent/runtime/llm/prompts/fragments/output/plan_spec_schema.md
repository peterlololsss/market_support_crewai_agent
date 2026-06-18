PlanSpec compact schema:
{
  "contract_version": "plan-spec",
  "plan_id": "stable short id for this plan",
  "selected_capability_id": "one id from Capability registry JSON",
  "user_intent_summary": "short semantic summary",
  "domain_scope": {
    "channel_id": "DomainContext channel id or unknown",
    "channel_kind": "bank|non_bank|unknown",
    "material_pack_option": null or "exact value from request.material_pack_options",
    "product_ids": [],
    "time_range": null or {"period": null, "start": null, "end": null, "label": null}
  },
  "required_artifacts": [],
  "allowed_artifacts": [],
  "forbidden_artifacts": [],
  "required_tools": [],
  "answerability_policy": "answer|send|clarify|abstain|refuse|handoff|smalltalk|no_reply",
  "output_schema_ref": "selected_capability_id:output_schema",
  "output_schema": null,
  "evidence_contract_ref": "selected_capability_id:evidence_contract",
  "evidence_contract": null or inline EvidenceContract,
  "steps": [
    {
      "step_id": "step-1",
      "description": "bounded deterministic or composition step",
      "uses_artifacts": [],
      "required_artifacts": [],
      "allowed_artifacts": [],
      "forbidden_artifacts": [],
      "required_tools": [],
      "evidence_query": null
    }
  ],
  "acceptance_criteria": [],
  "abstention_cases": [],
  "risk_flags": []
}

Rules:
- Select exactly one selected_capability_id from Capability registry JSON. Copy its artifact/tool/source boundaries into the matching PlanSpec fields.
- Sends: answerability_policy=send and selected_capability_id should be an action capability whose manifest matches the requested artifact/action.
- Knowledge answers: answerability_policy=answer and selected_capability_id should be an answer or summary capability whose manifest evidence contract can support the requested answer.
- Missing evidence or missing required artifact: use answerability_policy=abstain or clarify according to the selected capability's fallback/abstention guidance. Do not answer from history or model memory unless the EvidenceContract explicitly allows history.
- Handoff, refusal, smalltalk, and no-reply: use the matching handoff/general capability from Capability registry JSON.
- EvidenceContract required_scope_match names structured fields only: channel_id, channel_kind, material_pack_option, time_range, product_id, product_ids, artifact_type.
