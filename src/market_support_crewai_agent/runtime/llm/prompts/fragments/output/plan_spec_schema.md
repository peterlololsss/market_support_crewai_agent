PlanSpec compact schema:
{
  "contract_version": "plan-spec",
  "plan_id": "stable short id for this plan",
  "user_intent_summary": "short semantic summary",
  "plan_units": [
    {
      "unit_id": "unit-1",
      "selected_capability_id": "one id from Capability registry JSON",
      "domain_scope": {
        "channel_id": "DomainContext channel id or unknown",
        "channel_kind": "bank|non_bank|unknown",
        "material_pack_option": null or "exact value from available_artifacts material_pack.options",
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
      "evidence_contract_ref": null or "selected_capability_id:evidence_contract",
      "evidence_contract": null,
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
  ],
  "risk_flags": []
}

Rules:
- Output one plan_units item per atomic user intent. Each unit selects exactly one selected_capability_id from Capability registry JSON and copies its artifact/tool/source boundaries into the matching unit fields.
- Sends: answerability_policy=send and selected_capability_id should be an action capability whose manifest matches the requested artifact/action. Multiple requested send artifacts require multiple send units.
- Knowledge answers: answerability_policy=answer and selected_capability_id should be an answer or summary capability whose manifest evidence contract can support the requested answer.
- Mixed answer plus send requests require both an answer unit and a send unit.
- Missing user-resolvable slot: clarify only for artifact or material_pack_option. Missing adapter evidence, unavailable artifact, or missing report-scope coverage: do not clarify; select the requested capability and let deterministic evidence/decision return unable_to_answer or handoff. When send-vs-query wording is unclear, prefer the safest non-side-effect answer or abstention capability. Do not answer from history or model memory unless EvidenceContract explicitly allows history.
- Handoff, refusal, smalltalk, and no-reply: use the matching handoff/general capability from Capability registry JSON.
- EvidenceContract required_scope_match names structured fields only: channel_id, channel_kind, material_pack_option, time_range, product_id, product_ids, artifact_type.
