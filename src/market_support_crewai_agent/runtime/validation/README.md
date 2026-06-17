# Runtime Validation

Validation is phase-specific. Import the guard for the phase you are in; do not add wrapper pipelines or compatibility shims.

```text
request_input_guard.py          raw /reply request boundary, before runtime work
input_scope_guard.py            requested send scope vs policy/domain context
plan_spec_verifier.py           planner PlanSpec contract
capability_manifest_verifier.py capability manifest contract
execution_tool_guard.py         deterministic wrapper/tool permission checks
evidence_source_guard.py        source/artifact/channel/strategy/history checks
answerability.py                can the selected evidence answer the plan?
output_guard.py                 composer evidence-id/source-use checks
reply_validator.py              final ReplyResponse and action postconditions
reply_alignment_verifier.py     optional semantic verifier verdict schema
```

Shared helpers live in `guardrail_common.py`; shared result/enums live in `guardrail_types.py`.
