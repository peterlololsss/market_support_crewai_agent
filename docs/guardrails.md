# Guardrails

Last updated: 2026-06-17.

Guardrails are deterministic runtime checks. They are not safety-flavored prompt
text and they do not silently repair invalid model output.

## Phases

```text
input -> retrieval/evidence -> execution/tool -> output -> adapter
```

Implementation:

```text
src/market_support_crewai_agent/runtime/validation/request_input_guard.py
src/market_support_crewai_agent/runtime/validation/input_scope_guard.py
src/market_support_crewai_agent/runtime/validation/evidence_source_guard.py
src/market_support_crewai_agent/runtime/validation/execution_tool_guard.py
src/market_support_crewai_agent/runtime/validation/output_guard.py
src/market_support_crewai_agent/runtime/validation/reply_validator.py
```

## Input Guard

Input guard validates the request boundary and structured send scope. The
planner may emit `RequestedScope`; the guard compares it to the current
`PolicyManifest` and `DomainContext`.

## SendScope Policy

`SendScopePolicy` is derived from the request-scoped policy and current domain:

```text
allowed_capabilities
allowed_artifact_types
allowed_destinations
allowed_actions
required_user_confirmation
redaction_policy
```

Wrong-channel sends are blocked. Example: if the current `渠道` is "测试渠道A",
a request to send "测试渠道B的周报" must not be rewritten as A's `周报`.

Reason codes:

```text
send_scope_destination_outside_current_channel
ambiguous_destination_scope
unknown_strategy_scope
capability_not_allowed_by_send_scope_policy
action_not_allowed_by_send_scope_policy
```

## Retrieval / Evidence Guard

This phase enforces source scope before composition. It checks the selected
capability manifest and evidence contract.

Examples:

```text
材料包 question + only 周报 products -> abstain
月报 performance question + only 材料包 products -> abstain
selected 策略S1 + 策略S2材料包 evidence -> reject 策略S2 evidence
history-only evidence + allow_history=false -> reject
wrong 渠道 evidence -> reject
```

Common reason codes:

```text
required_evidence_missing
forbidden_source_type
source_type_not_allowed
artifact_type_not_allowed
history_source_not_current_artifact
source_not_evidence_by_default
channel_scope_mismatch
strategy_scope_mismatch
time_range_scope_mismatch
evidence_provenance_missing
```

## Execution / Tool Guard

The execution guard allows only fixed wrappers declared by policy and manifests:

```text
adapter_resolve.material_pack
adapter_resolve.weekly_report
adapter_resolve.monthly_report
adapter_resolve.sales_mention
document_mcp.get_documents
approved_static_knowledge.query
```

It blocks tool calls outside policy and artifact IDs that are not present in the
current `DomainContext`.

Reason codes:

```text
tool_not_allowed
invalid_artifact_id
```

## Output Guard

The output guard runs before `validate_reply`. It checks composed answers against
allowed evidence IDs and source scope. A composer answer must cite allowed
evidence when the capability requires evidence.

Reason codes:

```text
composer_evidence_ids_missing
composer_evidence_id_not_allowed
output_claims_supported
```

`validate_reply` then checks final public response shape and action legality:

```text
reply.kind is allowed
actions are in the validated directive
resolve_ref matches adapter evidence
report_scope is valid
reply.mentions are adapter-resolved
no pre-execution "已发送/请查收" claims
no raw locators or unsafe image markers
non-compliant replies use harness-owned refusal text
```

## Answerability Gate

`AnswerabilityGate` sits between evidence execution and output composition. It
returns:

```text
answer
clarify
abstain
```

This blocks the original source-mixing bug before the composer can answer.

Material-pack example:

```text
Input: 材料包里有哪些产品？
Evidence: 周报 has 产品A, no material_pack content
Output: unable_to_answer / material pack evidence missing
Blocked: listing 产品A from 周报
```

## Audit

Audit traces should contain enough to replay a decision without exposing raw
adapter internals:

```text
request/context ids
policy id/hash
canonical scope
PlanSpec / ExecutionPlan summary
guardrail decisions and reason codes
evidence source ids and scopes
BusinessFacts
AnswerabilityAssessment
ReplyResponse
validation result
optional alignment verdicts
```

Audit reason codes should be stable enough for CI assertions and dashboards.
