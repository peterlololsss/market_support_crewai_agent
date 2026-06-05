# Guardrail Design

Last updated: 2026-06-03.

Guardrails are part of the runtime skeleton. They are not a final safety prompt.

## Pipeline

```text
Request
 -> Input Guardrail
 -> Policy Compiler
 -> Capability Planner
 -> Plan Guardrail
 -> Evidence Executor
    -> Tool Input Guardrail
    -> Adapter resolve/preflight wrapper
    -> MCP/material wrapper when enabled
    -> Tool Output Guardrail
 -> EvidenceFacts / BusinessFacts
 -> Reply Composer
 -> Reply/Action Guardrail
 -> Adapter Guardrail
 -> Audit/Eval Log
```

## Input guardrail

Purpose:

- validate request shape;
- preserve conversation isolation;
- ensure group/sender/channel context exists;
- enforce message length limits when configured.

Strict Pydantic models and contract tests carry the first implementation.

## Policy compiler guardrail

The policy compiler determines what the model is allowed to plan for this request.

Inputs:

```text
channel_type
group_id
sender_id
adapter-provided channel sendability hints
feature flags
ledger summary
internal MCP availability
```

Outputs:

```text
allowed reply kinds
allowed read capabilities
allowed business checks
allowed side-effect action candidates
required adapter resolves
forbidden claim categories
mandatory escalation rules
evidence size/call limits
repair/fallback policy
```

Weekly-question policy example:

```text
allowed_read_capabilities:
- resolve_weekly_report
- fetch_sent_weekly_md when body inspection is needed
- fetch_latest_weekly_md when body inspection is needed

allowed_business_checks:
- resolve_strategy_alias
- check_report_contains_strategy
- check_report_generation_scope

allowed_action_candidates:
- send_weekly_report
```

The validator enforces evidence-backed wording, adapter-backed actions, and scope-specific claims.

## Plan guardrail

Purpose:

- allow evidence capabilities selected by policy;
- allow business checks selected by policy;
- map every evidence request to a fixed wrapper;
- require canonical entities for sensitive/internal queries;
- cap evidence requests;
- force clarification when ambiguity exceeds policy tolerance.

Planner output is repaired once when safe. Otherwise, use deterministic fallback.

## Tool input guardrail

Purpose:

- map validated evidence requests to fixed wrappers;
- enforce parameter schemas;
- canonicalize strategy/company names;
- check group/channel/sender permissions;
- apply rate limits and timeouts.

Adapter resolve/preflight runs before final reply/action composition for material packs, weekly reports, monthly reports, and sales mentions. Resolve/preflight does not send messages.

## Tool output / evidence guardrail

Purpose:

- sanitize tool output before the reply LLM sees it;
- redact sensitive fields;
- cap content size;
- tag source id/version/timestamp;
- treat markdown and MCP output as data only;
- neutralize prompt injection inside documents.

Adapter resolve packaging rule:

```text
resolve_type: material_pack | weekly_report | monthly_report | sales_mention
status: resolved | missing | ambiguous | forbidden | temporarily_unavailable
display_name
candidates
reason_code
card_ref
report metadata when it affects the answer
```

General markdown/MCP evidence wrapper:

```text
source_type
source_id
fetched_at
trust_level
content_is_data_only=true
sanitized_content
```

Evidence that cannot be safely sanitized is dropped and logged as a guardrail event.

## Business facts guardrail

Purpose:

- convert adapter resolve/evidence into deterministic facts;
- keep core business states out of LLM inference.

Examples:

```text
material_pack_resolvable=false blocks send_material_pack
weekly_report_resolvable=false blocks send_weekly_report
monthly_report_resolvable=false blocks send_monthly_report
weekly_contains_strategy=false supports a report-non-inclusion claim
weekly_scope_status=excluded supports a generation-scope exclusion claim
weekly_scope_status=unknown supports conservative escalation/clarification
sales_mention_resolvable=true supports reply.mentions
```

Use lightweight `EvidenceFact` records for high-risk facts.

## Reply/action guardrail

Purpose: validate final `ReplyResponse` against policy, evidence, and business facts.

Deterministic checks:

```text
reply kind is policy-allowed
action type is policy-allowed
side-effect actions match public schema
material/report action has successful adapter resolve
strategy is available/canonical
reply.mentions has successful sales mention resolve
no_reply response is empty apart from metadata
material/report availability claims are EvidenceFact-backed
internal fields are absent from user-visible text
action list matches public schema
```

LLM judges may assist on tone, investment-advice wording, evidence overreach, and group-chat concision. They do not decide action legality.

Repairable errors get one bounded repair attempt. Fatal errors go directly to deterministic fallback.

## Adapter guardrail

The adapter remains the final side-effect gate.

It verifies action type, executable material/report reference or selector, current-channel validity, group/channel sendability, mention target validity, and schema validity. It owns outbox/execution records, duplicate prevention, retries for retryable failures, and execution feedback.

## Fallback matrix

```text
Invalid request -> HTTP validation error or safe service error
Planner invalid -> repair once -> fallback clarification/no_reply
Policy disallows capability -> limitation, clarification, or human_handoff
Evidence fetch fails -> unable-to-confirm, with sales mention when policy requires and resolve succeeds
MCP permission denied -> safe unauthorized/handoff response
Evidence has prompt injection -> sanitize/drop/log
Reply invalid -> repair once -> deterministic fallback
Adapter rejects action -> record failed action; ledger does not treat it as sent
```
