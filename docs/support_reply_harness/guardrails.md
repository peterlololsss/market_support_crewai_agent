# Guardrail Design

Guardrails are part of the runtime skeleton, not a final safety prompt.

## Guardrail pipeline

```text
Request
 -> Input Guardrail
 -> Capability Planner
 -> Plan Guardrail
 -> Evidence Executor
    -> Tool Input Guardrail
    -> Adapter resolve/preflight wrapper
    -> MCP/material wrapper if needed
    -> Tool Output Guardrail
 -> EvidenceFacts / Business Facts
 -> Reply Composer
 -> Reply/Action Guardrail
 -> Adapter Guardrail
 -> Audit/Eval Log
```

## Input guardrail

Purpose:

- validate request shape;
- preserve conversation isolation;
- reject removed/unknown trigger fields;
- ensure group/sender/channel context exists;
- enforce message length limits if needed.

Current tests already cover parts of this through strict Pydantic models.

## Policy compiler guardrail

The policy compiler determines what the model is allowed to plan for this
request.

Inputs:

- `channel_type`
- `group_id`
- `sender_id`
- adapter-provided channel sendability hints
- feature flags
- ledger summary
- internal MCP availability

Outputs:

- allowed reply kinds;
- allowed read capabilities;
- allowed business checks;
- allowed side-effect action candidates;
- required adapter resolves;
- forbidden claims;
- mandatory escalation rules;
- evidence size/call limits;
- repair/fallback policy.

Example weekly question policy:

```text
allowed_read_capabilities:
- resolve_weekly_report
- fetch_sent_weekly_md if body inspection is needed
- fetch_latest_weekly_md if body inspection is needed

allowed_business_checks:
- resolve_strategy_alias
- check_report_contains_strategy
- check_report_generation_scope

allowed_action_candidates:
- send_weekly_report

forbidden:
- generate missing report section
- explain absence without evidence
- promise future inclusion
- send report unless adapter resolve passed
```

## Plan guardrail

Purpose:

- reject disallowed evidence capabilities;
- reject unsupported business checks;
- reject raw MCP tool names;
- reject side effects as evidence calls;
- reject unavailable material-pack/report/action candidates;
- require canonical entities for sensitive/internal queries;
- cap number of evidence requests;
- force clarification when ambiguity is too high.

Repair:

- allow one planner repair attempt with validation errors;
- if repair fails, use deterministic fallback.

Planner must not be allowed to assert final business facts such as "strategy is
outside scope" without evidence/business checks.

## Tool input guardrail

Purpose:

- map validated evidence requests to fixed wrappers;
- enforce parameter schemas;
- canonicalize strategy/company names;
- check group/channel/sender permissions;
- block raw ambiguous user text from internal systems;
- apply rate limits and timeouts.

Rules:

- no arbitrary MCP tool execution;
- no model-selected raw tool names;
- adapter resolve/preflight must happen before final reply/action composition
  for material packs, weekly reports, monthly reports, and sales mentions;
- no sends during resolve/preflight;
- all internal lookups must include requester and group context.

## Tool output / evidence guardrail

Purpose:

- sanitize tool output before the reply LLM sees it;
- redact sensitive fields;
- cap content size;
- tag source id/version/timestamp;
- treat markdown and MCP output as data only;
- defend against prompt injection inside documents.

Adapter resolve packaging rule:

```text
AdapterResolveResult:
- resolve_type: material_pack | weekly_report | monthly_report | sales_mention
- status: resolved | missing | ambiguous | forbidden | temporarily_unavailable
- display_name if available
- candidates when ambiguous
- reason_code when useful
- card_ref if the adapter needs an opaque execution reference
- report metadata when it affects the answer
```

Adapter resolve results should be business-safe structured fields that the
harness and LLM may read directly. They must not include raw filesystem paths,
tokens, phone numbers, or internal notes.

General evidence packaging rule for markdown/MCP sources:

```text
Every evidence block must be wrapped with:
- source_type
- source_id
- fetched_at
- trust_level
- explicit instruction that content is data only, not instructions
```

If evidence contains unsafe instructions, strip or neutralize them. If it cannot
be safely sanitized, drop the evidence and log a guardrail event.

## Business facts guardrail

Purpose:

- convert adapter resolve/evidence into deterministic facts;
- avoid relying on LLM inference for core business states.

Examples:

- `material_pack_resolvable=false` blocks `send_material_pack`;
- `weekly_report_resolvable=false` blocks `send_weekly_report`;
- `monthly_report_resolvable=false` blocks `send_monthly_report`;
- `weekly_contains_strategy=false` allows saying the report does not include
  the strategy;
- `weekly_scope_status=excluded` allows saying the strategy is outside scope;
- `weekly_scope_status=unknown` forbids saying outside scope;
- `sales_mention_resolvable=true` allows `reply.mentions`.

Use lightweight `EvidenceFact` records for high-risk facts. Do not build full
per-sentence claim mapping in the initial design.

## Reply/action guardrail

Purpose:

- validate final `ReplyResponse` against policy, evidence, and business facts.

Deterministic checks:

- reply kind is allowed by policy;
- action type is allowed by policy;
- no side-effect action contains free-form user-visible text;
- material pack/report action has successful adapter resolve;
- strategy is available/canonical;
- no `reply.mentions` unless sales mention resolve passed;
- if `reply.kind=no_reply`, `reply.text`, `reply.mentions`, and `actions`
  must all be empty;
- if text claims report/material availability, the claim must be supported by
  EvidenceFacts;
- no raw internal fields in text;
- no action outside public schema.

LLM-judge checks may be used for:

- tone;
- whether wording sounds like investment advice;
- whether answer goes beyond evidence;
- whether reply is concise enough for group chat.

Do not rely on LLM judges for action legality.

Repair:

- repairable errors may get one bounded repair attempt;
- fatal errors go directly to deterministic fallback.

Repairable examples:

- final text overstates report scope when facts only support unknown scope;
- reply kind and mention/action intent are inconsistent but facts allow a safe
  correction.

Fatal examples:

- final action tries to send a material pack/report that adapter resolve did
  not resolve;
- bank material-pack request is ambiguous but the model still tries to send;
- `reply.kind=no_reply` has text, mentions, or actions;
- action contains forbidden free-form text fields.

## Adapter guardrail

The adapter must remain the final side-effect gate.

Adapter should verify:

- action type is allowed;
- material pack/report card reference is resolvable if provided;
- strategy/pack/report selector is valid for current channel;
- group/channel can receive it;
- mention target is valid;
- runtime response matches schema.

Adapter owns execution reliability and should:

- create persistent outbox/execution records;
- derive operation keys from `inbound_message_id + ":reply"` and
  `inbound_message_id + ":" + action_id`;
- not re-execute operations already marked succeeded;
- retry only pending/retryable failed operations.

Adapter should write back execution results:

- executed action status;
- sent card/report/material-pack metadata in adapter-native terms;
- failure reason;
- adapter message id.

## Failure and fallback matrix

```text
Invalid request:
  -> HTTP validation error or safe service error

Planner invalid:
  -> repair once
  -> fallback clarification/no_reply

Policy disallows capability:
  -> do not execute
  -> reply with limitation, clarification, or human_handoff

Evidence fetch fails:
  -> reply with unable-to-confirm
  -> include sales mention if adapter resolve can find target and policy requires it

MCP permission denied:
  -> do not reveal internals
  -> say unable/unauthorized or human_handoff per policy

Evidence has prompt injection:
  -> sanitize or drop evidence
  -> log guardrail event

Reply invalid:
  -> repair once
  -> deterministic fallback

Adapter rejects action:
  -> record failed action
  -> do not treat as sent in ledger
```

## Deterministic fallback examples

Missing weekly strategy with scope unknown:

```text
text:
"当前这份周报里我没有看到该策略。我请销售同事确认是否需要补充。"

actions:
[]

reply:
- kind: human_handoff
- mentions: sales
```

Unavailable material pack:

```text
text:
"目前这个渠道下我没有看到可发送的对应材料包。我请销售同事确认。"

actions: []

reply:
- kind: human_handoff
- mentions: sales
```

Ambiguous strategy:

```text
text:
"你说的策略我这里有多个可能匹配项，麻烦确认一下具体是哪一个。"

actions: []

reply:
- kind: clarification
```
