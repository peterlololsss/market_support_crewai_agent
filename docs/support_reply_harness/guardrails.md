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
    -> MCP/material wrapper
    -> Tool Output Guardrail
 -> Business Facts
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
- `available_materials`
- `available_strategies`
- feature flags
- ledger summary
- material/internal MCP availability

Outputs:

- allowed read capabilities;
- allowed business checks;
- allowed terminal action candidates;
- forbidden claims;
- mandatory escalation rules;
- evidence size/call limits;
- repair/fallback policy.

Example weekly question policy:

```text
allowed_read_capabilities:
- fetch_sent_material_md
- fetch_latest_weekly_md

allowed_business_checks:
- resolve_strategy_alias
- check_report_contains_strategy
- check_report_generation_scope

allowed_action_candidates:
- send_text
- mention_sales
- ask_clarification
- no_reply

forbidden:
- generate missing report section
- explain absence without evidence
- promise future inclusion
- send material unless action preconditions pass
```

## Plan guardrail

Purpose:

- reject disallowed evidence capabilities;
- reject unsupported business checks;
- reject raw MCP tool names;
- reject side effects as evidence calls;
- reject unavailable material/action candidates;
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
- no writes in early phases;
- all internal lookups must include requester and group context.

## Tool output / evidence guardrail

Purpose:

- sanitize tool output before the reply LLM sees it;
- redact sensitive fields;
- cap content size;
- tag source id/version/timestamp;
- treat markdown and MCP output as data only;
- defend against prompt injection inside documents.

Evidence packaging rule:

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

- convert evidence into deterministic facts;
- avoid relying on LLM inference for core business states.

Examples:

- `material_available=false` blocks `send_material`;
- `weekly_contains_strategy=false` allows saying the report does not include
  the strategy;
- `weekly_scope_status=excluded` allows saying the strategy is outside scope;
- `weekly_scope_status=unknown` forbids saying outside scope;
- `must_mention_sales=true` requires `mention_sales`.

## Reply/action guardrail

Purpose:

- validate final `ReplyResponse` against policy, evidence, and business facts.

Deterministic checks:

- action type is allowed by policy;
- material type is available;
- strategy is available/canonical;
- no `send_material` unless action preconditions pass;
- no `mention_sales` without reason;
- if text claims material was sent, a `send_material` action must exist;
- if `must_mention_sales=true`, `mention_sales` must exist;
- no raw internal fields in text;
- no action outside public schema.

LLM-judge checks may be used for:

- tone;
- whether wording sounds like investment advice;
- whether answer goes beyond evidence;
- whether reply is concise enough for group chat.

Do not rely on LLM judges for action legality.

Repair:

- allow one reply repair attempt with validator errors;
- if still invalid, use deterministic fallback.

## Adapter guardrail

The adapter must remain the final side-effect gate.

Adapter should verify:

- action type is allowed;
- material exists;
- strategy exists;
- material id/version matches catalog;
- group/channel can receive it;
- mention target is valid;
- action is idempotent or safe to repeat;
- runtime response matches schema.

Adapter should write back:

- executed action status;
- material id/version;
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
  -> reply with limitation or mention_sales

Evidence fetch fails:
  -> reply with unable-to-confirm
  -> mention_sales if business-critical

MCP permission denied:
  -> do not reveal internals
  -> say unable/unauthorized or mention_sales per policy

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
"当前这份周报里我没有看到该策略。我帮你 @销售 确认是否需要补充。"

actions:
- mention_sales(reason="weekly_strategy_missing_scope_unknown")
```

Unavailable material:

```text
text:
"目前这个渠道下我没有看到可发送的对应材料，我帮你 @销售 确认。"

actions:
- mention_sales(reason="requested_material_unavailable")
```

Ambiguous strategy:

```text
text:
"你说的策略我这里有多个可能匹配项，麻烦确认一下具体是哪一个。"

actions:
- ask_clarification(text="请确认具体策略名称。")
```

