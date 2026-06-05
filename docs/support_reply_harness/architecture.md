# Support Reply Harness Architecture

Last updated: 2026-06-03.

## Problem restatement

The service is an external reasoning brain for a market support workflow. It receives structured WeCom chat context and returns one `ReplyResponse` for a WeCom adapter to execute.

The challenge is open-ended Chinese sales/support requests under strict boundaries around materials, reports, internal information, factual claims, escalation, and side effects.

## Current public contract

`ReplyRequest` includes:

```text
conversation_key
group_id
sender_id
message
is_group
context_id optional
group_name
dist_channel_name
sender_nickname
available_materials
available_strategies
channel_type
```

`ReplyResponse` separates reply semantics, primary user-visible text, customer-visible mentions, and side-effect action proposals.

```text
ReplyResponse
- contract_version
- response_id
- reply: PrimaryReply
- actions: list[SideEffectAction]

PrimaryReply
- kind: answer | clarification | human_handoff | unable_to_answer | no_reply
- text
- text_format: plain_text
- mentions: list[ReplyMention]

SideEffectAction
- send_material_pack
- send_weekly_report
- send_monthly_report
```

User-visible free-form reply text lives in `reply.text`. Sales mentions visible to the customer live in `reply.mentions`. The adapter owns execution reliability and final side-effect execution.

## Target runtime shape

```text
POST /reply
  -> Request validation
  -> Input guardrail
  -> Action ledger lookup
  -> Entity/canonicalization layer
  -> Policy compiler
  -> LLM planner
  -> Plan guardrail
  -> Evidence executor
      -> Tool input guardrail
      -> Adapter resolve/preflight wrapper
      -> Material/MCP wrapper when enabled
      -> Tool output guardrail
  -> EvidenceFact derivation
  -> BusinessFacts derivation
  -> Reply composer LLM
  -> Reply/action validator
  -> Repair once when safe
  -> Deterministic fallback when still unsafe
  -> Save conversation + audit trace
  -> Return ReplyResponse
  -> Adapter guardrail before real execution
```

## Source-of-truth hierarchy

Use this hierarchy when sources conflict:

1. Request contract and adapter-provided conversation/message identity.
2. Adapter resolve/preflight result for channel sendability, report/card existence, and sales mention target resolution.
3. Adapter-confirmed action ledger/execution result for what was actually sent.
4. Weekly/monthly report metadata returned by adapter resolve.
5. Permission-scoped internal MCP data.
6. Fetched markdown/report body when used.
7. Recent conversation turns.
8. LLM interpretation.

Planner conclusions are proposals. Evidence and deterministic business checks establish facts.

## Internal runtime concepts

### PolicyManifest

Request-scoped policy generated before planning.

Contains allowed reply kinds, allowed side-effect actions, required adapter resolves, allowed read capabilities, allowed deterministic business checks, forbidden claim categories, required escalation rules, evidence/resolve limits, and repair/fallback policy.

### ReplyPlan

LLM-generated and validated. It describes user need, detected entity candidates, evidence requests, requested business checks, candidate terminal actions, ambiguity flags, confidence, and unsupported-request notes.

`ReplyPlan` is not allowed to be a source of final business facts.

### AdapterResolveResult

Adapter-owned preflight result used before final reply/action composition. It answers whether a WeCom card or sales mention can be generated for the current channel.

Common fields:

```text
resolve_type: material_pack | weekly_report | monthly_report | sales_mention
status: resolved | missing | ambiguous | forbidden | temporarily_unavailable
display_name
candidates
reason_code
card_ref
```

Report-specific fields when relevant:

```text
period
report_date
strategy
contains_strategy
generated_strategies
scope_status: included | excluded | unknown
```

Adapter resolve results are business-safe structured data. They omit raw send targets, filesystem paths, tokens, phone numbers, internal notes, and adapter execution records.

### EvidenceFact

Lightweight internal fact records derived from adapter resolve and deterministic evidence. Initial fact types:

```text
report_contains_strategy
report_scope_status
material_pack_resolvable
weekly_report_resolvable
monthly_report_resolvable
sales_mention_resolvable
```

The initial harness uses these facts to validate high-risk claims and actions, without building full per-sentence claim mapping.

### BusinessFacts

Deterministic fact layer derived from request, ledger, metadata, and evidence.

Examples:

```text
strategy resolved / ambiguous / unknown
material pack available / ambiguous / unavailable
weekly or monthly report resolvable / unavailable
report contains strategy / does not contain strategy / unknown
report generation scope included / excluded / unknown
user permission allowed / denied
sales mention target resolved / not resolved
```

### ValidationResult

Machine-readable validation result with validity, severity, error codes, repairability, and fallback recommendation.

### AuditTrace

Replayable trace for incident review and eval debugging. It records request/context id, policy manifest id/hash, canonical entities, planner output, validation decisions, evidence calls/source ids, business facts, reply output, repair attempts, fallback reason, final actions, adapter execution status, and model/prompt/policy/validator versions.

## Capability taxonomy

Keep capability types separate.

Read/evidence capabilities:

```text
resolve_material_pack
resolve_weekly_report
resolve_monthly_report
resolve_sales_mention
fetch_latest_weekly_md
fetch_sent_weekly_md
query_internal_company_info
```

Deterministic business checks:

```text
resolve_strategy_alias
resolve_company_alias
check_material_pack_resolvable
check_weekly_report_resolvable
check_monthly_report_resolvable
check_report_contains_strategy
check_report_generation_scope
check_user_permission
check_channel_permission
```

Terminal action candidates:

```text
send_material_pack
send_weekly_report
send_monthly_report
```

Reply policies are prompt constraints plus validators, not callable tools.

## Weekly report absence semantics

If deterministic report evidence has `contains_strategy=false`, the reply may state that the report does not include the strategy.

If adapter-provided generated-strategy/scope metadata explicitly has `scope_status=excluded`, the reply may state that the strategy is outside the report generation scope.

If scope metadata is unavailable, the reply should use conservative wording and escalate or clarify according to policy.

The current xiaoyan adapter returns positive report-scope evidence when generated markdown explicitly matches the requested strategy. It does not emit negative exclusion evidence yet, so absent matches remain `unknown`.

## Material pack and report sending semantics

`send_material_pack`, `send_weekly_report`, and `send_monthly_report` are distinct action proposals.

A channel may have multiple strategies. A material pack may cover multiple strategies. Non-bank channels may have one default pack; bank channels may split strategies across multiple packs. If a bank-channel material-pack request does not specify enough strategy or pack information to resolve one pack, the harness asks for clarification.

The adapter is the source of truth for channel-scoped sendability and fetching. The harness waits for adapter resolve/preflight feedback before composing a final reply/action.

## Adapter identity and execution reliability

The adapter-to-harness request exposes harness-oriented identity fields:

```text
conversation_key
inbound_message_id
sender_id
receiver_id / bot_id when needed
send_time
context_id optional trace id
```

The adapter validates the full `ReplyResponse`, creates persistent outbox/execution records, executes the primary reply and actions, derives operation keys from inbound message identity plus reply/action identity, and writes adapter execution results for ledger/audit.

## Multi-agent decision

Use one orchestrated harness. Add more agents only when tool/permission surfaces diverge sharply, prompts become overloaded, or a separate compliance judge proves measurable safety value.
