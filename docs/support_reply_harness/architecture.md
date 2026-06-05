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

Legacy workflow prompts and historical send handlers are reference material only. They can contribute compliance
reason codes, adapter contract examples, standard send-copy ownership, and regression cases, but they are not copied as
runtime architecture and are not a source of deterministic intent routing.

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

For material/report sends, standard post-send wording belongs to the WeCom adapter execution layer. The harness returns
typed semantic actions and should not duplicate legacy post-send text in `reply.text` before adapter execution.

## Internal runtime concepts

### PolicyManifest

Request-scoped policy generated before planning.

Contains allowed reply kinds, allowed side-effect actions, required adapter resolves, allowed read capabilities, allowed deterministic business checks, forbidden claim categories, adapter-safe ledger summary, required escalation rules, evidence/resolve limits, and repair/fallback policy.

### ReplyPlan

LLM-generated and validated. It describes user need, detected entity candidates, evidence requests, requested business checks, candidate terminal actions, ambiguity flags, confidence, and unsupported-request notes.

`ReplyPlan` is not allowed to be a source of final business facts.

### AdapterResolveResult

Adapter-owned preflight result used before final reply/action composition. It answers whether a WeCom card or sales mention can be generated for the current channel.

Common fields:

```text
resolve_type: material_pack | weekly_report | monthly_report | sales_mention
status: resolved | missing | ambiguous | temporarily_unavailable
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
recent_executed_action
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
recent adapter-confirmed executed actions for “just sent” references
```

### ValidationResult

Machine-readable validation result with validity, severity, error codes, repairability, and fallback recommendation.

### AuditTrace

Replayable trace for incident review and eval debugging. It records request/context id, policy manifest id/hash,
canonical entities, planner output, validation decisions, evidence calls/source ids, business facts, reply output,
repair attempts, fallback reason, final actions, adapter execution status, per-CrewAI-stage latency/usage summaries,
and model/prompt/policy/validator versions.

CrewAI output metadata is compacted before audit storage. The trace keeps stage name, agent role, response format,
latency, usage metrics, structured-output type, raw-output length, and planning/todo counts. It does not store CrewAI
message history, raw prompts, raw plan text, or hidden execution content.

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

## Document MCP usage

Document MCP access is an evidence source, not an agent-owned browsing permission.

The planner may request document-backed evidence only when all of the following are true:

```text
1. the request is compliant;
2. adapter resolve, ledger history, and recent turns are insufficient;
3. the user is asking for knowledge or factual explanation, not for material/report sending;
4. the requested capability is allowed by the request-scoped PolicyManifest;
5. the entity/canonicalization layer has resolved enough product, strategy, company, or period context for a bounded query.
```

Typical document MCP use cases:

```text
company/product knowledge Q&A
strategy or index feature explanation backed by approved docs
standard Q&A wording lookup
```

Report body inspection is a future extension. The current document MCP exposes product/company/Q&A documents only.

Document MCP should not be used for:

```text
send_material_pack / send_weekly_report / send_monthly_report execution decisions
expected or target return requests
peer or competitor evaluation
contract or restricted internal document delivery
private contact handling
questions already answered by adapter resolve or action ledger
```

The runtime shape is:

```text
Planner LLM proposes query_internal_company_info evidence need
 -> plan validator checks PolicyManifest and canonical entities
 -> evidence executor calls a fixed Document MCP wrapper only for channel-permitted requests
 -> wrapper selects list_products/get_documents inputs, redacts, caps, and marks content as data-only
 -> document_context EvidenceFacts feed the composer prompt
 -> reply/action validator blocks knowledge_qa answers without document_context evidence
```

The MCP URL and raw tool names do not belong in customer-visible text or broad system prompts. The composer sees only
sanitized evidence and source summaries.

Current document MCP endpoint discovery:

```text
base URL: http://192.168.209.195:23000
streamable HTTP path: /mcp
required Accept header: application/json, text/event-stream
available tools: list_products, get_documents
```

These tools remain wrapper-only. They are not attached directly to CrewAI agents.

## Weekly report absence semantics

If deterministic report evidence has `contains_strategy=false`, the reply may state that the report does not include the strategy.

If adapter-provided generated-strategy/scope metadata explicitly has `scope_status=excluded`, the reply may state that the strategy is outside the report generation scope.

If scope metadata is unavailable, the reply should use conservative wording and escalate or clarify according to policy.

The current xiaoyan adapter returns positive report-scope evidence when generated markdown explicitly matches the requested strategy. It does not emit negative exclusion evidence yet, so absent matches remain `unknown`.

## Material pack and report sending semantics

`send_material_pack`, `send_weekly_report`, and `send_monthly_report` are distinct action proposals.

Report actions keep the public action type stable while the internal ReplyPlan carries a selector:
`report_scope=channel_all` means send the adapter-resolved channel report package, and
`report_scope=strategy` means the user asked for a specific strategy/product that must be confirmed as covered by the
report. Unknown or multi-strategy report ranges must clarify or hand off instead of sending.

Current adapter compatibility note: the existing WeCom action parser accepts only `type`, `action_id`, and `strategy` on
public action objects. Until the adapter contract is bumped, the harness does not emit public `selector` or `card_ref`
fields. It records action preconditions in the audit trace instead: action type/id, required resolve type/status,
internal report selector, plan/action/adapter strategy, whether an opaque adapter ref was available, and report scope
metadata.

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
