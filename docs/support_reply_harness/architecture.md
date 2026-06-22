# Support Reply Harness Architecture

Last updated: 2026-06-17.

## Problem restatement

The service is an external reasoning brain for a market support workflow. It receives structured WeCom chat context and returns one `ReplyResponse` for a WeCom adapter to execute.

The challenge is open-ended Chinese sales/support requests under strict boundaries around materials, reports, internal information, factual claims, escalation, and outbound actions.

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
available_artifacts
channel_type
```

`available_artifacts` is the adapter-provided artifact availability source for material packs, weekly reports, and
monthly reports. Each item uses `type=material_pack|weekly_report|monthly_report`; only `material_pack` may carry
`options`, and those options are material-pack routing choices, not a general strategy catalog. Empty material-pack
`options` means the channel has a single/current material pack, which is the common non-bank shape.

`ReplyResponse` separates reply semantics, primary user-visible text, customer-visible mentions, and outbound action proposals.

```text
ReplyResponse
- contract_version
- response_id
- reply: PrimaryReply
- actions: list[OutboundAction]

PrimaryReply
- kind: answer | clarification | human_handoff | unable_to_answer | no_reply
- text
- mentions: list[ReplyMention]

OutboundAction
- send_material_pack
- send_weekly_report
- send_monthly_report
```

User-visible free-form reply text lives in `reply.text`. Sales mentions visible to the customer live in `reply.mentions`. The adapter owns execution reliability and final outbound action execution.

Legacy workflow prompts and historical send handlers are reference material only. They can contribute compliance
reason codes, adapter contract examples, standard send-copy ownership, and regression cases, but they are not copied as
runtime architecture and are not a source of deterministic intent routing.

## Target runtime shape

```text
POST /reply
  -> Request validation
  -> Input validation
  -> Action ledger lookup
  -> Entity/canonicalization layer
  -> Policy compiler
  -> Context projection for planner
  -> LLM planner
  -> Plan validation
  -> Evidence executor
      -> Tool input validation
      -> Adapter resolve/preflight wrapper
      -> Document MCP, report-scope, or approved-knowledge wrapper when enabled
      -> Tool output validation
  -> EvidenceFact derivation
  -> BusinessFacts derivation
  -> Context projection for composer/verifier when needed
  -> ResponseDirective decision engine
  -> Deterministic renderer for action/refusal/clarification/handoff/unable/no_reply
  -> Reply composer LLM only for document-backed knowledge_answer
  -> Reply/action postcondition validator
  -> Save conversation + audit/runtime trace
  -> Return ReplyResponse
  -> Adapter validation before real execution
```

## Source-of-truth hierarchy

Use this hierarchy when sources conflict:

1. Request contract and adapter-provided conversation/message identity.
2. Adapter resolve/preflight result for channel sendability, artifact existence, and sales mention target resolution.
3. Adapter-confirmed action ledger/execution result for what was actually sent.
4. Weekly/monthly report metadata returned by adapter resolve.
5. Permission-scoped internal MCP data.
6. Fetched markdown/report body when used.
7. Recent conversation turns.
8. LLM interpretation.

Planner conclusions are proposals. Evidence and deterministic business facts establish facts.

For material/report sends, standard post-send wording belongs to the WeCom adapter execution layer. The harness returns
typed semantic actions and should not duplicate legacy post-send text in `reply.text` before adapter execution.

## Internal runtime concepts

### PolicyManifest

Request-scoped policy generated before planning.

Contains allowed reply modes, allowed capabilities, allowed outbound actions, allowed adapter resolves, allowed read capabilities, adapter-safe ledger summary, evidence call limits, and error-on-invalid policy.

### ExecutionPlan

Compiled deterministically from `PlanSpec`, policy, canonical context, and the capability registry. It describes user need, artifact kind, response mode, compliance, capabilities, adapter resolve specs, action intents, selected strategy, and the selected planner contract.

`ExecutionPlan` is not allowed to be a source of final business facts.

### ResponseDirective

Deterministic decision output derived from `ExecutionPlan`, `BusinessFacts`, evidence facts, request context, and policy.

`ResponseDirective` decides the reply mode, reply kind, text, mentions, action intents, and whether the knowledge composer is required. It is not public API; it is rendered to `ReplyResponse` or used to gate the knowledge composer.

### AdapterResolveResult

Adapter-owned preflight result used before final reply/action composition. It answers whether an adapter-safe artifact reference or sales mention can be generated for the current channel.

Common fields:

```text
resolve_type: material_pack | weekly_report | monthly_report | sales_mention
status: resolved | missing | ambiguous | forbidden | temporarily_unavailable
display_name
candidates
reason_code
resolve_ref
```

Report-specific fields when relevant:

```text
period
report_date
period_start
period_end
period_label
scope_complete
expected_product_count
generated_product_count
missing_product_count
report_sections
```

Adapter resolve results are business-safe structured data. They omit raw send targets, filesystem paths, tokens, phone numbers, internal notes, and adapter execution records.

### EvidenceFact

Lightweight internal fact records derived from adapter resolve and deterministic evidence. Initial fact types:

```text
material_pack_resolvable
weekly_report_resolvable
monthly_report_resolvable
sales_mention_resolvable
report_scope_summary
report_scope_match
report_scope_products
report_period
recent_executed_action
document_context
document_context_unavailable
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

Machine-readable validation result with validity, severity, error codes, and metadata.

### AuditTrace

Replayable trace for incident review and eval debugging. It records request/context id, policy manifest id/hash,
canonical scope, planner output, validation decisions, evidence calls/source ids, business facts, reply output,
invalid-output records, renderer decision reason, final actions, adapter execution status, per-CrewAI-stage latency/usage summaries,
and model ids, prompt profile ids, policy ids, and validator ids.

CrewAI output metadata is compacted before audit storage. The trace keeps stage name, agent role, response format,
latency, usage metrics, structured-output type, raw-output length, and planning/todo counts. It does not store CrewAI
message history, raw prompts, raw plan text, or hidden execution content.

### Context Is A Projection

Durable/current state and model-visible context are different layers. The
conversation transcript records what happened; `ModelVisibleContext` records
what the next planner, composer, or verifier is allowed to see.

`ContextProjectionManager` builds that view from request metadata, recent
history, action ledger summaries, `DomainContext`, `CanonicalContext`,
`PolicyManifest`, plan/validation state, evidence, `BusinessFacts`,
`AnswerabilityAssessment`, guardrail decisions, candidate response, and
alignment retry state.

Projection block types:

```text
recent_verbatim        last N turns, context-only
compacted_summary      deterministic older-history summary
large_result_preview   bounded preview with reload_handle
allowed_evidence       accepted evidence for the current plan
context_only           non-evidence state/history/ledger summaries
disallowed_evidence    rejected evidence with content redacted
app_state              request/canonical/domain/policy/plan/facts/guardrails
current_task           current user message and verifier candidate response
ephemeral              retry/alignment state
```

Conversation history is not claim evidence by default. Allowed evidence is
selected through the guardrail pipeline before prompt assembly; disallowed
evidence keeps source metadata but redacts content. Large content uses stable
previews and internal reload handles so prompts do not randomly truncate
documents or transcripts. Context pressure estimates and projection decisions
are audit metadata, not prompt text storage.

## Capability taxonomy

Keep capability types separate.

Read/evidence capabilities:

```text
resolve_material_pack
resolve_weekly_report
resolve_monthly_report
resolve_sales_mention
query_internal_company_info
```

Deterministic business fact fields:

```text
material_pack
weekly_report
monthly_report
sales_mention
recent_executed_actions
requested_material_pack_option_status
user_permission
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
 -> plan validator checks PolicyManifest and structured scope
 -> evidence executor calls a fixed Document MCP wrapper only for channel-permitted requests
 -> wrapper selects list_products/get_documents inputs, redacts, caps, and marks content as data-only
 -> document_context EvidenceFacts feed the composer prompt
 -> reply/action validator blocks knowledge_answer replies without document_context evidence
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

## Weekly report content semantics

Report-scope evidence answers questions about what products or sections are inside a weekly/monthly report. It is read evidence for knowledge answers, not a selector for sending a partial report.

If adapter-provided report-scope evidence is unavailable, the reply should use conservative wording and abstain, clarify, or hand off according to policy. The harness must not infer report contents from free-form query text.

## Material pack and report sending semantics

`send_material_pack`, `send_weekly_report`, and `send_monthly_report` are distinct action proposals.

Weekly and monthly report actions send the whole adapter-resolved report. They do not carry `report_scope`, `strategy`, or `material_pack_option`. Send actions always include adapter-safe `resolve_ref`; report actions also include `period` and `report_date` when the adapter supplied them. Questions about report contents use report-scope evidence and a knowledge answer instead of scoped send actions.

A material pack may be default-scoped or split across adapter-owned material-pack options. The harness proposes a typed
material-pack send for clear send requests, and the adapter resolve result decides whether the request is resolved,
ambiguous, or unavailable. If adapter resolve returns ambiguous candidates, the harness asks for clarification.

The adapter is the source of truth for channel-scoped sendability and fetching. The harness waits for adapter resolve/preflight feedback before composing a final reply/action.

## Adapter identity and execution reliability

The adapter-to-harness request exposes harness-oriented identity fields:

```text
conversation_key
context_id optional trace id
group_id
sender_id
group_name
dist_channel_name
sender_nickname
```

The adapter validates the full `ReplyResponse`, creates persistent outbox/execution records, executes the primary reply and actions, derives operation keys from inbound message identity plus reply/action identity, and writes adapter execution results for ledger/audit.

## Multi-agent decision

Use one orchestrated harness. Add more agents only when tool/permission surfaces diverge sharply, prompts become overloaded, or a separate compliance judge proves measurable safety value.
