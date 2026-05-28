# Support Reply Harness Architecture

## Problem restatement

The service is an external reasoning brain for a market support workflow. It
receives structured chat context and returns structured text/actions for a
WeWork adapter to execute. The challenge is to support open-ended Chinese sales
and support requests without enumerating every user phrasing, while still
enforcing strict boundaries around materials, reports, internal information,
claims, escalation, and side effects.

Core tension:

```text
The system needs enough model autonomy to understand ambiguous domain language,
but not enough autonomy to invent facts, bypass permissions, leak internal data,
or trigger unsupported actions.
```

## Current public contracts

`ReplyRequest` includes:

- `conversation_key`
- `group_id`
- `sender_id`
- `message`
- `is_group`
- `context_id` optional
- `group_name`
- `dist_channel_name`
- `sender_nickname`
- `available_materials`
- `available_strategies`
- `channel_type`

`ReplyResponse` includes:

- `text`
- `actions`

Supported action types:

- `send_text`
- `send_material`
- `mention_sales`
- `ask_clarification`
- `no_reply`

These are the current implemented contracts. The adapter can be deliberately
changed, and the proposed production contract below supersedes the mixed
`text + action text` model.

## Proposed production response contract

`ReplyResponse` should separate reply semantics, primary user-visible text,
customer-visible mentions, and side-effect actions.

```text
ReplyResponse:
- contract_version
- response_id
- reply: PrimaryReply
- actions: list[SideEffectAction]

PrimaryReply:
- kind: answer | clarification | human_handoff | unable_to_answer | no_reply
- text
- text_format: plain_text
- mentions: list[ReplyMention]

SideEffectAction:
- send_material_pack
- send_weekly_report
- send_monthly_report
```

Rules:

- `reply.text` is the only primary user-visible free-form reply text.
- `send_text` is removed.
- `ask_clarification` becomes `reply.kind=clarification`.
- `no_reply` becomes `reply.kind=no_reply`.
- `fallback` is not a reply kind; it is internal audit provenance.
- Customer-visible sales mentions belong to `reply.mentions`.
- Actions must not contain user-visible free-form fields such as `message`,
  `text`, `caption`, or `body`.
- Public `ReplyResponse` does not expose sending-side idempotency keys. The
  adapter owns execution reliability and derives operation keys from
  `inbound_message_id`, reply operation, and `action_id`.

Current scope rejected as over-engineering:

- `tenant` in the public contract;
- `business_unit` in the public contract;
- adapter-provided `environment`.

This is an internal group robot, not a multi-tenant SaaS platform. Deployment
environment, if needed, should come from runtime configuration.

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
      -> Material/MCP wrapper if needed
      -> Tool output guardrail
  -> Lightweight EvidenceFact derivation
  -> Deterministic business checks
  -> Reply composer LLM
  -> Reply/action validator
  -> Repair once if safe
  -> Deterministic fallback if still unsafe
  -> Save conversation + audit trace
  -> Return ReplyResponse
  -> Adapter guardrail before real execution
```

## Source-of-truth hierarchy

Use this hierarchy when sources conflict:

1. Request contract and adapter-provided conversation/message identity.
2. Adapter resolve/preflight result for channel sendability, report/card
   existence, and sales mention target resolution.
3. Adapter-confirmed action ledger / execution result for what was actually
   sent.
4. Weekly/monthly report metadata returned by adapter resolve.
5. Permission-scoped internal MCP data.
6. Fetched markdown/report body if used.
7. Recent conversation turns.
8. LLM interpretation.

Planner conclusions are not facts. The planner proposes evidence needs and
candidate actions; deterministic evidence and business checks establish facts.

## Internal runtime concepts

### PolicyManifest

Request-scoped policy generated before planning.

Contains:

- allowed reply kinds;
- allowed side-effect actions;
- required adapter resolves;
- allowed read capabilities if any;
- allowed deterministic business checks;
- forbidden claims/actions;
- required escalation rules;
- evidence/resolve limits;
- repair/fallback policy.

### ReplyPlan

LLM-generated but validated. It describes intent and evidence needs.

Contains:

- user need;
- detected entities;
- evidence requests;
- requested business checks;
- candidate terminal actions;
- ambiguity flags;
- confidence;
- unsupported request notes.

It must not contain final business facts as authority.

### AdapterResolveResult

Adapter-owned preflight result used before final reply/action composition. It
answers whether a WeWork card or sales mention can be generated for the current
channel.

Common fields:

- `resolve_type`: `material_pack`, `weekly_report`, `monthly_report`, or
  `sales_mention`;
- `status`: `resolved`, `missing`, `ambiguous`, `forbidden`, or
  `temporarily_unavailable`;
- `display_name` if available;
- `candidates` when ambiguous;
- `reason_code` when useful;
- `card_ref` if the adapter needs an opaque reference for card execution.

For weekly/monthly reports, adapter resolve may also return:

- `period`;
- `report_date`;
- `strategy`;
- `contains_strategy`;
- `generated_strategies`;
- `scope_status`: `included`, `excluded`, or `unknown`.

The resolve result should be business-safe structured data that the harness and
LLM may read directly. It should not include internal raw details such as
filesystem paths, tokens, phone numbers, or internal notes.

### EvidenceFact

Lightweight harness-generated fact list derived from adapter resolve and other
deterministic evidence. It is internal and not model-generated.

Initial fact types:

- `report_contains_strategy`;
- `report_scope_status`;
- `material_pack_resolvable`;
- `weekly_report_resolvable`;
- `monthly_report_resolvable`;
- `sales_mention_resolvable`.

The harness does not build full per-sentence claim mapping initially. It uses
EvidenceFacts to validate high-risk claims and actions.

### BusinessFacts

Deterministic fact layer, derived from request, ledger, metadata, and evidence.

Examples:

- strategy resolved or ambiguous;
- material pack available, ambiguous, or unavailable;
- weekly/monthly report resolvable or unavailable;
- report contains strategy or does not;
- report generation scope includes, excludes, or is unknown;
- user permission allowed or denied;
- sales mention target resolved or not resolved.

### ValidationResult

Machine-readable validation result.

Contains:

- valid/invalid;
- severity;
- error codes;
- repairable flag;
- fallback recommendation.

### AuditTrace

Replayable trace for incident review and eval debugging.

Contains:

- request id/context id;
- policy manifest hash/id;
- canonical entities;
- planner output;
- validation decisions;
- evidence calls and source ids;
- business facts;
- reply output;
- repair attempts;
- fallback reason;
- final actions;
- adapter execution status when available.

Decision source, reason codes, model/prompt/policy/validator versions, and
adapter resolve/execution references belong here, not in the public response.

## Capability taxonomy

Keep these separate.

### Read / evidence capabilities

- `resolve_material_pack`
- `resolve_weekly_report`
- `resolve_monthly_report`
- `resolve_sales_mention`
- `fetch_latest_weekly_md` if body inspection is needed
- `fetch_sent_weekly_md` if ledger-grounded body inspection is needed
- `query_internal_company_info`

These are executed by deterministic wrappers only after validation.

### Deterministic business checks

- `resolve_strategy_alias`
- `resolve_company_alias`
- `check_material_pack_resolvable`
- `check_weekly_report_resolvable`
- `check_monthly_report_resolvable`
- `check_report_contains_strategy`
- `check_report_generation_scope`
- `check_user_permission`
- `check_channel_permission`

Prefer deterministic implementations over LLM judgment.

### Terminal action candidates

- `send_material_pack`
- `send_weekly_report`
- `send_monthly_report`

Reply kinds such as `clarification`, `human_handoff`, and `no_reply` are not
actions. Customer-visible sales mentions are represented in `reply.mentions`,
with adapter-resolved targets.

### Reply policies

- `answer_only_from_evidence`
- `no_investment_advice`
- `no_unsupported_explanation`
- `no_raw_internal_data`
- `escalate_missing_scope_to_sales`
- `do_not_claim_action_without_action`

These are prompt constraints plus validators, not callable tools.

## Weekly report absence semantics

Important distinction:

```text
If markdown body lacks strategy XX:
  Allowed fact: "This report does not include XX."

If adapter-provided generated-strategy/scope metadata explicitly excludes XX:
  Allowed fact: "XX is not in this report generation scope."

If scope metadata is unavailable:
  Not allowed: "XX is not in generation scope."
  Allowed: "This report does not include XX; I will mention sales to confirm."
```

Therefore adapter resolve for weekly/monthly reports should return
`generated_strategies` or equivalent scope metadata when the harness is allowed
to make generation-scope claims. Without that metadata, the harness must use
conservative wording.

## Material pack and report sending semantics

Do not overload one `send_material` action.

- `send_material_pack`: sends an open-day/customer-facing material pack.
- `send_weekly_report`: sends a weekly report card.
- `send_monthly_report`: sends a monthly report card.

A channel has multiple strategies. A material pack may cover multiple
strategies. Non-bank channels may have one default pack; bank channels may split
strategies across multiple packs. If a bank-channel material-pack request does
not specify enough strategy or pack information to resolve one pack, the
harness must ask for clarification instead of guessing.

The adapter is the source of truth for channel-scoped sendability and fetching.
The harness must wait for adapter resolve/preflight feedback before composing
the final reply/action.

## Adapter identity and execution reliability

The adapter-to-harness request should expose harness-oriented identity fields:

- `conversation_key`: derived from raw WeWork `conversation_id`; used for
  history and ledger lookup scope.
- `inbound_message_id`: derived from raw WeWork `server_id`; used for duplicate
  detection, adapter operation-key derivation, and audit correlation.
- `sender_id`: derived from raw sender.
- `receiver_id` / `bot_id`: derived from raw receiver if needed.
- `send_time`: preserved as event timestamp.

Do not use `context_id` as a vague multipurpose field in the final contract.
If it remains during migration, document it as legacy and map it at the adapter
boundary.

The adapter owns execution reliability:

- validate the full `ReplyResponse` before executing anything;
- create persistent outbox/execution records;
- execute primary reply with `reply.mentions`;
- execute actions in order;
- derive operation keys from `inbound_message_id + ":reply"` and
  `inbound_message_id + ":" + action_id`;
- do not re-execute operations already marked succeeded;
- write adapter execution results for ledger/audit.

## Multi-agent decision

Initial recommendation:

```text
Do not build a true autonomous multi-agent Crew initially.
Build one orchestrated harness with two LLM stages.
```

Stages:

- Planner LLM: no tools, outputs validated `ReplyPlan`.
- Reply Composer LLM: no direct tools, sees sanitized evidence/business facts,
  outputs `ReplyResponse`.

Only add more agents later if tool/permission surfaces diverge sharply, prompts
become overloaded, or a separate compliance judge adds measurable safety.
