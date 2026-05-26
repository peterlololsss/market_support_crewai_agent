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

These public contracts should remain stable unless the adapter is deliberately
changed.

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
      -> Material/MCP wrapper
      -> Tool output guardrail
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

1. Request contract and adapter-provided availability.
2. Adapter-confirmed action ledger for what was actually sent.
3. Material catalog / material MCP metadata.
4. Weekly/monthly report metadata.
5. Permission-scoped internal MCP data.
6. Fetched markdown body.
7. Recent conversation turns.
8. LLM interpretation.

Planner conclusions are not facts. The planner proposes evidence needs and
candidate actions; deterministic evidence and business checks establish facts.

## Internal runtime concepts

### PolicyManifest

Request-scoped policy generated before planning.

Contains:

- allowed read capabilities;
- allowed deterministic business checks;
- allowed terminal action candidates;
- forbidden claims/actions;
- required escalation rules;
- evidence limits;
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

### EvidenceBundle

Harness-generated evidence package.

Contains per evidence item:

- source type: `weekly_md`, `material_md`, `internal_mcp`,
  `action_ledger`, or `request_context`;
- source id/version;
- fetched timestamp;
- sanitized content;
- trust level;
- metadata.

### BusinessFacts

Deterministic fact layer, derived from request, ledger, metadata, and evidence.

Examples:

- strategy resolved or ambiguous;
- material available or unavailable;
- report contains strategy or does not;
- report generation scope includes, excludes, or is unknown;
- user permission allowed or denied;
- material can be sent or cannot;
- sales mention is required or optional.

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

## Capability taxonomy

Keep these separate.

### Read / evidence capabilities

- `fetch_latest_weekly_md`
- `fetch_sent_material_md`
- `fetch_material_md`
- `list_available_materials`
- `query_internal_company_info`

These are executed by deterministic wrappers only after validation.

### Deterministic business checks

- `resolve_strategy_alias`
- `resolve_company_alias`
- `check_material_availability`
- `check_report_contains_strategy`
- `check_report_generation_scope`
- `check_user_permission`
- `check_channel_permission`

Prefer deterministic implementations over LLM judgment.

### Terminal action candidates

- `send_text`
- `send_material`
- `mention_sales`
- `ask_clarification`
- `no_reply`

The LLM may propose these, but validators and adapter enforce them.

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

If scope metadata explicitly excludes XX:
  Allowed fact: "XX is not in this report generation scope."

If scope metadata is unavailable:
  Not allowed: "XX is not in generation scope."
  Allowed: "This report does not include XX; I will mention sales to confirm."
```

Therefore material/report providers should eventually return:

- `material_id`
- `material_type`
- `strategy`
- `channel_type`
- `version`
- `published_at`
- `source_key`
- `generated_strategies` if available
- `scope_strategies` if available
- `markdown_body`

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

