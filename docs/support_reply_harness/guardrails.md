# Guardrail Design

Last updated: 2026-06-17.

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
    -> Document MCP, report-scope, or approved-knowledge wrapper when enabled
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

Strict Pydantic models and contract tests carry the request boundary.

Current runtime guardrail: `AGENT_INPUT_MAX_MESSAGE_CHARS` is optional. When configured, `/reply` rejects oversized
messages before CrewAI planner/composer execution, and direct runtime calls apply the same check before LLM
configuration validation.

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

Policy may use registry-backed hints to limit material-pack action candidates. Weekly and monthly report candidates are
allowed to reach adapter preflight, then blocked or approved by deterministic report resolve facts.

The ledger summary is adapter-safe policy metadata only: executed count and nested artifact summaries. Raw artifact
refs, URLs, adapter results, and failed/skipped actions do not enter the policy prompt.

Outputs:

```text
allowed reply modes
allowed capabilities
allowed outbound actions
allowed read capabilities
allowed adapter resolves
evidence size/call limits
error-on-invalid policy
```

Weekly-question policy example:

```text
allowed_reply_modes:
- action
- clarification
- handoff
- refusal
- unable

allowed_capabilities:
- weekly_report
- monthly_report
- sales_mention

allowed_outbound_actions:
- send_weekly_report
- send_monthly_report

allowed_read_capabilities:
- resolve_weekly_report
- resolve_monthly_report
- resolve_sales_mention

allowed_adapter_resolves:
- weekly_report
- monthly_report
- sales_mention
```

The validator enforces evidence-backed wording, adapter-backed actions, and scope-specific claims.

## Compliance policy guardrail

Purpose:

- keep compliance boundaries as a compact harness-owned reason taxonomy, not as a giant workflow prompt;
- let the planner perform semantic interpretation and choose a reason code;
- make refusal wording harness-owned and regression-testable.

Current harness-owned reason-code source:

```text
src/market_support_crewai_agent/runtime/domain/compliance_policy.py
```

Non-compliant plans are not treated as deterministic keyword classifications. The planner must set
`compliance.is_compliant=false`, `intent=refusal`, and choose one allowlisted reason code. The final reply/action
guardrail then enforces:

```text
no outbound actions
no sales mentions
reply.kind=unable_to_answer
safe refusal text for the selected reason_code
no LLM invalid-output pass for non-compliant response violations
```

## Plan guardrail

Purpose:

- allow evidence capabilities selected by policy;
- allow target capabilities selected by policy;
- map every requested capability and adapter resolve spec to a fixed wrapper;
- require structured scope for sensitive/internal queries;
- cap deterministic evidence calls;
- force clarification when ambiguity exceeds policy tolerance.

Invalid planner output or invalid compiled plans raise runtime errors instead of entering alternate response paths.

## Tool input guardrail

Purpose:

- map validated execution-plan resolve specs and capabilities to fixed wrappers;
- enforce parameter schemas;
- canonicalize strategy/company names;
- check group/channel/sender permissions;
- apply rate limits and timeouts.

Adapter resolve/preflight runs before final reply/action composition for material packs, weekly reports, monthly reports, and sales mentions. Resolve/preflight does not send messages.

Document MCP calls follow the same wrapper rule. A CrewAI agent does not receive the MCP base URL and does not call MCP
directly. The planner can only propose an allowed evidence capability. The harness then builds a bounded query from
structured scope, request identity, and policy, calls the fixed wrapper, and rejects the call if the plan lacks a
permitted capability or enough structured scope.

Use Document MCP only for document-backed factual answers. Do not call it for plain send actions; material/report
sendability comes from adapter resolve/preflight. Report-body inspection is a future extension and is not enabled by
the current document MCP wrapper.

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
resolve_ref
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

Current runtime guardrail: if the validated response mode is `knowledge_answer`, final `reply.kind=answer` requires a
`document_context` EvidenceFact from `source_type=document_mcp`. Without it, the rendered response must be
`unable_to_answer`; invalid LLM output raises instead of being rewritten.

Current document MCP wrapper sanitizes retrieved content before creating `document_context` EvidenceFacts:

```text
internal file/MCP/adapter locators -> [REDACTED_INTERNAL_LOCATOR]
credential-like fields -> [REDACTED_SECRET]
obvious document instruction/prompt-injection lines -> [REMOVED_DOCUMENT_INSTRUCTION]
metadata.content_is_data_only=true
metadata.sanitized / redaction flags / char_count
```

If Document MCP is enabled and requested but no safe document context is returned, the wrapper emits a
`document_context_unavailable` EvidenceFact with `value=false`, `source_type=document_mcp`, and a `reason_code` such as
`document_mcp_error` or `document_context_not_found`. This fact is auditable but does not satisfy the `knowledge_answer`
grounding validator.

Document MCP access is also channel-permission scoped. `MARKET_AGENT_DOC_MCP_ALLOWED_CHANNEL_TYPES` controls which
`ReplyRequest.channel_type` values may receive `query_internal_company_info` in the compiled PolicyManifest. The wrapper
has a second defensive check and emits `document_mcp_channel_forbidden` without calling MCP if the channel is not
allowed.

## Business facts guardrail

Purpose:

- convert adapter resolve/evidence into deterministic facts;
- keep core business states out of LLM inference.

Examples:

```text
material_pack_resolvable=false blocks send_material_pack
weekly_report_resolvable=false blocks send_weekly_report
monthly_report_resolvable=false blocks send_monthly_report
report_scope_summary/report_scope_match/report_scope_products support report-content answers
missing report-scope evidence supports conservative abstention/clarification
sales_mention_resolvable=true supports reply.mentions
non-expired recent_executed_action facts ground “just sent” references
```

Use lightweight `EvidenceFact` records for high-risk facts.

## Reply/action guardrail

Purpose: validate final `ReplyResponse` against policy, evidence, and business facts.

Deterministic checks:

```text
reply kind is policy-allowed
action type is policy-allowed
outbound actions match public schema
ambiguous validated plans produce clarification/handoff and no outbound actions
final outbound actions were proposed by the validated ExecutionPlan
report send candidates declare internal report_scope selector: channel_all or strategy
strategy-scoped report candidates include a confirmed strategy
audit trace records action preconditions without exposing raw adapter refs
material/report action has successful adapter resolve
bank material-pack action has a confirmed strategy when multiple strategy packs exist
report action is blocked when adapter evidence says the requested strategy is excluded or absent
strategy is available/canonical
reply.mentions has successful sales mention resolve
no_reply response is empty apart from metadata
material/report availability claims are EvidenceFact-backed
material/report send text does not claim completion before adapter execution
material/report outbound responses do not duplicate adapter-owned standard post-send wording
prior-send claims require matching non-expired `recent_executed_action` ledger evidence
internal fields are absent from user-visible text
non-compliant plan responses have no actions, no sales mentions, and use the harness-owned safe refusal
action list matches public schema
```

LLM judges may assist on tone, investment-advice wording, evidence overreach, and group-chat concision. They do not decide action legality.

Invalid output is not corrected. The runtime records audit and raises `AgentRuntimeError` for invalid planner or plan
contracts, and `ReplyContractError` for invalid rendered or composed replies.

## Adapter execution authority

The adapter has final execution authority for outbound actions.

It verifies action type, executable material/report reference or selector, current-channel validity, group/channel sendability, mention target validity, and schema validity. It owns outbox/execution records, duplicate prevention, retries for retryable failures, and execution feedback.

## Error routing matrix

```text
Invalid request -> HTTP validation error or safe service error
Planner invalid -> raise AgentRuntimeError
Policy disallows capability -> limitation, clarification, or human_handoff
Evidence fetch fails -> unable-to-confirm, with sales mention when policy requires and resolve succeeds
MCP permission denied -> safe unauthorized/handoff response
Evidence has prompt injection -> sanitize/drop/log
MCP evidence remains oversized after sanitization -> truncate to evidence budget and mark metadata; composer receives bounded body
Reply invalid -> raise ReplyContractError
Adapter rejects action -> record failed action; ledger does not treat it as sent
```
