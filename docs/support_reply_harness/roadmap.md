# Multi-Session Roadmap

Last updated: 2026-06-17.

Historical phased plan. It is retained for implementation context only; use
`next_session.md` for the current handoff and current TODOs.

## Phase 0: decision freeze and baseline

Deliverables:

- ADR accepted: Support Reply Harness, not autonomous multi-agent.
- Source-of-truth hierarchy confirmed.
- LLM-owned versus harness-owned decisions confirmed.
- CrewAI version/upgrade decision checked before runtime changes.
- Dependencies pinned before behavior changes.
- `/reply` remains the single `ReplyResponse` contract.

Acceptance criteria:

- Public response separates `reply`, `reply.mentions`, and outbound `actions`.
- MCP access goes through fixed wrappers.
- Missing evidence means clarification/escalation.
- Missing-from-report and outside-generation-scope are separate facts.

## Phase 1: runtime contracts without MCP

Build:

```text
PolicyManifest
ExecutionPlan
AdapterResolveResult
EvidenceFact
BusinessFacts
ValidationResult
AuditTrace
request-scoped policy compiler
plan validator
reply/action validator
deterministic renderer builder
initial audit trace object
```

Acceptance criteria:

- existing tests pass;
- new tests cover valid/invalid plans;
- new tests cover invalid final `ReplyResponse`;
- service can safely reply when planner/reply fails;
- audit trace records plan, resolve summaries, facts, validation errors, and error-on-invalid path.

## Phase 2: policy compiler

Goal: make behavior dynamic without scenario enumeration.

Inputs:

```text
channel_type
group_id
sender_id
adapter channel sendability hints
canonical strategy hints
adapter resolve capability
recent ledger summary
feature flags
internal MCP availability
```

Outputs:

```text
allowed read capabilities
allowed capabilities and adapter resolves
allowed action candidates
allowed reply kinds
required adapter resolves
forbidden claim categories
required escalations
evidence limits
error policy
renderer policy
```

## Phase 3: action ledger

Goal: ground “just sent” references.

Ledger entries include conversation identity, inbound message identity, group/sender identity, action id/type, sendable type, strategy, adapter metadata, status, timestamps, and adapter result.
The current runtime projects executed-only ledger records into `recent_executed_action` EvidenceFacts and
`BusinessFacts.recent_executed_actions` for composer grounding.

Acceptance criteria:

- “Just sent weekly report” resolves to ledger item;
- no entry means the runtime does not pretend to know;
- failed/proposed-only actions are not treated as sent;
- expired records are removed before ledger facts are built;
- audit links evidence/resolve to adapter metadata.

## Phase 4: entity canonicalization

Goal: prevent fuzzy user text from directly driving tools/actions.

Rules:

- deterministic dictionary/catalog first;
- alias table second;
- optional LLM extraction only as candidate generation;
- deterministic validation confirms;
- multiple candidates produce clarification;
- internal wrappers receive canonical identifiers.

## Phase 5: adapter resolve integration

Wrappers:

```text
resolve_material_pack
resolve_weekly_report
resolve_monthly_report
resolve_sales_mention
```

Acceptance criteria:

- adapter resolve runs before reply composition;
- missing strategy in body alone does not become missing scope;
- only explicit negative scope metadata permits generation-scope exclusion claims;
- unavailable material/report causes clarification, human_handoff, or unable_to_answer;
- bank-channel ambiguous material-pack requests clarify.

## Phase 6: business facts layer

Goal: keep factual decisions deterministic where possible.

BusinessFacts include material/report resolvability, sales mention resolvability, report inclusion/scope, user permission, channel permission, recent executed actions, and action preconditions.

## Phase 7: planner integration

Planner input includes request metadata, recent turns, adapter-safe executed-action ledger summary, canonical scope, policy manifest, and available materials/strategies.

Planner output is `PlanSpec`: one or more capability-scoped `plan_units`, each with domain scope, required artifacts/tools, answerability policy, output schema reference, evidence contract, and execution steps.

The deterministic compiler turns `PlanSpec` into `ExecutionPlan`; the model does not output final reply text or unchecked adapter execution details.

Planner output is validated and cannot be treated as factual authority.

## Phase 8: reply composer hardening

Current runtime shape: `DecisionEngine` produces `ResponseDirective` after evidence and business fact derivation. The deterministic renderer handles action, refusal, clarification, handoff, unable, and no_reply modes. The knowledge composer is called only when `ResponseDirective.requires_knowledge_composer=true`, which requires document_context evidence.

Composer input includes original message, bounded history, validated execution plan summary, adapter resolve results, EvidenceFacts, BusinessFacts, claim restrictions, and tone rules.

Acceptance criteria:

- public response follows `reply` / `mentions` / typed actions separation;
- text/action mismatches are caught;
- pre-execution success claims are blocked;
- missing strategy scope behavior matches policy.

## Phase 9: internal MCP integration

Current status: a minimal document MCP wrapper is implemented for `query_internal_company_info`. It is controlled by
settings, calls only `list_products` and `get_documents`, and returns bounded `document_context` EvidenceFacts. Broader
internal MCP capabilities below remain future work.

Current output guardrail status: document MCP evidence is sanitized for prompt-injection lines, internal locators, and
secret-like values. Oversized sanitized chunks are truncated to the evidence budget and marked in metadata instead of
blocking otherwise useful document context.

Fixed wrappers:

```text
query_internal_company_info
query_strategy_scope
query_customer_ownership when needed
```

Wrapper requirements: explicit schema, requester identity, group/channel context, permission check, field allowlist, redaction, size limit, audit log, timeout, and failure handling.

Document MCP integration is not a free-form CrewAI tool attachment. The planner may request a document evidence
capability; the harness validates the request, calls the fixed wrapper, sanitizes the result, and passes only evidence
summaries/chunks to the composer.

Acceptance criteria:

- MCP base URL is configured through settings, not prompts;
- planner prompt exposes allowed evidence capability names, not raw MCP endpoints;
- composer prompt receives sanitized evidence only;
- material/report send actions still rely on adapter resolve, not MCP text;
- MCP timeout, sensitive-field, oversized-payload, and prompt-injection cases fall back safely.

## Phase 10: adapter execution loop

Adapter validates action type, executable material/report reference or selector, channel validity, group sendability, mention target, and schema. It writes executed/failed status and adapter-native metadata back to the runtime ledger/audit path.

Current status: material/report send proposals are action-only at the harness boundary. The postcondition validator
rejects non-empty `reply.text` or mentions on side-effect action responses, so the WeCom adapter can own the standard
post-send follow-up copy without duplicate agent text.

## Phase 11: observability and audit

Audit records request/context id, inbound message id, conversation key, policy id/hash, canonical scope, compiled plan, response directive, validations, evidence ids, adapter resolves, EvidenceFacts, BusinessFacts, reply output, error-on-invalid path, final actions, adapter execution status, latency, token usage, and model/profile/version metadata.

Avoid logging secrets, unnecessary PII, and sensitive full evidence bodies.

Current status: audit traces record compact per-CrewAI-stage metadata from `LiteAgentOutput`, including latency,
usage metrics, agent role, response format, structured-output type, raw-output length, and planning/todo counts while
excluding raw prompts, CrewAI message history, and raw plan text.

## Phase 12: evaluation suite

Run `eval_plan.md` before adding autonomy or changing model/prompt/tool/policy behavior.

Production acceptance includes zero critical action violations, zero internal data leaks, no unsupported weekly absence explanations, adequate send precision, required handoffs, and prompt-injection resistance.

## Phase 13: production hardening

Work items: persistent conversation/ledger storage, adapter outbox records, timeout budgets, LLM retry policy, circuit breaker for MCP failures, feature flags, channel rollout, safe refusal messages, model/version pinning, eval acceptance checks, audit dashboard/log queries, and privacy/compliance review.

Current timeout status: `YANFU_LLM_TIMEOUT_SECONDS` is passed to CrewAI `LLM(timeout=...)` and also enforced as a
hard `asyncio.wait_for` budget around each planner/composer/invalid-output `kickoff_async` stage.

Current retry status: `CREWAI_MAX_RETRY_LIMIT` is passed to planner and composer CrewAI agents as `max_retry_limit`,
defaulting to CrewAI's default of 2.

## Minimum safe useful version

When scope must be cut, build in this order:

```text
PolicyManifest
reply/action validator
deterministic renderer
planner with typed ExecutionPlan
EvidenceFact with fake adapter resolve
adapter resolve for material packs/reports/sales mention
action ledger
real MCP/internal info
```

## Open decisions

- Where will action ledger live: runtime memory, existing adapter DB, or new table?
- Can adapter write back executed/failed action status reliably?
- Where exactly is weekly report generation scope stored?
- Are bank/non-bank material-pack resolution rules fully defined?
- Which internal MCP fields are allowed per group/channel/user?
- What is max acceptable `/reply` latency?
- Is full markdown content allowed in logs, or only source ids/hashes?
- Should CrewAI upgrade happen before work starts?
- Should runtime eventually use CrewAI Flow formally, or a local orchestrator with the same shape?

## Definition of done

The project is safe enough when every final action is validated, every evidence claim is traceable, unavailable material is never sent, weekly missing-strategy explanations are evidence-backed, required handoffs are present, prompt-injection tests pass, MCP data is permission-filtered and redacted, adapter has final execution guardrail, ledger resolves “just sent”, eval suite runs before behavior changes, and audit trace explains every reply/action.
