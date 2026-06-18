# Next Coding Session Handoff

Last updated: 2026-06-17.

This file is the practical starting point for the next coding agent session.

## Current implementation state

The harness now has the first production-shaped runtime path:

```text
Planner LLM -> PlanSpec -> validated ExecutionPlan -> EvidenceExecutor -> EvidenceFacts
-> BusinessFacts -> ResponseDirective -> deterministic renderer or knowledge composer
-> reply/action postcondition validator -> error-on-invalid
```

Implemented:

```text
ReplyResponse public boundary
PolicyManifest
PlanSpec and EvidenceContract planner/verifier boundary
ExecutionPlan and plan validation
generic PlanSpec verifier
adapter resolve/preflight wrapper
document MCP evidence wrapper, controlled by settings feature flags
EvidenceFact and BusinessFacts
ResponseDirective decision engine
reply/action postcondition validator
knowledge_answer document-grounding validator
deterministic response renderer
conversation history
adapter execution feedback ledger
audit trace with compact CrewAI stage usage metadata
runtime trace spans/events in audit and optional live logs
ContextProjectionManager and ModelVisibleContext before planner/composer/verifier prompts
configurable input message length guardrail
CrewAI planner/composer timeout budget
configurable CrewAI retry budget
action-only material/report send replies with adapter-owned standard follow-up copy
document MCP prompt-injection, secret/locator, and oversized-output guardrails
Runtime fake-dependency check script at `scripts/check_reply_runtime_fake_deps.py`
```

Document MCP is not attached directly to CrewAI agents. It is enabled only when
`MARKET_AGENT_DOC_MCP_ENABLED=true` and `MARKET_AGENT_DOC_MCP_BASE_URL` is configured. The planner sees only the
policy-allowed capability name `query_internal_company_info`; the fixed wrapper calls `/mcp` with `list_products` and
`get_documents`, ranks relevant docs first, appends the remaining small corpus, then passes bounded `document_context`
EvidenceFacts to the composer.

## Module ownership

Keep public contracts in:

```text
src/market_support_crewai_agent/schemas.py
```

Add runtime-only modules as needed:

```text
src/market_support_crewai_agent/runtime/domain/
src/market_support_crewai_agent/runtime/evidence/
src/market_support_crewai_agent/runtime/context/
src/market_support_crewai_agent/runtime/knowledge/
src/market_support_crewai_agent/runtime/llm/
src/market_support_crewai_agent/runtime/orchestration/
src/market_support_crewai_agent/runtime/state/
src/market_support_crewai_agent/runtime/validation/
```

Likely tests:

```text
tests/unit/domain/test_policy.py
tests/unit/validation/test_planning_guardrails.py
tests/unit/validation/test_structured_guardrails.py
tests/unit/evidence/test_evidence_executor.py
tests/unit/evidence/test_document_mcp.py
tests/unit/state/test_action_feedback.py
tests/unit/evidence/test_report_scope_evidence.py
tests/integration/runtime/test_reply_contract.py
```

## Current implementation focus

The bootstrap path is complete. New work should be a small extension to one of these existing seams:

```text
CapabilityManifest / EvidenceContract
PolicyManifest / PlanSpec compiler
EvidenceExecutor wrapper
BusinessFacts derivation
ResponseDirective / renderer
reply, output, answerability, or alignment validator
ContextProjectionManager / prompt assembly
ledger, audit, or runtime trace
```

Do not add a second planner, renderer, validator pipeline, or adapter contract shape.

## Current validator floor

Keep these deterministic checks green before adding model autonomy:

```text
outbound actions match public schema
reply.kind=no_reply has no text, mentions, or actions
send_material_pack requires material_pack_resolvable=true
send_weekly_report requires weekly_report_resolvable=true
send_monthly_report requires monthly_report_resolvable=true
reply.mentions requires sales_mention_resolvable=true
report-content claims require report_scope_summary/report_scope_match/report_scope_products evidence
action type is allowed by compiled policy
source/artifact/history use matches selected EvidenceContract
```

Use no LLM judge for action legality.

## Planner boundary

Planner outputs `PlanSpec`, not `ReplyResponse`, `ResponseDirective`, `BusinessFacts`, or final adapter evidence. `PlanSpec`
selects one capability manifest, declares domain scope, required/allowed/forbidden artifacts, required tools,
answerability policy, output schema ref, evidence contract ref or inline contract, steps, acceptance criteria,
abstention cases, and risk flags. The runtime compiles it to `ExecutionPlan` and the verifier validates it generically
through `PlanSpec` plus `EvidenceContract`.

## Evidence executor timing

Evidence executor now runs after execution-plan validation. It resolves only the adapter preflight specs present in
`ExecutionPlan.adapter_resolves`; the compiler must include `sales_mention` when deterministic handoff or unavailable
side-effect paths need it. Document MCP runs only for compliant `knowledge_answer` plans that request
`document_context` and only when the feature flag is enabled.

## Ledger timing

Ledger facts now feed evidence/business state before composition. The runtime converts only adapter-confirmed
`status=executed` ledger records into `recent_executed_action` EvidenceFacts and
`BusinessFacts.recent_executed_actions`; proposed, failed, skipped, and expired actions do not ground “just sent.”

## Current acceptance baseline

- Existing tests pass.
- New validator tests pass.
- Invalid planner output raises `AgentRuntimeError`.
- Invalid LLM `ReplyResponse` or invalid deterministic renderer output raises `ReplyContractError` after audit.
- Invalid LLM `ReplyResponse` cannot produce unavailable material-pack/report sends.
- Text/action mismatch is caught.
- Unsupported report-scope claims are caught.
- Deterministic renderer exists.
- Audit/runtime trace or structured validation result is available for debugging.
- `knowledge_answer` reply text requires document MCP evidence when that response mode is used.

## Context Is A Projection

Production prompt assembly now expects `PromptAssemblyContext(model_visible_context=...)`.
When adding source, evidence, or runtime state, project it through
`ContextProjectionManager` first instead of appending direct prompt text.

`ModelVisibleContext` separates:

```text
recent_verbatim / compacted_summary   conversation context only
large_result_preview                  stable preview + reload_handle
allowed_evidence                      current-plan accepted evidence
context_only                          ledger/history/app state
disallowed_evidence                   rejected source, content redacted
app_state/current_task/ephemeral      runtime state, current user task, retry state
```

Conversation history is not claim evidence unless an evidence contract allows
history and guardrails accept it. Context pressure and projection decisions are
recorded in prompt-program audit metadata; full prompt text and large payloads
stay out of audit.
