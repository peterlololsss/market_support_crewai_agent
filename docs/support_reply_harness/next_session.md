# Next Coding Session Handoff

Last updated: 2026-06-14.

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
`get_documents`, then passes bounded `document_context` EvidenceFacts to the composer.

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

## Minimal first implementation shape

The first code change should make the current runtime safer even if the LLM is still simple.

Recommended order:

1. Add internal enums/models for capability categories and validation results.
2. Add/confirm response enums/models: `reply.kind`, `reply.mentions`, `send_material_pack`, `send_weekly_report`, `send_monthly_report`.
3. Add `AdapterResolveResult` and lightweight `EvidenceFact`.
4. Add `compile_policy(request, ledger_summary=None)`.
5. Add `validate_reply(response, directive, plan, business_facts, evidence_facts, policy)`.
6. Add deterministic decision engine and response renderer.
7. Wire directive rendering/composer gating and reply validation in `reply_agent.py`.
8. Add tests that monkeypatch fake agent outputs and renderer outputs to ensure unsafe responses raise and audit is recorded.

## First validator cases

Start with deterministic checks:

```text
outbound actions match public schema
reply.kind=no_reply has no text, mentions, or actions
send_material_pack requires material_pack_resolvable=true
send_weekly_report requires weekly_report_resolvable=true
send_monthly_report requires monthly_report_resolvable=true
reply.mentions requires sales_mention_resolvable=true
generation-scope exclusion text requires report_scope_status=excluded
report-non-inclusion text requires report_contains_strategy=false
action type is allowed by compiled policy
```

Use no LLM judge for action legality in the first pass.

## First refusal cases

Unavailable material:

```text
reply.kind=human_handoff
reply.text="目前这个渠道下我没有看到可发送的对应材料，我帮你 @销售 确认。"
reply.mentions=[sales]
actions=[]
```

Ambiguous request:

```text
reply.kind=clarification
reply.text="我需要再确认一下你指的是哪一个材料或策略。"
actions=[]
```

No-reply response:

```text
reply.kind=no_reply
reply.text=""
reply.mentions=[]
actions=[]
```

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
- Audit trace or structured validation result is available for debugging.
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
