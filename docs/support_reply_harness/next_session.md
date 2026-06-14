# Next Coding Session Handoff

Last updated: 2026-06-14.

This file is the practical starting point for the next coding agent session.

## Current implementation state

The harness now has the first production-shaped runtime path:

```text
Planner LLM -> IntentFrame -> validated ExecutionPlan -> EvidenceExecutor -> EvidenceFacts
-> BusinessFacts -> ResponseDirective -> deterministic renderer or knowledge composer
-> reply/action postcondition validator -> error-on-invalid
```

Implemented:

```text
ReplyResponse public boundary
PolicyManifest
ExecutionPlan and plan validation
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
src/market_support_crewai_agent/runtime/policy.py
src/market_support_crewai_agent/runtime/planning.py
src/market_support_crewai_agent/runtime/evidence.py
src/market_support_crewai_agent/runtime/evidence_executor.py
src/market_support_crewai_agent/runtime/document_mcp.py
src/market_support_crewai_agent/runtime/business_facts.py
src/market_support_crewai_agent/runtime/decision.py
src/market_support_crewai_agent/runtime/response_renderer.py
src/market_support_crewai_agent/runtime/guardrails.py
src/market_support_crewai_agent/runtime/action_ledger.py
src/market_support_crewai_agent/runtime/audit.py
```

Likely tests:

```text
tests/test_policy.py
tests/test_planning_guardrails.py
tests/test_structured_guardrails.py
tests/test_evidence_executor.py
tests/test_document_mcp.py
tests/test_action_ledger.py
tests/test_weekly_scope.py
tests/test_prompt_injection.py
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

## Planner timing

Add planner after validators are in place.

Planner outputs `IntentFrame`, not `ReplyResponse` or `ExecutionPlan`. It proposes user need, artifact kind, action intent, compliance, strategy mentions, selected strategy, report scope, ambiguity slots, requested capabilities, and confidence. It does not call tools, invent capabilities, bypass policy, produce final reply text, output adapter resolves, output response mode, or claim final business facts.

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
