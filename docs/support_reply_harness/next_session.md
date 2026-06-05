# Next Coding Session Handoff

Last updated: 2026-06-03.

This file is the practical starting point for the next coding agent session.

## Before CrewAI runtime code

Follow `AGENTS.md` freshness requirements:

1. Run installed CrewAI version check.
2. Check PyPI latest.
3. Read the CrewAI changelog.
4. Consult the relevant live CrewAI docs page.
5. Update local docs when verified live behavior conflicts with local guidance.

## Objective

Implement Phase 1 without real MCP.

Build:

```text
internal runtime models
ReplyResponse models for current public adapter contract scope
PolicyManifest skeleton
AdapterResolveResult models
EvidenceFact models
plan validator
reply/action validator
deterministic fallback
audit trace skeleton
tests for validators
```

MCP tools, real material fetch, internal company lookup, and autonomous multi-agent flow come later.

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
src/market_support_crewai_agent/runtime/business_facts.py
src/market_support_crewai_agent/runtime/guardrails.py
src/market_support_crewai_agent/runtime/action_ledger.py
src/market_support_crewai_agent/runtime/audit.py
```

Likely tests:

```text
tests/test_policy.py
tests/test_planning_guardrails.py
tests/test_reply_guardrails.py
tests/test_evidence_executor.py
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
5. Add `validate_reply(response, policy, evidence_facts, request)`.
6. Add deterministic fallback builder.
7. Wire reply validation after current agent output in `reply_agent.py`.
8. Add tests that monkeypatch fake agent outputs to ensure unsafe responses are repaired or replaced.

## First validator cases

Start with deterministic checks:

```text
side-effect actions match public schema
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

## First fallback cases

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

No safe reply:

```text
reply.kind=no_reply
reply.text=""
reply.mentions=[]
actions=[]
```

## Planner timing

Add planner after validators are in place.

Planner outputs `ReplyPlan`, not `ReplyResponse`. It proposes user need, evidence requests, candidate terminal actions, required adapter resolves, ambiguity flags, and confidence. It does not call tools, invent capabilities, bypass policy, produce final reply text, or claim final business facts.

## Evidence executor timing

Add evidence executor after planner validation. Start with fake `AdapterResolveResult` fixtures. Then add adapter resolve wrappers for material packs, weekly reports, monthly reports, and sales mentions. Internal MCP wrappers come later.

## Ledger timing

Add ledger before relying on “just sent” semantics. Distinguish proposed, executed, and failed actions. Only executed adapter-confirmed actions ground “just sent.”

## Done for first coding session

- Existing tests pass.
- New validator tests pass.
- Invalid LLM `ReplyResponse` cannot produce unavailable material-pack/report sends.
- Text/action mismatch is caught.
- Unsupported report-scope claims are caught.
- Deterministic fallback exists.
- Audit trace or structured validation result is available for debugging.
