# Implementation Handoff

This file is for the next coding session. It turns the roadmap into practical
starting work while avoiding overbuilding.

## Before writing CrewAI code

Follow `AGENTS.md` freshness requirements:

1. Run installed CrewAI version check.
2. Check PyPI latest.
3. Read CrewAI changelog.
4. Consult relevant CrewAI docs page.
5. If docs conflict with local `AGENTS.md`, live docs win and `AGENTS.md`
   should be updated.

## Suggested first session objective

Implement Phase 1 without real MCP:

- internal runtime models;
- proposed production response models if adapter contract migration is in
  scope;
- policy manifest skeleton;
- adapter resolve result models;
- lightweight EvidenceFact models;
- plan validator;
- reply/action validator;
- deterministic fallback;
- audit trace skeleton;
- tests for validators.

Do not implement:

- MCP tools;
- real material fetch;
- internal company lookup;
- true multi-agent Crew.

## Proposed module ownership

Keep public contracts in:

```text
src/market_support_crewai_agent/schemas.py
```

Add runtime-only modules:

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

The first code change should make the current runtime safer even if the LLM is
still simple.

Recommended order:

1. Add internal enums/models for capability categories and validation results.
2. Add proposed response enums/models:
   `reply.kind`, `reply.mentions`, `send_material_pack`,
   `send_weekly_report`, and `send_monthly_report`.
3. Add `AdapterResolveResult` and lightweight `EvidenceFact`.
4. Add `compile_policy(request, ledger_summary=None)`.
5. Add `validate_reply(response, policy, evidence_facts, request)`.
6. Add deterministic fallback builder.
7. Wire reply validation after current agent output in `reply_agent.py`.
8. Add tests that monkeypatch fake agent outputs to ensure unsafe responses are
   repaired or replaced.

This creates safety value before adding planner/evidence complexity.

## First validator cases to implement

Start with deterministic checks:

- side-effect actions must not contain free-form text/message/caption fields.
- `reply.kind=no_reply` must have empty text, no mentions, and no actions.
- `send_material_pack` requires `material_pack_resolvable=true`.
- `send_weekly_report` requires `weekly_report_resolvable=true`.
- `send_monthly_report` requires `monthly_report_resolvable=true`.
- `reply.mentions` requires `sales_mention_resolvable=true`.
- If text says a strategy is outside report generation scope, require
  `report_scope_status=excluded`.
- If text says a report does not show a strategy, require
  `report_contains_strategy=false`.
- If action type is not allowed by compiled policy, block it.

Avoid LLM judges in the first validator pass.

## First fallback cases

Unavailable material:

```text
text:
"目前这个渠道下我没有看到可发送的对应材料，我帮你 @销售 确认。"

reply:
kind=human_handoff
mentions=[sales]

actions: []
```

Ambiguous request:

```text
text:
"我需要再确认一下你指的是哪一个材料或策略。"

reply:
kind=clarification

actions: []
```

No safe reply:

```text
text: ""
reply:
kind=no_reply
actions: []
```

## When to add planner

Add planner only after validators are in place.

Planner should output `ReplyPlan`, not `ReplyResponse`.

Planner may propose:

- user need;
- evidence requests;
- candidate terminal actions;
- required adapter resolves;
- ambiguity flags.

Planner must not:

- output final reply text;
- claim business facts;
- call tools;
- invent capabilities;
- bypass policy.

## When to add evidence executor

Add evidence executor after planner validates.

First use fake `AdapterResolveResult` fixtures. Then add adapter resolve
wrappers for material packs, weekly reports, monthly reports, and sales
mentions. Only later add internal MCP wrappers.

## When to add action ledger

Add ledger before relying on "just sent" semantics.

If adapter cannot write execution status yet, distinguish:

- proposed actions;
- executed actions;
- failed actions.

Only executed actions can ground "just sent."

Adapter owns execution reliability. The harness response should provide stable
`response_id` and `action_id`; the adapter derives operation keys from
`inbound_message_id + ":reply"` and `inbound_message_id + ":" + action_id`.

## CrewAI design note

Do not use a true multi-agent Crew initially.

Use either:

- the existing local orchestrator with two `Agent.kickoff_async` calls; or
- a CrewAI Flow-like structure if it helps state/routing.

The runtime harness should own sequencing and validation. Agents should not
delegate freely.

## Done for first coding session

The first session is complete when:

- existing tests pass;
- new validator tests pass;
- invalid LLM `ReplyResponse` cannot produce unavailable material-pack/report
  sends;
- text/action mismatch is caught;
- unsupported report-scope claims are caught;
- deterministic fallback exists;
- audit trace or structured validation result is available for debugging.
