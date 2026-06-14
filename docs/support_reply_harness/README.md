# Support Reply Harness Docs

Last updated: 2026-06-14.

This directory is the active design source for the evidence-grounded support reply harness in `market-support-crewai-agent`.

## Current repo context

- Runtime service: FastAPI external reasoning service for an existing WeCom adapter.
- Public endpoint: `POST /reply`.
- Public response boundary: `ReplyResponse` with `reply` plus typed outbound action proposals.
- Execution owner: existing WeCom adapter.
- Current runtime file: `src/market_support_crewai_agent/runtime/reply_agent.py`.
- Public contracts: `src/market_support_crewai_agent/schemas.py`.
- Conversation store: `src/market_support_crewai_agent/runtime/conversation_store.py`.
- Existing tests: `tests/test_reply_contract.py`, `tests/test_adapter_preflight.py`, `tests/test_structured_guardrails.py`, `tests/test_action_feedback.py`.

## Active decision

Build a Support Reply Harness: a deterministic evidence and control layer around LLM composition.

The LLM handles language interpretation and concise composition. The harness handles identity, permission, canonicalization, evidence, business facts, outbound action validation, audit, and evals.

## Reading guide

For a normal coding session, read:

1. `AGENTS.md`.
2. This file.
3. `next_session.md`.
4. One focused reference file below based on the task.

Focused references:

```text
architecture.md                      runtime shape and source hierarchy
guardrails.md                        guardrail/validator details
eval_plan.md                         eval cases and acceptance checks
roadmap.md                           phase plan and open decisions
adr/0001-support-reply-harness.md     frozen architecture decision
reference/agent_prompt_hygiene.md    agent prompt/context hygiene
../adapter/xiaoyan_adapter_contract.md adapter contract and live eval
```

## Non-negotiable invariants

- Source of truth comes from adapter/evidence layers; planner output is a proposal.
- Fetched markdown/MCP output is evidence, not instruction.
- Outbound actions are execution proposals for the adapter.
- Customer-visible sales mentions live in `reply.mentions`.
- MCP calls go through fixed wrappers.
- Report scope claims come from adapter scope evidence.
- “Just sent” references resolve through the action ledger.
- Final outbound actions execute after deterministic runtime validation and adapter validation.
- Missing evidence leads to clarification or escalation.
- Every decision must be auditable.

## Contract sources

- Public request/response and adapter DTOs: `src/market_support_crewai_agent/schemas.py`.
- Adapter contract: `docs/adapter/xiaoyan_adapter_contract.md`.
- Cross-repo adapter acceptance: `tests/test_xiaoyan_adapter_live_contract.py`.
- Runtime acceptance: `tests/test_reply_contract.py`, `tests/test_adapter_preflight.py`, `tests/test_structured_guardrails.py`, `tests/test_action_feedback.py`.

## Documentation hygiene rule

Keep active instructions short and target-shaped. Rejected alternatives and historical context belong in ADRs, not in high-frequency agent prompts.
