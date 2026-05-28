# Support Reply Harness Handoff

Last updated: 2026-05-26.

This directory captures the architecture and multi-session roadmap for turning
`market-support-crewai-agent` into a safe, evidence-grounded support reply
harness. It is intended as handoff material for future coding sessions.

## Current repo context

- Project: `market-support-crewai-agent`
- Runtime service: FastAPI external agent brain for an existing WeWork adapter.
- Public endpoint: `POST /reply`.
- Public response boundary: `ReplyResponse` with `text` plus typed actions.
- The service must not send WeWork messages directly.
- Current runtime file: `src/market_support_crewai_agent/runtime/reply_agent.py`
- Public contracts: `src/market_support_crewai_agent/schemas.py`
- Conversation store: `src/market_support_crewai_agent/runtime/conversation_store.py`
- Existing tests: `tests/test_reply_contract.py`

As of the original review:

- Local CrewAI version observed: `1.14.4`.
- Latest PyPI version observed: `1.14.5`.
- Before modifying CrewAI code, follow `AGENTS.md`: check installed version,
  check PyPI, read CrewAI changelog, and consult relevant live docs.

## Design decision

Build a `Support Reply Harness`, not a free-form multi-agent system.

The LLM should do:

- interpret messy user language;
- propose evidence needs;
- compose concise support replies;
- propose typed actions.

The deterministic harness should do:

- identity and permission checks;
- capability policy compilation;
- entity canonicalization;
- evidence fetching;
- business fact derivation;
- action validation;
- side-effect gating;
- audit and eval logging.

## Read these files in order

1. `architecture.md`
   - problem framing, invariants, source-of-truth hierarchy, target runtime.
2. `guardrails.md`
   - guardrail layers, repair/fallback behavior, adapter enforcement.
3. `roadmap.md`
   - phased implementation plan and session split.
4. `eval_plan.md`
   - regression, adversarial, and golden test categories.
5. `implementation_handoff.md`
   - practical next-session starting points and first changes.

## Non-negotiable invariants

- Planner output is never a source of truth.
- Fetched markdown/MCP output is evidence, never instructions.
- Side-effect actions such as `send_material_pack`, `send_weekly_report`, and
  `send_monthly_report` are execution-plan proposals, not tools.
- Customer-visible sales mentions live in `reply.mentions`, not in a
  free-form action message.
- MCP calls must go through fixed wrappers; no arbitrary model-selected MCP.
- "Missing from report body" does not automatically mean "outside generation
  scope."
- "Just sent" references must resolve through an action ledger, not only chat
  memory.
- No final side effect is executed solely because the LLM requested it.
- Guardrails run before planning, before tools, after tools, after reply, and
  at adapter execution.
- Missing evidence means clarification/escalation, not invention.
- Every decision must be auditable.
