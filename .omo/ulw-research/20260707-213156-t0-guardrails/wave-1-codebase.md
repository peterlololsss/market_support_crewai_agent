# Wave 1 Codebase Axis

Key findings:
- Existing seam: `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py` runs before direct-send and planner.
- Existing renderer consumes generic `GuardrailDecision.metadata` keys, so T0 should not add a downstream branch.
- `docs/support_reply_harness/guardrails.md` explicitly says input-policy rules should stay as data in one rule table instead of one guardrail class per topic.
- Risk in previous patch: hardcoded T0 branch and unconditional sales_mention capability.

EXPAND:
- Generic input-policy rule schema/registry.
- Metadata-driven handoff text remains the correct downstream boundary.
