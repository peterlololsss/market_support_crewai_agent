# Wave 1 OpenAI Axis

Sources:
- https://developers.openai.com/api/docs/guides/agents/guardrails-approvals
- https://developers.openai.com/api/docs/guides/agent-builder-safety
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/guides/function-calling
- https://developers.openai.com/cookbook/examples/partners/agentic_governance_guide/agentic_governance_cookbook

Key findings:
- Guardrails are lifecycle checks: input before main model, output before user-visible response, tool checks around tool calls, human review for side effects.
- Structured schemas are shape guarantees; deterministic business validators still own domain correctness.
- Untrusted input should be narrowed to structured fields/enums/validated JSON before it drives behavior.

EXPAND:
- None needed for current T0 fix.
