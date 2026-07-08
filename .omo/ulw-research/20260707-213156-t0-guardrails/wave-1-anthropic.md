# Wave 1 Anthropic Axis

Sources:
- https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks
- https://code.claude.com/docs/en/hooks
- https://code.claude.com/docs/en/mcp
- https://platform.claude.com/docs/en/test-and-evaluate/develop-tests

Key findings:
- Treat untrusted content as data, not instructions.
- Deterministic hooks are the right mechanism when behavior must always happen.
- Explicit approval/human interaction should be encoded at tool or lifecycle boundaries for side effects.

EXPAND:
- None needed for current T0 fix.
