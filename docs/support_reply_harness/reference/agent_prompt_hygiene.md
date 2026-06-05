# Agent Prompt and Documentation Hygiene

Last updated: 2026-06-03.

This guide is for writing repo instructions that coding agents can follow without absorbing stale design debris.

## Principle

Active instructions should describe the target state, not every historical wrong turn. Repeated failure modes should become code constraints, validators, tests, or ADRs.

## Use allowlists in high-frequency context

Good active instruction shape:

```text
Public side-effect action types:
- send_material_pack
- send_weekly_report
- send_monthly_report
```

Good implementation instruction shape:

```text
Use one canonical internal implementation path. Update callers and tests in the same change.
```

Good schema instruction shape:

```text
Canonical public response: ReplyResponse { reply, actions }.
reply.text is the primary user-visible text.
reply.mentions contains customer-visible sales mentions.
```

## Keep rejected alternatives out of routine prompts

Place historical rejected designs in ADRs, migration notes, or tests. Do not load them in `AGENTS.md` unless the current task is contract migration or history review.

Reason: naming an obsolete field, obsolete action, or obsolete compatibility strategy in active context can make a coding agent reproduce it.

## Encode recurring mistakes as tests

When a model repeatedly makes the same unsafe change, prefer:

```text
Pydantic forbid extra fields
schema allowlist
validator branch
golden test
contract smoke test
```

Use prose only for routing and intent.

## Compatibility policy wording

Prefer this wording:

```text
When replacing internal behavior, converge to one canonical path in the same patch.
A transition bridge needs an external published boundary, an active caller proven by tests, or an explicit ADR.
```

This prevents unrequested dual paths while still protecting real external contracts.

## Doc placement

Use this placement rule:

```text
AGENTS.md: active operational contract and doc routing
README.md: human run commands and project surface
docs/support_reply_harness/README.md: harness index and invariants
ADR: decisions, rejected alternatives, rationale
architecture.md: source hierarchy and runtime concepts
guardrails.md: validator/evidence behavior
eval_plan.md: regression/golden/adversarial tests
next_session.md: immediate implementation tasks
```

## Review checklist for active agent docs

Before committing active instructions, check:

- Can the agent start coding from this without reading five long files?
- Does the doc state canonical names rather than rejected names?
- Are compatibility bridges gated by external contract evidence?
- Are safety requirements encoded as validators/tests where possible?
- Are historical details in ADRs rather than routine context?
