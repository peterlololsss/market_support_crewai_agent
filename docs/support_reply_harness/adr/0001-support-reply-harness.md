# ADR 0001: Support Reply Harness, not autonomous multi-agent

Date: 2026-06-03.

## Status

Accepted.

## Context

The service is an external reasoning brain for a market support workflow. It receives structured WeCom chat context and returns one `ReplyResponse` for a WeCom adapter to execute.

The system must handle terse Chinese sales/support phrasing without enumerating every possible user message. It also must prevent unsupported claims, internal data leakage, permission bypass, and unsafe outbound actions.

The core tension is bounded autonomy: enough model interpretation to understand ambiguous domain language, but not enough model authority to invent facts or execute actions without deterministic evidence.

## Decision

Build one orchestrated Support Reply Harness.

Use deterministic runtime layers for:

- request validation;
- identity and permission checks;
- entity canonicalization;
- request-scoped policy compilation;
- evidence execution through fixed wrappers;
- lightweight EvidenceFact derivation;
- BusinessFacts derivation;
- reply/action validation;
- deterministic renderer;
- audit and eval logging.

Use LLM stages only for bounded language work:

```text
Planner LLM -> PlanSpec -> validated ExecutionPlan
Reply Composer LLM -> validated ReplyResponse
```

The planner proposes evidence needs and planned capabilities. Planner conclusions are not facts. The composer sees sanitized evidence/business facts and proposes the public response. Validators determine whether the final response is allowed.

## Consequences

The first safe build path is validators and contracts before autonomy. MCP integration, broad RAG, and additional agents come later.

The adapter has final execution authority for outbound actions. It validates the `ReplyResponse`, executes the primary reply and outbound action proposals, owns outbox/retry/idempotency, and writes execution feedback for ledger/audit.

The public `/reply` boundary remains stable unless an explicit contract change is accepted.

## Compatibility policy

Internal code should converge to one canonical path in the same change. A transition bridge needs an external published boundary, an active caller proven by tests, or a follow-up ADR.

## Prompt/context policy

Active agent instructions should describe the canonical target state and allowlists. Historical rejected designs stay in ADRs or tests, not in routine coding-agent context.
