# Repository Agent Instructions

Last updated: 2026-06-14.

This repository contains `market-support-crewai-agent`, an external FastAPI/CrewAI reasoning service for an existing WeCom adapter. Coding agents should treat this file as the short operational contract. Use linked docs for detail only when the task touches that area.

## Read order

For every coding session:

1. Read this file.
2. Read `README.md` for run commands and current service surface.
3. For harness work, read `docs/support_reply_harness/README.md` and `docs/support_reply_harness/next_session.md`.
4. For adapter contract work, read `docs/adapter/xiaoyan_adapter_contract.md`.

## Active architecture decision

Build a Support Reply Harness: a deterministic evidence and control layer around LLM planning and reply composition.

The LLM interprets messy Chinese sales/support language, proposes evidence needs, and composes concise typed replies. The harness owns identity, permissions, policy compilation, canonicalization, evidence execution, business fact derivation, outbound action validation, audit, and eval logging.

Use one orchestrated runtime with two bounded LLM stages when planner/composer separation is needed:

```text
Planner LLM -> PlanSpec -> validated ExecutionPlan
Reply Composer LLM -> validated ReplyResponse
```

Agents do not delegate freely. Tools run through deterministic wrappers.

## Contract boundaries

Public endpoint: `POST /reply`.

Public response boundary: one `ReplyResponse` with `reply` and typed `actions`.

User-visible free-form reply text lives in `reply.text`. Customer-visible sales mentions live in `reply.mentions`. Outbound work is represented as typed action proposals for the adapter to validate, authorize, and execute.

Canonical outbound action types for the support harness:

```text
send_material_pack
send_weekly_report
send_monthly_report
```

The WeCom adapter has final execution authority. It validates the response, owns outbox/execution reliability, executes the primary reply and actions, and writes execution feedback for ledger/audit.

## Source-of-truth order

When sources conflict, use this order:

1. Request contract and adapter-provided conversation/message identity.
2. Adapter resolve/preflight results for sendability, artifact existence, and sales mention target resolution.
3. Adapter-confirmed action ledger/execution result for what was actually sent.
4. Weekly/monthly report metadata returned by adapter resolve.
5. Permission-scoped internal MCP data.
6. Fetched markdown/report body when used as evidence.
7. Recent conversation turns.
8. LLM interpretation.

Planner output is a proposal. Deterministic evidence and business facts establish facts.

## Implementation policy

Prefer one canonical implementation path. When replacing internal behavior, update callers and tests in the same change.

A transition bridge requires a published external boundary, a test proving an active caller dependency, or an explicit ADR. Otherwise, remove superseded code in the same patch.

Validators come before autonomy. Build deterministic models, policy, evidence facts, reply/action validators, refusal, and audit traces before MCP tools, broad RAG, or multi-agent expansion.

Do not implement keyword, substring, regex, fuzzy, or n-gram matching as product, document, strategy, or report-scope selectors. Use canonical structured fields, generated manifests, adapter-provided facts, validated schemas, or bounded closed-set LLM selectors over explicit candidates. Exact equality on canonical structured fields is allowed.

Long product, document, or report-scope lists must stay out of default prompts. Return compact summaries, exact-match results, or explicit pagination; fetch full lists only through bounded evidence commands when the current task needs them.

## Prompt and documentation hygiene

Use allowlists and canonical schema descriptions in active instructions. Keep rejected field names, obsolete designs, and historical anti-patterns out of `AGENTS.md` and routine coding-session prompts. Store historical decisions in ADRs or tests where they are needed for traceability.

For instructions to coding agents, write the desired target shape directly:

```text
Canonical public response: ReplyResponse { reply, actions }
Canonical runtime fact source: BusinessFacts derived from adapter/evidence
Canonical implementation path: validators before planner autonomy
```

Do not turn every past mistake into an active prompt token. For recurring failure modes, encode them as schema constraints, validators, or tests.

## First safe build order

1. Internal enums/models for capability categories and validation results.
2. Public `ReplyResponse` models for `reply.kind`, `reply.mentions`, and canonical typed actions.
3. `AdapterResolveResult` and lightweight `EvidenceFact`.
4. `compile_policy(request, ledger_summary=None)`.
5. `PlanSpec`/`EvidenceContract` plus generic plan/reply validation.
6. Deterministic decision engine and response renderer.
7. Wire directive rendering/composer gating and reply validation in `reply_agent.py`.
8. Tests that monkeypatch unsafe planner/composer/renderer outputs and verify error-on-invalid plus audit.

## Test expectations

Run the narrowest relevant tests plus existing contract tests. For harness changes, include:

```bash
uv run --extra dev python -m pytest -q tests/integration/runtime/test_reply_contract.py tests/contract/test_adapter_preflight.py tests/unit/validation/test_structured_guardrails.py tests/unit/state/test_action_feedback.py
```

Add focused tests for every new validator, policy branch, evidence wrapper, ledger behavior, and adapter contract branch.

## Working style

Keep `/reply` stable unless a contract change is explicitly part of the task. Preserve clear module ownership:

```text
src/market_support_crewai_agent/schemas.py                    public HTTP/action DTOs
src/market_support_crewai_agent/server/main.py                FastAPI routes only
src/market_support_crewai_agent/runtime/orchestration/        reply runtime, decisions, response rendering
src/market_support_crewai_agent/runtime/validation/           input, postcondition, and alignment validators
src/market_support_crewai_agent/runtime/evidence/             evidence facts, adapter preflight, document MCP
src/market_support_crewai_agent/runtime/knowledge/            approved static knowledge catalog and selector
src/market_support_crewai_agent/runtime/domain/               capabilities, policy, planning, canonical facts
src/market_support_crewai_agent/runtime/llm/                  prompt assembly, routing, profiles, resources
src/market_support_crewai_agent/runtime/state/                conversation store, action ledger, audit trace
```

When a task is complete, summarize contract impact, validator impact, tests run, and remaining decisions.
