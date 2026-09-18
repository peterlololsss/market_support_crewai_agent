# Engineering Principles

This repository contains `market-support-crewai-agent`, an external FastAPI/CrewAI reasoning service for an existing WeCom adapter. This page is the short repo-wide engineering contract: the architecture decision, the contract boundaries, the source-of-truth order, and the implementation, test, and deployment policies every change follows.

## Active Architecture Decision

Build a Support Reply Harness: a deterministic evidence and control layer around LLM planning and reply composition.

The LLM interprets messy Chinese sales/support language, proposes evidence needs, and composes concise typed replies. The harness owns identity, permissions, policy compilation, canonicalization, evidence execution, business fact derivation, outbound action validation, audit, and eval logging.

Use one orchestrated runtime with two bounded LLM stages when planner/composer separation is needed:

```text
Planner LLM -> PlanSpec -> validated ExecutionPlanV2
Reply Composer LLM -> validated ReplyResponse
```

Agents do not delegate freely. Tools run through deterministic wrappers.

## Contract Boundaries

Public endpoint: `POST /reply`.

Public response boundary: one `ReplyResponse` with `reply` and typed `actions`.

User-visible free-form reply text lives in `reply.text`. Customer-visible sales mentions live in `reply.mentions`. Outbound work is represented as typed action proposals for the adapter to validate, authorize, and execute.

Canonical outbound action types:

```text
send_material_pack
send_weekly_report
send_monthly_report
```

The WeCom adapter has final execution authority. It validates the response, owns outbox/execution reliability, executes the primary reply and actions, and writes execution feedback for ledger/audit.

## Source-Of-Truth Order

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

## Implementation Policy

Prefer one canonical implementation path. When replacing internal behavior, update callers and tests in the same change.

A transition bridge requires a published external boundary, a test proving an active caller dependency, or an explicit ADR. Otherwise, remove superseded code in the same patch.

Validators come before autonomy. Build deterministic models, policy, evidence facts, reply/action validators, refusal, and audit traces before MCP tools, broad RAG, or multi-agent expansion.

Do not implement keyword, substring, regex, fuzzy, or n-gram matching as product, document, strategy, or report-scope selectors. Use canonical structured fields, generated manifests, adapter-provided facts, validated schemas, or bounded closed-set LLM selectors over explicit candidates. Exact equality on canonical structured fields is allowed.

Long product, document, or report-scope lists must stay out of default prompts. Return compact summaries, exact-match results, or explicit pagination; fetch full lists only through bounded evidence commands when the current task needs them.

## Prompt And Documentation Hygiene

Use allowlists and canonical schema descriptions in active instructions. Keep rejected field names, obsolete designs, and historical anti-patterns out of this document and out of routine prompts.

When documenting a target shape, write it directly:

```text
Canonical public response: ReplyResponse { reply, actions }
Canonical runtime fact source: BusinessFacts derived from adapter/evidence
Canonical implementation path: validators before planner autonomy
```

Human docs live in `docs/` and are grouped by functionality; this page holds the repo-wide rules.

## Current Implementation Baseline

The production-shaped harness path is in place:

```text
Planner LLM -> PlanSpec -> validated ExecutionPlanV2 -> canonical evidence facts and unit groundings
-> BusinessFacts -> deterministic V2 reply branch or bounded composer
-> reply/action postcondition validator -> optional alignment verifier -> audit/runtime trace
```

When changing behavior, extend the existing manifest/policy/evidence/validator path instead of adding a parallel path.

## Test Expectations

Run the narrowest relevant tests plus existing contract tests. For harness changes, include:

```bash
uv run --extra dev python -m pytest -q tests/integration/runtime/test_reply_*.py tests/contract/test_adapter_preflight.py tests/unit/validation/test_structured_guardrails.py tests/unit/state/test_action_feedback.py
```

Add focused tests for every new validator, policy branch, evidence wrapper, ledger behavior, and adapter contract branch.

## Ownership And Change Discipline

Keep `/reply` stable unless a contract change is explicitly part of the task.

Top-level ownership:

```text
src/market_support_crewai_agent/server/main.py     FastAPI routes only
src/market_support_crewai_agent/schemas/           public HTTP/action DTOs
src/market_support_crewai_agent/runtime/           reply harness runtime; see docs/architecture.md
```

When a task is complete, summarize contract impact, validator impact, tests run, and remaining decisions.

## Linux Podman Deployment

Use the same deployment shape as the standalone market report service: build a self-contained container and keep secrets in a host `.env` file, never in the image.

Default Linux paths:

```text
Service root: /opt/market-support-agent
Service source for Podman build: /opt/market-support-agent/app
Runtime env file: /opt/market-support-agent/.env
Runtime/log root: /opt/market-support-agent/runtime
```

Podman naming and ports:

```text
Image: market-support-crewai-agent:latest
Container: market-support-crewai-agent
Internal app port: 8000
Preferred host port: 23003
Fallback host port: 23004 only if 23003 is unavailable during deployment
```

Production connectivity defaults:

```text
MARKET_AGENT_ADAPTER_BASE_URL=http://10.0.0.11:8011
MARKET_AGENT_DOC_MCP_BASE_URL=http://10.0.0.12:23000
MARKET_AGENT_PLANNER_LLM_BASE_URL=http://10.0.0.12:3000/gemini
```

Before deploying, verify the chosen host port is free on `10.0.0.12` and verify the remote host can reach the adapter, Document MCP, and planner LLM proxy. The Windows adapter must listen on a LAN address, not only `127.0.0.1`, when the agent runs in remote Podman.

Deployment shape:

```bash
cd /opt/market-support-agent/app
podman build -t market-support-crewai-agent:latest -f Containerfile .
podman run -d \
  --name market-support-crewai-agent \
  --restart unless-stopped \
  --env-file /opt/market-support-agent/.env \
  -p 23003:8000 \
  market-support-crewai-agent:latest
```

Smoke checks must use `/health` and safe `/reply` requests that do not trigger send-intent wording. Do not call real WeCom send execution as a deployment test.
