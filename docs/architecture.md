# Runtime Architecture

Last updated: 2026-07-21.

`market-support-crewai-agent` is a deterministic support-reply harness around bounded LLM stages. The LLM interprets messy Chinese sales/support language and proposes intent or wording. The runtime owns source scope, policy, evidence execution, business facts, validation, audit, and outbound action safety.

## Public Boundary

The public endpoint is `POST /reply`.

The public response is always:

```text
ReplyResponse { reply, actions }
```

`reply.text` is user-visible wording. `reply.mentions` carries customer-visible sales mentions. `actions` are typed outbound proposals only. The WeCom adapter validates, authorizes, executes, and records the final execution result.

Canonical outbound action types:

```text
send_material_pack
send_weekly_report
send_monthly_report
```

The wire scenes remain `direct` and `group`. Human-facing operational aliases are prose only:
`internal_dm == identity.scene="direct"` and `external_group == identity.scene="group"`. They do not add an enum value,
request field, route, or runtime.

## Admission And Identity

Both scenes enter one shared Reply Harness runtime. The deployment owns one canonical tenant; a caller-provided v2
identity is accepted only when its `tenant_ref` exactly equals that configured tenant. The service does not route among
caller-selected tenants.

For `/reply`, admission has one fixed order. Direct-only gates are skipped for group traffic, but their relative order
does not change:

```text
service-key configuration and authentication
-> request body and contract schema
-> deployment tenant equality
-> internal-DM enablement flag
-> direct-audit key
-> adapter API key
-> authenticated adapter compatibility
-> reply reservation
```

Only after these gates does the route normalize `ReplyRequestV2` once into `VerifiedRequestEnvelopeV1` and enter the
shared service/lifecycle path. A failed gate performs no reservation, state, recall, evidence, audit, or LLM work.

Versioned feedback uses a deliberately different late-event order:

```text
service-key authentication
-> action-feedback.v2 schema
-> deployment tenant equality
-> issued-response/effect correlation
-> feedback prepare and commit
```

A correlated direct feedback event is still accepted after the internal-DM flag is turned off. Feedback does not repeat
the direct flag, audit-key, or adapter-compatibility gates because it binds to an effect the server already issued.

`GET /health` remains network-free process liveness only. It returns
`{"status":"ok","service":"market-support-crewai-agent"}` and does not expose readiness, tenant, adapter, build, or
request-derived state.

## Runtime Flow

```text
ReplyRequestV2
-> HTTP admission and deployment identity
-> VerifiedRequestEnvelopeV1
-> reply reservation
-> ConversationStore + ActionLedger
-> DomainContextV1Builder
-> PolicyManifestV2
-> input_policy.match
-> direct_send.match
-> question_recall.collect
-> strict planner/composer/verifier stage DTO projection when an LLM is needed
-> Planner LLM emits PlanSpec
-> finalize_execution_plan_v2 -> ExecutionPlanV2
-> validate_execution_plan_v2
-> EvidenceExecutor.execute_v2 -> canonical facts, resolve bindings, and unit groundings
-> deterministic renderer or compose_v2_reply
-> _validate_v2_reply final postconditions
-> optional alignment verifier
-> audit/runtime trace
-> ReplyResponse
```

Direct and group turns do not fork this pipeline. Direct policy closes recall and effects before planning/evidence;
external groups retain their distribution evidence, adapter preflight, typed proposals, and output postconditions.

Conversation state remains keyed by the full verified identity:

```text
ConversationStateKey(surface, adapter_namespace, tenant_ref, scene, subject_ref, principal_ref)
```

`principal_ref` is part of every group key. Transcript, clarification, replay, payload, ledger, receipt, and audit state
therefore remain private per principal even when two users share the same group subject.

The deterministic pre-planner order is intentional:

```text
input_policy.match      request-policy handoff rules
direct_send.match      narrow closed-set send commands
question_recall.collect fail-open approved-static Q&A or advisory document-QA candidates
```

A high-confidence approved-static recall hit may compile a `knowledge_answer` plan, but it still uses the normal evidence, renderer/composer, and validator path. Low-confidence, risky, Document-MCP-derived, or completion-like recall stays fail-open and continues to the planner.

## Domain Model

The runtime reasons over structured scope, not free-text product or report names.

```text
DistributionChannel
  -> Strategy*
      -> Product*
      -> material_pack Artifact*
  -> weekly_report Artifact*
  -> monthly_report Artifact*
```

Core concepts live under `src/market_support_crewai_agent/runtime/policy/` and related runtime packages:

```text
DistributionChannel(id, name, kind)
Strategy(id, name, channel_id)
Product(id, name, channel_id, strategy_ids)
Artifact(id, artifact_type, scope, source_type, fact_types)
ArtifactScope(channel_id, strategy_id, product_ids, time_range)
DomainContextV1(channel, strategies, products, artifacts)
```

`材料包` is not interchangeable with `周报` or `月报`. Weekly/monthly report evidence cannot satisfy material-pack questions unless a capability manifest explicitly allows that fallback. Built-in material-pack capabilities do not allow it.

## Source Precedence

When sources conflict, use this order:

1. Request contract and adapter-provided conversation/message identity.
2. Adapter resolve/preflight results for sendability, artifact existence, and sales mention target resolution.
3. Adapter-confirmed action ledger/execution result for what was actually sent.
4. Weekly/monthly report metadata returned by adapter resolve.
5. Permission-scoped internal MCP data.
6. Fetched markdown/report body when used as evidence.
7. Recent conversation turns.
8. LLM interpretation.

Planner output is a proposal. Evidence facts and `BusinessFacts` establish runtime facts.

## Module Map

```text
src/market_support_crewai_agent/server/main.py         FastAPI app and routes
src/market_support_crewai_agent/server/lifespan.py     startup/shutdown task orchestration
src/market_support_crewai_agent/server/auth.py         service-key dependency
src/market_support_crewai_agent/server/adapter_compatibility.py  process-cached adapter compatibility client
src/market_support_crewai_agent/schemas/               public HTTP and action DTOs (conversation, adapter, reply, feedback, health)
src/market_support_crewai_agent/runtime/service.py     reply service entry
src/market_support_crewai_agent/runtime/lifecycle.py   one reply turn: reservation, admission, candidate build, alignment, commit
src/market_support_crewai_agent/runtime/lifecycle_*.py turn protocols, admission, audit/journal proposals, history projection
src/market_support_crewai_agent/runtime/pipeline.py    candidate build pipeline
src/market_support_crewai_agent/runtime/planning_flow.py planner path orchestration
src/market_support_crewai_agent/runtime/planning/      input policy, direct send, PlanSpec, planner LLM
src/market_support_crewai_agent/runtime/policy/        capabilities, compliance policy, ontology
src/market_support_crewai_agent/runtime/context/       strict role-specific model input projection
src/market_support_crewai_agent/runtime/evidence/      canonical evidence facts, grounding, admission, and execution
src/market_support_crewai_agent/runtime/integrations/  adapter, Document MCP, CrewAI wrappers
src/market_support_crewai_agent/runtime/decisions/     deterministic business-fact projections
src/market_support_crewai_agent/runtime/rendering/     V2 composer projection, output authority, and response IDs
src/market_support_crewai_agent/runtime/validation/    request, reply, locator, and alignment validators
src/market_support_crewai_agent/runtime/recall/        approved static knowledge and question/document recall
src/market_support_crewai_agent/runtime/prompts/       prompt assembly, routing, profiles, resources
src/market_support_crewai_agent/runtime/state/         conversation store, action ledger, transaction coordinator, audit state
src/market_support_crewai_agent/runtime/observability/ audit trace and runtime trace helpers
```

Every production module stays under 250 pure lines; each directory owns one responsibility, and moved symbols are imported from their canonical owner rather than re-exported. Repo-wide rules live in `engineering-principles.md`.
