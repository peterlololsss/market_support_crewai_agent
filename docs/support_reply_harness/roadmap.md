# Multi-Session Roadmap

This is intentionally split into phases so future sessions can continue safely.

## Phase 0 - Decision freeze and baseline

Goal: prevent architectural drift before coding.

Deliverables:

- Architecture decision record: "Support Reply Harness, not autonomous
  multi-agent."
- Confirm source-of-truth hierarchy.
- Confirm LLM-owned vs harness-owned decisions.
- Confirm whether CrewAI upgrade from `1.14.4` to `1.14.5` is needed now.
- Pin dependencies before behavior changes.
- Keep existing `/reply` contract stable.

Acceptance criteria:

- `ReplyResponse` remains the external boundary.
- No arbitrary MCP tools go directly to LLM.
- Missing evidence means clarification/escalation, not invention.
- "Missing from report" and "outside generation scope" are different facts.

## Phase 1 - Runtime contracts without MCP

Goal: build skeleton harness with fake/no-op evidence.

Build:

- `PolicyManifest`
- `ReplyPlan`
- `EvidenceBundle`
- `BusinessFacts`
- `ValidationResult`
- `AuditTrace`
- request-scoped policy compiler
- plan validator
- reply/action validator
- deterministic fallback builder
- initial audit trace object

Do not build:

- real MCP integration;
- complex multi-agent system;
- report parsing;
- internal company lookup.

Acceptance criteria:

- Existing tests still pass.
- New tests cover valid and invalid plans.
- New tests cover invalid final `ReplyResponse`.
- Service can still reply when planner/reply fails via fallback.
- Audit trace records plan, validation errors, repair/fallback path.

## Phase 2 - Policy compiler

Goal: make behavior dynamic without scenario enumeration.

Inputs:

- `channel_type`
- `group_id`
- `sender_id`
- `available_materials`
- `available_strategies`
- recent action ledger summary
- feature flags
- material/internal MCP availability

Outputs:

- allowed read capabilities;
- allowed business checks;
- allowed action candidates;
- forbidden claims;
- required escalations;
- evidence limits;
- repair policy;
- fallback policy.

Acceptance criteria:

- Tests prove policies differ by channel/material availability.
- Planner cannot bypass policy by wording.
- Policy manifest is included in audit trace.
- Feature flags can disable internal info safely.

## Phase 3 - Action ledger

Goal: ground "just sent" references.

Build:

- `ActionLedger` interface.
- Initial in-memory implementation.
- Later persistent implementation.
- Ledger write on successful adapter-executed action when possible.
- If adapter cannot write back yet, runtime may record proposed actions but
  must mark them as unconfirmed.

Ledger entry fields:

```text
conversation_key
context_id/message_id
group_id
sender_id
action_type
material_type
strategy
material_id
version
status: proposed | executed | failed
created_at
executed_at
adapter_result
```

Acceptance criteria:

- "Just sent weekly report" resolves to ledger item, not generic latest.
- If no ledger entry exists, system does not pretend it knows.
- Failed/proposed-only actions are not treated as sent.
- Audit trace links evidence fetch to ledger material id/version.

## Phase 4 - Entity canonicalization

Goal: prevent fuzzy user text from directly driving tools/actions.

Entities:

- strategy names;
- securities names if relevant;
- company/customer names;
- material types;
- temporal references such as "just now", "this week", "latest".

Rules:

- deterministic dictionary/catalog first;
- alias table second;
- optional LLM-assisted extraction only as a candidate generator;
- deterministic validation must confirm;
- multiple candidates require clarification;
- raw ambiguous entity must not be sent to internal MCP as final identifier.

Acceptance criteria:

- Strategy aliases resolve to canonical names.
- Unknown strategy does not become fabricated.
- Multiple candidates force clarification or sales mention.
- Raw user text is not sent directly into internal wrappers.

## Phase 5 - Material/weekly evidence integration

Goal: fetch whole markdown safely.

Build wrappers:

- `list_materials`
- `fetch_latest_weekly_md`
- `fetch_sent_material_md`
- `fetch_material_md`

Evidence metadata required:

- `material_id`
- `material_type`
- `strategy`
- `channel_type`
- `version`
- `published_at`
- `source_key`
- `generated_strategies` if available
- `scope_strategies` if available
- `markdown_body`

Acceptance criteria:

- Full markdown fetch works under size limit.
- Prompt injection inside markdown is ignored.
- Missing strategy in body only does not become missing scope.
- Missing strategy in explicit scope metadata triggers mention_sales.
- Unavailable weekly material forces clarification or mention_sales.

## Phase 6 - Business facts layer

Goal: keep factual decisions out of LLM where possible.

Business checks:

- `check_material_availability`
- `check_report_contains_strategy`
- `check_report_generation_scope`
- `check_user_permission`
- `check_channel_permission`
- `check_action_preconditions`

Acceptance criteria:

- Reply composer receives business facts.
- Validator checks final reply against business facts.
- If `must_mention_sales=true`, final response has `mention_sales`.
- If `can_send_material=false`, final response has no `send_material`.

## Phase 7 - Planner integration

Goal: make planner useful but not authoritative.

Planner input:

- request metadata;
- recent turns summary;
- ledger summary;
- canonical entities;
- policy manifest;
- available materials/strategies.

Planner output:

- user need;
- detected entities;
- evidence requests;
- business checks requested;
- action candidates;
- ambiguities;
- confidence.

Planner must not output:

- final user text;
- final business facts;
- raw tool calls;
- arbitrary MCP tool names;
- unsupported explanations.

Acceptance criteria:

- Planner handles diverse phrasings without new branches.
- Planner cannot invent capabilities.
- Planner cannot request direct side effects.
- Invalid planner outputs are repaired or safely fall back.

## Phase 8 - Reply composer hardening

Goal: generate final `ReplyResponse` from validated facts.

Reply input:

- original user message;
- bounded recent turns;
- validated plan summary;
- evidence bundle;
- business facts;
- allowed action candidates;
- forbidden claims;
- tone rules.

Acceptance criteria:

- Public response contract remains unchanged.
- Invalid text/action mismatches are caught.
- "I sent it" without `send_material` is caught.
- Missing strategy scope behavior matches policy.

## Phase 9 - Internal MCP integration

Goal: safely query internal company/customer/strategy info.

Fixed wrappers only:

- `query_internal_company_info`
- `query_strategy_scope`
- `query_customer_ownership` if needed later

Wrapper requirements:

- explicit schema;
- requester identity;
- group/channel context;
- permission check before call;
- field allowlist;
- redaction;
- result size limit;
- audit log;
- timeout;
- failure fallback.

Acceptance criteria:

- Permission denied returns safe response.
- Missing company name asks clarification.
- Ambiguous company name asks clarification.
- Sensitive fields are removed before reply LLM sees evidence.
- Internal MCP failure does not crash `/reply`.

## Phase 10 - Adapter feedback loop

Goal: adapter remains final side-effect gate.

Adapter should verify:

- action type allowed;
- material exists;
- strategy exists;
- material id/version matches catalog;
- group/channel can receive it;
- mention target valid;
- idempotency;
- schema validity.

Adapter should write back:

- executed action status;
- material id/version;
- failure reason;
- adapter message id.

Acceptance criteria:

- Bad `send_material` is refused by adapter.
- Ledger records executed vs failed.
- Repeated requests do not duplicate unsafe actions.
- Runtime resolves "just sent" from adapter-confirmed ledger.

## Phase 11 - Observability and audit

Goal: make every decision replayable.

Log/audit:

- request id/context id;
- conversation key;
- policy manifest id/hash;
- canonical entities;
- planner output;
- plan validation result;
- evidence requests;
- evidence source ids/versions;
- business facts;
- reply output;
- reply validation result;
- repair attempts;
- fallback reason;
- final actions;
- adapter execution status;
- latency;
- token usage;
- model/version.

Avoid logging:

- full sensitive internal data;
- credentials;
- unnecessary raw user PII;
- full markdown if compliance disallows it.

Acceptance criteria:

- Any bad reply can be traced to plan/evidence/validator.
- Eval failures can replay with same evidence ids.
- Guardrail blocks are measurable.
- Latency and LLM cost are visible.

## Phase 12 - Evaluation suite

Goal: prevent regression before adding autonomy.

See `eval_plan.md`.

Acceptance criteria before production:

- zero critical action violations in golden set;
- zero internal data leakage in adversarial set;
- high precision for `send_material`;
- mandatory `mention_sales` never missed in missing-scope cases;
- no unsupported "why" explanations in weekly absence cases.

## Phase 13 - Production hardening

Goal: make runtime reliable under real usage.

Work items:

- persistent conversation/ledger storage;
- idempotency keys;
- request timeout budget;
- evidence fetch timeout budget;
- LLM retry policy;
- circuit breaker for MCP failures;
- feature flags per capability;
- rollout by group/channel;
- safe fallback messages;
- model/version pinning;
- eval gate before model upgrades;
- audit dashboard or log queries;
- privacy/compliance review.

Acceptance criteria:

- MCP down does not break all replies.
- LLM invalid output does not break all replies.
- Feature flag can disable internal info instantly.
- Known safe fallback always exists.
- Production logs support incident review.

## Session split

Recommended new-session breakdown:

1. Architecture ADR + internal contracts.
2. Validators first.
3. Policy compiler.
4. Planner stage.
5. Action ledger.
6. Material evidence skeleton.
7. Real material/weekly integration.
8. Reply composer hardening.
9. Internal MCP integration.
10. Adapter feedback loop.
11. Eval harness.
12. Production hardening.

## If scope must be cut

Minimum safe useful version:

1. `PolicyManifest`
2. reply/action validator
3. deterministic fallback
4. planner with typed `ReplyPlan`
5. `EvidenceBundle` with fake provider
6. material weekly fetch
7. action ledger
8. real MCP/internal info

Do not start with MCP. Do not start with multi-agent. Do not start with fancy
RAG. Do not expand the public action schema unless the adapter needs it.

## Top risks

- Treating planner conclusions as facts.
- Letting markdown prompt injection override policy.
- Treating missing report text as missing generation scope.
- Sending material based on fuzzy strategy match.
- Querying internal MCP with raw ambiguous text.
- No adapter feedback, making "just sent" unreliable.
- Conversation memory leaking across users/groups.
- LLM reply text disagreeing with action list.
- Overusing LLM judge for deterministic safety.
- Adding multiple agents before contracts are stable.
- No eval gate before prompt/model/version changes.
- Internal MCP returning fields the user should not see.

## Open decisions

- Where will action ledger live: runtime memory, existing adapter DB, new table?
- Can adapter write back executed/failed action status?
- Does material metadata include `generated_strategies` or `scope_strategies`?
- What exactly is weekly report generation scope and where is it stored?
- Are `available_strategies` authoritative or only UI hints?
- Who is sales mention target: adapter decides, or runtime provides reason only?
- Are bank/non-bank policy differences already defined?
- Which internal MCP fields are allowed per group/channel/user?
- What is max acceptable `/reply` latency?
- Is full markdown content allowed in logs, or only source ids/hashes?
- Should CrewAI upgrade from `1.14.4` to `1.14.5` happen before work starts?
- Should runtime eventually use CrewAI Flow formally, or a local orchestrator
  with the same shape?

## Definition of done

The project is safe enough when:

- every final action is validated against policy and request availability;
- every evidence claim is traceable to source id/version or business fact;
- unavailable material is never sent;
- missing strategy in weekly report never produces invented explanation;
- mandatory sales escalation is never missed in golden cases;
- markdown/MCP prompt injection tests pass;
- internal MCP data is permission-filtered and redacted;
- adapter has final execution guardrail;
- action ledger can resolve "just sent" reliably;
- eval suite runs before prompt/model/tool changes;
- audit trace can explain why any reply/action happened.

