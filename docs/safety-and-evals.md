# Safety And Evals

Last updated: 2026-07-20.

The harness is safety-first: model output is a proposal until deterministic validators, evidence checks, and adapter preflight make it safe to answer or propose an action.

## Guardrail Pipeline

Reject unsafe state at the active V2 boundary where it first appears:

```text
request_input_guard             raw /reply request checks
validate_execution_plan_v2      finalized plan and policy checks
ground_execution_plan_v2        canonical evidence admission and unit grounding
_validate_v2_composer_output    evidence, media, action, mention, and reply ceilings
_validate_v2_reply              final scene and reply/action postconditions
alignment verifier/loop         optional bounded semantic verification and remediation
```

Guardrails return machine-readable reason codes. They should not silently repair unsafe model output.

## Answerability

Each validated V2 plan unit declares its answerability policy. Canonical evidence execution binds accepted facts and derived `UnitBusinessFactsV1` to that unit before rendering:

```text
answer   -> the bounded composer may use only admitted unit evidence
clarify  -> the deterministic V2 branch asks for the missing bounded input
handoff  -> the deterministic V2 branch uses only adapter-resolved mention authority
abstain  -> the deterministic V2 branch returns a safe unable response
```

This prevents material-pack questions from being answered with unrelated weekly/monthly report evidence.

## Selector Rules

Do not use keyword, substring, regex, fuzzy, or n-gram matching as product, document, strategy, or report-scope selectors.

Allowed alternatives:

- Canonical structured fields from the request, adapter, or manifests.
- Exact equality on canonical IDs or normalized structured fields.
- Adapter resolve/preflight results.
- Validated schemas and closed-set LLM selection over explicit candidates.
- Bounded evidence commands that fetch only the current task's needed list.

Recall and search may propose candidates, but final product/document/report scope must still come from the deterministic policy/evidence/validator path.

## Adapter Network Boundary

Authenticated adapter requests are pinned to one parsed origin before and after I/O. All redirects are rejected without
a second request. HTTPS is allowed normally; HTTP is limited to literal loopback, RFC1918, or link-local addresses and
must run only on an authenticated private or encrypted tunnel network. Public or hostname-based cleartext endpoints
fail closed.

## Eval And Test Commands

For harness behavior changes, run the narrowest relevant tests plus contract coverage. Common commands:

```bash
uv run --extra dev python -m pytest -q tests/integration/runtime/test_reply_*.py tests/contract/test_adapter_preflight.py tests/unit/validation/test_structured_guardrails.py tests/unit/state/test_action_feedback.py
uv run python scripts/check_no_semantic_keyword_matching.py
uv run python scripts/check_prompt_registry.py
```

For focused reply-contract coverage, run the direct action, group policy, feedback
correlation, and observed-only CrewAI completion capture checks:

```bash
CREWAI_MAX_RETRY_LIMIT=0 uv run --extra dev python -m pytest -q tests/integration/runtime/test_reply_contract_deterministic_actions.py tests/integration/runtime/test_reply_contract_artifact_clarification.py tests/integration/runtime/test_reply_contract_group_handoff.py tests/integration/runtime/test_reply_contract_group_policy.py tests/integration/runtime/test_reply_contract_internal_dm_knowledge.py tests/unit/server/test_feedback_route_correlation.py tests/unit/orchestration/test_crewai_provider_transport.py::test_factory_capture_records_only_real_crewai_completion_invocation
```

For the single-enterprise scene boundary, run the documented-contract test and focused amendment matrix:

```bash
uv run --extra dev python -m pytest -q tests/contract/test_documented_scene_contract.py
uv run --extra dev python -m pytest -q tests/unit/settings/test_settings.py tests/unit/identity/test_deployment_identity.py tests/unit/server/test_scene_admission.py tests/unit/server/test_feedback_route_correlation.py tests/contract/test_adapter_client_contract.py tests/contract/test_adapter_preflight.py tests/integration/runtime/test_reply_*.py tests/integration/runtime/test_scene_activation.py tests/integration/runtime/test_scene_http_scripts.py tests/unit/policy/test_policy_authority_v2.py tests/unit/recall/test_direct_media_policy.py tests/unit/validation/test_structured_guardrails.py tests/unit/state/test_action_feedback.py tests/unit/state/test_scene_principal_isolation.py
```

The repository-required harness suite and complete guards are (the literal
repository command currently reports 88 tests because the seven relocated
preflight-failure cases are not part of its path list):

```bash
uv run --extra dev python -m pytest -q tests/integration/runtime/test_reply_*.py tests/contract/test_adapter_preflight.py tests/unit/validation/test_structured_guardrails.py tests/unit/state/test_action_feedback.py
uv run python scripts/check_prompt_registry.py
uv run python scripts/check_no_semantic_keyword_matching.py
uv run --extra dev python scripts/check_reply_acceptance.py
uv run python scripts/check_reply_runtime_fake_deps.py
uv run --extra dev python -m pytest -q
```

The preserved 95-test harness gate includes the relocated preflight-failure
module explicitly:

```bash
uv run --extra dev python -m pytest -q tests/integration/runtime/test_reply_*.py tests/contract/test_adapter_preflight.py tests/contract/test_adapter_preflight_failures.py tests/unit/validation/test_structured_guardrails.py tests/unit/state/test_action_feedback.py
```

Real loopback HTTP proof uses only typed fake dependencies and fixture credentials:

```bash
mkdir -p .omo/evidence/manual-scene-http
uv run python scripts/serve_reply_fake_deps.py --host 127.0.0.1 --port 18080 --adapter-port 18081 --api-key fixture-secret --adapter-api-key fixture-adapter-secret --tenant-ref tenant:test --internal-dm-enabled true --mode compatible >.omo/evidence/manual-scene-http/server.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
uv run python scripts/check_scene_http_contract.py --base-url http://127.0.0.1:18080 --api-key fixture-secret --tenant-ref tenant:test --output .omo/evidence/manual-scene-http/result.json
kill "$server_pid"
wait "$server_pid" || test $? -eq 143
trap - EXIT
```

Closed startup modes must each use a fresh process:

```bash
for mode in tenant-unconfigured dm-disabled adapter-missing-fields adapter-wrong-scene adapter-wrong-tenant adapter-wrong-version; do
  uv run python scripts/serve_reply_fake_deps.py --host 127.0.0.1 --port 18080 --adapter-port 18081 --api-key fixture-secret --adapter-api-key fixture-adapter-secret --tenant-ref tenant:test --internal-dm-enabled true --mode "$mode" >>.omo/evidence/manual-scene-http/error-server.log 2>&1 &
  server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
  uv run python scripts/check_scene_http_contract.py --base-url http://127.0.0.1:18080 --api-key fixture-secret --tenant-ref tenant:test --expect-mode "$mode" --append-output .omo/evidence/manual-scene-http/error.json
  kill "$server_pid"
  wait "$server_pid" || test $? -eq 143
  trap - EXIT
done
```

These HTTP checks must finish with `send_spy_count=0` and closed ports 18080/18081. Never substitute production
credentials, an external adapter execution endpoint, or a real WeCom send as a smoke test. `/health` proves liveness,
not readiness.

For prompt changes, run the machine-consumed prompt safety set covering registry, structure, budgets, baseline/provenance seals, and stage-input boundaries:

```bash
uv run --extra dev python -m pytest -q tests/unit/llm/test_prompt_program_registry_fixture.py tests/unit/llm/test_prompt_budget_fixture.py tests/unit/llm/test_prompt_program_structure.py tests/unit/llm/test_stage_input_privacy.py
```

For live adapter checks, use safe resolve/preflight/read-only scenarios. Do not trigger real WeCom sends as a deployment or smoke test.

## Audit Expectations

Every unsafe or abstained path should leave enough trace to answer:

- What did the user ask?
- Which capability or deterministic path was selected?
- Which evidence facts were accepted or rejected?
- Which source, artifact, channel, strategy, product, and time scope applied?
- Which validator produced the final allow/clarify/abstain/refuse decision?
- Which typed actions were proposed and which were actually executed by the adapter?
