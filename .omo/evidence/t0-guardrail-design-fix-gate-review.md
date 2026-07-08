# T0 Guardrail Design Fix Gate Review

## recommendation
APPROVE

## blockers
None.

## originalIntent
Verify that T0-related messages now route to human handoff through an injectable input-policy rule-table shape, not through a hardcoded T0 branch or one class per guardrail.

## desiredOutcome
- `T0`, `t0`, fullwidth `Ｔ ０`, and `T+0` match an input guardrail handoff plan with no outbound actions.
- Non-T0 messages fall through to planner via `no_match`.
- `InputPolicyRule` and `rules=` provide the injected rule tuple seam.
- Handoff plans respect `PolicyManifest` sales mention capability/resolve allowlists.
- Downstream rendering remains metadata-driven through generic guardrail metadata.
- Evidence artifacts include T0-scoped code review with `remove-ai-slops`/`programming` coverage and a criterion-level manual QA matrix.

## userOutcomeReview
Approved from the user's perspective. The shipped artifact implements the requested guardrail design shape and demonstrates the expected `/reply` outcome: Q85 returns HTTP 200, `reply.kind=human_handoff`, and no actions. The prior blockers are resolved by `.omo/evidence/t0-guardrail-design-fix-code-review.md` and `.omo/ulw-research/20260707-213156-t0-guardrails/manual-qa-matrix.md`.

## direct gate review
- Current diff adds one generic frozen `InputPolicyRule` dataclass/table and `rules=` injection on `match_input_policy`; no per-guardrail class was introduced.
- `match_input_policy` loops over rules and delegates to generic matching/result construction; no T0-specific branch in the matcher.
- `_handoff_result` includes `sales_mention` capability/resolve only when both `policy.allowed_capabilities` and `policy.allowed_adapter_resolves` allow it.
- Tests now cover T0 variants, non-T0 fallthrough, policy-denied sales mention edge, injected rule table, and metadata-driven renderer behavior.
- Direct slop pass found no unresolved overfit/slop: the new table/dataclass is the requested seam, tests assert behavior rather than only deletion/removal, and changed files are below 250 pure LOC (`input_policy.py` 115, `test_input_policy.py` 138).
- Direct programming pass found no `Any`, `object`, `cast`, `type: ignore`, broad exception, or oversized-file blocker in scoped files; dataclasses are frozen/slots and constants use `Final`.

## checked artifact paths
- `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py`
- `tests/unit/domain/test_input_policy.py`
- `src/market_support_crewai_agent/runtime/orchestration/workflow.py`
- `docs/support_reply_harness/README.md`
- `docs/support_reply_harness/next_session.md`
- `.omo/evidence/t0-guardrail-design-fix-code-review.md`
- `.omo/evidence/t0-guardrail-design-fix-gate-review.md`
- `.omo/ulw-research/20260707-213156-t0-guardrails/manual-qa-matrix.md`
- `.omo/ulw-research/20260707-213156-t0-guardrails/notepad.md`
- `tmp/t0_q85_handoff.md`
- `tmp/t0_q85_handoff_gate_rereview.md`

## verification rerun by gate reviewer
- `uv run --extra dev python -m pytest -q tests/unit/domain/test_input_policy.py` -> `5 passed in 0.21s`.
- `uv run --extra dev python -m pytest -q tests/integration/runtime/test_reply_contract.py tests/contract/test_adapter_preflight.py tests/unit/validation/test_structured_guardrails.py tests/unit/state/test_action_feedback.py tests/unit/domain/test_input_policy.py` -> `114 passed in 4.61s`.
- `uv run ruff check src/market_support_crewai_agent/runtime/domain/planning/input_policy.py tests/unit/domain/test_input_policy.py` -> `All checks passed!`.
- `uv run basedpyright src/market_support_crewai_agent/runtime/domain/planning/input_policy.py tests/unit/domain/test_input_policy.py` -> `0 errors, 0 warnings, 0 notes`.
- `git diff --check -- src/market_support_crewai_agent/runtime/domain/planning/input_policy.py tests/unit/domain/test_input_policy.py` -> no whitespace errors; Git emitted only LF-to-CRLF working-copy warnings.
- `PYTHONIOENCODING=utf-8 CREWAI_VERBOSE=false MARKET_AGENT_TRACE_LOG_EVENTS=false MARKET_AGENT_REPLY_ALIGNMENT_VERIFIER_ENABLED=false uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --ids 85 --parallel 1 --output-md tmp/t0_q85_handoff_gate_rereview.md` -> status `200`, kind `human_handoff`, actions `-`; runtime trace shows `input_policy.matched` and `evidence.resolve_types=["sales_mention"]`.

## exact evidence gaps
None remaining.
