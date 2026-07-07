# ULW Notepad: Prompt Harness Design

## Bootstrap
Tier: HEAVY — user explicitly requested ultrawork and a designed prompt architecture for the harness, touching prompt/context boundaries and likely tests.
Skills:
- omo:ulw-loop — requested ultrawork evidence-bound execution.
- omo:ulw-research — user asked to search web examples; we need external best-practice synthesis.
- omo:programming — expected .py tests/code changes; Python reference loaded.
- ponytail active — avoid speculative abstraction, but user explicitly rejected minimal patch; use it to keep design lean.
- openai-docs/official web docs — use only if OpenAI product prompt/agent docs are cited; fallback browsing restricted to official OpenAI domains for OpenAI claims.

## Success Criteria
1. Prompt architecture separates durable taxonomy/rules from eval examples and manifest data; no Xiaoyan fixture question text is required in prompt tests.
2. Planner prompt gives capability cards as compact, schema-like contracts rather than example stuffing, while preserving harness boundaries: policy allowlist, evidence contracts, adapter authority, source hierarchy.
3. Regression coverage proves exact-question leakage is removed and category-level routing guidance remains visible.
4. Real-surface evidence: generated planner prompt dump/parsing shows no fixture leakage and includes the new prompt sections/cards.
5. External research synthesis cites official/primary sources and maps findings to this harness.

## Manual QA Scenarios
- CLI/data surface: `PYTHONIOENCODING=utf-8 python scripts/...` or inline command to assemble planner prompt for representative requests; PASS if prompt has no fixture exact hits and has contract sections.
- Test surface: `uv run --extra dev python -m pytest -q tests/unit/llm/test_prompt_router.py ...`; PASS if targeted tests pass.

## Evidence Ledger


## 2026-07-07 continuation

- User added requirement: document the job of each prompt section/role for future reference, including manifest responsibility.
- Planner subagent returned a file-level architecture plan; closed after receipt.
- External primary-source research fetched into `.omo/ulw-research/20260707-prompt-harness/`:
  - OpenAI prompt engineering docs and Practical Guide to Building Agents PDF.
  - Anthropic Building Effective Agents and Writing Effective Tools for Agents.
  - Google ADK Simple agents docs.
  - CrewAI Agents docs.
  - LangGraph URL tried but 404, excluded from evidence.
- RED proof command:
  `uv run --extra dev python -m pytest -q tests/unit/llm/test_prompt_no_eval_fixture_leakage.py tests/unit/llm/test_prompt_router.py tests/unit/context/test_context_projection.py`
  Result: 4 failed, 14 passed. Failures prove exact Xiaoyan question leakage, missing documented section roles, and `capability_cards` with `|pos=`/`|neg=` instead of `capability_contracts`.

## GREEN/QA evidence

- Prompt hygiene/snapshot slice:
  `uv run --extra dev python -m pytest -q tests/unit/llm/test_prompt_no_eval_fixture_leakage.py tests/unit/llm/test_prompt_router.py tests/unit/context/test_context_projection.py tests/unit/llm/test_prompt_snapshots.py`
  Result: 23 passed.
- Focused prompt/capability/contract suite:
  `uv run --extra dev python -m pytest -q tests/unit/llm/test_prompt_no_eval_fixture_leakage.py tests/unit/llm/test_prompt_router.py tests/unit/llm/test_prompt_assembler.py tests/unit/llm/test_prompt_snapshots.py tests/unit/context/test_context_projection.py tests/unit/domain/test_capability_registry.py tests/contract/test_plan_spec_contract.py`
  Result: 51 passed.
- Prompt registry:
  `uv run python scripts/check_prompt_registry.py` -> ok.
- Semantic keyword guard:
  `uv run python scripts/check_no_semantic_keyword_matching.py` -> failed on pre-existing, untouched `runtime/state/latency_report.py:124` and `:127` text membership/.find checks.
- Harness baseline:
  `uv run --extra dev python -m pytest -q tests/integration/runtime/test_reply_contract.py tests/contract/test_adapter_preflight.py tests/unit/validation/test_structured_guardrails.py tests/unit/state/test_action_feedback.py`
  Result: 109 passed.
- Real prompt assembly QA:
  `python - <<'PY' ... ReplyRequest -> compile_policy -> route_intent -> select_prompt_program ...`
  Result saved to `.omo/ulw-loop/prompt-harness-20260707/evidence/prompt_contract_check.json`: prompt_chars=47045, fixture_hit_count=0, capability_contracts=true, capability_cards=false, pos/neg pipe=false, projected examples=false, section roles=true.

## Reviewer

- Reviewer agent `Code Reviewer` returned PASS, recommendation APPROVE, no CRITICAL/HIGH/MEDIUM/LOW findings.
- Reviewer ran scoped tests: `uv run --extra dev python -m pytest -q tests/unit/llm/test_prompt_no_eval_fixture_leakage.py tests/unit/llm/test_prompt_router.py tests/unit/context/test_context_projection.py` -> 18 passed.
- Reviewer residual risk: full suite not rerun in review turn; known semantic-keyword guard failure in untouched `runtime/state/latency_report.py` remains unrelated.
