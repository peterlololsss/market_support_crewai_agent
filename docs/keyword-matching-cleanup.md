# Keyword Matching Cleanup

Last updated: 2026-06-17.

## Removed Matchers

| Area | Removed production behavior | Replacement |
|---|---|---|
| Strategy canonicalization | Built-in strategy alias/typo tables, numeric substring aliases, generic `指增` fallback, index-enhancement token detection, fuzzy similarity, semantic-description overlap, and unused local entity resolver. | Adapter/domain structured fields plus bounded planner capability selection; no local keyword resolver is production authority. |
| SendScope | `send_scope_guard.py` compatibility shim and historical text target extraction API. | `input_guard` using structured `RequestedScope`, `DomainContext`, policy actions, canonical channel IDs, and strategy IDs/names. |
| Report send defaulting | Message-token check for unnamed strategy report requests. | Planner-owned `report_scope`, `ambiguity_slots`, and canonical strategy status. |
| Report-scope refetch | Free-text verifier query normalization such as mapping “product list” text to `report_scope_products`. | `ReplyAlignmentVerdict` requires typed `refined_evidence_query` values: `report_scope_products` or `report_scope_summary`. |
| Reply sent/report claim guarding | Final validation scans over reply text for send/report/material tokens and report exclusion templates. | Action legality, source scope, and answerability are enforced through `ExecutionPlan`, `BusinessFacts`, direct guard functions, `ComposerReplyOutput.evidence_ids`, and plan-spec verification. |
| Material product claim scan | Regex scan for `Product...` strings in final answer text. | Composer must cite allowed evidence IDs; evidence contract and plan-spec checks decide whether answer output is valid. |
| Document MCP block ranking | Local semantic term/n-gram/strategy substring block scoring. | Closed-set document selection chooses document IDs before fetch; local bounding now truncates selected document text without semantic ranking. |
| Prompt router audit field | Empty `matched_keywords` field. | Removed. The router returns only non-authoritative audit hints derived from `PolicyManifest`. |

## Original Failure Mode

The bug this cleanup prevents: a user asks "材料包里有哪些产品？", no current
`material_pack` content exists, but a `周报` contains 产品A. Keyword or broad
knowledge matching must not list 产品A. The correct result is abstention or
clarification with a material-pack evidence reason.

## Remaining Legacy Heuristics

None. The cleanup intentionally removed compatibility adapters instead of wrapping old matchers.

## Allowlisted Exact Uses

| Location | Reason |
|---|---|
| `runtime/evidence/document_mcp.py` prompt-injection and locator/secret regexes | Safety sanitizers, not semantic routing or canonicalization. |
| `runtime/evidence/document_mcp.py` JSON-RPC `"error"` key check | Exact protocol validation. |
| `runtime/validation/reply_validator.py` raw locator and image-marker checks | Exact output safety validators. |
| `runtime/validation/reply_validator.py` pre-execution send-claim sanitizer | Strips unsafe composer wording before validation; it is not allow/block authority. |
| `runtime/llm/prompting/router.py` model-family routing | Deterministic routing from configured model ID, unrelated to business semantics. |
| `DocumentProductCandidate.keywords` metadata | Adapter/MCP-provided candidate metadata for closed-set LLM selection; production code does not keyword-score it. |

## Banned Semantic Selection

Do not use keyword, substring, regex, fuzzy, edit-distance, token, or n-gram
matching to select:

```text
渠道
策略
产品
材料包
周报
月报
report scope
document scope
```

Allowed replacements:

```text
canonical structured fields
adapter-provided IDs/metadata
DomainContext entities
CapabilityManifest / EvidenceContract
exact equality on canonical fields
bounded closed-set LLM selectors over explicit candidates
```

## CI Guard

Run:

```bash
uv run python scripts/check_no_semantic_keyword_matching.py
```

The core acceptance script also runs this guard before runtime checks:

```bash
uv run python scripts/check_reply_acceptance.py
```
