# Domain Model

Last updated: 2026-06-17.

The agent must reason over typed business scope, not free-text keywords.

## Hierarchy

```text
渠道 DistributionChannel
  -> kind: bank | non_bank | unknown
  -> 策略 Strategy*
      -> 产品 Product*
      -> 材料包 Artifact(material_pack)*
  -> 周报 Artifact(weekly_report)*
  -> 月报 Artifact(monthly_report)*
```

- `渠道` comes from the adapter request/preflight context.
- Bank channels may have multiple `策略`; missing strategy for a bank
  `材料包` request requires clarification.
- Non-bank channels may have one strategy; a single strategy may be inferred but
  must still appear in `DomainContext`/canonical scope metadata.
- Future non-bank multiple-strategy channels are allowed; they should use the
  same ambiguity path as bank channels.

## Core Types

Implementation:

```text
src/market_support_crewai_agent/runtime/domain/ontology.py
```

Runtime types:

```text
DistributionChannel(id, name, kind)
Strategy(id, name, channel_id)
Product(id, name, channel_id, strategy_ids)
Artifact(id, artifact_type, scope, source_type, fact_types)
ArtifactScope(channel_id, strategy_id, product_ids, time_range)
DomainContext(channel, strategies, products, artifacts)
```

IDs are stable runtime IDs. User text is never enough to select a product,
strategy, or artifact by nearest match.

## Artifact Distinctions

`材料包` is not a report.

```text
material_pack
  source: adapter_material_pack_content or adapter_resolve for sendability
  answers: product list, open calendar, material-pack-specific content
  cannot be replaced by: 周报, 月报, document_context, history

weekly_report
  source: adapter_resolve and adapter_report_scope
  answers: 周报 period, 周报 scope, 周报 performance/product report questions
  cannot be replaced by: 材料包 or 月报

monthly_report
  source: adapter_resolve and adapter_report_scope
  answers: 月报 period, 月报 scope, 月报 performance/product report questions
  cannot be replaced by: 材料包 or 周报
```

Weekly/monthly reports cannot satisfy `material_pack` evidence unless the
selected capability manifest explicitly allows fallback. Built-in material-pack
capabilities do not allow fallback.

## Source Precedence

When sources conflict:

1. Request contract and adapter-provided identity.
2. Adapter resolve/preflight for sendability, artifact existence, and mention
   resolution.
3. Adapter-confirmed action ledger for executed sends.
4. Weekly/monthly report metadata from adapter resolve.
5. Permission-scoped internal MCP data.
6. Fetched markdown/report body when used as evidence.
7. Recent conversation turns.
8. LLM interpretation.

Planner output is never a business fact. Evidence and `BusinessFacts` establish
facts.

## Canonicalization

`CanonicalEntityResolver` resolves against explicit `DomainContext` entities,
exact names/aliases, structured defaults, or bounded closed-set selectors.

Expected behavior:

```text
材料包里有哪些产品 + no material_pack content -> abstain
策略S1材料包里有哪些产品 + S1/S2 packs -> use S1 only
产品A appears under two 策略 -> ambiguous/clarify
产品C unknown -> unresolved/clarify or abstain, not nearest 产品A/产品B
```

Exact equality on canonical structured fields is allowed. Substring, fuzzy,
regex, n-gram, or nearest-keyword product/strategy/artifact selection is banned.

## Evidence Scope

Every evidence fact carries source and scope metadata:

```text
fact_type
source_type
source_id
resolve_type
artifact_type
ArtifactScope(channel_id, strategy_id, product_ids, time_range)
SourceMetadata(provenance, channel_id, strategy_id, product_ids, time_range)
```

`retrieval_source_guard` rejects evidence when:

```text
source type is forbidden
artifact type is not allowed
channel scope mismatches current 渠道
strategy scope mismatches selected 策略
time range mismatches requested 周报/月报 period
history is used while allow_history=false
provenance is missing when required
```
