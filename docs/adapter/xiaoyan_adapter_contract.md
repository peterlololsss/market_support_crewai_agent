# Xiaoyan WeCom Current Adapter Contract

Last updated: 2026-06-17.

The `xiaoyan_wecom` backend provides adapter preflight/resolve for `market-support-crewai-agent`. Contract models live in `src/market_support_crewai_agent/schemas.py`. Cross-repo acceptance lives in `tests/live/test_xiaoyan_adapter_live_contract.py`.

This document describes the single current adapter contract. Older adapter payload shapes are removed and are not accepted by the agent runtime. The adapter must return current capabilities, current resolve results, current batch resolve results, and current action feedback.

## Endpoint surface

```text
GET  /health
GET  /adapter/capabilities
GET  /adapter/metrics
POST /adapter/resolve
POST /adapter/resolve/batch
POST /adapter/report-scope
POST /actions/feedback
```

## Capabilities

`GET /adapter/capabilities` returns service metadata for `xiaoyan-wecom-market-agent-adapter`, contract versions `adapter-resolve`, `adapter-resolve-batch`, and `adapter-action`, endpoint paths, supported resolve types, status values, request-size limits, batch limits, cache settings, and optional auth metadata.

## Resolve request

`POST /adapter/resolve` accepts one `AdapterResolveRequest`:

```json
{
  "resolve_type": "material_pack",
  "dist_name": "银河证券",
  "material_pack_option": "指增"
}
```

Supported resolve types:

```text
material_pack
weekly_report
monthly_report
sales_mention
```

`material_pack_option` is accepted only for `resolve_type=material_pack`. Weekly and monthly report resolve requests do not accept strategy, material-pack option, or report-scope selectors; they resolve the whole current report for the channel.

## Batch resolve

`POST /adapter/resolve/batch` accepts a JSON object with `requests` and preserves result order:

```json
{
  "requests": [
    {"resolve_type": "material_pack", "dist_name": "银河证券", "material_pack_option": "指增"},
    {"resolve_type": "weekly_report", "dist_name": "银河证券"},
    {"resolve_type": "monthly_report", "dist_name": "银河证券"},
    {"resolve_type": "sales_mention", "dist_name": "银河证券"}
  ]
}
```

Each result uses `AdapterResolveResult` with `contract_version=adapter-resolve`, typed status, display name, reason code, and adapter evidence needed by the runtime. When `status=resolved`, `resolve_ref` is required.

Resolve metadata may include:

```text
material_pack_option
period
report_date
period_start
period_end
period_label
scope_complete
expected_product_count
generated_product_count
missing_product_count
report_sections
```

Adapter public payloads are projections from adapter-owned records into typed DTOs. Public references such as `resolve_ref` and `material_id` are opaque adapter identifiers.

`ReplyRequest.material_pack_options` and `AdapterResolveResult.material_pack_options` are material-pack routing options
only. They are not a general strategy catalog. Empty `material_pack_options` means the channel has no extra
material-pack scope for the harness to choose before resolve; the adapter still owns final material-pack selection and
may return `resolved`, `ambiguous`, `missing`, `forbidden`, or `temporarily_unavailable` from resolve/preflight.

Raw send targets, URLs, filesystem paths, receiver identifiers, credentials, and internal execution records stay in adapter storage.

Agent-returned send actions carry the adapter-safe `resolve_ref` needed for execution. Material-pack actions may also carry `material_pack_option` when the current request explicitly selected one of `ReplyRequest.material_pack_options`. Weekly and monthly report actions carry only `resolve_type`, `period`, and `report_date` in addition to `resolve_ref`; they do not carry `report_scope`, `strategy`, or `material_pack_option`. The adapter must execute from `resolve_ref`; it must not re-select artifacts by guessing from free-form reply text.

`POST /adapter/report-scope` is a bounded read command for report-content evidence: which products, sections, and counts are present inside a weekly/monthly report. It is not a send-action selector. It accepts `material_type`, `dist_name`, `command`, optional `period`, and command-specific fields:

```text
summary        compact counts and report sections only
match          bounded exact/closed-set match result for one query
list_products  explicit paginated products only
```

Default `/adapter/resolve` and report-scope `summary` payloads must not include full product lists. Product lists are returned only by `list_products` or bounded match results. Report-scope facts come from adapter-owned manifest/helper outputs, not markdown keyword scanning.

Report period/date questions should be answerable from `/adapter/resolve` metadata alone. `period_start`, `period_end`, and `period_label` are optional public metadata fields for this purpose and must not require a report-scope product lookup.

Removed locator fields such as `card_ref`, URLs, filesystem paths, and raw MCP locators are not accepted in agent-facing payloads.

The adapter owns standard post-send follow-up wording for material packs, weekly reports, and monthly reports. The
harness returns semantic outbound action proposals and should not send duplicate "already sent / please check" text before
adapter execution.

## Runtime preflight requirement

The reply runtime verifies `/adapter/capabilities`, then collects preflight checks with `/adapter/resolve/batch` before LLM composition. Matching `status=resolved` is required before the runtime can return material/report/sales outbound action proposals.

Missing report-scope evidence remains `unknown`. The runtime can use positive report-content evidence when the adapter supplies it for knowledge answers. It must not use report-scope evidence to partially send a report.

## Live adapter eval

Start the real adapter server from `xiaoyan_wecom`:

```bash
cd /Users/ivan/PycharmProjects/xiaoyan_wecom
uv run --with requests --with python-dotenv python scripts/market_agent_adapter_server.py --host 127.0.0.1 --port 8011
```

Run live contract tests from this repo:

```bash
cd /Users/ivan/PycharmProjects/market_support_crewai_agent
MARKET_AGENT_LIVE_ADAPTER_BASE_URL=http://127.0.0.1:8011 uv run --extra dev python -m pytest -q tests/live/test_xiaoyan_adapter_live_contract.py
```

To test a real channel's current sendability:

```bash
MARKET_AGENT_LIVE_ADAPTER_DIST_NAME=银河证券
```

To test positive report-scope evidence without depending on production report data:

```bash
cd /Users/ivan/PycharmProjects/xiaoyan_wecom
python3 scripts/market_agent_adapter_scope_fixture.py
```

The fixture-backed live eval sets `MARKET_AGENT_LIVE_ADAPTER_EXPECT_REPORT_SCOPE=1`, then verifies that the real adapter returns a resolved `/adapter/report-scope` summary. To test material-pack routing, set `MARKET_AGENT_LIVE_MATERIAL_PACK_OPTION`; the live preflight eval verifies that only `material_pack` resolve receives that option.

## Action feedback

Adapter execution feedback is accepted at:

```text
POST /actions/feedback
```

The runtime action ledger stores adapter-confirmed executions. Repeated identical feedback payloads are idempotent. The runtime includes only recent `status=executed` adapter actions in prompts so “just sent” references are grounded by adapter-confirmed execution.

Accepted feedback action categories reflect adapter execution metadata and remain adapter-safe:

```text
send_material_pack
send_weekly_report
send_monthly_report
mention_sales
send_text
```

`material_id` is an opaque adapter reference, and `adapter_result` carries sanitized execution metadata.

## Metrics

The adapter client exposes typed `metrics()` for cache/uptime checks during live eval and performance testing. Metrics include sanitized transport counters by canonical route name, status code, and duration aggregate.
