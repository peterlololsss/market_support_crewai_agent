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
outbound_message_target
```

`outbound_message_target` is available only to the DM outbound lifecycle. It
uses the adapter's strict shape instead of the distributor artifact shape:

```json
{
  "resolve_type": "outbound_message_target",
  "target_kind": "group",
  "target_name": "银河客户群"
}
```

The result contains only `status`, `reason_code`, `display_name`,
`target_kind`, `target_count`, `resolved_count`, and the opaque `resolve_ref`.

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

Adapter public payloads are projections from adapter-owned records into typed DTOs. Public references such as `resolve_ref`
and feedback `artifact.artifact_ref` are opaque adapter identifiers.

`ReplyRequest.available_artifacts` and `AdapterResolveResult.available_artifacts` are the sole adapter-provided
artifact-availability source. Each item uses `type=material_pack|weekly_report|monthly_report`; material-pack items may
carry `options` when the adapter exposes explicit material-pack routing choices. If explicit options are present and the
user did not select one, the harness asks a `material_pack_option` clarification before sending. Empty material-pack
`options` means the channel has a single/current material pack and the harness should not ask the user to pick a
strategy-like category before resolve. This is the common non-bank shape:

```json
{"available_artifacts": [{"type": "material_pack", "options": []}]}
```

The adapter still owns final material-pack selection and may return `resolved`, `ambiguous`, `missing`, `forbidden`, or
`temporarily_unavailable` from resolve/preflight.

Raw send targets, URLs, filesystem paths, receiver identifiers, credentials, and internal execution records stay in adapter storage.

Agent-returned send actions carry the adapter-safe `resolve_ref` needed for execution. Material-pack actions may also carry `material_pack_option` when the current request explicitly selected one of `available_artifacts[type=material_pack].options`. Weekly and monthly report actions carry only `resolve_type`, `period`, and `report_date` in addition to `resolve_ref`; they do not carry `report_scope`, `strategy`, or `material_pack_option`. The adapter must execute from `resolve_ref`; it must not re-select artifacts by guessing from free-form reply text.

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

## Direct-message outbound lifecycle

When `ReplyRequest.is_group=false`, the harness accepts every adapter-forwarded
DM sender and exposes only `query_internal_company_info` plus the adapter-owned
prepared outbound capability. It does not expose current-chat material/report
sends, sales mentions, or any sender allowlist of its own. The adapter retains
final authorization and execution authority.

The harness checks `GET /adapter/capabilities` for a ready
`outbound_messaging` capability whose action types are exactly the lifecycle
actions supported by xiaoyan_wecom. A complete initial request returns:

```json
{
  "type": "prepare_outbound_message",
  "action_id": "act-1",
  "target": {
    "kind": "group",
    "name": "银河客户群",
    "resolve_ref": "outbound-target:..."
  },
  "content": {"kind": "text", "text": "请查收本周更新"}
}
```

The adapter prepares immutable state and reports its sanitized
`confirmation_ref` through `/actions/feedback`. Only a later DM in the same
conversation may return:

```json
{
  "type": "execute_prepared_outbound_message",
  "action_id": "act-1",
  "confirmation_ref": "wecom-adapter-confirmation:..."
}
```

Execute never repeats or mutates target/content. Missing target/content is a
clarification with no action, and a confirmation ref not present in
adapter-confirmed prepare feedback fails closed.

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

Artifact execution metadata is nested under `artifact`; flat execution fields such as `resolve_ref`, `material_type`,
`material_pack_option`, `material_id`, and `version` are not accepted.

```json
{
  "action_type": "send_weekly_report",
  "status": "executed",
  "action_id": "act-1",
  "artifact": {
    "type": "weekly_report",
    "resolve_ref": "weekly:adapter-confirmed-resolve-ref",
    "artifact_ref": "weekly_report:opaque-artifact-ref",
    "period": "20260529",
    "report_date": "2026-05-29"
  },
  "adapter_result": {"ok": true}
}
```

For material packs, `artifact.type=material_pack` and the selected material-pack routing value, when present, lives at
`artifact.option`. `artifact.artifact_ref` is an opaque adapter reference, and `adapter_result` carries sanitized
execution metadata.

## Metrics

The adapter client exposes typed `metrics()` for cache/uptime checks during live eval and performance testing. Metrics include sanitized transport counters by canonical route name, status code, and duration aggregate.
