# Xiaoyan WeCom Adapter Contract

Last updated: 2026-06-03.

The `xiaoyan_wecom` backend provides adapter preflight/resolve for `market-support-crewai-agent`. Contract models live in `src/market_support_crewai_agent/schemas.py`. Cross-repo acceptance lives in `tests/test_xiaoyan_adapter_live_contract.py`.

## Endpoint surface

```text
GET  /health
GET  /adapter/capabilities
GET  /adapter/metrics
POST /adapter/resolve
POST /adapter/resolve/batch
POST /actions/feedback
```

## Capabilities

`GET /adapter/capabilities` returns service metadata for `xiaoyan-wecom-market-agent-adapter`, contract versions `adapter-resolve.v1` and `adapter-resolve-batch.v1`, endpoint paths, supported resolve types, status values, request-size limits, batch limits, cache settings, and optional auth metadata.

## Resolve request

`POST /adapter/resolve` accepts one `AdapterResolveRequest`:

```json
{
  "resolve_type": "material_pack",
  "dist_name": "银河证券",
  "strategy": "指增"
}
```

Supported resolve types:

```text
material_pack
weekly_report
monthly_report
sales_mention
```

## Batch resolve

`POST /adapter/resolve/batch` accepts a JSON object with `requests` and preserves result order:

```json
{
  "requests": [
    {"resolve_type": "material_pack", "dist_name": "银河证券", "strategy": "指增"},
    {"resolve_type": "weekly_report", "dist_name": "银河证券", "strategy": "指增"},
    {"resolve_type": "monthly_report", "dist_name": "银河证券", "strategy": "指增"},
    {"resolve_type": "sales_mention", "dist_name": "银河证券"}
  ]
}
```

Each result uses `AdapterResolveResult` with `contract_version=adapter-resolve.v1`, typed status, display name, reason code, and adapter evidence needed by the runtime.

Report scope evidence uses:

```text
contains_strategy
generated_strategies
scope_status
strategy
period
report_date
```

Adapter public payloads are projections from adapter-owned records into typed DTOs. Public references such as `card_ref` and `material_id` are opaque adapter identifiers.

Raw send targets, URLs, filesystem paths, receiver identifiers, credentials, and internal execution records stay in adapter storage.

Current action compatibility: the existing WeCom action consumer accepts only `type`, `action_id`, and `strategy` on
agent-returned action objects. The harness therefore keeps report selector and `card_ref` details in validation/audit
state for now, and does not emit public `selector` or `card_ref` fields until the adapter action contract is explicitly
bumped.

The adapter owns standard post-send follow-up wording for material packs, weekly reports, and monthly reports. The
harness returns semantic side-effect proposals and should not send duplicate "already sent / please check" text before
adapter execution.

## Runtime preflight requirement

The reply runtime verifies `/adapter/capabilities`, then collects preflight checks with `/adapter/resolve/batch` before LLM composition. Matching `status=resolved` is required before the runtime can return material/report/sales side-effect action proposals.

Missing report scope evidence remains `unknown`. The runtime can use positive scope evidence when the adapter supplies it.

## Live adapter smoke

Start the real adapter server from `xiaoyan_wecom`:

```bash
cd /Users/ivan/PycharmProjects/xiaoyan_wecom
uv run --with requests --with python-dotenv python scripts/market_agent_adapter_server.py --host 127.0.0.1 --port 8011
```

Run live contract tests from this repo:

```bash
cd /Users/ivan/PycharmProjects/market_support_crewai_agent
MARKET_AGENT_LIVE_ADAPTER_BASE_URL=http://127.0.0.1:8011 uv run pytest tests/test_xiaoyan_adapter_live_contract.py
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

The fixture-backed live smoke sets `MARKET_AGENT_LIVE_ADAPTER_STRATEGY` and `MARKET_AGENT_LIVE_ADAPTER_EXPECT_SCOPE=1`, then verifies that the real adapter returns `contains_strategy=true` and `scope_status=included`.

## Action feedback

Adapter execution feedback is accepted at:

```text
POST /actions/feedback
```

The runtime action ledger stores adapter-confirmed executions. Repeated identical feedback payloads are idempotent. The runtime includes only recent `status=executed` adapter actions in prompts so “just sent” references are grounded by adapter-confirmed execution.

Accepted feedback action categories reflect adapter execution metadata and remain adapter-safe. `material_id` is an opaque adapter reference, and `adapter_result` carries sanitized execution metadata.

## Metrics

The adapter client exposes typed `metrics()` for cache/uptime checks during live smoke and performance testing. Metrics include sanitized transport counters by canonical route name, status code, and duration aggregate.
