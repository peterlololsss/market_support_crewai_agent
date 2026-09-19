# WeCom Adapter Contract

Last updated: 2026-07-20.

The `assistant_wecom` backend provides adapter preflight/resolve for `market-support-crewai-agent`. Contract models live in `src/market_support_crewai_agent/schemas.py`. Cross-repo acceptance lives in `tests/live/test_assistant_adapter_live_contract.py`.

This document describes the single current adapter contract. The agent ingress boundary is a breaking
`reply-request.v2` contract: older `/reply` payload shapes are removed and are not accepted by the agent runtime. There
is no legacy wrapper, compatibility bridge, synthesized legacy request, or legacy/v2 route union. Legacy payloads fail
closed at the request boundary before stateful runtime work or model-visible calls. The adapter must return current
capabilities, current resolve results, current batch resolve results, and current action feedback.

The runtime still returns one public `ReplyResponse { reply, actions }`. The WeCom adapter keeps final execution
authority: it validates the reply/actions, owns outbox reliability, executes the primary reply and typed action proposals,
and records execution feedback.

## Endpoint surface

```text
GET  /health
POST /reply
GET  /adapter/capabilities
GET  /adapter/metrics
POST /adapter/resolve
POST /adapter/resolve/batch
POST /adapter/report-scope
POST /actions/feedback
```

## Agent reply request

`POST /reply` accepts only one current request contract:

```text
contract_version = reply-request.v2
```

The agent service requires its configured service key on `/reply` and `/actions/feedback`, and every v2 identity must
exactly match the agent deployment's configured canonical tenant. The request cannot select an adapter or switch the
deployment tenant. The wire values remain `direct` and `group`; `internal_dm` and `external_group` are prose aliases
only and must not appear as fields or scene values.

The runtime does not accept the old unversioned shape with `conversation_key`, `group_id`, `sender_id`, and `is_group`.
Do not send both legacy and v2 fields. Unknown contract versions and mixed legacy/v2 payloads are rejected at admission
instead of being normalized into a compatibility object.

Group request example:

```json
{
  "contract_version": "reply-request.v2",
  "request_id": "req:group-message-001",
  "message": "请发一下周报",
  "context_id": "ctx:origin-message-001",
  "identity": {
    "contract_version": "conversation-identity.v1",
    "surface": "wecom",
    "scene": "group",
    "tenant_ref": "tenant:primary",
    "group_ref": "group:opaque-001",
    "principal_ref": "principal:opaque-001"
  },
  "presentation": {
    "contract_version": "group-presentation.v1",
    "conversation_name": "Example group",
    "principal_name": "Example user"
  },
  "business_scope": {
    "kind": "distribution",
    "dist_channel_name": "示例券商",
    "channel_type": "non_bank",
    "available_artifacts": [
      {"type": "weekly_report", "options": []}
    ]
  },
  "grants": {
    "contract_version": "principal-grants.v1",
    "read_capabilities": ["resolve_weekly_report", "resolve_sales_mention"],
    "outbound_actions": ["send_weekly_report"],
    "mention_types": ["sales"]
  }
}
```

Direct reply-only request example:

```json
{
  "contract_version": "reply-request.v2",
  "request_id": "req:direct-message-001",
  "message": "介绍一下公司",
  "context_id": "ctx:origin-message-002",
  "identity": {
    "contract_version": "conversation-identity.v1",
    "surface": "wecom",
    "scene": "direct",
    "tenant_ref": "tenant:primary",
    "direct_thread_ref": "direct:opaque-001",
    "principal_ref": "principal:opaque-001"
  },
  "presentation": {
    "contract_version": "direct-presentation.v1",
    "principal_name": "Example user"
  },
  "business_scope": {"kind": "unscoped"},
  "grants": {
    "contract_version": "principal-grants.v1",
    "read_capabilities": ["query_internal_company_info"],
    "outbound_actions": [],
    "mention_types": []
  }
}
```

V2 grant lists are required. Empty lists mean deny-all; the runtime does not infer omitted legacy grants or fill missing
authority from artifact availability.

A direct request is unscoped and reply-only. Its read grants are either empty or exactly
`["query_internal_company_info"]`, and outbound-action and mention grants are empty. Direct recall, adapter business
resolves, actions, mentions, and media are disabled. Group requests retain distribution scope and per-principal state;
the adapter remains final authority for group preflight, action validation, execution, and feedback.

## Capabilities

`GET /adapter/capabilities` returns service metadata for `assistant-wecom-market-agent-adapter`, contract versions
`adapter-resolve`, `adapter-resolve-batch`, and `adapter-action`, endpoint paths, supported resolve types, status values,
request-size limits, batch limits, cache settings, and optional auth metadata.

Scene compatibility adds only these five optional additive fields:

```json
{
  "supported_scenes": ["direct", "group"],
  "reply_request_contract_versions": ["reply-request.v2"],
  "action_feedback_contract_versions": ["action-feedback.v2"],
  "conversation_identity_contract_versions": ["conversation-identity.v1"],
  "deployment_tenant_ref": "tenant:primary"
}
```

The lists are bounded, duplicate-free canonical lists. Additional well-formed future version strings are allowed;
unknown scene values, blank/control-character versions, malformed tenants, duplicates, and unknown object fields are
rejected by the strict schema.

Current group resolve/preflight remains compatible with a field-absent group capability payload: when all five fields
are absent, the existing service/version/endpoint/resolve/status checks still apply. If any additive field is present,
it must still parse strictly, but group traffic does not require these fields merely to answer or preflight.

Direct admission is stricter. Before state reservation, the agent requires an adapter API key and an authenticated
capability response containing `direct`, `reply-request.v2`, `action-feedback.v2`, `conversation-identity.v1`, and a
`deployment_tenant_ref` exactly equal to the agent deployment tenant. The direct check reuses the client's existing
successful-capabilities cache and does not call a resolve or send endpoint.

These fields are an external adapter release prerequisite. A fake/local contract pass does not prove that the deployed
adapter advertises them. Production must keep `MARKET_AGENT_INTERNAL_DM_ENABLED=false` until the real adapter read-only
capability test passes.

## Resolve request

`POST /adapter/resolve` accepts one `AdapterResolveRequest`:

```json
{
  "resolve_type": "material_pack",
  "dist_name": "示例券商",
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
    {"resolve_type": "material_pack", "dist_name": "示例券商", "material_pack_option": "指增"},
    {"resolve_type": "weekly_report", "dist_name": "示例券商"},
    {"resolve_type": "monthly_report", "dist_name": "示例券商"},
    {"resolve_type": "sales_mention", "dist_name": "示例券商"}
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

`ReplyRequestV2.business_scope.available_artifacts` and `AdapterResolveResult.available_artifacts` are the sole
adapter-provided artifact-availability source. Each item uses `type=material_pack|weekly_report|monthly_report`;
material-pack items may carry `options` when the adapter exposes explicit material-pack routing choices. If explicit
options are present and the user did not select one, the harness asks a `material_pack_option` clarification before
sending. Empty material-pack `options` means the channel has a single/current material pack and the harness should not
ask the user to pick a strategy-like category before resolve. This is the common non-bank shape inside
`business_scope`:

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

## Authenticated transport boundary

The agent sends adapter bearer credentials only to the configured origin. Adapter calls never follow `3xx` responses,
including same-origin redirects, and the response URL is checked against the configured origin before its body is
accepted. There is no retry or redirect fallback.

Adapter base URLs may use `https://`. Cleartext `http://` is restricted to literal loopback, RFC1918, or link-local
addresses so the documented `http://10.0.0.11:8011` deployment remains supported. That exception provides no
transport encryption: operators must keep it on an authenticated private network or an encrypted tunnel. Public or
hostname-based HTTP endpoints, URL userinfo, query/fragment components, malformed authorities, and other schemes fail
closed before network I/O.

## Runtime preflight requirement

The reply runtime verifies `/adapter/capabilities`, then collects preflight checks with `/adapter/resolve/batch` before LLM composition. Matching `status=resolved` is required before the runtime can return material/report/sales outbound action proposals.

Missing report-scope evidence remains `unknown`. The runtime can use positive report-content evidence when the adapter supplies it for knowledge answers. It must not use report-scope evidence to partially send a report.

## Live adapter eval

Start the real adapter server from `assistant_wecom`:

```bash
cd ~/projects/assistant_wecom
uv run --with requests --with python-dotenv python scripts/market_agent_adapter_server.py --host 127.0.0.1 --port 8011
```

Run live contract tests from this repo:

```bash
cd ~/projects/market_support_crewai_agent
MARKET_AGENT_LIVE_ADAPTER_BASE_URL=http://127.0.0.1:8011 uv run --extra dev python -m pytest -q tests/live/test_assistant_adapter_live_contract.py
```

The release prerequisite is checked without sending anything:

```bash
MARKET_AGENT_LIVE_ADAPTER_BASE_URL=http://127.0.0.1:8011 uv run --extra dev python -m pytest -q tests/live/test_assistant_adapter_live_contract.py -m live -k capabilities
```

To test a real channel's current sendability:

```bash
MARKET_AGENT_LIVE_ADAPTER_DIST_NAME=示例券商
```

To test positive report-scope evidence without depending on production report data:

```bash
cd ~/projects/assistant_wecom
python3 scripts/market_agent_adapter_scope_fixture.py
```

The fixture-backed live eval sets `MARKET_AGENT_LIVE_ADAPTER_EXPECT_REPORT_SCOPE=1`, then verifies that the real adapter returns a resolved `/adapter/report-scope` summary. To test material-pack routing, set `MARKET_AGENT_LIVE_MATERIAL_PACK_OPTION`; the live preflight eval verifies that only `material_pack` resolve receives that option.

## Action feedback

Adapter execution feedback is accepted at:

```text
POST /actions/feedback
```

### Versioned feedback contract

`action-feedback.v2` is the typed feedback contract for an issued `ReplyResponse`. Its identity is the same verified
`conversation-identity.v1` used by the corresponding `reply-request.v2`; the runtime derives the full
`ConversationStateKey` from that verified identity. Neither a raw conversation key nor a serialized state-key reference is
accepted as feedback authority.

```json
{
  "contract_version": "action-feedback.v2",
  "feedback_id": "fb:execution-001",
  "request_id": "req:group-message-001",
  "response_id": "resp-0123456789abcdef0123456789abcdef",
  "identity": {
    "contract_version": "conversation-identity.v1",
    "surface": "wecom",
    "scene": "group",
    "tenant_ref": "tenant:primary",
    "group_ref": "group:opaque-001",
    "principal_ref": "principal:opaque-001"
  },
  "executions": [
    {
      "action_type": "send_weekly_report",
      "status": "executed",
      "action_id": "act-0123456789abcdef0123456789abcdef",
      "artifact": {
        "type": "weekly_report",
        "resolve_ref": "weekly:opaque-resolve",
        "artifact_ref": "weekly:opaque-result",
        "period": "20260529",
        "report_date": "2026-05-29"
      },
      "adapter_result": {"ok": true}
    }
  ]
}
```

The v2 body is strict: `feedback_id`, `request_id`, `response_id`, identity, and `executions` are required; it allows
zero to five executions and at most 64 KiB. Send executions are unique by `action_id`; response-level effects are
unique by `action_type`; duplicates are rejected before any issued-response lookup. `response_id` is
`resp-` followed by 32 lowercase hexadecimal characters, and `action_id` is `act-` followed by 32 lowercase
hexadecimal characters.

Every send execution must bind exactly to the issued effect: action type, action ID, artifact type, opaque
`resolve_ref`, material-pack option, and report period/date must match. `artifact_ref` is required only for an
`executed` send and is forbidden for `failed` or `skipped`. `send_text` and `mention_sales` are response-level effects:
they forbid both an action ID and artifact. `adapter_result` is sanitized metadata only; credentials, signatures,
tokens, cookies, URLs, paths, raw identity/recipient fields, and resolve references inside that metadata are rejected.

`action-feedback.v2` is active. A submission must resolve to the issued response for that verified identity before the coordinator
prepares and commits the feedback. Repeated `(ConversationStateKey, feedback_id)` submissions with the same body are
receipt-replayed rather than applied twice; a conflicting reuse of `feedback_id` is rejected. The issued response, its
effects, and its feedback receipts remain correlated by the opaque server-generated response/action IDs. Adapters must
retain the IDs returned with the issued response and must not manufacture them.

Late v2 feedback follows service-key authentication, body/schema validation, deployment-tenant equality, and immutable
issued-response/effect correlation. It does not recheck the DM flag or adapter compatibility. This permits a matching
effect issued while internal DM was enabled to be recorded after an operator turns the flag off, without admitting a
new direct reply.

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

For material packs, `artifact.type=material_pack` and the selected material-pack routing value, when present, lives at
`artifact.option`. `artifact.artifact_ref` is an opaque adapter reference, and `adapter_result` carries sanitized
execution metadata.

## Metrics

The adapter client exposes typed `metrics()` for cache/uptime checks during live eval and performance testing. Metrics include sanitized transport counters by canonical route name, status code, and duration aggregate.
