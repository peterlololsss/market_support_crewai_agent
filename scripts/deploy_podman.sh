#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-market-support-crewai-agent:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-market-support-crewai-agent}"
HOST_IP="${HOST_IP:-10.0.0.12}"
HOST_PORT="23003"
APP_PORT="8000"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="${APP_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_ROOT="${SERVICE_ROOT:-$(dirname -- "$APP_ROOT")}"
ENV_FILE="${ENV_FILE:-$SERVICE_ROOT/.env}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$SERVICE_ROOT/runtime}"
PODMAN_ENV_FILE=""

DEFAULT_ADAPTER_BASE_URL="http://10.0.0.11:8011"
DEFAULT_DOC_MCP_BASE_URL="http://10.0.0.12:23000"
DEFAULT_PLANNER_LLM_BASE_URL="http://10.0.0.12:3000/gemini"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

http_code() {
  curl -sS --connect-timeout 5 --max-time 15 -o /dev/null -w "%{http_code}" "$@"
}

require_reachable() {
  local name="$1"
  local url="$2"
  local code
  code="$(http_code "$url" || true)"
  [[ "$code" != "000" ]] || die "$name is not reachable: $url"
  echo "ok: $name reachable ($url, HTTP $code)"
}

require_2xx() {
  local name="$1"
  local url="$2"
  shift 2
  local code
  code="$(http_code "$@" "$url" || true)"
  [[ "$code" == 2* ]] || die "$name is not healthy: $url returned HTTP $code"
  echo "ok: $name healthy ($url, HTTP $code)"
}

env_value() {
  local key="$1"
  local fallback="$2"
  local value="${!key-}"
  if [[ -z "$value" ]]; then
    value="$(awk -F= -v key="$key" 'index($0, key "=") == 1 {sub(/^[^=]*=/, ""); print; exit}' "$PODMAN_ENV_FILE")"
  fi
  printf '%s' "${value:-$fallback}"
}

need_cmd curl
need_cmd podman

[[ -f "$APP_ROOT/Containerfile" ]] || die "Containerfile not found under APP_ROOT=$APP_ROOT"
[[ -f "$ENV_FILE" ]] || die "env file not found: $ENV_FILE"

mkdir -p "$RUNTIME_ROOT"
rm -f "$RUNTIME_ROOT/container.env"
PODMAN_ENV_FILE="$(mktemp "$RUNTIME_ROOT/container.env.XXXXXX")"
trap 'rm -f "$PODMAN_ENV_FILE"' EXIT
tr -d '\r' < "$ENV_FILE" > "$PODMAN_ENV_FILE"
chmod 600 "$PODMAN_ENV_FILE"

ADAPTER_BASE_URL="$(env_value MARKET_AGENT_ADAPTER_BASE_URL "$DEFAULT_ADAPTER_BASE_URL")"
DOC_MCP_BASE_URL="$(env_value MARKET_AGENT_DOC_MCP_BASE_URL "$DEFAULT_DOC_MCP_BASE_URL")"
PLANNER_LLM_BASE_URL="$(env_value MARKET_AGENT_PLANNER_LLM_BASE_URL "$DEFAULT_PLANNER_LLM_BASE_URL")"
ADAPTER_API_KEY="$(env_value MARKET_AGENT_ADAPTER_API_KEY "")"
APP_API_KEY="$(env_value MARKET_AGENT_API_KEY "")"
DEPLOYMENT_TENANT_REF="$(env_value MARKET_AGENT_DEPLOYMENT_TENANT_REF "")"
INTERNAL_DM_ENABLED="$(env_value MARKET_AGENT_INTERNAL_DM_ENABLED "false")"

[[ "$APP_API_KEY" =~ [^[:space:]] ]] || die "MARKET_AGENT_API_KEY must be nonblank"
[[ "$DEPLOYMENT_TENANT_REF" =~ ^tenant:[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$ ]] || \
  die "MARKET_AGENT_DEPLOYMENT_TENANT_REF must be a canonical tenant ref"

adapter_headers=()
if [[ -n "$ADAPTER_API_KEY" ]]; then
  adapter_headers=(-H "X-API-Key: ${ADAPTER_API_KEY}")
fi

require_2xx "adapter" "${ADAPTER_BASE_URL%/}/health" "${adapter_headers[@]}"
require_reachable "Document MCP" "${DOC_MCP_BASE_URL%/}/mcp"
require_reachable "planner LLM proxy" "$PLANNER_LLM_BASE_URL"

echo "building image: $IMAGE_NAME"
podman build -t "$IMAGE_NAME" -f "$APP_ROOT/Containerfile" "$APP_ROOT"

if podman container exists "$CONTAINER_NAME"; then
  echo "replacing container: $CONTAINER_NAME"
  podman rm -f "$CONTAINER_NAME" >/dev/null
fi

if (echo >/dev/tcp/127.0.0.1/"$HOST_PORT") >/dev/null 2>&1; then
  die "port $HOST_PORT is already in use; fixed deployment port is 23003"
fi

echo "starting container on fixed port $HOST_PORT"
podman run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --env-file "$PODMAN_ENV_FILE" \
  -e MARKET_AGENT_ADAPTER_BASE_URL="$ADAPTER_BASE_URL" \
  -e MARKET_AGENT_DOC_MCP_BASE_URL="$DOC_MCP_BASE_URL" \
  -e MARKET_AGENT_PLANNER_LLM_BASE_URL="$PLANNER_LLM_BASE_URL" \
  -p "$HOST_PORT:$APP_PORT" \
  "$IMAGE_NAME" >/dev/null

app_headers=(-H "X-API-Key: ${APP_API_KEY}")

health_url="http://127.0.0.1:$HOST_PORT/health"
for _ in $(seq 1 30); do
  if [[ "$(http_code "$health_url" || true)" == 2* ]]; then
    break
  fi
  sleep 1
done
require_2xx "agent health" "$health_url"

reply_url="http://127.0.0.1:$HOST_PORT/reply"

group_reply_payload="$(cat <<JSON
{"contract_version":"reply-request.v2","request_id":"req:deploy-group-smoke","message":"服务连通性检查，请简短回答。","context_id":"ctx:deploy-group-smoke","identity":{"contract_version":"conversation-identity.v1","surface":"wecom","scene":"group","tenant_ref":"$DEPLOYMENT_TENANT_REF","group_ref":"group:deploy-smoke","principal_ref":"principal:deploy-smoke"},"presentation":{"contract_version":"group-presentation.v1","conversation_name":"deployment smoke","principal_name":"deployment smoke"},"business_scope":{"kind":"distribution","dist_channel_name":"deployment smoke","channel_type":"unknown","available_artifacts":[]},"grants":{"contract_version":"principal-grants.v1","read_capabilities":[],"outbound_actions":[],"mention_types":[]}}
JSON
)"
if ! curl -fsS --max-time 180 "${app_headers[@]}" -H "Content-Type: application/json" "$reply_url" --data "$group_reply_payload" >/dev/null; then
  podman logs --tail 120 "$CONTAINER_NAME" >&2 || true
  die "safe group /reply smoke failed"
fi

case "${INTERNAL_DM_ENABLED,,}" in
  true|1|yes|on)
    direct_reply_payload="$(cat <<JSON
{"contract_version":"reply-request.v2","request_id":"req:deploy-direct-smoke","message":"请介绍内部支持服务。","context_id":"ctx:deploy-direct-smoke","identity":{"contract_version":"conversation-identity.v1","surface":"wecom","scene":"direct","tenant_ref":"$DEPLOYMENT_TENANT_REF","direct_thread_ref":"direct:deploy-smoke","principal_ref":"principal:deploy-smoke"},"presentation":{"contract_version":"direct-presentation.v1","principal_name":"deployment smoke"},"business_scope":{"kind":"unscoped"},"grants":{"contract_version":"principal-grants.v1","read_capabilities":["query_internal_company_info"],"outbound_actions":[],"mention_types":[]}}
JSON
)"
    if ! curl -fsS --max-time 180 "${app_headers[@]}" -H "Content-Type: application/json" "$reply_url" --data "$direct_reply_payload" >/dev/null; then
      podman logs --tail 120 "$CONTAINER_NAME" >&2 || true
      die "safe direct /reply smoke failed"
    fi
    ;;
  false|0|no|off|"") ;;
  *) die "MARKET_AGENT_INTERNAL_DM_ENABLED has an invalid boolean value" ;;
esac

echo "deployed: http://$HOST_IP:$HOST_PORT"
echo "health:   http://$HOST_IP:$HOST_PORT/health"
