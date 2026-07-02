#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-xiaoyan@192.168.209.195}"
REMOTE_ROOT="/data/xiaoyan/market_support_crewai_agent"
REMOTE_APP="$REMOTE_ROOT/app"
REMOTE_UPLOAD="$REMOTE_ROOT/app.upload"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

need_cmd ssh
need_cmd tar

ssh "$REMOTE" "test -f '$REMOTE_ROOT/.env' || { echo 'ERROR: missing env file: $REMOTE_ROOT/.env' >&2; exit 1; }"

echo "uploading current tree to $REMOTE:$REMOTE_APP"
tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.pytest_cache' \
  --exclude='tmp' \
  --exclude='.env' \
  --exclude='*/__pycache__' \
  -czf - . | ssh "$REMOTE" "set -euo pipefail; [ '$REMOTE_APP' = '/data/xiaoyan/market_support_crewai_agent/app' ]; mkdir -p '$REMOTE_ROOT/runtime'; rm -rf '$REMOTE_UPLOAD'; mkdir -p '$REMOTE_UPLOAD'; tar xzf - -C '$REMOTE_UPLOAD'; test -f '$REMOTE_UPLOAD/Containerfile'; rm -rf '$REMOTE_APP'; mv '$REMOTE_UPLOAD' '$REMOTE_APP'; bash '$REMOTE_APP/scripts/deploy_podman.sh'"
