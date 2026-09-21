#!/usr/bin/env bash
# Head ExecStartPost: wait for HTTP /health then /v1/models. ExecStartPost
# starts as soon as the launcher process starts, before Docker necessarily
# creates the container. systemd kills this probe if ExecStart exits;
# readiness only owns the health deadline. Match HTTP, not log greps.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${HERE}/env.sh"

BASE_URL="http://127.0.0.1:${GLM53_PORT}"
DEADLINE=$((SECONDS + GLM53_HEALTH_TIMEOUT_SEC))
MODEL="$GLM53_SERVED_MODEL_NAME"

printf 'GLM53_READINESS state=waiting_for_health url=%s at=%s\n' \
  "$BASE_URL/health" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
until /usr/bin/curl -fsS --max-time 3 "$BASE_URL/health" >/dev/null; do
  if (( SECONDS >= DEADLINE )); then
    printf 'GLM53_READINESS_FAILED reason=health_timeout seconds=%s at=%s\n' \
      "$GLM53_HEALTH_TIMEOUT_SEC" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >&2
    exit 1
  fi
  sleep 10
done

/usr/bin/curl -fsS --max-time 5 "$BASE_URL/v1/models" \
  | /usr/bin/python3 -c 'import json,sys
wanted = sys.argv[1]
data = json.load(sys.stdin)["data"]
assert any(model["id"] == wanted for model in data), wanted' "$MODEL"
printf 'GLM53_READINESS state=ready model=%s at=%s\n' \
  "$MODEL" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
