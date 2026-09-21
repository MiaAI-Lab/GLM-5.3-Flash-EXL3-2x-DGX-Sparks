#!/usr/bin/env bash
# Head ExecStartPre: wait for SSH, then restart the worker watcher so it is
# listening before start.sh docker-runs rank 1. Strict host-key checking only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${HERE}/env.sh"

SSH=(
  /usr/bin/ssh
  -o BatchMode=yes
  -o StrictHostKeyChecking=yes
  -o UserKnownHostsFile="$GLM53_KNOWN_HOSTS"
  -o ConnectTimeout=5
  -o ConnectionAttempts=1
  "$GLM53_WORKER_SSH"
)
DEADLINE=$((SECONDS + GLM53_COORD_TIMEOUT_SEC))

printf 'GLM53_COORDINATOR state=waiting_for_worker_ssh worker=%s at=%s\n' \
  "$GLM53_WORKER_SSH" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
until "${SSH[@]}" /usr/bin/true >/dev/null 2>&1; do
  if (( SECONDS >= DEADLINE )); then
    printf 'GLM53_COORDINATOR_FAILED state=waiting_for_worker_ssh worker=%s at=%s\n' \
      "$GLM53_WORKER_SSH" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >&2
    exit 1
  fi
  sleep 5
done

printf 'GLM53_COORDINATOR state=restarting_worker_watcher worker=%s at=%s\n' \
  "$GLM53_WORKER_SSH" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
"${SSH[@]}" /usr/bin/sudo -n /usr/bin/systemctl restart "$GLM53_WORKER_UNIT"

until "${SSH[@]}" /usr/bin/systemctl is-active --quiet "$GLM53_WORKER_UNIT"; do
  if (( SECONDS >= DEADLINE )); then
    printf 'GLM53_COORDINATOR_FAILED state=waiting_for_worker_service worker=%s at=%s\n' \
      "$GLM53_WORKER_SSH" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >&2
    exit 1
  fi
  sleep 2
done

printf 'GLM53_COORDINATOR state=worker_watcher_ready at=%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
