#!/usr/bin/env bash
# Rank 0 runs ./start.sh from the kit checkout (detached docker run, then
# start.sh exits after /health). Rank 1 waits for the worker container the
# head creates over SSH — it must not run start.sh a second time.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${HERE}/env.sh"

RANK="${1:?usage: launch-glm53.sh <0|1>}"

case "$RANK" in
  0)
    cd "$GLM53_REPO"
    # TAIL=1 would steal the systemd process; docker wait owns the container.
    exec env -u TAIL ./start.sh
    ;;
  1)
    docker rm -f "$GLM53_CONTAINER_WORKER" >/dev/null 2>&1 || true
    deadline=$((SECONDS + GLM53_WORKER_WAIT_SEC))
    printf 'GLM53_WORKER state=waiting_for_head_launcher container=%s at=%s\n' \
      "$GLM53_CONTAINER_WORKER" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    until docker container inspect "$GLM53_CONTAINER_WORKER" >/dev/null 2>&1; do
      if (( SECONDS >= deadline )); then
        printf 'GLM53_WORKER_FAILED reason=container_timeout seconds=%s at=%s\n' \
          "$GLM53_WORKER_WAIT_SEC" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >&2
        exit 1
      fi
      sleep 2
    done
    if [[ "$(docker inspect -f '{{.State.Running}}' "$GLM53_CONTAINER_WORKER")" != "true" ]]; then
      docker logs --tail 80 "$GLM53_CONTAINER_WORKER" >&2 || true
      printf 'GLM53_WORKER_FAILED reason=container_exited at=%s\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >&2
      exit 1
    fi
    printf 'GLM53_WORKER state=container_running at=%s\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    ;;
  *)
    printf 'rank must be 0 or 1\n' >&2
    exit 2
    ;;
esac
