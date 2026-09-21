#!/usr/bin/env bash
# Run the rank launcher, then wait on the real container exit so systemd
# owns lifecycle. Docker --restart stays off (start.sh default).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${HERE}/env.sh"

RANK="${1:?usage: run-glm53-systemd.sh <0|1>}"
[[ "$RANK" == "0" || "$RANK" == "1" ]] || {
  echo "rank must be 0 or 1" >&2
  exit 2
}

if [[ "$RANK" == "0" ]]; then
  NAME="$GLM53_CONTAINER_HEAD"
else
  NAME="$GLM53_CONTAINER_WORKER"
fi

printf 'GLM53_SYSTEMD_LAUNCH rank=%s container=%s at=%s\n' \
  "$RANK" "$NAME" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
"${HERE}/launch-glm53.sh" "$RANK"

set +e
container_exit="$(/usr/bin/docker wait "$NAME")"
wait_rc=$?
set -e

if [[ "$wait_rc" -ne 0 ]]; then
  printf 'GLM53_DOCKER_WAIT_FAILED rank=%s wait_rc=%s at=%s\n' \
    "$RANK" "$wait_rc" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >&2
  exit "$wait_rc"
fi

if [[ ! "$container_exit" =~ ^[0-9]+$ ]] || (( container_exit > 255 )); then
  printf 'GLM53_CONTAINER_EXIT_INVALID rank=%s value=%q at=%s\n' \
    "$RANK" "$container_exit" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >&2
  exit 1
fi

printf 'GLM53_CONTAINER_EXITED rank=%s exit=%s at=%s\n' \
  "$RANK" "$container_exit" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >&2
exit "$container_exit"
