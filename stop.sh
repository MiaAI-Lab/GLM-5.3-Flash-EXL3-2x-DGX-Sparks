#!/usr/bin/env bash
# stop.sh — stop the GLM-5.3-Flash EXL3 vLLM server(s)
#
# Stops whichever stack is running (both, if both exist). Names and SSH
# come from the launchers: TP=2 via start.sh (.env), TP=3 via start-tp3.sh
# (.env then .env.tp3), so rank-2 fabric/user pins are not duplicated here.
#   TP=2  glm53-exl3-head + glm53-exl3-worker
#   TP=3  glm53-exl3-tp3-head + glm53-exl3-tp3-w1 + glm53-exl3-tp3-w2
# Weights and compile caches stay on disk so a later start restarts fast.
#
# Usage:
#   ./stop.sh           stop the running stack(s). Local heads are detected;
#                       if neither head is here, both launchers still run so
#                       orphaned workers (including TP=3 rank 2) are removed.
#   ./stop.sh tp2       TP=2 only  (./start.sh stop)
#   ./stop.sh tp3       TP=3 only  (./start-tp3.sh stop)
#   ./stop.sh all       both, skip detection
#
# Equivalent to ./start.sh stop and/or ./start-tp3.sh stop.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" \
        | sed -e '/^set -euo pipefail/d' -e 's/^# \{0,1\}//'
}

have_container() {
    docker inspect "$1" >/dev/null 2>&1
}

# Names from the same files the launchers source. TP=2 never reads .env.tp3.
# Subshell so .env does not leak into start.sh as "caller overrides".
tp2_head_name() {
    (
        CONTAINER_HEAD="glm53-exl3-head"
        if [ -f "$SCRIPT_DIR/.env" ]; then
            set -a
            # shellcheck disable=SC1091
            source "$SCRIPT_DIR/.env"
            set +a
        fi
        printf '%s' "${CONTAINER_HEAD:-glm53-exl3-head}"
    )
}

tp3_head_name() {
    (
        CONTAINER_HEAD="glm53-exl3-tp3-head"
        if [ -f "$SCRIPT_DIR/.env" ]; then
            set -a
            # shellcheck disable=SC1091
            source "$SCRIPT_DIR/.env"
            set +a
        fi
        if [ -f "$SCRIPT_DIR/.env.tp3" ]; then
            set -a
            # shellcheck disable=SC1091
            source "$SCRIPT_DIR/.env.tp3"
            set +a
        fi
        printf '%s' "${CONTAINER_HEAD:-glm53-exl3-tp3-head}"
    )
}

stop_tp2() {
    "$SCRIPT_DIR/start.sh" stop
}

stop_tp3() {
    "$SCRIPT_DIR/start-tp3.sh" stop
}

want_tp2=0
want_tp3=0
cmd="${1:-}"
case "$cmd" in
    "" )
        if have_container "$(tp2_head_name)"; then want_tp2=1; fi
        if have_container "$(tp3_head_name)"; then want_tp3=1; fi
        if [ "$want_tp2" = 0 ] && [ "$want_tp3" = 0 ]; then
            # Neither head is local — still tear down workers so a removed
            # TP=3 head cannot leave w1/w2 up. Skip TP=3 if it was never
            # configured (avoids start-tp3.sh copying .env.tp3.example).
            want_tp2=1
            if [ -f "$SCRIPT_DIR/start-tp3.sh" ] && [ -f "$SCRIPT_DIR/.env.tp3" ]; then
                want_tp3=1
            fi
        fi
        ;;
    tp2|2) want_tp2=1 ;;
    tp3|3)
        want_tp3=1
        ;;
    all)
        want_tp2=1
        want_tp3=1
        ;;
    -h|--help|help) usage; exit 0 ;;
    *)
        echo "unknown argument: $cmd (try ./stop.sh --help)" >&2
        exit 1
        ;;
esac

rc=0
if [ "$want_tp2" = 1 ]; then
    stop_tp2 || rc=$?
fi
if [ "$want_tp3" = 1 ]; then
    if [ ! -f "$SCRIPT_DIR/start-tp3.sh" ]; then
        echo "ERROR: start-tp3.sh not found" >&2
        rc=1
    else
        stop_tp3 || rc=$?
    fi
fi
exit "$rc"
