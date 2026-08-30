#!/usr/bin/env bash
set -uo pipefail

DEPLOY_DIR="/home/spark/GLM-5.3-Flash-EXL3-2x-DGX-Sparks"
WORKER="spark@10.0.0.2"
HEAD_IP="10.0.0.1"
WORKER_IP="10.0.0.2"
POLL_SECONDS=10
HEALTH_INTERVAL=30
MAX_HEALTH_FAILURES=3
LOCK_PID_FILE="$DEPLOY_DIR/logs/cluster.lock.pid"

log() { printf '[glm53-watchdog] %s\n' "$*"; }

start_in_progress() {
    local holder
    holder="$(tr -d '[:space:]' <"$LOCK_PID_FILE" 2>/dev/null || true)"
    [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null
}

nodes_ready() {
    ip -4 addr show dev enp1s0f1np1 2>/dev/null | grep -q "inet ${HEAD_IP}/" || return 1
    docker info >/dev/null 2>&1 || return 1
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER" \
        "ip -4 addr show dev enp1s0f1np1 | grep -q 'inet ${WORKER_IP}/' && docker info >/dev/null 2>&1" \
        >/dev/null 2>&1 || return 1
}

cluster_healthy() {
    [ "$(docker inspect -f '{{.State.Running}}' glm53-exl3-head 2>/dev/null || true)" = "true" ] || return 1
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER" \
        "test \"\$(docker inspect -f '{{.State.Running}}' glm53-exl3-worker 2>/dev/null || true)\" = true" \
        >/dev/null 2>&1 || return 1
    curl -fsS -m 5 http://127.0.0.1:8888/health >/dev/null 2>&1 || return 1
    curl -fsS -m 5 http://127.0.0.1:8888/v1/models 2>/dev/null \
        | grep -q 'GLM-5.3-Flash-EXL3' || return 1
}

wait_for_nodes() {
    local announced=0
    until nodes_ready; do
        if [ "$announced" -eq 0 ]; then
            log "waiting for local Docker, CX7 ${HEAD_IP}, and worker ${WORKER}"
            announced=1
        fi
        sleep "$POLL_SECONDS"
    done
    log "both Sparks and Docker daemons are ready"
}

start_cluster() {
    if start_in_progress; then
        local holder
        holder="$(tr -d '[:space:]' <"$LOCK_PID_FILE" 2>/dev/null || true)"
        log "start.sh already running (pid ${holder:-?}); not launching a second restart"
        return 1
    fi
    log "starting worker first, then head, from persistent weights and JIT caches"
    if SKIP_PULL=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 "$DEPLOY_DIR/start.sh" restart; then
        log "GLM cluster startup completed"
        return 0
    fi
    log "startup failed; will wait for prerequisites and retry"
    return 1
}

cleanup_on_signal() {
    log "watchdog stopping; leaving containers running (systemctl stop runs start.sh stop)"
    exit 0
}
trap cleanup_on_signal TERM INT

log "watchdog started"
while true; do
    wait_for_nodes

    if cluster_healthy; then
        log "GLM cluster already healthy"
    elif start_in_progress; then
        log "start.sh in progress; waiting for it to finish before health monitoring"
        while start_in_progress; do sleep "$POLL_SECONDS"; done
        continue
    else
        start_cluster || { sleep 30; continue; }
    fi

    failures=0
    while true; do
        sleep "$HEALTH_INTERVAL"
        if start_in_progress; then
            failures=0
            continue
        fi
        if cluster_healthy; then
            failures=0
            continue
        fi
        failures=$((failures + 1))
        log "health failure ${failures}/${MAX_HEALTH_FAILURES}"
        if [ "$failures" -ge "$MAX_HEALTH_FAILURES" ]; then
            log "cluster unhealthy; waiting for both Sparks before recovery"
            break
        fi
    done
done
