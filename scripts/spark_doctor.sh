#!/usr/bin/env bash
# ============================================================================
# spark_doctor.sh — Hardware, RoCEv2 & Preflight Diagnostics for 2x DGX Spark
# ============================================================================
#
# Read-only preflight for the Head and Worker nodes:
# 1. Inter-node connectivity: ICMP ping and passwordless SSH to the worker
# 2. RoCEv2 / CX7 interfaces: presence and link state (Head + Worker)
# 3. GPU presence and the GPU_MEM_UTIL sanity band
# 4. Docker daemon access and local GHCR image presence
# 5. Serving endpoint reachability on PORT (/health, /v1/models), if running
#
# Model weights, engine load and token generation are not tested, so a clean run
# is a connectivity/config preflight, not a serving-readiness verdict.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

HEAD_IP="${HEAD_IP:-10.0.0.1}"
WORKER_IP="${WORKER_IP:-10.0.0.2}"
WORKER_USER="${WORKER_USER:-$USER}"
PORT="${PORT:-8888}"
IMAGE="${IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3}"
HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f1np1}"
WORKER_CX7_IF="${WORKER_CX7_IF:-enp1s0f0np0}"
HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"
WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

ok() {
    printf "  \033[1;32m[PASS]\033[0m %s\n" "$1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

warn() {
    printf "  \033[1;33m[WARN]\033[0m %s\n" "$1"
    WARN_COUNT=$((WARN_COUNT + 1))
}

fail() {
    printf "  \033[1;31m[FAIL]\033[0m %s\n" "$1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

section() {
    printf "\n\033[1;36m=== %s ===\033[0m\n" "$1"
}

printf "\033[1;35m"
cat << 'EOF'
  ____                   _      ____             _             
 / ___| _ __   __ _ _ __| | __ |  _ \  ___   ___| |_ ___  _ __ 
 \___ \| '_ \ / _` | '__| |/ / | | | |/ _ \ / __| __/ _ \| '__|
  ___) | |_) | (_| | |  |   <  | |_| | (_) | (__| || (_) | |   
 |____/| .__/ \__,_|_|  |_|\_\ |____/ \___/ \___|\__\___/|_|   
       |_|        2x DGX Spark Diagnostics & Doctor            
EOF
printf "\033[0m\n"
printf "Target Head: %s | Worker: %s@%s | Port: %s\n" "$HEAD_IP" "$WORKER_USER" "$WORKER_IP" "$PORT"

# ----------------------------------------------------------------------------
# 1. SSH & IP Connectivity
# ----------------------------------------------------------------------------
section "1. Inter-Node Connectivity & SSH"

if ping -c 1 -W 2 "$WORKER_IP" >/dev/null 2>&1; then
    ok "Worker IP ($WORKER_IP) is reachable via ICMP ping."
else
    fail "Worker IP ($WORKER_IP) is NOT responding to ping."
fi

if ssh -o BatchMode=yes -o ConnectTimeout=5 "${WORKER_USER}@${WORKER_IP}" "echo ready" >/dev/null 2>&1; then
    ok "Passwordless SSH to ${WORKER_USER}@${WORKER_IP} is configured."
else
    fail "Passwordless SSH to ${WORKER_USER}@${WORKER_IP} failed. Check ~/.ssh/authorized_keys."
fi

# ----------------------------------------------------------------------------
# 2. RoCEv2 / CX7 InfiniBand Interfaces
# ----------------------------------------------------------------------------
section "2. RoCEv2 & CX7 High-Speed Interconnect"

if ip link show "$HEAD_CX7_IF" >/dev/null 2>&1; then
    head_state=$(ip -brief link show "$HEAD_CX7_IF" | awk '{print $2}')
    if [ "$head_state" = "UP" ]; then
        ok "Head CX7 interface '$HEAD_CX7_IF' is UP."
    else
        warn "Head CX7 interface '$HEAD_CX7_IF' is present but state is '$head_state'."
    fi
else
    warn "Head CX7 interface '$HEAD_CX7_IF' not found in ip link (check HEAD_CX7_IF in .env)."
fi

if ssh -o BatchMode=yes -o ConnectTimeout=5 "${WORKER_USER}@${WORKER_IP}" "ip link show '$WORKER_CX7_IF'" >/dev/null 2>&1; then
    ok "Worker CX7 interface '$WORKER_CX7_IF' verified."
else
    warn "Worker CX7 interface '$WORKER_CX7_IF' check returned error."
fi

if command -v ibv_devinfo >/dev/null 2>&1; then
    if ibv_devinfo -d "$HEAD_CX7_IB" 2>/dev/null | grep -q "PORT_ACTIVE"; then
        ok "Head RDMA device '$HEAD_CX7_IB' port state is ACTIVE (RoCEv2 ready)."
    else
        warn "Head RDMA device '$HEAD_CX7_IB' is not PORT_ACTIVE."
    fi
else
    warn "ibv_devinfo not found on host (ibverbs utilities optional on host)."
fi

# ----------------------------------------------------------------------------
# 3. GPU Hardware & Unified Memory (UMA)
# ----------------------------------------------------------------------------
section "3. GPU Hardware & Unified Memory (UMA)"

if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    ok "Head GPU detected: $gpu_name (Total Memory: $gpu_mem)."
    
    # GPU_MEM_UTIL: accept exactly what start.sh's validate_numeric_config accepts
    # (0 < util <= 1), then warn outside the 0.70 - 0.95 band. An unjudgeable value
    # is a FAIL, never a silent PASS; the band is sanity only, not a UMA admission model.
    if ! [[ "$GPU_MEM_UTIL" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0+)?)$ ]] \
       || ! awk -v u="$GPU_MEM_UTIL" 'BEGIN { exit !(u > 0 && u <= 1) }'; then
        fail "GPU_MEM_UTIL must be greater than 0 and at most 1 (got: '$GPU_MEM_UTIL'); start.sh refuses this value, so the UMA budget cannot be judged."
    elif awk -v u="$GPU_MEM_UTIL" 'BEGIN { exit !(u > 0.95) }'; then
        warn "GPU_MEM_UTIL=$GPU_MEM_UTIL is above the 0.95 ceiling this check warns on, risking UMA OOM on mixed load."
    elif awk -v u="$GPU_MEM_UTIL" 'BEGIN { exit !(u < 0.70) }'; then
        warn "GPU_MEM_UTIL=$GPU_MEM_UTIL is below the 0.70 floor this check warns on; the 1.75M token KV pool may be under-allocated on GB10."
    else
        ok "GPU_MEM_UTIL=$GPU_MEM_UTIL is inside the 0.70 - 0.95 band this check accepts (band sanity only, not a memory-admission guarantee)."
    fi
else
    fail "nvidia-smi not found on host. Ensure NVIDIA drivers are installed."
fi

# ----------------------------------------------------------------------------
# 4. Docker & GHCR Image
# ----------------------------------------------------------------------------
section "4. Docker & Container Runtime"

if command -v docker >/dev/null 2>&1; then
    ok "Docker CLI is available."
    if docker info >/dev/null 2>&1; then
        ok "Docker daemon is running and accessible."
    else
        fail "Cannot connect to Docker daemon. Check permissions (sudo usermod -aG docker $USER)."
    fi
    
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        ok "Image '$IMAGE' is present locally."
    else
        warn "Image '$IMAGE' not found locally (start.sh will pull on start)."
    fi
else
    fail "Docker is not installed."
fi

# ----------------------------------------------------------------------------
# 5. Live Service Endpoint Reachability (if running)
# ----------------------------------------------------------------------------
section "5. Live Service Endpoint Reachability"

health_url="http://127.0.0.1:${PORT}/health"
if command -v curl >/dev/null 2>&1; then
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$health_url" || echo "000")
    if [ "$http_code" = "200" ]; then
        ok "vLLM /health endpoint returned HTTP 200 (endpoint reachable; engine load and token generation are not verified)."
        
        # Probe model list
        models_out=$(curl -s --max-time 3 "http://127.0.0.1:${PORT}/v1/models" || echo "")
        if echo "$models_out" | grep -q "GLM-5.3"; then
            ok "OpenAI /v1/models lists a GLM-5.3 model (endpoint reachable)."
        fi
    elif [ "$http_code" = "000" ]; then
        warn "Service is not currently running on port $PORT (start via ./start.sh)."
    else
        warn "Health check returned HTTP $http_code."
    fi
fi

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
section "Doctor Summary"
printf "Results: \033[1;32m%d Passed\033[0m | \033[1;33m%d Warnings\033[0m | \033[1;31m%d Failures\033[0m\n" \
    "$PASS_COUNT" "$WARN_COUNT" "$FAIL_COUNT"

if [ "$FAIL_COUNT" -eq 0 ]; then
    printf "\n\033[1;32m✔ Preflight passed: no failures in the checks above.\033[0m\n"
    printf "  Scope: connectivity, config and container checks only - model weights, engine load and token generation are not tested.\n\n"
    exit 0
else
    printf "\n\033[1;31m✘ Please resolve the failure items above before running ./start.sh\033[0m\n\n"
    exit 1
fi
