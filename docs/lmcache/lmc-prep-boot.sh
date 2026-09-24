#!/bin/bash
# lmc-prep-boot.sh — release stale CUDA-IPC mappings before a champion boot.
#
# WHY: when vLLM is SIGKILLed (start.sh stop() uses `docker rm -f`), its
# connector never runs UNREGISTER_KV_CACHE, so each node's LMCache server keeps
# a CUDA-IPC mapping of the DEAD worker's KV tensors (~17.4 GiB). The worker
# preflight then sees MemAvailable < 99.35 GiB and refuses to boot.
#
# FIX: commit the server container (preserving its L2 cache in the image
# layer), then recreate it — a fresh process holds no stale mapping.
# Run on BOTH nodes. Usage:  bash lmc-prep-boot.sh <image-prefix>
set -u
PREFIX="${1:-glm53-lmc-live}"

prep(){
  local host="$1"
  ssh -o ConnectTimeout=8 "$host" "
    set -u
    if ! docker ps --format '{{.Names}}' | grep -qx lmcache-mp; then
      echo '  lmcache-mp not running; nothing to prep'
      exit 0
    fi
    L2=\$(docker exec lmcache-mp sh -c 'ls /home/cshintov/lmc-l2 2>/dev/null | wc -l' 2>/dev/null || echo 0)
    docker commit lmcache-mp ${PREFIX}-snap:test >/dev/null 2>&1
    docker rm -f lmcache-mp >/dev/null 2>&1
    sleep 2
    bash ~/start-lmc-server.sh ${PREFIX}-snap:test >/dev/null 2>&1
    sleep 5
    echo \"  L2 files preserved: \$L2\"
    free -g | sed -n '2p' | awk '{print \"  MemAvailable=\"\$7\"G\"}'
    nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  gpu_app_mem=/' | head -3
    timeout 3 bash -c 'echo > /dev/tcp/127.0.0.1/5555' 2>/dev/null && echo '  :5555 LISTEN' || echo '  :5555 DOWN'
  "
}

echo "=== gb10a ==="; prep gb10a
echo "=== gb10b ==="; prep gb10b
