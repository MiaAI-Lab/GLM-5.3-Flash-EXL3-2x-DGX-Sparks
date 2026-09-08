#!/usr/bin/env bash
# Final live configuration: adaptive-k (2,4,7) + FP8 dense,kda at 850k, KV pool capped at the stock level (15.0 GiB)
# so the FP8 memory saving becomes host headroom instead of a bigger KV pool. Boots from the overnight-decode worktree.
# Usage: boot_live_best.sh <phase-dir>
set -uo pipefail
R=/home/mia/NewModels/glm-5.3-flash-sm120/logs/overnight-decode-20260907T224521Z
cp "$R/C1/override.json" "$HOME/.cache/vllm-glm53-flash/glm53_adaptive_k.json"
exec "$R/scripts/boot_candidate.sh" "$1" MAX_MODEL_LEN=850000 GLM53_ADAPTIVE_K_SET=2,4,7 GLM53_DENSE_FP8=dense,kda \
  'EXTRA_ARGS=--cudagraph-capture-sizes 1 2 3 4 5 6 8 9 10 12 15 16 20 24 32 --kv-cache-memory-bytes 16106127360'
