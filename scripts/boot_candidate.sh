#!/usr/bin/env bash
# Boot from the overnight-decode worktree with the adaptive-k knob ON (union set), 131k, profiler config kept for parity with B0.
# Usage: boot_candidate.sh <phase-dir> [extra env assignments...]
set -uo pipefail
PH="$1"; shift; mkdir -p "$PH"
W=/home/mia/NewModels/glm-5.3-flash-sm120/.claude/worktrees/overnight-decode
cd "$W"
export MAX_MODEL_LEN=131072
export EXTRA_ARGS='--cudagraph-capture-sizes 1 2 3 4 5 6 8 9 10 12 15 16 18 20 24 32 --profiler-config {"profiler":"torch","torch_profiler_dir":"/root/.cache/vllm/prof-decode","delay_iterations":2,"max_iterations":3,"torch_profiler_with_stack":false,"ignore_frontend":true}'
export GLM53_ADAPTIVE_K=ema
export GLM53_ADAPTIVE_K_SET=2,3,4,5,7
for kv in "$@"; do export "$kv"; done
env | grep -E "^(MAX_MODEL_LEN|EXTRA_ARGS|GLM53_ADAPTIVE_K)" > "$PH/boot-env.txt"
date -u +%FT%TZ > "$PH/boot-start.txt"
SKIP_SHIP=1 ./start.sh restart > "$PH/boot.log" 2>&1; echo "start.sh exit=$?" >> "$PH/boot.log"
date -u +%FT%TZ > "$PH/boot-end.txt"
