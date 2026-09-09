#!/usr/bin/env bash
# Run the whole quality panel against the live server and file it under logs/quality-<RUN>/<ARM>/.
# usage: RUN=quality-20260909 scripts/quality/run_arm.sh <arm-name> [code_eval extra args...]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ARM="$1"; shift
RUN="${RUN:-quality-$(date -u +%Y%m%dT%H%M%SZ)}"
D="$ROOT/logs/$RUN/$ARM"; mkdir -p "$D"
DATA="${DATA:-/tmp/claude-1000/-home-mia-NewModels-glm-5-3-flash-sm120/c97b7ef7-8f4e-4fd2-a975-b8d485bba2aa/scratchpad/data}"
date -u +%FT%TZ > "$D/start.txt"
docker inspect glm53-exl3-head --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^(ABLIT|GLM53_ADAPTIVE_K|GLM53_DENSE_FP8|EXTRA_ARGS)=' > "$D/server-env.txt" 2>/dev/null
cp "$HOME/.cache/vllm-glm53-flash/glm53_adaptive_k.json" "$D/adaptive_k.json" 2>/dev/null
echo "[$ARM] kl_panel"; python3 "$ROOT/scripts/quality/kl_panel.py" capture "$D/kl.json" > "$D/kl.log" 2>&1; tail -6 "$D/kl.log"
echo "[$ARM] toolcall_eval"; python3 "$ROOT/scripts/quality/toolcall_eval.py" "$D/toolcall.json" > "$D/toolcall.log" 2>&1; tail -1 "$D/toolcall.log"
echo "[$ARM] code_eval"; python3 "$ROOT/scripts/quality/code_eval.py" "$D/code.json" --data "$DATA" "$@" > "$D/code.log" 2>&1; grep -A3 '"humaneval"\|"mbpp"' "$D/code.log" | grep pass_rate
date -u +%FT%TZ > "$D/end.txt"; echo "[$ARM] done -> $D"
