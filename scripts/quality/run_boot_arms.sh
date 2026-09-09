#!/usr/bin/env bash
# On a boot with adaptive-k enabled: run the full panel at k=7 (JSON toggle off) and then with
# adaptive-k on, from the same weights — the second KL capture doubles as the noise floor.
# usage: RUN=quality-20260909 scripts/quality/run_boot_arms.sh <arm-prefix>   e.g. bf16-ablit0
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
P="$1"; J="$HOME/.cache/vllm-glm53-flash/glm53_adaptive_k.json"
cp "$J" "$J.bak-$(date -u +%Y%m%dT%H%M%SZ)"
echo '{"mode":"off","set":"2,4,7","alpha":0.25,"margin":1.0,"min_steps":4,"saturate":"max"}' > "$J"; sleep 20
"$ROOT/scripts/quality/run_arm.sh" "$P-k7" --workers 3 --n-he 40 --n-mbpp 40
echo '{"mode":"ema","set":"2,4,7","alpha":0.25,"margin":1.0,"min_steps":4,"saturate":"max"}' > "$J"; sleep 20
"$ROOT/scripts/quality/run_arm.sh" "$P-adaptk" --workers 3 --n-he 40 --n-mbpp 40
echo ALLDONE
