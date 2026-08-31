#!/usr/bin/env bash
# vllm#53030 gate: piecewise-graph BatchDescriptor collision can silently pin
# spec-decode acceptance at exactly 1.00 per position. Run after EVERY
# graph-enabled boot, once real traffic (or a bench) has produced >100 drafts.
# Healthy: pos0 ratio well below 1.0 and monotone decay across positions.
# Source: tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark docs/OPEN-PROBLEMS.md #7.
set -euo pipefail
BASE="${1:-http://192.168.100.126:8888}"

metrics=$(curl --noproxy '*' -fsS --max-time 10 "${BASE}/metrics")
drafts=$(printf '%s\n' "$metrics" | grep -F 'spec_decode_num_drafts_total{' | grep -oE '[0-9.e+]+$' || echo 0)
drafts=${drafts%%.*}
if [ "${drafts:-0}" -lt 100 ]; then
  echo "SKIP: only ${drafts:-0} drafts since boot (<100); send a bench round first"
  exit 0
fi

echo "drafts_total=${drafts}"
fail=0
printf '%s\n' "$metrics" | grep -F 'accepted_tokens_per_pos_total{' | while IFS= read -r line; do
  pos=$(printf '%s' "$line" | grep -oE 'position="[0-9]+"' | grep -oE '[0-9]+')
  val=$(printf '%s' "$line" | grep -oE '[0-9.e+]+$'); val=${val%%.*}
  ratio=$(awk -v a="$val" -v d="$drafts" 'BEGIN{printf "%.4f", a/d}')
  echo "pos${pos}: accepted=${val} ratio=${ratio}"
done

pos0=$(printf '%s\n' "$metrics" | grep -F 'accepted_tokens_per_pos_total{' | grep -F 'position="0"' | grep -oE '[0-9.e+]+$'); pos0=${pos0%%.*}
ratio0=$(awk -v a="$pos0" -v d="$drafts" 'BEGIN{printf "%.4f", a/d}')
pinned=$(awk -v r="$ratio0" 'BEGIN{print (r>0.999)?1:0}')
if [ "$pinned" = "1" ]; then
  echo "FAIL: pos0 acceptance ${ratio0} pinned at ~1.00 over ${drafts} drafts — vllm#53030 signature."
  echo "      Spec decode is silently broken under CUDA graphs. Restart with ENFORCE_EAGER=1 to confirm, then investigate capture sizes."
  exit 1
fi
echo "PASS: pos0 acceptance ${ratio0} (healthy decay expected across positions)"
