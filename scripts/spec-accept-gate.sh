#!/usr/bin/env bash
# vllm#53030 gate: piecewise-graph BatchDescriptor collision can silently pin
# spec-decode acceptance at exactly 1.00 per position. Run after EVERY
# graph-enabled boot, once real traffic (or a bench) has produced >100 drafts.
# Healthy: pos0 ratio well below 1.0 and monotone decay across positions.
# Source: tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark docs/OPEN-PROBLEMS.md #7.
#
# Usage: spec-accept-gate.sh [metrics-base-url]  (default http://127.0.0.1:8888)
# Exit codes: 0 PASS, or SKIP when <100 drafts have accumulated since boot;
#             1 FAIL: pos0 pinned at ~1.00 (the vllm#53030 signature);
#             2 cannot judge: the endpoint is unreachable, or a required
#             series is missing, ambiguous or unreadable. An unreadable
#             acceptance curve must never be reported as PASS, FAIL or SKIP.
set -euo pipefail
BASE="${1:-http://127.0.0.1:8888}"

die() { echo "ERROR: $*" >&2; exit 2; }

# Prometheus samples are `metric{labels} value [timestamp]`; the value may be
# fractional or scientific (1.5e+03). ${v%%.*} read the exponent form as its
# leading digit, so a live counter looked almost empty and a healthy start was
# reported as SKIP. Prints the count as a whole number, and exits non-zero when
# the sample carries no finite non-negative decimal or scientific value, when an
# overflowing exponent reaches +Inf (1e999), or when the count exceeds 2^53:
# awk holds doubles, so past that the printed integer is no longer exact for the
# bash compares below, and 1e300 (300 digits) compared false against the
# <100-drafts test — a silent PASS for a curve that was never measured.
sample() {
  awk -v line="$1" 'BEGIN{
    sub(/^.*\}/, "", line)                # drop the metric name and labels
    if (!match(line, /[^[:space:]]+/)) exit 1
    v = substr(line, RSTART, RLENGTH)     # value token, timestamp ignored
    if (v !~ /^\+?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$/) exit 1
    n = v + 0                             # +Inf from an overflow fails it too
    if (n > 2^53) exit 1
    printf "%.0f", int(n)
  }'
}

metrics=$(curl --noproxy '*' -fsS --max-time 10 "${BASE}/metrics") ||
  die "cannot read ${BASE}/metrics (curl rc $?)"

# Two label sets (e.g. one endpoint serving two model_name values) have no
# defensible aggregate here; concatenating them silently corrupts every ratio.
# A missing denominator is not "no traffic" either: it used to fall through
# `|| echo 0` into the <100-drafts SKIP, reporting "cannot judge" as "quiet".
drafts_series=$(printf '%s\n' "$metrics" | grep -F 'spec_decode_num_drafts_total{' || true)
drafts_n=$(printf '%s\n' "$drafts_series" | grep -cF 'spec_decode_num_drafts_total{' || true)
[ "$drafts_n" -ge 1 ] ||
  die "no spec_decode_num_drafts_total series in /metrics — cannot judge acceptance without a drafts denominator"
[ "$drafts_n" -le 1 ] ||
  die "${drafts_n} spec_decode_num_drafts_total series in /metrics — ambiguous label sets, refusing to pick a denominator"
drafts=$(sample "$drafts_series") ||
  die "unreadable spec_decode_num_drafts_total value (${drafts_series}) — cannot judge acceptance"
if [ "$drafts" -lt 100 ]; then
  echo "SKIP: only ${drafts} drafts since boot (<100); send a bench round first"
  exit 0
fi

echo "drafts_total=${drafts}"

# Per-position series must exist once >100 drafts have run; their absence used
# to end the script mid-run (rc 1, with nothing but drafts_total printed).
pos_series=$(printf '%s\n' "$metrics" | grep -F 'accepted_tokens_per_pos_total{' || true)
[ -n "$pos_series" ] ||
  die "no accepted_tokens_per_pos_total series in /metrics — model is not spec-decoding, or the metric name drifted; cannot judge acceptance"
# A matched series without a numeric position label is not judgeable evidence:
# it used to fail the position list under pipefail and take the whole run down
# rc 1 with no diagnostic, or silently judge only the labelled subset. The label
# list boundary is part of the pattern: a bare `position="0"` also matched
# inside `notposition="0"`, whose value was then judged as position 0.
POS_LABEL='[{,]position="'
unlabeled=$(printf '%s\n' "$pos_series" | grep -vcE "${POS_LABEL}[0-9]+\"" || true)
[ "$unlabeled" -eq 0 ] ||
  die "${unlabeled} accepted_tokens_per_pos_total series without a numeric position label — cannot judge acceptance"
positions=$(printf '%s\n' "$pos_series" | grep -oE "${POS_LABEL}[0-9]+\"" | grep -oE '[0-9]+' | sort -n | uniq)
printf '%s\n' "$positions" | grep -qx '0' ||
  die 'per-position series present but position="0" is missing — cannot judge the pinned-1.00 signature'

for pos in $positions; do
  n=$(printf '%s\n' "$pos_series" | grep -cE "${POS_LABEL}${pos}\"" || true)
  [ "$n" -eq 1 ] ||
    die "${n} series for position=\"${pos}\" — ambiguous label sets, refusing to aggregate them into one ratio"
  line=$(printf '%s\n' "$pos_series" | grep -E "${POS_LABEL}${pos}\"" || true)
  val=$(sample "$line") ||
    die "unreadable accepted_tokens_per_pos_total value at position=\"${pos}\" (${line}) — cannot judge acceptance"
  ratio=$(awk -v a="$val" -v d="$drafts" 'BEGIN{printf "%.4f", a/d}')
  echo "pos${pos}: accepted=${val} ratio=${ratio}"
  if [ "$pos" = 0 ]; then pos0="$val"; fi
done

ratio0=$(awk -v a="$pos0" -v d="$drafts" 'BEGIN{printf "%.4f", a/d}')
pinned=$(awk -v r="$ratio0" 'BEGIN{print (r>0.999)?1:0}')
if [ "$pinned" = "1" ]; then
  echo "FAIL: pos0 acceptance ${ratio0} pinned at ~1.00 over ${drafts} drafts — vllm#53030 signature."
  echo "      Spec decode is silently broken under CUDA graphs. Restart with ENFORCE_EAGER=1 to confirm, then investigate capture sizes."
  exit 1
fi
echo "PASS: pos0 acceptance ${ratio0} (healthy decay expected across positions)"
