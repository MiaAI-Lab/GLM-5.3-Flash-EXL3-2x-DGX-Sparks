#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Resumable, checksum-verified fetch of the GLM-5.3-Flash fidelity suite
# (BF16 reference hidden states + shared lm_head + sealed token windows).
#
# Why curl and not huggingface_hub: this box reaches neither API host through the
# python client. huggingface.co is DNS-poisoned (AliDNS hands back Twitter IPs,
# DNSPod a Facebook IP) and DoH is blocked, so only hf-mirror.com is usable - and
# huggingface_hub 1.28 still fetches the *tree listing* straight from
# huggingface.co, which fails no matter what HF_ENDPOINT says. The mirror's
# /resolve/ path redirects to us.aws.cdn.hf.co, which is reachable and answers
# range requests: ~0.8 MB/s per connection.
#
# Two failure modes this script exists to survive, both measured here:
#   1. stalls - a connection sits at 0 B/s for minutes, small files as often as
#      big ones. --speed-limit/--speed-time abort those, and the loop retries.
#   2. silent corruption on resume - after a stall abort, resuming from the local
#      breakpoint produced files with valid HTTP status and wrong bytes (a 14 KB
#      token window that failed to parse at char 4096, a file left at exactly
#      4096 bytes, a zero-byte file, all reported "200"). Never trust a completed
#      curl here: SHA256SUMS is fetched first and every file is verified before it
#      is put in place, with a fall-back to a fresh transfer if the resume is bad.
#
# What is actually distributed: only shard 0 of 10. capture-manifest-full.json
# claims complete:true / contexts:5120 in a directory holding 512 hidden files;
# capture-manifest-shard.json is the truth (contiguous indices 0-511, stride 1),
# so the token windows to fetch are 0..511, not 0..5119.
#
#   bash fetch-fidelity-suite.sh SELECT=metadata          # manifests + reports + SHA256SUMS
#   bash fetch-fidelity-suite.sh SELECT=head              # shared lm_head (1.27 GB)
#   bash fetch-fidelity-suite.sh SELECT=tokens            # windows 0..TOKENS_UPTO-1
#   bash fetch-fidelity-suite.sh SELECT=ref  PARALLEL=8   # 512 reference shards, 8.6 GB
#   bash fetch-fidelity-suite.sh SELECT=fp8  PARALLEL=8   # 512 FP8 shards, scorer validation
#   bash fetch-fidelity-suite.sh --verify                 # check every fetched file
#   bash fetch-fidelity-suite.sh --verify --purge-bad     # delete bad ones, then rerun SELECT
set -euo pipefail

REPO="${REPO:-malaiwah/GLM-5.3-Flash-fidelity-suite-v1}"
DEST="${DEST:-/srv/models/fidelity-suites/GLM-5.3-Flash-fidelity-suite-v1}"
BASE="https://hf-mirror.com/datasets/$REPO/resolve/main"
LIST="${LIST:-/tmp/fidelity-files.txt}"     # every path in the repo, one per line
PARALLEL="${PARALLEL:-4}"
MAX_TIME="${MAX_TIME:-14400}"
SELECT="${SELECT:-metadata}"
TOKENS_UPTO="${TOKENS_UPTO:-512}"
LANE_REF="reference-bf16-shard0"
LANE_FP8="as-served-fp8-shard0"

# The repo's file list comes from the mirror's API (one call, ~80 s) and is cached,
# so repeated SELECT runs cost nothing.
file_list() {
  if [ ! -s "$LIST" ]; then
    curl -4 -fsSL --retry 10 --retry-delay 5 --retry-all-errors -m 900 \
      "https://hf-mirror.com/api/datasets/$REPO" \
      | jq -r '.siblings[] | .rfilename' > "$LIST.tmp"
    [ -s "$LIST.tmp" ] && mv "$LIST.tmp" "$LIST" || { rm -f "$LIST.tmp"; return 1; }
  fi
  grep -E "$1" "$LIST"
}

METADATA='SHA256SUMS README.md llms.txt suite/suite-manifest.json
head/head-extraction.json
reports/clean-scope-recompute.md reports/environment.json reports/image-pin.txt reports/nvidia-smi.txt
reports/qualify-bf16.json reports/qualify-fp8.json reports/report-fp8-vs-bf16.json
reports/report-fp8-vs-bf16-scorefrom1024.json reports/head-equality-fp8.json
reports/determinism-bf16.json reports/determinism-fp8.json reports/determinism-noise-bf16.json
reports/determinism-noise-fp8.json reports/determinism-guarded-bf16.json
reports/determinism-kernelpin-bf16.json reports/determinism-stackpin-bf16.json
reports/determinism-stackpin-nodeepgemm-bf16.json
reports/gen-check.json reports/gen-snippet.txt reports/head-extraction.json
reports/crosscheck-brandonmusic.json reports/fp8-on-brandon-panel.json reports/tr3-4bpw-packed-kld.json
reports/k6-packed-kld.json reports/k6-five-run-kld.json reports/dione-q4-packed-kld.json
reports/dione-3.0bpw-packed-kld.json reports/turbo-4.05bpw-packed-kld.json
reports/vcruz-k2-2bpw-packed-kld.json reports/vcruz-k2-2bpw-attributable.json
reports/vcruz-k2-2bpw-hub-digest-verification.json reports/tensor-stats-bf16.json
reports/tensor-stats-fp8.json reports/stack-provenance-retro.json
reports/kvdelta-bf16-fp8kv-FAILED.txt reports/kvdelta-fp8-fp8kv-FAILED.txt
reports/engine-logs/README.md
reference-bf16-shard0/capture-manifest-full.json reference-bf16-shard0/capture-manifest-shard.json
reference-bf16-shard0/capture-cut-point.json as-served-fp8-shard0/capture-manifest-full.json
as-served-fp8-shard0/capture-manifest-shard.json as-served-fp8-shard0/capture-cut-point.json'

case "$SELECT" in
  metadata) FILES="$METADATA" ;;
  head)     FILES="$(file_list "^head/(head|final_norm)\.safetensors$")" ;;
  ref)      FILES="$(file_list "^$LANE_REF/hidden_[0-9]+\.safetensors$")" ;;
  fp8)      FILES="$(file_list "^$LANE_FP8/hidden_[0-9]+\.safetensors$")" ;;
  tokens)   # the window index is the number in context-NNNN.json; with
            # -F"[.-]" that is field 2 ("suite/tokens/context", "0001", "json").
            # Field 3 is "json" -> 0, which used to select all 5,120 windows.
            FILES="$(file_list "^suite/tokens/context-[0-9]{1,4}\.json$" \
                        | awk -F'[.-]' -v n="$TOKENS_UPTO" '$2 + 0 < n')" ;;
  all)      FILES="$(file_list '.')" ;;
  *)        FILES="${SELECT//,/ }" ;;
esac

expected_sha() {
  awk -v n="./$1" '$2 == n {print $1; exit}' "$DEST/SHA256SUMS" 2>/dev/null
}

fetch_one() {
  local f="$1"
  local part="$DEST/$f.part"
  local expected code tries resume
  expected=$(expected_sha "$f")
  for tries in 1 2 3 4 5 6; do
    # First attempts resume; after two bad results the partial is thrown away and
    # the transfer starts clean, because a bad breakpoint never gets better.
    if [ "$tries" -le 2 ]; then resume="-C -"; else resume=""; rm -f "$part"; fi
    code=$(curl -4 -fsSL --create-dirs $resume \
               --connect-timeout 15 --speed-limit 4096 --speed-time 25 \
               -m "$MAX_TIME" -o "$part" -w '%{http_code}' "$BASE/$f" 2>/dev/null) || code=000
    if [ "$code" = "200" ] || [ "$code" = "206" ]; then
      if [ -z "$expected" ] || [ "$(sha256sum "$part" | cut -d' ' -f1)" = "$expected" ]; then
        mv "$part" "$DEST/$f"
        echo "ok $code $(stat -c%s "$DEST/$f") $f"
        return 0
      fi
      echo "bad-sha $f (attempt $tries)"
    fi
    sleep 3
  done
  rm -f "$part"
  echo "FAIL $f"
  return 1
}
export -f fetch_one expected_sha
export BASE DEST MAX_TIME

case "${1:-}" in
  --verify)
    [ -s "$DEST/SHA256SUMS" ] || { echo "SHA256SUMS not fetched yet"; exit 1; }
    checked=0; bad=0; absent=0
    while read -r sum name; do
      name="${name#./}"
      if [ ! -f "$DEST/$name" ]; then absent=$((absent + 1)); continue; fi
      checked=$((checked + 1))
      if [ "$(sha256sum "$DEST/$name" | cut -d' ' -f1)" != "$sum" ]; then
        echo "BAD $name"
        bad=$((bad + 1))
        [ "${2:-}" = "--purge-bad" ] && rm -f "$DEST/$name"
      fi
    done < "$DEST/SHA256SUMS"
    echo "verified=$checked corrupt=$bad not-fetched=$absent"
    [ "$bad" -eq 0 ]
    ;;
esac

echo "SELECT=$SELECT files=$(echo "$FILES" | wc -w) dest=$DEST parallel=$PARALLEL"
# shellcheck disable=SC2086
printf '%s\n' $FILES | xargs -P "$PARALLEL" -I{} bash -c 'fetch_one "$@"' _ {}

echo "--- unresolved"
missing=0
for f in $FILES; do
  [ -s "$DEST/$f" ] || { echo "  MISSING $f"; missing=$((missing + 1)); }
done
echo "missing=$missing"
du -sh "$DEST"
