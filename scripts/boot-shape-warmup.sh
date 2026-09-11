#!/usr/bin/env bash
# boot-shape-warmup.sh — burn DFlash2 / sampler / kpool shapes after /health.
#
# glm53-flash DFlash2 k=7:
#   BLOCK_SIZE = min(256, next_pow2(scheduled_tokens + num_query_per_req))
#   num_query_per_req = 1 + k = 8
# BLOCK 8 is unreachable (min scheduled 1 → 9 → 16). Do not copy DSpark's
# next_pow2(s+6) ladder or 9500-token 8192-chunk arms.
#
# Non-fatal: the launcher WARNs on nonzero exit. Pair with a persistent
# TRITON_CACHE_DIR + TILELANG_CACHE_DIR so each shape compiles once per image.
#
# Usage: boot-shape-warmup.sh [base_url] [model]
# Env:
#   GLM53_WARMUP_REQ_TIMEOUT       per-request curl --max-time (default 240)
#   GLM53_WARMUP_MAX_CONCURRENCY   resolved --max-num-seqs (default 4)
#   GLM53_WARMUP_DFLASH_K          speculative tokens (default 7)
#   GLM53_WARMUP_TRITON_CACHE_DIR  host Triton cache (sampler postcondition)
#   GLM53_WARMUP_BEARER / VLLM_API_KEY
#   WARMUP_CURL                    test seam
set -u

BASE="${1:-http://127.0.0.1:8888}"
MODEL="${2:-GLM-5.3-Flash-EXL3}"
CURL_BIN="${WARMUP_CURL:-curl}"
REQ_TIMEOUT="${GLM53_WARMUP_REQ_TIMEOUT:-240}"
MAX_CONCURRENCY="${GLM53_WARMUP_MAX_CONCURRENCY:-4}"
DFLASH_K="${GLM53_WARMUP_DFLASH_K:-7}"
case "$MAX_CONCURRENCY" in
  ''|*[!0-9]*|0)
    echo "boot-shape-warmup: invalid GLM53_WARMUP_MAX_CONCURRENCY=${MAX_CONCURRENCY@Q}; using 4" >&2
    MAX_CONCURRENCY=4
    ;;
esac
case "$DFLASH_K" in
  ''|*[!0-9]*) DFLASH_K=7 ;;
esac
NONCE="$$-$(date +%s)"

AUTH_ARGS=()
if [ -n "${GLM53_WARMUP_BEARER:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${GLM53_WARMUP_BEARER}")
elif [ -n "${VLLM_API_KEY:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi

next_pow2() {
  local n=$1 p=1
  while [ "$p" -lt "$n" ]; do p=$((p * 2)); done
  printf '%s' "$p"
}

# k=7 → +8. Pick one s per live BLOCK in {16,32,64,128,256}.
LADDER_S=(1 24 56 120 248)
# Long-prefill rungs: trigger BuildPrefillChunkMetadataKernel (1, 2 and
# partial MNBT=7168 chunks). 65536 covers agent-sized contexts; >128k prompts
# can still compile one more specialization.
# Prefills do not affect the DFlash BLOCK shapes above (decode is 1 query).
PREFILL_S=(3584 7168 14336 65536)

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mk_prompt() {
  local n=$1 tag=$2 body
  body=$(printf 'warm %.0s' $(seq 1 "$n"))
  printf '[warmup %s %s] The following is filler context, ignore it: %s Reply with OK.' \
    "$NONCE" "$tag" "$body"
}

fire() {
  local tag=$1 words=$2 thinking=$3 out=$4 profile=${5:-bounded} prompt payload sample_fields thinking_json
  prompt=$(mk_prompt "$words" "$tag")
  if [ "$thinking" = "true" ]; then thinking_json=true; else thinking_json=false; fi
  if [ "$profile" = "serve-default" ]; then
    payload='{"model":"'"$MODEL"'","messages":[{"role":"user","content":"'"$prompt"'"}],"temperature":0}'
  elif [ "${profile#sampling}" != "$profile" ]; then
    case "$profile" in
      # Hub generation_config.json stamps top_p=0.95. Omitting top_p on a
      # k-only arm compiles TOPK+TOPP, never k-only. top_p=1.0 / top_k=0
      # are how glm53-flash's sampler drops the p / k tensors (None).
      sampling-k)  sample_fields='"top_k":40,"top_p":1.0' ;;
      sampling-p)  sample_fields='"top_k":0,"top_p":0.9' ;;
      *)           sample_fields='"top_k":40,"top_p":0.9' ;;
    esac
    payload='{"model":"'"$MODEL"'","messages":[{"role":"user","content":"'"$prompt"'"}],"max_tokens":24,"temperature":0.8,'"$sample_fields"',"chat_template_kwargs":{"enable_thinking":'"$thinking_json"'}}'
  else
    payload='{"model":"'"$MODEL"'","messages":[{"role":"user","content":"'"$prompt"'"}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"enable_thinking":'"$thinking_json"'}}'
  fi
  if "$CURL_BIN" -fsS --max-time "$REQ_TIMEOUT" "${AUTH_ARGS[@]}" \
      "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
      -d "$payload" >/dev/null 2>>"$tmpdir/errors"; then
    echo ok > "$out"
  else
    echo fail > "$out"
  fi
}

burst() {
  local arm=$1 c=$2 words=$3 profile=${4:-bounded} thinking=${5:-false} i t0 t1
  for i in $(seq 1 "$c"); do : > "$tmpdir/${arm}-${i}"; done
  t0=$(date +%s)
  for i in $(seq 1 "$c"); do
    fire "${arm}-${i}" "$words" "$thinking" "$tmpdir/${arm}-${i}" "$profile" &
  done
  wait
  t1=$(date +%s)
  echo "  arm ${arm}: C=${c} x ~${words} tok, profile=${profile}, think=${thinking}, $((t1 - t0))s"
}

SAMPLER_KERNEL=_topk_topp_kernel

sampler_cache_combos() {
  local root=$1 ttir kuse puse combo
  for ttir in "$root"/*/"$SAMPLER_KERNEL.ttir"; do
    [ -f "$ttir" ] || continue
    kuse=$(grep -oE '%K[^A-Za-z0-9_]' "$ttir" | wc -l)
    puse=$(grep -oE '%P[^A-Za-z0-9_]' "$ttir" | wc -l)
    if [ "$kuse" -gt 1 ] && [ "$puse" -gt 1 ]; then combo=k+p
    elif [ "$kuse" -gt 1 ]; then combo=k-only
    elif [ "$puse" -gt 1 ]; then combo=p-only
    else combo=neither; fi
    printf '%s\n' "$combo"
  done | sort -u
}

verify_sampler_cache() {
  local root="${GLM53_WARMUP_TRITON_CACHE_DIR:-}" combos combo n missing=""
  if [ -z "$root" ] || [ ! -d "$root" ]; then
    echo "  sampler-cache postcondition: SKIPPED (GLM53_WARMUP_TRITON_CACHE_DIR unset or not a directory)"
    return 0
  fi
  combos=$(sampler_cache_combos "$root")
  for combo in k-only p-only k+p; do
    n=$(printf '%s\n' "$combos" | grep -cx "$combo")
    [ "$n" -ge 1 ] || missing="${missing} ${combo}:0/1"
  done
  if [ -z "$missing" ]; then
    echo "  sampler-cache postcondition: MET — ${SAMPLER_KERNEL} constexpr combos on this rank:"
    printf '%s\n' "$combos" | sed 's/^/    /'
    return 0
  fi
  echo "  sampler-cache postcondition: unmet —${missing} (constexpr combos)"
  return 1
}

mk_ladder_prompt() {
  local n=$1 out="hello" i
  for ((i = 1; i < n; i++)); do out="$out hello"; done
  printf '%s' "$out"
}

verify_ladder_rung() {
  local s=$1 tag=${2:-ladder} prompt want_block got resp t0 t1 qpad
  qpad=$((DFLASH_K + 1))
  : > "$tmpdir/$tag-$s"
  prompt=$(mk_ladder_prompt "$s")
  want_block=$(next_pow2 $((s + qpad)))
  if [ "$want_block" -gt 256 ]; then want_block=256; fi
  printf '{"model":"%s","prompt":"%s"}' "$MODEL" "$prompt" > "$tmpdir/$tag-$s.tok.json"
  if ! resp=$("$CURL_BIN" -fsS --max-time 30 "${AUTH_ARGS[@]}" \
        "$BASE/tokenize" -H "Content-Type: application/json" \
        --data-binary "@$tmpdir/$tag-$s.tok.json" \
        2>>"$tmpdir/errors"); then
    echo "boot-shape-warmup: tokenize verify FAILED for rung ${tag} s=${s}: POST /tokenize errored — rung skipped, BLOCK ${want_block} NOT warmed" >&2
    echo fail > "$tmpdir/$tag-$s"
    return 0
  fi
  got=$(printf '%s\n' "$resp" | grep -o '"count"[[:space:]]*:[[:space:]]*[0-9]*' | head -n 1 | grep -o '[0-9]*$')
  if [ -z "$got" ]; then
    echo "boot-shape-warmup: tokenize verify FAILED for rung ${tag} s=${s}: no usable \"count\" in /tokenize response — rung skipped, BLOCK ${want_block} NOT warmed" >&2
    echo fail > "$tmpdir/$tag-$s"
    return 0
  fi
  if [ "$got" -ne "$s" ]; then
    echo "boot-shape-warmup: tokenize verify FAILED for rung ${tag} s=${s}: /tokenize reported ${got} tokens, need exactly ${s} — rung skipped, BLOCK ${want_block} NOT warmed" >&2
    echo fail > "$tmpdir/$tag-$s"
    return 0
  fi
  t0=$(date +%s)
  # Long rungs exceed ARG_MAX as a curl argument; stage the JSON in a file.
  printf '{"model":"%s","prompt":"%s","max_tokens":1,"temperature":0}' \
    "$MODEL" "$prompt" > "$tmpdir/$tag-$s.json"
  if "$CURL_BIN" -fsS --max-time "$REQ_TIMEOUT" "${AUTH_ARGS[@]}" \
      "$BASE/v1/completions" -H "Content-Type: application/json" \
      --data-binary "@$tmpdir/$tag-$s.json" \
      >/dev/null 2>>"$tmpdir/errors"; then
    echo ok > "$tmpdir/$tag-$s"
    t1=$(date +%s)
    echo "  ${tag} s=${s}: tokenize ${got}/${s} -> BLOCK ${want_block} fired ($((t1 - t0))s)"
  else
    echo fail > "$tmpdir/$tag-$s"
    echo "  ${tag} s=${s}: tokenize ${got}/${s} -> BLOCK ${want_block} request FAILED"
  fi
}

ladder() {
  local s
  for s in "${LADDER_S[@]}"; do
    verify_ladder_rung "$s"
  done
}

prefill() {
  local s
  for s in "${PREFILL_S[@]}"; do
    verify_ladder_rung "$s" prefill
  done
}

if ! "$CURL_BIN" -fsS --max-time 10 "${AUTH_ARGS[@]}" "$BASE/v1/models" >/dev/null 2>&1; then
  echo "boot-shape-warmup: API not reachable at $BASE — skipping sweep" >&2
  exit 1
fi

echo "boot-shape-warmup: sweeping DFlash2 k=${DFLASH_K} / sampler / kpool shapes"
total_t0=$(date +%s)

ladder
prefill

EXPECTED_CHAT_REQUESTS=6
burst c1        1 32 bounded false
burst think-c1  1 16 bounded true
burst short-c1  1 8 serve-default
burst samp-k    1 8 sampling-k false
burst samp-p    1 8 sampling-p false
burst samp-kp   1 8 sampling-kp false
if [ "$MAX_CONCURRENCY" -ge 2 ]; then
  burst short-c2 2 8 serve-default
  EXPECTED_CHAT_REQUESTS=$((EXPECTED_CHAT_REQUESTS + 2))
fi
if [ "$MAX_CONCURRENCY" -ge 3 ]; then
  burst samp-kp-c3 3 8 sampling-kp false
  EXPECTED_CHAT_REQUESTS=$((EXPECTED_CHAT_REQUESTS + 3))
fi
if [ "$MAX_CONCURRENCY" -ge 4 ]; then
  burst short-c4 4 8 serve-default
  EXPECTED_CHAT_REQUESTS=$((EXPECTED_CHAT_REQUESTS + 4))
fi
if [ "$MAX_CONCURRENCY" -gt 4 ]; then
  echo "boot-shape-warmup: WARN: MAX_NUM_SEQS=${MAX_CONCURRENCY}; batch shapes above C=4 are not pre-warmed" >&2
fi

SAMPLER_POSTCOND=ok
verify_sampler_cache || SAMPLER_POSTCOND=fail

total=0 ok_count=0
for f in "$tmpdir"/*-*; do
  [ -f "$f" ] || continue
  case "$f" in *.json) continue;; esac
  total=$((total + 1))
  [ "$(cat "$f")" = "ok" ] && ok_count=$((ok_count + 1))
done
EXPECTED_REQUESTS=$(( ${#LADDER_S[@]} + ${#PREFILL_S[@]} + EXPECTED_CHAT_REQUESTS ))
if [ "$total" -ne "$EXPECTED_REQUESTS" ]; then
  echo "boot-shape-warmup: internal error: tallied $total outcomes for $EXPECTED_REQUESTS scheduled requests" >&2
  exit 1
fi
total_t1=$(date +%s)
echo "boot-shape-warmup: ${ok_count}/${total} requests ok in $((total_t1 - total_t0))s"

if [ "$ok_count" -lt "$total" ]; then
  echo "boot-shape-warmup: $((total - ok_count)) request(s) failed — uncovered shapes may JIT mid-serve" >&2
  sed -n '1,5p' "$tmpdir/errors" >&2 2>/dev/null || true
  exit 1
fi
if [ "$SAMPLER_POSTCOND" != ok ]; then
  echo "boot-shape-warmup: sampler-cache postcondition UNMET — ${SAMPLER_KERNEL} variants may JIT mid-serve" >&2
  exit 1
fi
exit 0
