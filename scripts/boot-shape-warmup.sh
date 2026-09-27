#!/usr/bin/env bash
# boot-shape-warmup.sh — burn DFlash2 / sampler / kpool shapes after /health.
#
# glm53-flash DFlash2 k=7:
#   BLOCK_SIZE = min(256, next_pow2(scheduled_tokens + num_query_per_req))
#   num_query_per_req = 1 + k = 8
# BLOCK 8 is unreachable (min scheduled 1 → 9 → 16). Do not copy DSpark's
# next_pow2(s+6) ladder or 9500-token 8192-chunk arms.
#
# Exit: 0 every shape warmed; 1 warmup incomplete (nonfatal — the launcher
# WARNs, an unwarmed shape is a cache miss, not a broken engine); 3 degenerate
# engine (fatal — the launchers collect logs and fail the start instead of
# READY). Pair with a persistent TRITON_CACHE_DIR + TILELANG_CACHE_DIR so each
# shape compiles once per image.
#
# Degenerate-engine canary (on by default): a boot can answer /health and still
# generate garbage — #249 reports every reply as "!!!!" with DFlash acceptance
# ~0, which today shows up only as the unbounded arms timing out and gets filed
# as "uncovered shapes may JIT mid-serve". Two checks over requests the sweep
# already sends, and only on positive evidence: a request that fails, an
# unreadable counter or too little traffic is reported as "cannot judge", never
# as degeneration.
#   content     the bounded temperature-0 c1 arm ("Reply with OK.", thinking
#               off) must carry an OK token in choices[0].message.content,
#               decoded by a JSON parser (never message.reasoning_content); a
#               body that is not a chat-completions reply is not judged
#   acceptance  the two spec-decode counter families are sampled before and
#               after the sweep and must pair per label set: unchanged label
#               sets, every counter finite, integral and non-decreasing, and no
#               generation sample (a *_created or process_start_time_seconds
#               series the exporter exposes) may move. Only then are the
#               per-series deltas summed, and the sweep must have drafted >=
#               GLM53_WARMUP_CANARY_MIN_DRAFTS tokens and accepted at least one.
#               Anything else — no baseline sample, a changed, unpaired or
#               missing series, a reset, a restart — is "cannot judge", never
#               "accepted nothing".
#
# Two snapshots cannot see a restart that both reset a series and climbed past
# its pre-restart value in between unless the exporter exposes a generation
# sample. That gap is irreducible here: it stays a "cannot judge" risk, never a
# claim of health.
#
# Usage: boot-shape-warmup.sh [base_url] [model]
# Env:
#   GLM53_WARMUP_REQ_TIMEOUT       per-request curl --max-time (default 240)
#   GLM53_WARMUP_MAX_CONCURRENCY   resolved --max-num-seqs (default 4)
#   GLM53_WARMUP_DFLASH_K          speculative tokens (default 7)
#   GLM53_WARMUP_TRITON_CACHE_DIR  host Triton cache (sampler postcondition)
#   GLM53_WARMUP_BEARER / VLLM_API_KEY
#   GLM53_WARMUP_CANARY            1 (default) = canary on, 0 = off
#   GLM53_WARMUP_CANARY_MIN_DRAFTS drafted tokens needed to judge acceptance (default 64)
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
CANARY="${GLM53_WARMUP_CANARY:-1}"
CANARY_MIN_DRAFTS="${GLM53_WARMUP_CANARY_MIN_DRAFTS:-64}"
case "$CANARY_MIN_DRAFTS" in
  ''|*[!0-9]*) CANARY_MIN_DRAFTS=64 ;;
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
  # The reply body is kept as *.json (the tally skips it) for the c1 canary.
  if "$CURL_BIN" -fsS --max-time "$REQ_TIMEOUT" "${AUTH_ARGS[@]}" \
      "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
      -d "$payload" >"$out.resp.json" 2>>"$tmpdir/errors"; then
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

# One /metrics scrape into $1. Returns nonzero and leaves the file empty when
# the endpoint is unreadable or the transfer broke, so a partial exposition can
# never be judged.
metrics_snapshot() {
  local body=$1 rc=0
  "$CURL_BIN" -fsS --max-time 10 "${AUTH_ARGS[@]}" "$BASE/metrics" >"$body" 2>/dev/null || rc=$?
  if [ "$rc" != "0" ] || [ ! -s "$body" ]; then
    : > "$body"
    return 1
  fi
  return 0
}

# Drafted/accepted acceptance from two /metrics samples, paired per label set.
# Prints one line:
#   judged <drafted> <accepted> <series>
#   unjudged <reason>
# Only per-series deltas of the two spec-decode families are summed: a label set
# that appears, disappears, goes missing or resets between the samples makes the
# whole delta unusable, never a zero total. Generation samples (the counter's
# own *_created sample or process_start_time_seconds, when the exporter exposes
# them) must not move between the samples, which catches a restart that already
# climbed past its pre-restart counters between the two reads.
spec_delta() {
  python3 -S -c '
import math, sys

# A Prometheus counter is exposed as <stem>_total plus a <stem>_created
# creation sample, so _created attaches to the stem: the _total suffix is not
# part of the creation name.
DRAFT_STEM = "vllm:spec_decode_num_draft_tokens"
ACCEPT_STEM = "vllm:spec_decode_num_accepted_tokens"
DRAFT = DRAFT_STEM + "_total"
ACCEPT = ACCEPT_STEM + "_total"
GENERATION = (DRAFT_STEM + "_created", ACCEPT_STEM + "_created", "process_start_time_seconds")


def load(path):
    counters = {DRAFT: {}, ACCEPT: {}}
    generation = {}
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "{" in line:
                name, _, rest = line.partition("{")
                labels, sep, value = rest.rpartition("}")
                if not sep:
                    return None
                labels, value = labels.strip(), value.strip()
            else:
                name, _, value = line.partition(" ")
                labels, value = "", value.strip()
            if name not in counters and name not in GENERATION:
                continue
            try:
                number = float(value)
            except ValueError:
                return None
            if not math.isfinite(number):
                return None
            if name in counters:
                # Counters are token counts: a fractional value is a broken
                # sample, and a repeated label set would double-count it.
                if number != int(number) or labels in counters[name]:
                    return None
                counters[name][labels] = number
            else:
                if (name, labels) in generation:
                    return None
                generation[(name, labels)] = number
    return counters, generation


def labels_of(counters, name):
    return ", ".join(sorted(counters[name])) or "(no labels)"


before = load(sys.argv[1])
after = load(sys.argv[2])
if before is None or after is None:
    print("unjudged", "a /metrics sample carries a counter line that is not a finite integer", sep="\t")
    raise SystemExit(0)
c0, g0 = before
c1, g1 = after

if set(g0) != set(g1) or any(g0[key] != g1[key] for key in g0):
    print("unjudged", "a generation sample moved between the samples — the engine restarted", sep="\t")
    raise SystemExit(0)

for side, counters in (("before", c0), ("after", c1)):
    for label, name in (("drafted", DRAFT), ("accepted", ACCEPT)):
        if not counters[name]:
            print("unjudged", "the " + label + " counter family is missing " + side + " the sweep", sep="\t")
            raise SystemExit(0)
    if set(counters[DRAFT]) != set(counters[ACCEPT]):
        print("unjudged", "drafted and accepted series are unpaired " + side + " the sweep (drafted: "
              + labels_of(counters, DRAFT) + "; accepted: " + labels_of(counters, ACCEPT) + ")", sep="\t")
        raise SystemExit(0)

if set(c0[DRAFT]) != set(c1[DRAFT]) or set(c0[ACCEPT]) != set(c1[ACCEPT]):
    print("unjudged", "the spec-decode label set changed during the sweep (before: "
          + labels_of(c0, DRAFT) + "; after: " + labels_of(c1, DRAFT) + ")", sep="\t")
    raise SystemExit(0)

drafted = accepted = 0
for labels in sorted(c0[DRAFT]):
    if c1[DRAFT][labels] < c0[DRAFT][labels] or c1[ACCEPT][labels] < c0[ACCEPT][labels]:
        print("unjudged", "the counters went backwards for " + labels + " ("
              + str(int(c0[DRAFT][labels])) + "/" + str(int(c0[ACCEPT][labels])) + " -> "
              + str(int(c1[DRAFT][labels])) + "/" + str(int(c1[ACCEPT][labels]))
              + ") — engine restarted mid-sweep", sep="\t")
        raise SystemExit(0)
    drafted += int(c1[DRAFT][labels] - c0[DRAFT][labels])
    accepted += int(c1[ACCEPT][labels] - c0[ACCEPT][labels])

print("judged", drafted, accepted, len(c0[DRAFT]), sep="\t")
' "$1" "$2"
}

# The answer channel of the reply, parsed as JSON: choices[0].message.content
# (decoded), never reasoning_content. Prints one line:
#   content<TAB><decoded answer>  message.content is a string (may be empty)
#   reasoning                     content is not a string, but reasoning_content is
#   schema                        valid JSON, but not a chat-completions body
#   unreadable                    missing, not JSON, or not UTF-8
#   none                          no string content field at all
# A body this reader cannot parse is never judged: transport garbage and schema
# drift are "cannot judge", not evidence of a degenerate model.
reply_verdict() {
  python3 -S -c '
import json, sys

try:
    with open(sys.argv[1], "rb") as stream:
        body = json.load(stream)
except (OSError, ValueError):
    print("unreadable")
    raise SystemExit(0)

message = None
if isinstance(body, dict):
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        candidate = choices[0].get("message")
        if isinstance(candidate, dict):
            message = candidate
if message is None:
    print("schema")
    raise SystemExit(0)

content = message.get("content")
if isinstance(content, str):
    print("content", content, sep="\t")
elif isinstance(message.get("reasoning_content"), str) and message["reasoning_content"]:
    print("reasoning")
else:
    print("none")
' "$1"
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

# n copies of "hello", single-space separated, no trailing space. One printf
# with the format reused per argument — appending to a growing string made the
# 65536 rung quadratic in prompt length.
mk_ladder_prompt() {
  local n=$1 out
  out=$(printf 'hello %.0s' $(seq 1 "$n"))
  printf '%s' "${out% }"
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

if [ "$CANARY" != "0" ]; then
  metrics_snapshot "$tmpdir/metrics.before" || true
fi

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

# Degenerate-engine canary (see header). Runs before the tally so a broken
# engine is reported as broken, not as missing JIT coverage.
DEGENERATE=""
if [ "$CANARY" != "0" ]; then
  c1_body="$tmpdir/c1-1.resp.json"
  if [ "$(cat "$tmpdir/c1-1" 2>/dev/null)" = "ok" ]; then
    c1_line=$(reply_verdict "$c1_body")
    case "${c1_line%%$'\t'*}" in
      content)
        c1_reply=${c1_line#*$'\t'}
        if [ -z "$c1_reply" ]; then
          DEGENERATE="${DEGENERATE}; content: c1 returned an empty message on the answer channel"
        elif ! printf '%s' "$c1_reply" | grep -Eqi '(^|[^[:alnum:]])(ok|okay)([^[:alnum:]]|$)'; then
          # A standalone OK / OK(ay) token; substring matching would accept
          # "broken" or "look" and pass a garbage engine.
          DEGENERATE="${DEGENERATE}; content: c1 (temperature 0, \"Reply with OK.\") answered ${c1_reply:0:40}"
        fi
        ;;
      reasoning)
        echo "boot-shape-warmup: canary: c1 answered on reasoning_content with thinking off — content not judged" >&2
        ;;
      schema)
        echo "boot-shape-warmup: canary: c1 reply is not a chat-completions body — content not judged" >&2
        ;;
      unreadable)
        echo "boot-shape-warmup: canary: c1 reply body is not readable JSON — content not judged" >&2
        ;;
      *)
        echo "boot-shape-warmup: canary: c1 reply has no readable answer channel — content not judged" >&2
        ;;
    esac
  else
    echo "boot-shape-warmup: canary: c1 request did not complete — content not judged" >&2
  fi

  if [ ! -s "$tmpdir/metrics.before" ]; then
    # A whole-boot total is not a delta: a boot that drafted nothing before the
    # sweep would look like a sweep with no acceptance.
    echo "boot-shape-warmup: canary: no /metrics sample before the sweep — acceptance not judged (a delta needs a baseline)" >&2
  elif ! metrics_snapshot "$tmpdir/metrics.after"; then
    echo "boot-shape-warmup: canary: /metrics was unreadable after the sweep — acceptance not judged" >&2
  else
    delta=$(spec_delta "$tmpdir/metrics.before" "$tmpdir/metrics.after") || delta=""
    case "$delta" in
      judged*)
        read -r _ drafted accepted series <<<"$delta"
        echo "  canary: DFlash accepted ${accepted}/${drafted} drafted tokens during the sweep (${series} series)"
        if [ "$accepted" -eq 0 ]; then
          if [ "$drafted" -ge "$CANARY_MIN_DRAFTS" ]; then
            DEGENERATE="${DEGENERATE}; acceptance: 0/${drafted} drafted tokens accepted during the sweep"
          else
            echo "boot-shape-warmup: canary: only ${drafted} drafted tokens (< ${CANARY_MIN_DRAFTS}) — acceptance not judged" >&2
          fi
        fi
        ;;
      unjudged*)
        echo "boot-shape-warmup: canary: acceptance not judged (${delta#unjudged*$'\t'})" >&2
        ;;
      *)
        echo "boot-shape-warmup: canary: acceptance not judged (the /metrics samples could not be compared)" >&2
        ;;
    esac
  fi
fi

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

if [ -n "$DEGENERATE" ]; then
  echo "boot-shape-warmup: DEGENERATE ENGINE — ${DEGENERATE#; }. The engine answers /health but its output is not trustworthy; restart the kit (a later boot of the same image is usually fine). GLM53_WARMUP_CANARY=0 skips this check." >&2
  if [ "$ok_count" -lt "$total" ]; then
    echo "boot-shape-warmup: $((total - ok_count)) warmup request(s) also failed (unbounded arms that never stop are expected on a degenerate engine)" >&2
  fi
  exit 3
fi

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
