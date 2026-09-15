#!/usr/bin/env bash
# scripts/spec-accept-gate.sh: exit-code contract against synthetic /metrics.
#
# A stub `curl` first on PATH serves an inert fixture file, so no server, GPU,
# container or network is involved. Each case pins a behaviour the gate's
# callers depend on being distinguishable:
#   1. >100 drafts and no per-position series  -> rc 2, named diagnostic
#      (before this, the run died rc 1 printing only drafts_total)
#   2. a matched per-position series with no numeric position label -> rc 2,
#      named diagnostic (before this, the run died rc 1 with no output, or
#      judged only the labelled part of the curve)
#   3. a missing or unreadable drafts denominator -> rc 2 (the missing family
#      used to fall through `|| echo 0` into the <100-drafts SKIP, reporting
#      "cannot judge" as "not enough traffic")
#   4. scientific counts (1.5e+03, with the optional sample timestamp) are
#      read as 1500, not truncated to their leading digit 1
#   5. position="0" absent while others exist  -> rc 2, named diagnostic
#   6. two label sets for one series           -> rc 2, no ratio emitted
#   7. pinned pos0 -> rc 1 FAIL, healthy decay -> rc 0 PASS, <100 drafts ->
#      rc 0 SKIP: the three verdicts stay distinct from the rc 2 diagnostics
#   8. default endpoint is loopback; an explicit argument still wins
#   9. a count that cannot be an exact integer for the bash compares (1e300)
#      or that overflows to +Inf (1e999) is refused rc 2 instead of passing
#      silently or FAILing on an infinite ratio, and a phantom
#      `notposition="0"` label is not a position-0 series
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
GATE="$HERE/../scripts/spec-accept-gate.sh"
[ -f "$GATE" ] || { echo "spec-accept-gate.sh not found" >&2; exit 1; }
fail=0

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir "$WORK/bin"
export GATE_FIXTURE="$WORK/metrics.txt" GATE_ARGV="$WORK/curl-argv.txt"
cat > "$WORK/bin/curl" <<'SHIM'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$GATE_ARGV"
[ -n "${GATE_CURL_FAIL:-}" ] && exit 7
cat "$GATE_FIXTURE"
SHIM
chmod +x "$WORK/bin/curl"
export PATH="$WORK/bin:$PATH"

# fixture <drafts_value> [<position>:<accepted_value> ...] -> $GATE_FIXTURE
# Values are written verbatim, so a fixture can carry the fractional,
# scientific and timestamped shapes a scrape target emits.
fixture() {
    { echo "# TYPE vllm:spec_decode_num_drafts_total counter"
      echo "vllm:spec_decode_num_drafts_total{model_name=\"glm\"} ${1}"
      echo "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter"
      shift
      for pair in "$@"; do
          echo "vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name=\"glm\",position=\"${pair%%:*}\"} ${pair##*:}"
      done
    } > "$GATE_FIXTURE"
}

out=""
rc=0
run() { : > "$GATE_ARGV"; out="$("$GATE" "$@" 2>&1)"; rc=$?; }
check() { # $1 description, $2 expected rc, $3 required output substring
    if [ "$rc" != "$2" ]; then
        echo "FAIL $1: rc $rc (want $2): $out"; fail=1; return
    fi
    case "$out" in
        *"$3"*) echo "ok   $1 (rc $rc)" ;;
        *) echo "FAIL $1: rc $rc but output lacks [$3]: $out"; fail=1 ;;
    esac
}
no_ratio() { # $1 description: the run must not have judged any position
    case "$out" in
        *"ratio="*) echo "FAIL $1: emitted a ratio anyway: $out"; fail=1 ;;
        *) echo "ok   $1: no ratio emitted" ;;
    esac
}

# --- missing, unlabelled and mismatched series ----------------------------
fixture 150.0
run
check "150 drafts, no per-position series" 2 "no accepted_tokens_per_pos_total series"

# Every matched series carrying no position label: the position list used to
# fail under pipefail and take the whole run down rc 1, printing nothing.
fixture 150.0
printf '%s\n' 'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="glm"} 7.75' >> "$GATE_FIXTURE"
run
check "unlabelled per-position series" 2 "without a numeric position label"
no_ratio "unlabelled per-position series"

# Labelled and unlabelled series mixed: judging the labelled subset alone would
# report a verdict for a curve that was never fully read.
fixture 150.0 0:150.0 1:20.0
printf '%s\n' 'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="glm"} 7.75' >> "$GATE_FIXTURE"
run
check "partly unlabelled per-position family" 2 "without a numeric position label"
no_ratio "partly unlabelled per-position family"

# `notposition="0"` ends in `position="0"`, so the bare substring match judged
# its value as the position-0 series and reported FAIL for a curve that has no
# position 0 in it at all.
fixture 150.0 1:20.0
printf '%s\n' 'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="glm",notposition="0"} 150.0' >> "$GATE_FIXTURE"
run
check "notposition label is not a position" 2 "without a numeric position label"
no_ratio "notposition label is not a position"

# A genuine position-0 series next to that phantom: pre-fix the bare substring
# count saw two position-0 series and reported the misleading rc 2 "ambiguous
# label sets"; the phantom series is unattributable, not a second curve.
fixture 100.0 0:80.0
printf '%s\n' 'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="glm",notposition="0"} 200.0' >> "$GATE_FIXTURE"
run
check "genuine pos0 beside notposition" 2 "without a numeric position label"
no_ratio "genuine pos0 beside notposition"

# The same phantom on a series that does carry a genuine label: pre-fix
# `notposition="4"` was collected as position 4 and judged as `pos4`.
fixture 100.0 0:80.0
printf '%s\n' 'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="glm",position="3",notposition="4"} 7.0' >> "$GATE_FIXTURE"
run
check "phantom position beside a genuine label" 0 "PASS: pos0 acceptance"
case "$out" in
    *"pos4:"*) echo "FAIL notposition=\"4\" was judged as a position: $out"; fail=1 ;;
    *) echo "ok   notposition=\"4\" was not judged as a position" ;;
esac

fixture 150.0 1:90.0 2:60.0
run
check "position 0 missing" 2 'position="0" is missing'

fixture 150.0 0:150.0 1:20.0
printf '%s\n' 'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="other",position="0"} 3.0' >> "$GATE_FIXTURE"
run
check "two label sets for position 0" 2 "ambiguous label sets"
no_ratio "two label sets for position 0"

fixture 150.0 0:120.0 1:90.0
printf '%s\n' 'vllm:spec_decode_num_drafts_total{model_name="other"} 7.0' >> "$GATE_FIXTURE"
run
check "two label sets for the denominator" 2 "ambiguous label sets"

# --- the drafts denominator is required, and must be readable -------------
# No drafts family at all: `|| echo 0` used to make this SKIP rc 0, i.e. the
# "cannot judge" case reported as "not enough traffic".
{ echo "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter"
  echo 'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="glm",position="0"} 150.0'
} > "$GATE_FIXTURE"
run
check "no drafts family" 2 "no spec_decode_num_drafts_total series"

fixture NaN 0:80.0
run
check "unreadable drafts value" 2 "unreadable spec_decode_num_drafts_total value"

fixture 150.0 0:NaN
run
check "unreadable position value" 2 "unreadable accepted_tokens_per_pos_total value"

# 1e300 is a finite double but not an exact integer: the printed 300-digit
# count silently compared false against the <100-drafts test, so the run went
# on and PASSed a curve it never measured. 1e999 overflows to +Inf, which made
# the position ratio infinite and reported FAIL instead of refusing the input.
fixture 1e300 0:80.0
run
check "out-of-range drafts value" 2 "unreadable spec_decode_num_drafts_total value"

fixture 150.0 0:1e999
run
check "position value overflowing to infinity" 2 "unreadable accepted_tokens_per_pos_total value"
no_ratio "position value overflowing to infinity"

fixture 150.0 0:1e300
run
check "out-of-range position value" 2 "unreadable accepted_tokens_per_pos_total value"
no_ratio "out-of-range position value"

# --- scientific counts are counts, not leading digits ---------------------
# 1.5e+03 drafts is 1500: a healthy run, not a <100-drafts SKIP. The samples
# carry the optional trailing timestamp a scrape target may append.
fixture "1.5e+03 1757900000000" "0:1.2e+03 1757900000000" "1:9e+02 1757900000000"
run
check "scientific counts" 0 "PASS: pos0 acceptance"
case "$out" in
    *"drafts_total=1500"*) echo "ok   1.5e+03 drafts read as 1500" ;;
    *) echo "FAIL scientific drafts misread: $out"; fail=1 ;;
esac
case "$out" in
    *"ratio=0.8000"*) echo "ok   1.2e+03/1.5e+03 accepted ratio is 0.8000" ;;
    *) echo "FAIL scientific accepted misread: $out"; fail=1 ;;
esac

# --- the PASS / FAIL / SKIP verdicts stay distinct ------------------------
fixture 150.0 0:150.0 1:150.0
run
check "pinned pos0" 1 "FAIL: pos0 acceptance"

fixture 100.0 0:80.0 1:60.0 2:40.0
run
check "healthy decay at exactly 100 drafts" 0 "PASS: pos0 acceptance"

fixture 99.0 0:70.0
run
check "99 drafts" 0 "SKIP: only 99 drafts"

# --- endpoint -------------------------------------------------------------
fixture 100.0 0:80.0 1:60.0
run
check "loopback default verdict" 0 "PASS"
case "$(cat "$GATE_ARGV")" in
    *"http://127.0.0.1:8888/metrics"*) echo "ok   default endpoint is loopback" ;;
    *) echo "FAIL default endpoint: $(cat "$GATE_ARGV")"; fail=1 ;;
esac
case "$(cat "$GATE_ARGV")" in
    *"192.168."*) echo "FAIL default endpoint is still site-specific: $(cat "$GATE_ARGV")"; fail=1 ;;
    *) echo "ok   no site-specific host requested" ;;
esac

run "http://127.0.0.1:9999"
case "$(cat "$GATE_ARGV")" in
    *"http://127.0.0.1:9999/metrics"*) echo "ok   explicit argument overrides the default" ;;
    *) echo "FAIL explicit endpoint: $(cat "$GATE_ARGV")"; fail=1 ;;
esac

# --- unreachable endpoint -------------------------------------------------
export GATE_CURL_FAIL=1
run
unset GATE_CURL_FAIL
check "unreachable endpoint" 2 "cannot read http://127.0.0.1:8888/metrics"

[ "$fail" = 0 ] && echo "spec-accept-gate tests: PASS"
exit $fail
