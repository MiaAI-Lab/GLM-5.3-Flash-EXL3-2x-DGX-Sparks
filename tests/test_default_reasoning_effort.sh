#!/usr/bin/env bash
# GLM53_DEFAULT_REASONING_EFFORT: launch-time enum guard + the serve-args flag.
#
# Two independent claims are checked against start.sh's own text, not a replica:
#   1. validate_numeric_config accepts exactly "" | low | high | max and rejects
#      everything else (notably `medium`, which files/chat_template.jinja does
#      NOT recognize and would silently render as Max).
#   2. BOTH inner scripts (head rank 0 and headless worker rank 1) append
#      --default-chat-template-kwargs exactly once, with space-free JSON, when
#      the knob is set -- and append nothing at all when it is empty.
# Values are passed through the environment so adversarial strings are
# exercised safely rather than interpolated into source.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
START="$HERE/../start.sh"
[ -f "$START" ] || { echo "start.sh not found" >&2; exit 1; }
fail=0

# ---------------------------------------------------------------- guard ----
guard="$(sed -n '/^# GLM53 numeric config guard (begin)$/,/^# GLM53 numeric config guard (end)$/p' "$START")"
[ -n "$guard" ] || { echo "numeric config guard block not found" >&2; exit 1; }
printf '%s\n' "$guard" > "/tmp/_effort_guard.$$"

guard_rc() { # reads the value from the environment; echoes validate_numeric_config's rc
    GLM53_DEFAULT_REASONING_EFFORT="$1" bash -c '
        source "/tmp/_effort_guard.'$$'"
        GPU_MEM_UTIL=0.87 MAX_MODEL_LEN=1000000 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048
        validate_numeric_config' >/dev/null 2>&1
    echo $?
}

check_guard() {
    got="$(guard_rc "$1")"
    if [ "$got" = "$2" ]; then echo "ok   guard [$1] -> rc $got"
    else echo "FAIL guard [$1] -> rc $got want $2"; fail=1; fi
}

check_guard ""       0
check_guard "low"    0
check_guard "high"   0
check_guard "max"    0
check_guard "medium" 2
check_guard "High"   2
check_guard "MAX"    2
check_guard " high"  2
check_guard "high "  2
check_guard "junk"   2
check_guard "1"      2
check_guard 'high;id' 2

# an UNSET knob must also pass (the guard reads ${VAR-}, not $VAR under set -u)
if bash -c 'set -u; source "/tmp/_effort_guard.'$$'"
    GPU_MEM_UTIL=0.87 MAX_MODEL_LEN=1000000 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048
    validate_numeric_config' >/dev/null 2>&1; then
    echo "ok   guard [<unset>] -> rc 0"
else
    echo "FAIL guard [<unset>] must pass with the var unset"; fail=1
fi

# the rejection message must name the knob and the four legal values
msg="$(GLM53_DEFAULT_REASONING_EFFORT=medium bash -c '
    source "/tmp/_effort_guard.'$$'"
    GPU_MEM_UTIL=0.87 MAX_MODEL_LEN=1000000 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048
    validate_numeric_config' 2>&1 >/dev/null || true)"
case "$msg" in
    *GLM53_DEFAULT_REASONING_EFFORT*low*high*max*) echo "ok   guard error names knob and enum" ;;
    *) echo "FAIL guard error text: [$msg]"; fail=1 ;;
esac

# ----------------------------------------------------------- serve args ----
# Slice each inner script out of its quoted heredoc, then keep only the ARGS
# construction (ARGS=( ... up to the config.json existence check).
args_block() { # $1 = HEAD_SCRIPT | WORKER_SCRIPT
    awk -v var="$1" '
        index($0, "cat > \"$" var "\" <<") { inblk = 1; next }
        inblk && $0 == "EOF" { exit }
        inblk { print }
    ' "$START" | sed -n '/^ARGS=(/,/^\[ -f "${MODEL_DIR}\/config.json"/p' | sed '$d'
}

for var in HEAD_SCRIPT WORKER_SCRIPT; do
    args_block "$var" > "/tmp/_effort_args_${var}.$$"
    if ! [ -s "/tmp/_effort_args_${var}.$$" ]; then
        echo "FAIL could not slice ARGS block for $var"; fail=1; continue
    fi
    if ! grep -q -- '--default-chat-template-kwargs' "/tmp/_effort_args_${var}.$$"; then
        echo "FAIL $var ARGS block has no --default-chat-template-kwargs"; fail=1
    fi
done

emit() { # $1 = HEAD_SCRIPT|WORKER_SCRIPT, $2 = knob value; prints argv one per line
    GLM53_DEFAULT_REASONING_EFFORT="$2" bash -c '
        say() { :; }
        SERVED_MODEL_NAME=m PORT=8888 TP=2 NNODES=2 HEAD_IP=10.0.0.1 MASTER_PORT=29500
        ENFORCE_EAGER=0 QUANTIZATION=none MAX_MODEL_LEN=1000000 GPU_MEM_UTIL=0.87
        MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048 KV_CACHE_DTYPE=fp8
        SPEC_METHOD=none MTP_TOKENS=0 CHAT_TEMPLATE= LANGUAGE_MODEL_ONLY=0
        LIMIT_MM= SKIP_MM_PROFILING=0 EXTRA_ARGS=
        source "/tmp/_effort_args_'"$1"'.'$$'"
        printf "%s\n" "${ARGS[@]}"' 2>/dev/null
}

check_emit() { # $1 rank var, $2 value, $3 expected JSON ("" = flag must be absent)
    local out n json
    out="$(emit "$1" "$2")"
    n="$(printf '%s\n' "$out" | grep -c -- '^--default-chat-template-kwargs$' || true)"
    json="$(printf '%s\n' "$out" | grep -A1 -- '^--default-chat-template-kwargs$' | sed -n '2p')"
    if [ -z "$3" ]; then
        if [ "$n" = "0" ]; then echo "ok   $1 [$2] -> no flag"
        else echo "FAIL $1 [$2] -> emitted $n flag(s): [$json]"; fail=1; fi
        return
    fi
    if [ "$n" != "1" ]; then
        echo "FAIL $1 [$2] -> flag emitted $n times (want exactly 1)"; fail=1; return
    fi
    if [ "$json" != "$3" ]; then
        echo "FAIL $1 [$2] -> JSON [$json] want [$3]"; fail=1; return
    fi
    case "$json" in
        *" "*) echo "FAIL $1 [$2] -> JSON contains a space: [$json]"; fail=1; return ;;
    esac
    echo "ok   $1 [$2] -> $json"
}

for var in HEAD_SCRIPT WORKER_SCRIPT; do
    check_emit "$var" ""     ""
    check_emit "$var" "low"  '{"reasoning_effort":"low"}'
    check_emit "$var" "high" '{"reasoning_effort":"high"}'
    check_emit "$var" "max"  '{"reasoning_effort":"max"}'
done

# the two ranks must build the same flag from the same knob
h="$(emit HEAD_SCRIPT high | grep -A1 -- '^--default-chat-template-kwargs$' | sed -n '2p')"
w="$(emit WORKER_SCRIPT high | grep -A1 -- '^--default-chat-template-kwargs$' | sed -n '2p')"
if [ "$h" = "$w" ] && [ -n "$h" ]; then echo "ok   both ranks build the identical flag"
else echo "FAIL rank flags differ: head [$h] worker [$w]"; fail=1; fi

# ------------------------------------------------------------- wiring ------
launcher="$(cat "$START")"
case "$launcher" in
    *'_cli_default_effort_set="${GLM53_DEFAULT_REASONING_EFFORT+1}"'*)
        echo "ok   caller capture is setness-aware" ;;
    *) echo "FAIL caller capture is not setness-aware"; fail=1 ;;
esac
case "$launcher" in
    *'[ -n "${_cli_default_effort_set}" ] && GLM53_DEFAULT_REASONING_EFFORT="$_cli_default_effort"'*)
        echo "ok   caller value wins over .env" ;;
    *) echo "FAIL caller override line missing"; fail=1 ;;
esac
case "$launcher" in
    *'-e "GLM53_DEFAULT_REASONING_EFFORT=${GLM53_DEFAULT_REASONING_EFFORT-}"'*)
        echo "ok   knob reaches both containers via nccl_common" ;;
    *) echo "FAIL knob is not exported into the containers"; fail=1 ;;
esac
# default must stay empty: this PR changes no behaviour until an operator opts in
case "$launcher" in
    *'GLM53_DEFAULT_REASONING_EFFORT="${GLM53_DEFAULT_REASONING_EFFORT-}"'*)
        echo "ok   launcher default is empty (unchanged behaviour)" ;;
    *) echo "FAIL launcher default is not empty"; fail=1 ;;
esac

rm -f "/tmp/_effort_guard.$$" "/tmp/_effort_args_HEAD_SCRIPT.$$" "/tmp/_effort_args_WORKER_SCRIPT.$$"
[ "$fail" = 0 ] && echo "default reasoning effort tests: PASS"
exit $fail
