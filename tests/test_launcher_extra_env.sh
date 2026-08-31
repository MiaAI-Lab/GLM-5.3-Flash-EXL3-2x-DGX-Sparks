#!/usr/bin/env bash
# GLM53_EXTRA_ENV: NAME=VALUE entries are appended to nccl_common as -e pairs; bad names/values/owned names abort; only names logged.
# Values are passed through the environment (not interpolated into source) so adversarial strings are exercised safely.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
blk="$(sed -n '/^    # Extra container env for diagnostics/,/^    fi$/p' "$HERE/../start.sh")"
[ -n "$blk" ] || { echo "extra-env block not found" >&2; exit 1; }
printf '%s\n' "$blk" | sed 's/^    //; s/^local _kv _name _value _names=""$/_names=""/' > /tmp/_extra_env_blk.$$
run() { # reads GLM53_EXTRA_ENV from the environment; prints the joined -e list, the log line, or ERR
  GLM53_EXTRA_ENV="$1" bash -c 'die() { echo "DIE: $*" >&2; exit 2; }; log() { echo "LOG:$*"; }; nccl_common=(); source /tmp/_extra_env_blk.'$$'; printf "%s " "${nccl_common[@]}"; echo' 2>/dev/null || echo ERR
}
fail=0
check() { got="$(run "$1" | tr '\n' '|' | sed 's/[[:space:]]*|/|/g; s/|$//')"; if [ "$got" = "$2" ]; then echo "ok   [$1]"; else echo "FAIL [$1] -> [$got] want [$2]"; fail=1; fi; }
check "" ""
check "VLLM_DEBUG_WORKSPACE=1" "LOG:extra container env (both ranks): VLLM_DEBUG_WORKSPACE|-e VLLM_DEBUG_WORKSPACE=1"
check "A=1 B_2=x" "LOG:extra container env (both ranks): A B_2|-e A=1 -e B_2=x"
check "A=a=b" "LOG:extra container env (both ranks): A|-e A=a=b"
check "A=" "LOG:extra container env (both ranks): A|-e A="
check "A=/tmp/x.log:1" "LOG:extra container env (both ranks): A|-e A=/tmp/x.log:1"
check "bad-name=1" "ERR"
check "novalue" "ERR"
check "1ABC=2" "ERR"
check 'A=x;id' "ERR"
check 'A=$(id)' "ERR"
check 'A=a b' "ERR"
check 'A=*' "ERR"
check 'A="q"' "ERR"
check "NCCL_DEBUG=INFO" "ERR"
check "VLLM_API_KEY=leak" "ERR"
check "GLM53_MIXED_PREFILL_CHUNK=0" "ERR"
check "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0" "ERR"
# secret redaction: the value must not appear in the log line
out="$(run "VLLM_DEBUG_WORKSPACE=s3cr3t" | head -1)"; case "$out" in *s3cr3t*) echo "FAIL log leaks value"; fail=1;; *) echo "ok   [log redacts values]";; esac
rm -f /tmp/_extra_env_blk.$$
exit $fail
