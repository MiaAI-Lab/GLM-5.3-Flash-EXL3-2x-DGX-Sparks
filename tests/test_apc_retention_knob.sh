#!/usr/bin/env bash
# Unit test for the GLM53_APC_RETENTION_INTERVAL knob logic in start.sh (no server needed).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
fn="$(sed -n '/^# --- glm53 apc-retention knob/,/^# --- end glm53 apc-retention knob/p' "$HERE/../start.sh")"
[ -n "$fn" ] || { echo "knob block not found in start.sh" >&2; exit 1; }
eval "$fn"
fail=0
check() { # $1 input, $2 expected exported value or "UNSET" or "ERR"
    unset VLLM_PREFIX_CACHE_RETENTION_INTERVAL
    if glm53_apc_retention_env "$1" 2>/dev/null; then got="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL-UNSET}"; else got=ERR; fi
    if [ "$got" = "$2" ]; then echo "ok   '$1' -> $got"; else echo "FAIL '$1' -> $got (want $2)"; fail=1; fi
}
check ""        UNSET
check 14336     14336
check 3584      3584
check 0003584   3584
check 7168      7168
check 0         ERR
check 08        ERR
check 3585      ERR
check -3584     ERR
check abc       ERR
check "14336 "  ERR
check 1003520   ERR
check 1000000   ERR
check 996352    996352
check 18446744073709565952 ERR
check 00000000014336 ERR
exit $fail
