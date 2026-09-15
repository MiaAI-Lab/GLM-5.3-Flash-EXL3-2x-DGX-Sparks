#!/usr/bin/env bash
# A/B/A decode bench around a GB10 SM clock lock.
#
# GB10 reports `Supported Clocks: N/A` (no --applications-clocks), so the only
# lever is `nvidia-smi -lgc MIN,MAX`. Stock max SM clock is 3003 MHz; the
# hypothesis under test is that capping the boost ceiling stops thermal
# oscillation and raises *sustained* decode throughput.
#
# Decode is the axis E3 does not touch (the grouped MoE tier never runs on
# decode-sized steps), so this is measured with tests/bench_decode.py
# --structured: warmed, temp 0, thinking off, count-1-to-200, median tok/s.
#
# TP=2 over CX7 means one slow rank stalls the collective: lock BOTH ranks or
# neither. Both nodes need a sudo password here, so `lock`/`unlock` are run by
# hand and `phase` needs no privileges.
#
#   ./tests/bench_clock_lock.sh state
#   ./tests/bench_clock_lock.sh phase a1 logs/clocklock-<TS>
#   ./tests/bench_clock_lock.sh lock 2200      # prompts, both ranks
#   ./tests/bench_clock_lock.sh phase b  logs/clocklock-<TS>
#   ./tests/bench_clock_lock.sh unlock         # prompts, both ranks
#   ./tests/bench_clock_lock.sh phase a2 logs/clocklock-<TS>
#   ./tests/bench_clock_lock.sh report logs/clocklock-<TS>
set -euo pipefail

cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && . ./.env && set +a || true
WORKER="${WORKER_USER:-zurih}@${WORKER_IP:-10.0.0.2}"
BASE="http://127.0.0.1:${PORT:-8888}"
QUERY='clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu'

die() { echo "error: $*" >&2; exit 1; }

gpu_state() {
    echo "head:   $(nvidia-smi --query-gpu="$QUERY" --format=csv,noheader)"
    echo "worker: $(ssh -o BatchMode=yes -o ConnectTimeout=8 "$WORKER" \
        "nvidia-smi --query-gpu=$QUERY --format=csv,noheader" 2>/dev/null || echo unreachable)"
}

# Telemetry for the duration of a phase, both ranks, 1 Hz.
start_telemetry() {
    local out=$1
    nvidia-smi --query-gpu="$QUERY" --format=csv -l 1 > "$out/head.csv" 2>&1 &
    echo $! > "$out/.head.pid"
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$WORKER" \
        "nvidia-smi --query-gpu=$QUERY --format=csv -l 1" > "$out/worker.csv" 2>&1 &
    echo $! > "$out/.worker.pid"
}

stop_telemetry() {
    local out=$1 p
    for r in head worker; do
        if [[ -f "$out/.$r.pid" ]]; then
            p=$(cat "$out/.$r.pid"); kill "$p" 2>/dev/null || true
            rm -f "$out/.$r.pid"
        fi
    done
    # the remote nvidia-smi survives the ssh client; reap it
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$WORKER" \
        "pkill -f 'nvidia-smi --query-gpu' || true" 2>/dev/null || true
}

case "${1:-}" in
state)
    gpu_state
    ;;

phase)
    name="${2:?phase name}"; outdir="${3:?outdir}"
    [[ -d $outdir ]] || die "no such dir: $outdir"
    d="$outdir/$name"; mkdir -p "$d"
    code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$BASE/health" || true)
    [[ $code == 200 ]] || die "server not healthy (/health -> $code)"
    gpu_state | tee "$d/gpu-before.txt"
    trap 'stop_telemetry "$d"' EXIT
    start_telemetry "$d"
    python3 tests/bench_decode.py --phase "$name" --out "$d/decode.json" \
        --structured --runs 3 2>&1 | tee "$d/bench.log"
    stop_telemetry "$d"; trap - EXIT
    gpu_state | tee "$d/gpu-after.txt"
    echo "[$name] receipts -> $d"
    ;;

lock)
    mhz="${2:?MHz}"
    echo "locking both ranks to ${mhz},${mhz} MHz (sudo will prompt twice)"
    sudo nvidia-smi -lgc "${mhz},${mhz}"
    ssh -t "$WORKER" "sudo nvidia-smi -lgc ${mhz},${mhz}"
    gpu_state
    ;;

unlock)
    echo "resetting both ranks to stock clocks (sudo will prompt twice)"
    sudo nvidia-smi -rgc
    ssh -t "$WORKER" "sudo nvidia-smi -rgc"
    gpu_state
    ;;

report)
    outdir="${2:?outdir}"
    python3 - "$outdir" <<'PY'
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])


def med_power(csv: Path):
    """Median power/temp/SM clock over the phase, GPU-busy samples only."""
    if not csv.exists():
        return None
    pw, tp, sm = [], [], []
    for line in csv.read_text().splitlines()[1:]:
        f = [x.strip() for x in line.split(",")]
        if len(f) < 5:
            continue
        try:
            util = float(f[4].split()[0])
            if util < 5:          # idle between runs is not the workload
                continue
            sm.append(float(f[0].split()[0]))
            pw.append(float(f[2].split()[0]))
            tp.append(float(f[3].split()[0]))
        except (ValueError, IndexError):
            continue
    if not pw:
        return None
    return {
        "sm_mhz": round(statistics.median(sm)),
        "power_w": round(statistics.median(pw), 1),
        "temp_c": round(statistics.median(tp), 1),
        "peak_temp_c": round(max(tp), 1),
        "samples": len(pw),
    }


rows = {}
for name in ("a1", "b", "a2"):
    d = root / name
    j = d / "decode.json"
    if not j.exists():
        continue
    rec = json.loads(j.read_text())
    runs = [r for r in rec.get("runs", []) if r.get("tok_s")]
    tps = [r["tok_s"] for r in runs]
    # bench_decode records full `text` per run; without it there is nothing to
    # compare, and a set of Nones would collapse to a false "identical".
    texts = [r["text"] for r in runs if r.get("text")]
    rows[name] = {
        "decode_tok_s_median": round(statistics.median(tps), 2) if tps else None,
        "decode_tok_s_runs": [round(t, 2) for t in tps],
        "texts_identical": len(set(texts)) == 1 if len(texts) == len(runs) and runs else None,
        "head": med_power(d / "head.csv"),
        "worker": med_power(d / "worker.csv"),
    }

print(json.dumps(rows, indent=2))

a1, b, a2 = (rows.get(k, {}).get("decode_tok_s_median") for k in ("a1", "b", "a2"))
if a1 and a2:
    drift = abs(a1 - a2) / ((a1 + a2) / 2) * 100
    print(f"\nunlocked drift a1 vs a2 : {drift:.2f}%  ({a1} -> {a2})")
    if drift > 1.0:
        print("  A/B/A UNSTABLE (>1%): the delta below is not trustworthy")
    if b:
        base = (a1 + a2) / 2
        print(f"locked vs unlocked mean : {(b - base) / base * 100:+.2f}%  ({base:.2f} -> {b})")
PY
    ;;

*)
    sed -n '2,25p' "$0"; exit 1
    ;;
esac
