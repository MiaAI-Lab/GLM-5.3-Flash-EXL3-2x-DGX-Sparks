#!/usr/bin/env python3
"""KLD / logprob panel for quantization A/B (dense-FP8 groups, KDA first).

Self-contained fixed text set; no external fixtures.

    GLM53_BENCH_BASE=http://127.0.0.1:8000 python3 tests/bench_kda_fp8.py capture --out kda_fp8.json
    python3 tests/bench_kda_fp8.py compare a.json b.json

Capture requires teacher-forced prompt_logprobs on the fixed text set.
Generation-logprob fallback receipts are not comparable after histories
diverge and are not accepted. Compare reports shared-top-k KL(A||B),
argmax agreement, and top-1 NLL delta (B - A; positive = less top-1 mass).
This is a screening panel, not full numerical qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

BASE = os.environ.get("GLM53_BENCH_BASE", "http://127.0.0.1:8888")
MODEL = "GLM-5.3-Flash-EXL3"

TEXTS = {
    "prose": (
        "Explain how a hash map handles collisions, covering separate chaining "
        "with linked lists, open addressing with linear probing, load factor "
        "thresholds, and amortized resizing cost. Be thorough and precise."
    ),
    "code": (
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
        "    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n"
        "    return quicksort(left) + middle + quicksort(right)\n"
        "Explain the average-case time complexity of this implementation."
    ),
    "arithmetic": (
        "A train travels 120 km in 2 hours, then 180 km in the next 3 hours. "
        "What is its average speed over the whole journey? Show each step."
    ),
    "structured": "Count from 1 to 40. Output only the numbers, separated by spaces.",
    "dense_tokens": (
        " ".join(
            f"Entry {i}: node NODE{i % 7} reported checksum CK-{i:06d} after "
            f"the maintenance window; temperature {40 + (i * 7) % 23} C, "
            f"fan duty {30 + (i * 13) % 60} percent, no faults."
            for i in range(40)
        )
    ),
}


def _post(path: str, body: dict, timeout: float = 600.0):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


def capture(out: str) -> int:
    res: dict = {"base": BASE, "model": MODEL, "texts": {}}
    for name, text in TEXTS.items():
        rec: dict = {"prompt_chars": len(text), "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}
        st, d = _post("/v1/completions", {
            "model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0,
            "echo": True, "logprobs": 1, "prompt_logprobs": 20,
        })
        pl = (d["choices"][0].get("prompt_logprobs") or []) if st == 200 else []
        if not isinstance(pl, list) or len(pl) < 2 or not any(pl):
            print(f"{name}: prompt_logprobs unavailable; capture is unqualified", file=sys.stderr)
            return 2
        rec["mode"] = "prompt_logprobs"
        rec["positions"] = pl
        n = len(rec["positions"])
        print(f"{name}: mode={rec['mode']} positions={n}", flush=True)
        res["texts"][name] = rec
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res))
    print("wrote", out)
    return 0


def _dist(pos: dict) -> tuple[dict, str]:
    if not isinstance(pos, dict) or not pos:
        raise ValueError("missing position distribution")
    probs = {}
    for token, value in pos.items():
        logprob = value.get("logprob") if isinstance(value, dict) else value
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
            raise ValueError("invalid log probability")
        if not math.isfinite(logprob) or logprob > 0:
            raise ValueError("invalid log probability")
        probs[token] = math.exp(logprob)
    return probs, max(probs, key=probs.get)


def compare(a_path: str, b_path: str) -> int:
    with open(a_path) as source:
        a_receipt = json.load(source)
    with open(b_path) as source:
        b_receipt = json.load(source)
    a, b = a_receipt.get("texts"), b_receipt.get("texts")
    if (
        not isinstance(a, dict) or not a or not isinstance(b, dict)
        or a.keys() != b.keys()
        or not a_receipt.get("model")
        or a_receipt.get("model") != b_receipt.get("model")
    ):
        print("UNQUALIFIED: empty or incompatible receipt sets")
        return 2
    bad, invalid = 0, False
    print(f"{'text':12s} {'mode':20s} {'pos':>5s} {'meanKL':>9s} {'argmax':>7s} {'dNLL':>9s}")
    for name in a:
        left, right = a[name], b[name]
        if not isinstance(left, dict) or not isinstance(right, dict):
            print(f"{name}: UNQUALIFIED record")
            invalid = True
            continue
        pa, pb = left.get("positions"), right.get("positions")
        fingerprint = left.get("prompt_sha256")
        if (
            left.get("mode") != "prompt_logprobs" or right.get("mode") != "prompt_logprobs"
            or not isinstance(fingerprint, str) or len(fingerprint) != 64
            or fingerprint != right.get("prompt_sha256")
            or not isinstance(pa, list) or not isinstance(pb, list)
            or len(pa) < 2 or len(pa) != len(pb)
        ):
            print(f"{name}: UNQUALIFIED context or position mismatch")
            invalid = True
            continue
        kls, dnll, agree = [], [], 0
        try:
            for x, y in zip(pa[1:], pb[1:]):
                ax, ta = _dist(x)
                by, tb = _dist(y)
                keys = ax.keys() & by.keys()
                if not keys or any(ax[k] <= 0 or by[k] <= 0 for k in keys):
                    raise ValueError("no usable shared probability support")
                za, zb = sum(ax[k] for k in keys), sum(by[k] for k in keys)
                kls.append(sum((ax[k] / za) * math.log((ax[k] / za) / (by[k] / zb)) for k in keys))
                agree += ta == tb
                dnll.append(math.log(max(ax.values())) - math.log(max(by.values())))
        except ValueError as exc:
            print(f"{name}: UNQUALIFIED: {exc}")
            invalid = True
            continue
        n = len(kls)
        mean_kl = sum(kls) / n
        print(f"{name:12s} {left['mode']:20s} {n:5d} {mean_kl:9.2e} "
              f"{agree / n:7.3f} {sum(dnll) / n:9.2e}")
        if mean_kl > 0.01 or agree / n < 0.99:
            bad += 1
    if invalid:
        print("UNQUALIFIED")
        return 2
    print("FLAG" if bad else "WITHIN KL/ARGMAX THRESHOLDS (shared top-k screening only)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "capture":
        return capture(args.out)
    return compare(args.a, args.b)


if __name__ == "__main__":
    raise SystemExit(main())
