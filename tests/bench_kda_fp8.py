#!/usr/bin/env python3
"""KLD / logprob panel for quantization A/B (dense-FP8 groups, KDA first).

Self-contained fixed text set; no external fixtures.

    GLM53_BENCH_BASE=http://127.0.0.1:8000 python3 tests/bench_kda_fp8.py capture --out kda_fp8.json
    python3 tests/bench_kda_fp8.py compare a.json b.json

Capture tries prompt_logprobs (echo, max_tokens=1) and falls back to
generation logprobs (max_tokens=64, logprobs=5) when the server does not
return prompt distributions. Compare reports, per text: positions, mean
KL(A||B) over the shared top-k support (nats), argmax agreement, and the
top-1 NLL delta. All sourcing is temperature-0 deterministic.
"""

from __future__ import annotations

import argparse
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
        rec: dict = {"prompt_chars": len(text)}
        st, d = _post("/v1/completions", {
            "model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0,
            "echo": True, "logprobs": 1, "prompt_logprobs": 20,
        })
        pl = (d["choices"][0].get("prompt_logprobs") or []) if st == 200 else []
        if pl and any(pl):
            rec["mode"] = "prompt_logprobs"
            rec["positions"] = pl
        else:
            st2, d2 = _post("/v1/completions", {
                "model": MODEL, "prompt": text, "max_tokens": 64,
                "temperature": 0, "logprobs": 5,
            })
            lg = (d2["choices"][0].get("logprobs") or {}) if st2 == 200 else {}
            toks = lg.get("tokens") or []
            tops = lg.get("top_logprobs") or []
            rec["mode"] = "generation_logprobs"
            rec["positions"] = [
                {"rank1": {"token": t, "logprob": 0.0}, **(tp or {})}
                for t, tp in zip(toks, tops)
            ]
            rec["generated_text"] = "".join(toks)
        n = len(rec["positions"])
        print(f"{name}: mode={rec['mode']} positions={n}", flush=True)
        res["texts"][name] = rec
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res))
    print("wrote", out)
    return 0


def _dist(pos: dict) -> tuple[dict, str]:
    probs = {}
    for k, v in pos.items():
        if isinstance(v, dict) and "logprob" in v:
            probs[k] = math.exp(v["logprob"])
        elif isinstance(v, dict) and "token" in v:
            pass
    top = max(probs.items(), key=lambda kv: kv[1])[0] if probs else ""
    return probs, top


def compare(a_path: str, b_path: str) -> int:
    a = json.load(open(a_path))["texts"]
    b = json.load(open(b_path))["texts"]
    bad = 0
    print(f"{'text':12s} {'mode':20s} {'pos':>5s} {'meanKL':>9s} {'argmax':>7s} {'dNLL':>9s}")
    for name in a:
        pa, pb = a[name]["positions"], b[name]["positions"]
        kls, agree, n, dnll = [], 0, 0, []
        for x, y in zip(pa[1:], pb[1:]):
            if not x or not y:
                continue
            ax, ta = _dist(x)
            by, tb = _dist(y)
            keys = set(ax) & set(by)
            if not keys:
                continue
            za, zb = sum(ax[k] for k in keys), sum(by[k] for k in keys)
            if za <= 0 or zb <= 0:
                continue
            kl = sum((ax[k] / za) * math.log((ax[k] / za) / (by[k] / zb)) for k in keys)
            kls.append(kl)
            agree += ta == tb
            n += 1
            la = max((v.get("logprob", -1e9) for v in x.values() if isinstance(v, dict)), default=0.0)
            lb = max((v.get("logprob", -1e9) for v in y.values() if isinstance(v, dict)), default=0.0)
            dnll.append(-la - -lb)
        mean_kl = sum(kls) / max(1, len(kls))
        print(f"{name:12s} {a[name]['mode']:20s} {n:5d} {mean_kl:9.2e} "
              f"{agree / max(1, n):7.3f} {sum(dnll) / max(1, len(dnll)):9.2e}")
        if mean_kl > 0.01 or (n and agree / n < 0.99):
            bad += 1
    print("FLAG" if bad else "PANEL CLEAN")
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
