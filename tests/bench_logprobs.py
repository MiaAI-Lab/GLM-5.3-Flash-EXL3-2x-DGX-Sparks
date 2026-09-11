#!/usr/bin/env python3
"""Capture prompt logprobs (top-20) for fixed texts; compare two captures.

Usage:
  python3 bench_logprobs.py capture OUT.json
  python3 bench_logprobs.py compare A.json B.json
"""
import json
import math
import sys
import urllib.request

BASE = "http://127.0.0.1:8888"
MODEL = "GLM-5.3-Flash-EXL3"


def texts():
    out = {}
    out["prose_mla"] = (
        "Multi-head latent attention compresses the key and value projections into a "
        "single low-rank latent vector. During inference the KV cache therefore stores "
        "the compressed latent plus a small decoupled rotary component, which cuts memory "
        "traffic and raises the maximum batch size. The sparse indexer then selects the "
        "top-k most relevant blocks for each query so that attention cost grows sublinearly "
        "with context length. Operators who deploy such a model must balance the page size "
        "of the cache against the alignment requirements of every hybrid group, because a "
        "single misaligned group forces the whole prefix lookup back to whole-page "
        "granularity."
    ) * 4
    out["log_rows"] = "\n".join(
        f"Entry {i:04d}: node NODE{i % 7} reported checksum CK-{i * 37:06d} after the "
        f"maintenance window; the operator logged temperature {40 + (i * 7) % 23} C, fan "
        f"duty {30 + (i * 13) % 60} percent, and no faults."
        for i in range(60)
    )
    out["code"] = (
        "def cumulative_sum(values):\n"
        "    total = 0\n"
        "    out = []\n"
        "    for value in values:\n"
        "        total += value\n"
        "        out.append(total)\n"
        "    return out\n\n"
        "def bucketize(scores, edges):\n"
        "    buckets = [[] for _ in range(len(edges) + 1)]\n"
        "    for score in scores:\n"
        "        for idx, edge in enumerate(edges):\n"
        "            if score < edge:\n"
        "                buckets[idx].append(score)\n"
        "                break\n"
        "        else:\n"
        "            buckets[-1].append(score)\n"
        "    return buckets\n"
    ) * 3
    out["mixed"] = (
        "Question: explain why the sky is blue and why sunsets are red, then compute "
        "the arithmetic mean of 17, 23, 41, and 99. Answer in numbered steps. "
        "Step 1: Rayleigh scattering scales as one over wavelength to the fourth power, "
        "so short blue wavelengths scatter far more strongly than long red ones. "
        "Step 2: near the horizon the optical path is longer and blue light is removed, "
        "leaving the red end of the spectrum. "
        "Step 3: the arithmetic mean is (17 + 23 + 41 + 99) / 4 = 180 / 4 = 45."
    ) * 3
    return out


def capture(out_path):
    res = {}
    for name, text in texts().items():
        body = {
            "model": MODEL,
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0,
            "echo": True,
            "logprobs": 1,
            "prompt_logprobs": 20,
        }
        req = urllib.request.Request(
            BASE + "/v1/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=600) as resp:
            d = json.loads(resp.read().decode())
        pl = d["choices"][0].get("prompt_logprobs") or []
        res[name] = pl
        nll = [
            -max(v["logprob"] for v in pos.values() if v.get("rank") == 1)
            for pos in pl[1:] if pos
        ]
        print(name, "positions", len(pl), "mean top1 nll", round(sum(nll) / max(1, len(nll)), 5))
    json.dump(res, open(out_path, "w"))


def compare(a_path, b_path):
    A = json.load(open(a_path))
    B = json.load(open(b_path))
    print(f"{'text':12s} {'pos':>6} {'mean_KL':>10} {'argmax_agree':>13}")
    for name in A:
        pa, pb = A[name], B[name]
        kls, agree, n = [], 0, 0
        for x, y in zip(pa[1:], pb[1:]):
            if not x or not y:
                continue
            ax = {k: math.exp(v["logprob"]) for k, v in x.items()}
            by = {k: math.exp(v["logprob"]) for k, v in y.items()}
            keys = set(ax) & set(by)
            if not keys:
                continue
            za = sum(ax[k] for k in keys)
            zb = sum(by[k] for k in keys)
            kls.append(sum((ax[k] / za) * math.log((ax[k] / za) / (by[k] / zb)) for k in keys))
            ta = max(x.items(), key=lambda kv: kv[1]["logprob"])[0]
            tb = max(y.items(), key=lambda kv: kv[1]["logprob"])[0]
            agree += ta == tb
            n += 1
        print(f"{name:12s} {n:6d} {sum(kls)/max(1,len(kls)):10.4f} {agree/max(1,n):13.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(sys.argv[2])
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("usage: capture OUT | compare A B")
