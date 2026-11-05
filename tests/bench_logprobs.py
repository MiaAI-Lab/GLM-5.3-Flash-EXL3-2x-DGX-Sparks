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
        nll, empty, unscorable = [], 0, 0
        for pos in pl[1:]:
            if not pos:
                empty += 1
                continue
            top1 = [v["logprob"] for v in pos.values() if v.get("rank") == 1]
            if not top1:
                unscorable += 1
                continue
            nll.append(-max(top1))
        mean = f"{sum(nll) / len(nll):.5f}" if nll else "unavailable"
        print(name, "positions", len(pl), "scored", len(nll), "empty", empty,
              "unscorable", unscorable, "mean top1 nll", mean)
    with open(out_path, "w") as fh:
        json.dump(res, fh)


def compare(a_path, b_path):
    with open(a_path) as fh:
        A = json.load(fh)
    with open(b_path) as fh:
        B = json.load(fh)
    for label, absent in (("B", sorted(set(A) - set(B))), ("A", sorted(set(B) - set(A)))):
        if absent:
            print(f"not compared (missing from {label}): {', '.join(absent)}")
    print("legend: mean_cond_KL = mean over each position's shared top-20 tokens, renormalised"
          " on that overlap; it is not a full-distribution KL")
    print("        excl = positions excluded (an empty side, or no shared token);"
          " unavailable = nothing comparable")
    print(f"{'text':12s} {'pos':>6} {'excl':>5} {'mean_cond_KL':>13} {'argmax_agree':>13}")
    for name in A:
        if name not in B:
            continue
        pa, pb = A[name], B[name]
        if len(pa) != len(pb):
            print(f"{name:12s} unavailable: capture lengths differ (A={len(pa)} B={len(pb)})")
            continue
        kl_sum, agree, n = 0.0, 0, 0
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
            kl_sum += sum((ax[k] / za) * math.log((ax[k] / za) / (by[k] / zb)) for k in keys)
            ta = max(x.items(), key=lambda kv: kv[1]["logprob"])[0]
            tb = max(y.items(), key=lambda kv: kv[1]["logprob"])[0]
            agree += ta == tb
            n += 1
        kl = f"{kl_sum / n:13.4f}" if n else f"{'unavailable':>13}"
        ag = f"{agree / n:13.3f}" if n else f"{'unavailable':>13}"
        print(f"{name:12s} {n:6d} {max(0, len(pa) - 1) - n:5d} {kl} {ag}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "capture":
        capture(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("usage: capture OUT | compare A B")
