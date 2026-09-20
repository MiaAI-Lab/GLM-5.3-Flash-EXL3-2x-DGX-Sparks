#!/usr/bin/env python3
"""Prefix-cache receipt for the NVMe tier: cold -> GPU-hit -> (restart) -> NVMe-hit.

Sends the same long prompt three times and reports TTFT / total time and
``usage.prompt_tokens_details.cached_tokens`` (needs
``--enable-prompt-tokens-details``, which OFFLOAD_NVME=1 adds). Run once before
and once after ``./start.sh restart``; the post-restart first call is the
NVMe restore (GPU prefix cache is empty after a restart).

    python3 tests/bench_nvme_restore.py --tokens 40000 --tag before
    ./start.sh restart
    python3 tests/bench_nvme_restore.py --tokens 40000 --tag after --once
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request


def build_prompt(n_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
             "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
             "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
    # ~1 token per short word; deterministic per seed so hashes match across runs.
    body = " ".join(rng.choice(words) for _ in range(n_tokens))
    return f"Here is a long transcript:\n{body}\n\nReply with the single word OK."


def one(url: str, model: str, prompt: str, key: str | None) -> dict:
    req = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    hdr = {"Content-Type": "application/json"}
    if key:
        hdr["Authorization"] = f"Bearer {key}"
    t0 = time.perf_counter()
    r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(req).encode(), headers=hdr)
    ttft = None
    usage = None
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            obj = json.loads(payload)
            if ttft is None and obj.get("choices") and obj["choices"][0].get("delta", {}).get("content"):
                ttft = time.perf_counter() - t0
            if obj.get("usage"):
                usage = obj["usage"]
    total = time.perf_counter() - t0
    cached = (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens")
    return {"ttft_s": ttft, "total_s": total, "prompt_tokens": (usage or {}).get("prompt_tokens"), "cached_tokens": cached}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "GLM-5.3-Flash-EXL3"))
    ap.add_argument("--tokens", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--tag", default="")
    ap.add_argument("--once", action="store_true", help="single call (post-restart NVMe restore)")
    ap.add_argument("--out", default="results/nvme-restore.jsonl")
    a = ap.parse_args()
    key = os.environ.get("VLLM_API_KEY")
    prompt = build_prompt(a.tokens, a.seed)
    labels = ["restore"] if a.once else ["cold", "gpu-hit", "gpu-hit-2"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for label in labels:
        res = one(a.url, a.model, prompt, key)
        res.update({"label": label, "tag": a.tag, "tokens_requested": a.tokens, "ts": time.time()})
        print(json.dumps(res))
        with open(a.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        time.sleep(3)


if __name__ == "__main__":
    main()
