#!/usr/bin/env python3
"""Non-streaming N-agent multi-turn prefix-cache stress.

Each agent has a private growing prefix (agent salt at position 0, then common
filler). Per turn every agent appends a unique chunk and all agents fire one
request concurrently. Records prompt_tokens / cached_tokens per request,
prefix-cache counter deltas and wall time.
"""
import argparse
import concurrent.futures as cf
import json
import math
import re
import secrets
import time
import urllib.request

BASE = "http://127.0.0.1:8888"
MODEL = "GLM-5.3-Flash-EXL3"
HITS_RE = re.compile(r"^vllm:prefix_cache_hits_total(?:\{[^}]*\})?\s+(\S+)", re.M)
QRY_RE = re.compile(r"^vllm:prefix_cache_queries_total(?:\{[^}]*\})?\s+(\S+)", re.M)
SALT = secrets.token_hex(8)


def metrics() -> dict:
    raw = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    return {
        "hits": float(HITS_RE.search(raw).group(1)),
        "queries": float(QRY_RE.search(raw).group(1)),
    }


def filler(n_tokens: int, tag: str) -> str:
    return f"[{tag}] " + "the " * n_tokens


def request(prompt: str, timeout: float = 900) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + "\nReply with OK."}],
        "temperature": 0,
        "max_tokens": 4,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode())
    u = obj.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": int(u.get("prompt_tokens") or 0),
        "cached_tokens": int(det.get("cached_tokens") or 0),
        "wall_s": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--base-tokens", type=int, default=60000)
    ap.add_argument("--chunk-tokens", type=int, default=6000)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    histories = [filler(args.base_tokens, f"A{a} {SALT}") for a in range(args.agents)]
    rows = []
    t_start = time.time()
    for turn in range(args.turns):
        for a in range(args.agents):
            histories[a] += filler(args.chunk_tokens, f"A{a} t{turn}")
        before = metrics()
        with cf.ThreadPoolExecutor(max_workers=args.agents) as ex:
            results = [
                f.result()
                for f in [ex.submit(request, h) for h in histories]
            ]
        after = metrics()
        for a, r in enumerate(results):
            r.update({"turn": turn, "agent": a})
            rows.append(r)
        hit = sum(r["cached_tokens"] for r in results)
        prom = sum(r["prompt_tokens"] for r in results)
        gq = after["queries"] - before["queries"]
        gh = after["hits"] - before["hits"]
        print(
            f"{args.label} turn {turn}: prompt={prom} cached={hit} "
            f"ratio={hit/prom if prom else 0:.3f} "
            f"global={(gh/gq if gq else float('nan')):.3f} "
            f"max_wall={max(r['wall_s'] for r in results):.1f}s "
            f"sum_wall={sum(r['wall_s'] for r in results):.1f}s",
            flush=True,
        )
        with open(args.out, "w") as fh:
            json.dump({"label": args.label, "args": vars(args), "rows": rows}, fh, indent=2)
    print(f"{args.label} done elapsed={time.time()-t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
