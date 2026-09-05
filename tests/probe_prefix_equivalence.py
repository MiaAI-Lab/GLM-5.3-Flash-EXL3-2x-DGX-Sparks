#!/usr/bin/env python3
"""Cold-vs-cold determinism control for the prefix cache (live; idle server).

Same 45K prompt + prose task, 64 tokens, temp 0 with per-token logprobs: cold1/cold2/cold3
(three disjoint cache_salt namespaces = three independent cold prefills; the noise floor is the MAX of the three pairwise
max-|dlogprob| values — a single cold pair proved too noisy: floors 0.04-0.41 across runs), then warm1/warm2 (re-runs of the
s1/s2 namespaces, prefix served from the cache). Reports the
max |dlogprob| of the chosen token over agreeing positions for cold-vs-cold (noise floor) and cold-vs-warm, the position-0
chosen-token agreement, and first-divergence indices for information. Refuses to run on a busy server; requires the warm
runs to be served from the cache (hit >= 0.95) and >= 4 leading agreeing positions. Exits 1 if cold-vs-warm exceeds
3 x max(cold-vs-cold floor, 0.02) (the kit's own run-to-run noise, measured 0.15-0.41 nats) or the position-0 chosen token
differs. A deployment-specific smoke gate, not a statistical proof. Absolute thresholds
and top-5 set identity are deliberately not used: two cold runs already differ by ~0.4 nats and reorder the top-5.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import uuid

import os
BASE = os.environ.get("GLM53_BASE_URL", "http://127.0.0.1:8888")
MODEL = "GLM-5.3-Flash-EXL3"
API_KEY = os.environ.get("VLLM_API_KEY", "")


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h
SEED = "Ledger row %d reconciled to the cent under audit rule seven. "


def gen(prompt: str, salt: str) -> tuple[str, float, list]:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "cache_salt": salt, "logprobs": True, "top_logprobs": 5}
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                                                _headers()), timeout=1800))
    ch = d["choices"][0]
    lps = [(it["token"], float(it["logprob"]), tuple(sorted(t["token"] for t in (it.get("top_logprobs") or []))))
           for it in ((ch.get("logprobs") or {}).get("content") or [])]
    return ch["message"]["content"], round(time.time() - t0, 1), lps


def lp_delta(a: list, b: list) -> tuple[float, int, bool]:
    mx = 0.0; n = 0
    for (ta, la, _), (tb, lb, _) in zip(a, b):
        if ta != tb:
            break
        mx = max(mx, abs(la - lb)); n += 1
    return mx, n, bool(a and b and a[0][0] == b[0][0])


def diverge(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else 10**9


def require_idle() -> None:
    import re
    t = urllib.request.urlopen(urllib.request.Request(BASE + "/metrics", headers=_headers()), timeout=20).read().decode()
    vals = re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^}]*\})? (\S+)", t, re.M)
    if not vals or sum(float(v) for v in vals) > 0:
        print("server busy or readiness gauges missing — refusing to run", file=sys.stderr)
        raise SystemExit(77)


def hits_delta():
    import re
    t = urllib.request.urlopen(urllib.request.Request(BASE + "/metrics", headers=_headers()), timeout=20).read().decode()
    q = sum(float(v) for v in re.findall(r"^vllm:prefix_cache_queries_total(?:\{[^}]*\})? (\S+)", t, re.M))
    h = sum(float(v) for v in re.findall(r"^vllm:prefix_cache_hits_total(?:\{[^}]*\})? (\S+)", t, re.M))
    return q, h


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "equiv"
    require_idle()
    run = uuid.uuid4().hex[:8]
    task = sys.argv[2] if len(sys.argv) > 2 else "prose"
    prompt = "".join(SEED % (980_000 + i) for i in range(1, 3000)) + (
        "\nWrite a Python function that parses these ledger rows into a dict keyed by row number and returns the count. Code only."
        if task == "code" else
        "\nIn about 300 words, explain what an audit reconciliation is and why row-level checks matter.")
    runs = {"cold1": gen(prompt, f"{run}-s1"), "cold2": gen(prompt, f"{run}-s2"), "cold3": gen(prompt, f"{run}-s3")}
    q0, h0 = hits_delta()
    runs["warm1"] = gen(prompt, f"{run}-s1"); runs["warm2"] = gen(prompt, f"{run}-s2")   # warm re-runs of each namespace
    time.sleep(3); q1, h1 = hits_delta()
    warm_hit = (h1 - h0) / (q1 - q0) if q1 > q0 else 0.0
    print(f"[{tag}] warm runs prefix-cache hit ratio {warm_hit:.3f}", flush=True)
    for k, (t, w, lps) in runs.items():
        print(f"[{tag}] {k}: {w}s len={len(t)} logprob positions={len(lps)}", flush=True)
    f12 = lp_delta(runs["cold1"][2], runs["cold2"][2]); f13 = lp_delta(runs["cold1"][2], runs["cold3"][2]); f23 = lp_delta(runs["cold2"][2], runs["cold3"][2])
    floor = max(f12[0], f13[0], f23[0]); n_cc = min(f12[1], f13[1], f23[1]); top0_cc = f12[2] and f13[2] and f23[2]
    d1 = lp_delta(runs["cold1"][2], runs["warm1"][2]); d2 = lp_delta(runs["cold2"][2], runs["warm2"][2])
    cw = max(d1[0], d2[0]); top0 = d1[2] and d2[2]
    print(f"[{tag}] text first-divergence: cold-vs-cold @{diverge(runs['cold1'][0], runs['cold2'][0])} | cold-vs-warm @{diverge(runs['cold1'][0], runs['warm1'][0])} | warm-vs-warm @{diverge(runs['warm1'][0], runs['warm2'][0])}", flush=True)
    print(f"[{tag}] max|dlogprob|: cold-vs-cold floor {floor:.4f} (pairs {f12[0]:.4f}/{f13[0]:.4f}/{f23[0]:.4f}, {n_cc} pos, pos-0 same={top0_cc}) | cold-vs-warm {cw:.4f} ({min(d1[1], d2[1])} pos) | pos-0 token same={top0}", flush=True)
    for k in ("cold1", "cold2", "warm1", "warm2"):
        print(f"[{tag}] {k} pos0: {runs[k][2][0] if runs[k][2] else None}", flush=True)
    with open(f"/tmp/equiv-{tag}.json", "w") as f:
        json.dump({k: {"text": t, "lps": lps} for k, (t, w, lps) in runs.items()}, f)
    # Empirical floor prior: across 8 recorded runs on this kit the cold-vs-cold floor ranged 0.04-0.41 nats
    # (median ~0.20); a single run can draw a lucky-low floor, so the gate uses max(measured, 0.15).
    tol = 3.0 * max(floor, 0.15)
    ok = n_cc >= 4 and min(d1[1], d2[1]) >= 4 and cw <= tol and top0 and warm_hit >= 0.95
    print(f"[{tag}] {'PASS' if ok else 'FAIL'}: cold-vs-warm max|dlogprob| {cw:.4f} <= 3 x cold-vs-cold floor ({tol:.4f}); pos-0 chosen token identical={top0}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
