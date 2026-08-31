#!/usr/bin/env python3
"""Sparse prefix-cache retention gate (live; server must be idle). Exits nonzero on any failed threshold.

Checks, in order:
 1. equivalence gate (numerical, sampling-free): temp 0 with per-token logprobs (top_logprobs=5), 64 tokens, code + prose.
    A fresh 45K prefix is generated COLD twice under two cache_salt namespaces (cold-vs-cold noise floor) and WARM twice
    (same salt, prefix served from the cache). Over the positions where two runs emit the same token, we compare the chosen
    token's logprob over the leading agreeing positions (>= 4 required): cold-vs-warm max |dlogprob| must be <= 3 x
    max(cold-vs-cold max, 0.02) (the deployment's own run-to-run
    numerical noise band, measured 0.15-0.41 nats on this kit: TP=2 reduction order / EXL3 / fp8 KV), and the position-0
    chosen token must be identical (position 0 depends only on the cached prefix state). Absolute thresholds and top-5
    set identity are NOT used: two cold runs already differ by up to ~0.4 nats and can reorder the top-5. Text may
    diverge later (near-tie argmax flips are inherent); that is not what the gate measures. This is a deployment-specific
    smoke gate (one cold-vs-cold pair per task as the baseline), not a statistical equivalence proof.
    DFlash acceptance (accepted/drafted, over a 400-token generation) warm must be >= 0.85 x cold; missing counters fail.
 2. 3-turn conversation: each re-turn hit >= 0.95.
 3. pair ladder 45K/56K/66K/80K: A cold, B cold, A after B, B after A — re-turn hits >= 0.95 (<= 66K) / 0.90 (80K).
 4. root 55K + subagent sharing ~20K then diverging: subagent hit >= 0.40 of its prompt (the 14336 boundary), root re-turn >= 0.95.
Usage: tests/validate_apc_retention.py [tag]   (prints SUMMARY json; exit 1 on failure)
"""
from __future__ import annotations

import json
import re
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
RUN = uuid.uuid4().hex[:8]  # unique namespace per run: "cold" really is cold


def metrics() -> dict:
    t = urllib.request.urlopen(urllib.request.Request(BASE + "/metrics", headers=_headers()), timeout=20).read().decode()

    def g(k: str) -> float:
        vals = re.findall(rf"^vllm:{k}(?:\{{[^}}]*\}})? (\S+)", t, re.M)
        if not vals:
            raise SystemExit(f"metric vllm:{k} missing — cannot evaluate gates")
        return sum(float(v) for v in vals)  # sum over label sets (single engine here)

    return {"q": g("prefix_cache_queries_total"), "h": g("prefix_cache_hits_total"),
            "acc": g("spec_decode_num_accepted_tokens_total"), "drafts": g("spec_decode_num_draft_tokens_total")}


def chat(messages: list, max_tokens: int, salt: str, logprobs: bool = False) -> dict:
    m1 = metrics(); t0 = time.time(); ttft = None; out = []; lps = []
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}, "cache_salt": salt}
    if logprobs:
        body["logprobs"] = True; body["top_logprobs"] = 5
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 _headers())
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            d = line[5:].strip()
            if d == b"[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            ch = j.get("choices") or []
            if ch and ch[0]["delta"].get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                out.append(ch[0]["delta"]["content"])
            if ch and ch[0].get("logprobs") and ch[0]["logprobs"].get("content"):
                for item in ch[0]["logprobs"]["content"]:
                    lps.append((item.get("token"), float(item.get("logprob", 0.0)),
                                tuple(sorted(t.get("token") for t in (item.get("top_logprobs") or [])))))
    wall = time.time() - t0
    time.sleep(3)
    m2 = metrics()
    q = m2["q"] - m1["q"]; h = m2["h"] - m1["h"]; dr = m2["drafts"] - m1["drafts"]; ac = m2["acc"] - m1["acc"]
    if q <= 0:
        raise SystemExit("prefix_cache_queries did not advance — server busy or metrics stale")
    return {"text": "".join(out), "ttft": round(ttft if ttft is not None else wall, 2), "wall": round(wall, 1),
            "hit": round(h / q, 3), "accept": round(ac / dr, 3) if dr > 0 else None, "drafts": dr,
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "lps": lps}


def user(text: str) -> list:
    return [{"role": "user", "content": text}]


def mk(n: int, t: int) -> str:
    return "".join(SEED % (t + i) for i in range(1, n))


def lp_delta(a: list, b: list) -> tuple[float, int, bool]:
    """(max |dlogprob| over agreeing positions, number of agreeing positions, position-0 chosen token identical)."""
    mx = 0.0; n = 0
    for (ta, la, _), (tb, lb, _) in zip(a, b):
        if ta != tb:
            break
        mx = max(mx, abs(la - lb)); n += 1
    same0 = bool(a and b and a[0][0] == b[0][0])
    return mx, n, same0


def diverge(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else 10**9


def require_idle() -> None:
    t = urllib.request.urlopen(urllib.request.Request(BASE + "/metrics", headers=_headers()), timeout=20).read().decode()
    vals = re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^}]*\})? (\S+)", t, re.M)
    if not vals:
        print("readiness gauges (num_requests_running/waiting) missing — refusing to run the gate", file=sys.stderr)
        raise SystemExit(77)
    busy = sum(float(v) for v in vals)
    if busy > 0:
        print(f"server busy ({busy:g} requests running/waiting) — refusing to run the gate", file=sys.stderr)
        raise SystemExit(77)


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "apc"
    require_idle()
    fails: list[str] = []
    summary: dict = {"tag": tag, "run": RUN}

    def check(cond: bool, msg: str) -> None:
        print(f"[{tag}] {'ok  ' if cond else 'FAIL'} {msg}", flush=True)
        if not cond:
            fails.append(msg)

    # 1. numerical equivalence gate (logprobs) with cold-vs-cold noise floor, plus acceptance over a long generation
    for label, task in (("code", "\nWrite a Python function that parses these ledger rows into a dict keyed by row number and returns the count. Code only."),
                        ("prose", "\nIn about 300 words, explain what an audit reconciliation is and why row-level checks matter.")):
        prefix = mk(3000, 910_000 if label == "code" else 920_000)
        c1 = chat(user(prefix + task), 64, f"{RUN}-{label}-c1", logprobs=True)
        c2 = chat(user(prefix + task), 64, f"{RUN}-{label}-c2", logprobs=True)
        w1 = chat(user(prefix + task), 64, f"{RUN}-{label}-c1", logprobs=True)
        w2 = chat(user(prefix + task), 64, f"{RUN}-{label}-c2", logprobs=True)   # warm re-run of cold2's namespace
        check(w1["hit"] >= 0.95, f"equiv {label}: warm hit {w1['hit']} >= 0.95")
        check(len(c1["lps"]) >= 8 and len(w1["lps"]) >= 8, f"equiv {label}: logprobs returned ({len(c1['lps'])}/{len(w1['lps'])} positions)")
        floor, n_cc, _ = lp_delta(c1["lps"], c2["lps"])
        cw = max(lp_delta(c1["lps"], w1["lps"])[0], lp_delta(c2["lps"], w2["lps"])[0])
        n_cw = min(lp_delta(c1["lps"], w1["lps"])[1], lp_delta(c2["lps"], w2["lps"])[1])
        top0 = lp_delta(c1["lps"], w1["lps"])[2] and lp_delta(c2["lps"], w2["lps"])[2]
        tol = 3.0 * max(floor, 0.15)  # 0.15 = empirical lower-quartile of cold-vs-cold floors (8 runs)
        print(f"[{tag}] equiv {label}: cold-vs-cold max|dlp| {floor:.4f} over {n_cc} pos (noise floor) | cold-vs-warm max|dlp| {cw:.4f} over {n_cw} pos | pos-0 token same={top0} | warm ttft {w1['ttft']}s", flush=True)
        check(n_cc >= 4 and n_cw >= 4, f"equiv {label}: >= 4 leading positions compared (cold-vs-cold {n_cc}, cold-vs-warm {n_cw})")
        check(w1["hit"] >= 0.95 and w2["hit"] >= 0.95, f"equiv {label}: both warm runs served from the cache ({w1['hit']}, {w2['hit']})")
        check(cw <= tol, f"equiv {label}: cold-vs-warm max|dlogprob| {cw:.4f} <= 3 x cold-vs-cold floor ({tol:.4f})")
        check(top0, f"equiv {label}: position-0 chosen token identical cold vs warm")
        # acceptance over a 400-token generation (cold namespace vs warm namespace)
        ca = chat(user(prefix + task), 400, f"{RUN}-{label}-c3")
        wa = chat(user(prefix + task), 400, f"{RUN}-{label}-c3")
        check(ca["drafts"] > 0 and wa["drafts"] > 0 and ca["accept"] is not None and wa["accept"] is not None,
              f"equiv {label}: DFlash draft counters advanced (cold {ca['drafts']}, warm {wa['drafts']})")
        if ca["accept"] is not None and wa["accept"] is not None:
            check(wa["accept"] >= 0.85 * ca["accept"], f"equiv {label}: acceptance warm {wa['accept']} >= 0.85 x cold {ca['accept']}")
        summary[f"equiv_{label}"] = {"floor": floor, "cw": cw, "n_cw": n_cw, "top0": top0, "accept_cold": ca["accept"], "accept_warm": wa["accept"]}

    # 2. multi-turn
    conv = user(mk(3000, 910_000) + "\nSummarise row 10 in one sentence.")
    hits = []
    for turn in range(3):
        r = chat(conv, 60, f"{RUN}-mt")
        hits.append(r["hit"])
        print(f"[{tag}] multiturn {turn + 1}: prompt {r['prompt_tokens']} hit {r['hit']} ttft {r['ttft']}s", flush=True)
        conv = conv + [{"role": "assistant", "content": r["text"]}, {"role": "user", "content": f"Now row {11 + turn}, one sentence."}]
    check(all(h >= 0.95 for h in hits[1:]), f"multiturn re-turn hits {hits[1:]} >= 0.95")
    summary["multiturn_hits"] = hits

    # 3. pair ladder
    summary["pairs"] = {}
    for n, label, thr in ((3000, "45K", 0.95), (3700, "56K", 0.95), (4400, "66K", 0.95), (5300, "80K", 0.90)):
        A = user(mk(n, 700_000 + n) + "\nReply OK."); B = user(mk(n, 800_000 + n) + "\nReply OK.")
        rows = [("A cold", chat(A, 4, f"{RUN}-pA{n}")), ("B cold", chat(B, 4, f"{RUN}-pB{n}")),
                ("A after B", chat(A, 4, f"{RUN}-pA{n}")), ("B after A", chat(B, 4, f"{RUN}-pB{n}"))]
        for lab, r in rows:
            print(f"[{tag}] pair {label} {lab:10s} ttft {r['ttft']:5.1f}s hit {r['hit']}", flush=True)
        check(rows[2][1]["hit"] >= thr and rows[3][1]["hit"] >= thr, f"pair {label}: re-turn hits {rows[2][1]['hit']}/{rows[3][1]['hit']} >= {thr}")
        summary["pairs"][label] = {lab: r["hit"] for lab, r in rows}

    # 4. subagent divergence
    root = user(mk(3650, 930_000) + "\nReply OK."); sub = user(mk(1300, 930_000) + mk(600, 940_000) + "\nReply OK.")
    r0 = chat(root, 4, f"{RUN}-root"); r1 = chat(sub, 4, f"{RUN}-root"); r2 = chat(root, 4, f"{RUN}-root")
    print(f"[{tag}] subagent: root cold hit {r0['hit']} | sub hit {r1['hit']} | root re-turn hit {r2['hit']} ttft {r2['ttft']}s", flush=True)
    check(r1["hit"] >= 0.40, f"subagent hit {r1['hit']} >= 0.40 (14336-boundary hit on a ~29K prompt)")
    check(r2["hit"] >= 0.95, f"root re-turn hit {r2['hit']} >= 0.95 after the subagent")
    summary["subagent"] = {"sub_hit": r1["hit"], "root_return_hit": r2["hit"]}

    summary["fails"] = fails
    print("SUMMARY " + json.dumps(summary), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
