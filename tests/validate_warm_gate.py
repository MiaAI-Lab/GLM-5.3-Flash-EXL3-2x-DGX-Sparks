#!/usr/bin/env python3
"""Mixed-prefill gate v2 live validation (server idle; policy `skip` expected). Exits nonzero on a failed threshold.

 A. warm the 50K conversation W (prefill only); start a long generation G on a distinct cold 50K prefix; once G streams,
    send W + a short follow-up (uncached remainder well under WARM_TOKENS) -> TTFT_warm <= 3 s (v1 `skip`: 15-17 s),
    and its `cached_tokens` (if the server reports prompt_tokens_details) must be >= 90 % of its prompt.
 B. control just above the warm window: W + ~6K new tokens (> WARM_TOKENS) during G -> must NOT bypass: TTFT >= MAX_WAIT_MS.
 C. while G decodes, a cold 20K read C: time-to-first-SERVICE (the moment vllm:num_requests_running rises to 2, polled at
    5 Hz) must be within [0.9 x MAX_WAIT_MS, MAX_WAIT_MS + 2.5 s] — the deadline bounds admission, NOT TTFT: the first
    content token only arrives after the whole prefill (20K at LATE_CAP tokens/step), which is reported for information.
    G's stream rate before / during / after is reported in SSE chunks per second (DFlash2 emits several tokens per chunk,
    so this is NOT tokens/s; only ratios are meaningful).
Env: GLM53_BASE_URL (default http://127.0.0.1:8888), VLLM_API_KEY (bearer), GLM53_MIXED_PREFILL_WARM_TOKENS/MAX_WAIT_MS
(defaults 3584 / 1500, used for the thresholds).
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.request
import uuid

BASE = os.environ.get("GLM53_BASE_URL", "http://127.0.0.1:8888")
MODEL = "GLM-5.3-Flash-EXL3"
API_KEY = os.environ.get("VLLM_API_KEY", "")
WARM_TOKENS = int(os.environ.get("GLM53_MIXED_PREFILL_WARM_TOKENS", "3584"))
MAX_WAIT_S = int(os.environ.get("GLM53_MIXED_PREFILL_MAX_WAIT_MS", "1500")) / 1000.0
SEED = "Ledger row %d reconciled to the cent under audit rule seven. "
RUN = uuid.uuid4().hex[:8]


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def require_idle() -> None:
    t = urllib.request.urlopen(urllib.request.Request(BASE + "/metrics", headers=_headers()), timeout=20).read().decode()
    vals = re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^}]*\})? (\S+)", t, re.M)
    if not vals:
        print("readiness gauges missing — refusing to run", file=sys.stderr)
        raise SystemExit(77)
    if sum(float(v) for v in vals) > 0:
        print("server busy — refusing to run", file=sys.stderr)
        raise SystemExit(77)


def mk(n: int, t: int) -> str:
    return "".join(SEED % (t + i) for i in range(1, n))


def stream(text: str, max_tokens: int, rec: dict) -> None:
    t0 = time.time(); rec["t0"] = t0; rec["chunks"] = []
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}, "cache_salt": f"{RUN}"}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), _headers())
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            d = line[5:].strip()
            if d == b"[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                rec["usage"] = j["usage"]
            ch = j.get("choices") or []
            if ch and ch[0]["delta"].get("content"):
                if "ttft" not in rec:
                    rec["ttft"] = time.time() - t0
                rec["chunks"].append(time.time())
    rec["wall"] = time.time() - t0


def rate(rec: dict, a: float, b: float) -> float:
    return round(len([t for t in rec.get("chunks", []) if a <= t <= b]) / max(b - a, 1e-6), 1)


def running_now() -> float:
    t = urllib.request.urlopen(urllib.request.Request(BASE + "/metrics", headers=_headers()), timeout=10).read().decode()
    return sum(float(v) for v in re.findall(r"^vllm:num_requests_running(?:\{[^}]*\})? (\S+)", t, re.M))


def time_to_service(baseline: float, t0: float, stop: dict, limit: float = 120.0):
    """Poll num_requests_running until it exceeds the baseline; return seconds since t0 (or None)."""
    while time.time() - t0 < limit and not stop.get("done"):
        try:
            if running_now() > baseline:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(0.2)
    return None


def cached(rec: dict):
    u = rec.get("usage") or {}
    return (u.get("prompt_tokens_details") or {}).get("cached_tokens"), u.get("prompt_tokens")


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "warm-gate"
    require_idle()
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"[{tag}] {'ok  ' if cond else 'FAIL'} {msg}", flush=True)
        if not cond:
            fails.append(msg)

    W = mk(3000, 950_000); G = mk(3000, 960_000) + "\nWrite a 600-word essay on the history of double-entry bookkeeping."
    C = mk(1300, 970_000) + "\nReply OK."
    r = {}; stream(W + "\nReply OK.", 4, r); print(f"[{tag}] warm-up W prefill {r['wall']:.1f}s", flush=True)
    solo = {}; stream(G, 300, solo); solo_rate = rate(solo, solo["t0"] + solo["ttft"], solo["t0"] + solo["wall"])
    print(f"[{tag}] solo generation {solo_rate} chunks/s (ttft {solo['ttft']:.1f}s)", flush=True)

    g = {}; th = threading.Thread(target=stream, args=(G + " Then write another 900 words on ledgers.", 900, g), daemon=True); th.start()
    while "ttft" not in g and th.is_alive():
        time.sleep(0.2)
    time.sleep(3)
    # A. warm follow-up
    w = {}; tw = time.time(); stream(W + "\nOne sentence: what is row 7 about?", 30, w)
    c_tok, p_tok = cached(w)
    print(f"[{tag}] A warm follow-up during G: TTFT {w['ttft']:.2f}s cached_tokens={c_tok} prompt={p_tok}", flush=True)
    check(w["ttft"] <= 3.0, f"A: warm follow-up TTFT {w['ttft']:.2f}s <= 3.0 s (v1 skip: 15-17 s)")
    if c_tok is not None and p_tok:
        check(c_tok >= 0.9 * p_tok, f"A: cached_tokens {c_tok} >= 90 % of prompt {p_tok} (bypass attributed to the cache hit)")
    else:
        print(f"[{tag}] note: server does not report prompt_tokens_details.cached_tokens; attribution by TTFT only", flush=True)
    before = rate(g, tw - 3, tw)
    time.sleep(2)
    # B. control just above the warm window
    b = {}; stream(W + "\n" + mk(max(WARM_TOKENS // 12 + 200, 400), 990_000) + "\nReply OK.", 4, b)
    print(f"[{tag}] B control (> WARM_TOKENS new tokens) during G: TTFT {b['ttft']:.2f}s", flush=True)
    check(b["ttft"] >= MAX_WAIT_S * 0.9, f"B: above-window follow-up NOT bypassed (TTFT {b['ttft']:.2f}s >= {MAX_WAIT_S:.1f}s)")
    time.sleep(2)
    # C. cold arrival: admission time via the running-count transition, TTFT for information
    base = running_now()
    c = {}; stop = {}; tts_box = {}
    def _poll():
        tts_box["v"] = time_to_service(base, tc, stop)
    tc = time.time(); pth = threading.Thread(target=_poll, daemon=True); pth.start(); stream(C, 4, c); stop["done"] = True; pth.join(timeout=5)
    tts = tts_box.get("v")
    print(f"[{tag}] C cold 20K arrival during G: time-to-first-service {tts if tts is None else round(tts, 2)}s (running {base:g} -> {base+1:g}) | TTFT {c['ttft']:.2f}s wall {c['wall']:.1f}s", flush=True)
    check(tts is not None and MAX_WAIT_S * 0.9 <= tts <= MAX_WAIT_S + 2.5,
          f"C: cold arrival admitted after the deadline (time-to-first-service {tts} s in [{MAX_WAIT_S*0.9:.1f}, {MAX_WAIT_S+2.5:.1f}] s)")
    during = rate(g, tc, tc + c["wall"]); after = rate(g, tc + c["wall"], tc + c["wall"] + 5)
    th.join(timeout=900)
    print(f"[{tag}] G chunks/s: before {before} | during cold prefill {during} | after {after} | solo {solo_rate}", flush=True)
    summary = {"tag": tag, "run": RUN, "warm_ttft": round(w["ttft"], 2), "warm_cached": c_tok, "control_ttft": round(b["ttft"], 2),
               "cold_time_to_service": None if tts is None else round(tts, 2), "cold_ttft": round(c["ttft"], 2), "cold_wall": round(c["wall"], 1), "g_before": before, "g_during": during,
               "g_after": after, "solo": solo_rate, "fails": fails}
    print("SUMMARY " + json.dumps(summary), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
