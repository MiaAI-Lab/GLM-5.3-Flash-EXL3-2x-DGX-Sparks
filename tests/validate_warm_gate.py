#!/usr/bin/env python3
"""Exploratory observation of the opt-in mixed-prefill gate (server idle; policy `skip`).

This is NOT a qualification. It reports what one local run observed, and it
establishes no TTFT or time-to-first-service guarantee: `vllm:num_requests_running`
is an admission counter, not a measure of compute service, and G's SSE chunk rate
is a host-side liveness proxy, not tokens/s.

 A. warm follow-up W (uncached remainder under WARM_TOKENS) while a long generation G holds a decoder: reports TTFT and,
    when the server reports prompt_tokens_details, how much of that prompt was cached.
 B. control just above the warm window (> WARM_TOKENS new tokens) during G: reports whether it was admitted at once or held.
 C. cold arrival during G: polls the admission-counter transition (the running count rising, 5 Hz) and reports TTFT
    separately. Admission order is all the counter can show; the prefill itself is not observed.

Every measurement carries the peer's liveness in its own window: a window in which G produced no chunk has no decode peer,
so the observation is recorded as void instead of compared. A capture that errored, was dropped or ended without a finish
reason is void as well and is never reported as a completed observation.

Exit: 77 nothing to observe (gate disabled at 0/0, or the server is busy), 2 no usable observation, 1 a predeclared
expectation was observed not to hold, 0 all predeclared expectations were observed to hold. Codes 0 and 1 describe one
exploratory run; neither qualifies either knob.

Env: GLM53_BASE_URL (default http://127.0.0.1:8888), VLLM_API_KEY (bearer), GLM53_MIXED_PREFILL_WARM_TOKENS/MAX_WAIT_MS
(the values the server runs with; both features are opt-in and default to 0, and a run against a disabled gate refuses to
start rather than scoring it).
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
WARM_TOKENS = int(os.environ.get("GLM53_MIXED_PREFILL_WARM_TOKENS", "0"))
MAX_WAIT_MS = int(os.environ.get("GLM53_MIXED_PREFILL_MAX_WAIT_MS", "0"))
MAX_WAIT_S = MAX_WAIT_MS / 1000.0
COMPLETE_REASONS = ("stop", "length")  # anything else is a void capture, not an observation
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


def require_gate_enabled() -> None:
    """Both gate features are opt-in (0 = off); a disabled gate has nothing to observe."""
    if WARM_TOKENS <= 0 or MAX_WAIT_MS <= 0:
        print(f"mixed-prefill gate disabled (WARM_TOKENS={WARM_TOKENS}, MAX_WAIT_MS={MAX_WAIT_MS}); "
              "export the values the server runs with to observe anything; a 0/0 run is not "
              "scored as a pass or a failure", file=sys.stderr)
        raise SystemExit(77)


def mk(n: int, t: int) -> str:
    return "".join(SEED % (t + i) for i in range(1, n))


def stream(text: str, max_tokens: int, rec: dict) -> None:
    """One streaming capture.

    `rec['complete']` is set only for a stream that ended with a reported finish
    reason. An HTTP/JSON failure, a dropped read or a stream that ends without
    one is recorded as void with the reason, so a capture failure is never read
    as a completed observation.
    """
    t0 = time.time()
    rec.update(t0=t0, chunks=[], complete=False, finish_reason=None, void_reason="stream did not start")
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}, "cache_salt": f"{RUN}"}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), _headers())
    try:
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
                for ch in j.get("choices") or []:
                    if ch.get("delta", {}).get("content"):
                        if "ttft" not in rec:
                            rec["ttft"] = time.time() - t0
                        rec["chunks"].append(time.time())
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
    except Exception as exc:  # propagate the failure instead of scoring a zero
        rec["void_reason"] = f"{type(exc).__name__}: {exc}"
    else:
        if rec["finish_reason"] in COMPLETE_REASONS:
            rec["void_reason"] = None
            rec["complete"] = True
        else:
            rec["void_reason"] = f"stream ended with finish_reason={rec['finish_reason']!r}"
    rec["wall"] = time.time() - t0
    if "ttft" not in rec:
        rec["void_reason"] = rec["void_reason"] or "no content token in the capture"
        rec["complete"] = False


def observation(rec: dict) -> str:
    """'' when `rec` is a complete observation, else the reason it is void."""
    if not rec.get("complete"):
        return rec.get("void_reason") or "capture did not complete"
    if "ttft" not in rec:
        return "capture reported no content token"
    return ""


def peer_decoding(rec: dict, a: float, b: float) -> bool:
    """True when the peer stream produced at least one chunk inside [a, b]."""
    return any(a <= t <= b for t in rec.get("chunks", []))


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
    require_gate_enabled()
    unmet: list[str] = []
    void: list[str] = []

    def check(rec: dict, cond: bool, msg: str) -> None:
        """One predeclared expectation; a void capture can never satisfy it."""
        reason = observation(rec)
        if reason:
            void.append(f"{msg} ({reason})")
            print(f"[{tag}] void    {msg} — not observed: {reason}", flush=True)
            return
        print(f"[{tag}] {'holds  ' if cond else 'not-met'} {msg}", flush=True)
        if not cond:
            unmet.append(msg)

    def peer_gate(rec: dict, a: float, b: float, msg: str) -> bool:
        """Require the peer to be decoding inside the window the measurement covers."""
        if peer_decoding(rec, a, b):
            return True
        void.append(f"{msg} (no peer chunk in the window)")
        print(f"[{tag}] void    {msg} — no peer decode in the window", flush=True)
        return False

    def ttft(rec: dict) -> str:
        return f"{rec['ttft']:.2f}s" if "ttft" in rec else "n/a"

    W = mk(3000, 950_000); G = mk(3000, 960_000) + "\nWrite a 600-word essay on the history of double-entry bookkeeping."
    C = mk(1300, 970_000) + "\nReply OK."
    r = {}; stream(W + "\nReply OK.", 4, r); print(f"[{tag}] warm-up W prefill {r['wall']:.1f}s", flush=True)
    solo = {}; stream(G, 300, solo)
    if observation(solo):
        print(f"[{tag}] solo reference unavailable: {observation(solo)}", flush=True)
        solo_rate = None
    else:
        solo_rate = rate(solo, solo["ttft"], solo["wall"])
        print(f"[{tag}] solo generation {solo_rate} chunks/s (ttft {solo['ttft']:.1f}s)", flush=True)

    g = {}; th = threading.Thread(target=stream, args=(G + " Then write another 900 words on ledgers.", 900, g), daemon=True); th.start()
    while "ttft" not in g and th.is_alive():
        time.sleep(0.2)
    time.sleep(3)
    # A. warm follow-up, observed only while the peer is actually decoding
    w = {}; tw = time.time(); stream(W + "\nOne sentence: what is row 7 about?", 30, w)
    c_tok, p_tok = cached(w)
    print(f"[{tag}] A warm follow-up during G: TTFT {ttft(w)} cached_tokens={c_tok} prompt={p_tok}", flush=True)
    peer_gate(g, tw, tw + w.get("wall", 0.0), "A: peer decoding during the warm follow-up")
    check(w, w.get("ttft", 9e9) <= 3.0, f"A: warm follow-up TTFT {ttft(w)} <= 3.0 s")
    if c_tok is not None and p_tok:
        check(w, c_tok >= 0.9 * p_tok,
              f"A: cached_tokens {c_tok} >= 90 % of prompt {p_tok} (bypass attributed to the cache hit)")
    else:
        print(f"[{tag}] note: server does not report prompt_tokens_details.cached_tokens; attribution by TTFT only", flush=True)
    before = rate(g, tw - 3, tw)
    time.sleep(2)
    # B. control just above the warm window
    b = {}; stream(W + "\n" + mk(max(WARM_TOKENS // 12 + 200, 400), 990_000) + "\nReply OK.", 4, b)
    print(f"[{tag}] B control (> WARM_TOKENS new tokens) during G: TTFT {ttft(b)}", flush=True)
    check(b, b.get("ttft", 0.0) >= MAX_WAIT_S * 0.9,
          f"B: above-window follow-up NOT bypassed (TTFT {ttft(b)} >= {MAX_WAIT_S:.1f}s)")
    time.sleep(2)
    # C. cold arrival: only the admission-counter transition is observed here.
    base = running_now()
    c = {}; stop = {}; tts_box = {}
    def _poll():
        tts_box["v"] = time_to_service(base, tc, stop)
    tc = time.time(); pth = threading.Thread(target=_poll, daemon=True); pth.start(); stream(C, 4, c); stop["done"] = True; pth.join(timeout=5)
    tts = tts_box.get("v")
    print(f"[{tag}] C cold 20K arrival during G: admission (running {base:g} -> {base+1:g}) observed at "
          f"{tts if tts is None else round(tts, 2)}s | TTFT {ttft(c)} wall {c.get('wall', 0.0):.1f}s", flush=True)
    if tts is None:
        void.append("C: admission-counter transition not observed")
        print(f"[{tag}] void    C: the admission counter never rose; nothing observed", flush=True)
    else:
        check(c, MAX_WAIT_S * 0.9 <= tts <= MAX_WAIT_S + 2.5,
              f"C: admission counter rose {tts} s after submit, in [{MAX_WAIT_S*0.9:.1f}, {MAX_WAIT_S+2.5:.1f}] s "
              f"(admission order only; compute service is not measured)")
    during = rate(g, tc, tc + c.get("wall", 0.0)); after = rate(g, tc + c.get("wall", 0.0), tc + c.get("wall", 0.0) + 5)
    th.join(timeout=900)
    print(f"[{tag}] G chunks/s: before {before} | during cold arrival {during} | after {after} | solo {solo_rate}", flush=True)
    scope = ("exploratory observation of one run: not a qualification, and no TTFT or "
             "time-to-first-service guarantee is established")
    summary = {"tag": tag, "run": RUN, "scope": scope, "warm_ttft": w.get("ttft"), "warm_cached": c_tok,
               "control_ttft": b.get("ttft"), "admission_delay_s": None if tts is None else round(tts, 2),
               "cold_ttft": c.get("ttft"), "cold_wall": c.get("wall"), "g_before": before, "g_during": during,
               "g_after": after, "solo": solo_rate, "void": void, "unmet": unmet,
               "captures": {"warm": w.get("finish_reason"), "control": b.get("finish_reason"),
                            "cold": c.get("finish_reason")}}
    print("SUMMARY " + json.dumps(summary), flush=True)
    print(f"[{tag}] {scope}", flush=True)
    if unmet:
        print(f"[{tag}] {len(unmet)} predeclared expectation(s) observed not to hold; this neither passes nor fails the gate", flush=True)
        return 1
    if void:
        print(f"[{tag}] no observation: {len(void)} capture(s)/window(s) were void", flush=True)
        return 2
    print(f"[{tag}] every predeclared expectation was observed to hold in this one run", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
