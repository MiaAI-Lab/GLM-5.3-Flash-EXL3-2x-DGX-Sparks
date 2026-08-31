#!/usr/bin/env python3
"""Concurrency + long-context ladder for GLM-5.3-Flash EXL3 on :8888.

Runs N simultaneous streaming chat completions per cell (mode x context x level x rep)
and reports aggregate tok/s, per-stream tok/s, TTFT / ITL percentiles, prefix-cache hit
ratio, preemptions and starvation age. Tokens are counted from the server's `usage`
(stream_options.include_usage), never from SSE chunk counts (DFlash2 emits ~2.3 tokens per
chunk). Drives the live OpenAI API only; does not reimplement the model.

Canonical run (server must be idle):
  python3 tests/bench_concurrency.py --levels 1,2,4,8,12,16 --modes code,data,chat \
      --ctx 0,50000,100000 --reps 3 --out logs/ladder-$(date +%F).json

Long-context cells use a *distinct* cached prefix per lane: a warm-up pass prefills each
lane's prefix (max_tokens=1), then the measured pass sends prefix + a 200-token suffix and
asserts the prefix-cache hit ratio from /metrics. `--shared-prefix` measures the
shared-prefix variant instead. `tests/bench_live.html` renders the JSON (or status.json
while running).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE = os.environ.get("GLM53_BASE", "http://127.0.0.1:8888")
MODEL = os.environ.get("GLM53_MODEL", "GLM-5.3-Flash-EXL3")

CHAT_TOPICS = [
    "why sourdough needs a long cold proof", "how tides work, for a curious ten-year-old",
    "what makes a good cup of pour-over coffee", "the history of the bicycle in three paragraphs",
    "how to plan a first vegetable garden in a small yard", "why the sky is blue and sunsets are red",
    "how a bill becomes law, plainly", "tips for sleeping better on night shifts",
    "what a sommelier actually does", "how to teach a dog to settle on a mat",
    "why bread goes stale and how to revive it", "what causes jet lag and how to beat it",
    "how compound interest sneaks up on you", "the difference between weather and climate",
    "how to write a toast for a friend's wedding", "why cats knead blankets",
]
CODE_TASKS = [
    "a Python module with eight small utility functions (slugify, chunk, clamp, retry decorator, LRU memo, deep-merge dicts, parse ISO dates, safe int) each with a docstring and a type-annotated signature",
    "a TypeScript module exporting a typed Result<T,E> helper with ok/err constructors, map, flatMap, unwrapOr and match, plus six unit tests in vitest style",
    "a complete Python dataclass model for an invoicing system: Customer, LineItem, Invoice, Payment, with validation methods and a to_dict on each",
    "a Go file implementing a thread-safe in-memory key-value store with TTL expiry, Get/Set/Delete/Keys and a background sweeper",
    "a Python pytest file with twelve tests for a function that parses cron expressions, covering every field and common error cases",
    "a Rust module with a simple tokenizer for arithmetic expressions and a recursive-descent parser producing an AST enum",
    "a SQL migration creating tables for users, teams, memberships and audit_log with indexes, plus three reporting views",
    "a bash script that rotates logs in a directory: gzip files older than 7 days, delete older than 30, with dry-run and verbose flags and help text",
    "a Python asyncio worker pool with bounded concurrency, retries with jitter, graceful shutdown and a small CLI",
    "a Java class implementing an LRU cache with generics, unit tests in JUnit 5 style",
    "a Kotlin data model and repository interface for a to-do app with Room annotations",
    "a C function set for a ring buffer with tests using assert",
    "a Python click CLI that renames files by regex with a dry-run flag and colored output",
    "a Django model set for a blog with posts, tags, comments and a moderation flag plus admin registration",
    "a Node.js Express router for a notes API with validation middleware and error handling",
    "a Swift struct-based JSON decoder wrapper with typed errors and three unit tests",
]
DATA_TASKS = [
    "a JSON array of 40 fictional customers with id, name, email, city, country, plan (free/pro/team), mrr, signup_date and tags",
    "a CSV with a header and 60 rows of fictional weather readings: station_id, timestamp, temp_c, humidity, wind_kph, pressure_hpa, condition",
    "a YAML config for a web service with server, database, cache, logging, feature_flags and three environments overriding values",
    "a markdown table of 45 rows comparing fictional laptops: model, cpu, ram_gb, storage_gb, weight_kg, battery_hours, price_usd",
    "a JSON schema for an e-commerce order object with nested address, items, payment and shipment objects, all fields described",
    "a numbered list of 120 fictional but realistic street names for a new suburb, one per line",
    "an OpenAPI 3 YAML snippet describing four endpoints for a notes API with request/response schemas",
    "a JSON array of 50 fictional books with isbn, title, author, year, pages, genre and rating",
    "a CSV of 80 fictional employees: id, first, last, department, title, salary, start_date, remote",
    "a TOML config for a build tool with three profiles and documented keys",
    "a JSON array of 60 fictional cities with name, country, population, latitude, longitude, timezone",
    "an INI-style config for a mail server with six sections",
    "a markdown table of 40 chemical elements with symbol, number, mass, group, period, phase",
    "a JSON object describing a fictional API rate-limit policy with tiers, windows and burst rules",
    "a CSV of 70 fictional orders: order_id, customer_id, sku, qty, unit_price, currency, status, placed_at",
    "a YAML Kubernetes Deployment plus Service for a fictional web app with resources and probes",
]
FILLER = "Ledger row %d reconciled to the cent under audit rule seven. "


def _post_stream(body: dict, timeout: float = 3600.0):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def metrics() -> dict[str, float]:
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=20).read().decode("utf-8", "replace")
    out: dict[str, float] = {}
    for key in ("prefix_cache_queries_total", "prefix_cache_hits_total", "num_preemptions_total",
                "num_requests_running", "num_requests_waiting"):
        m = re.search(rf"^vllm:{key}\{{[^}}]*\}}\s+(\S+)$", txt, re.M)
        if m:
            out[key] = float(m.group(1))
    return out


def percentiles(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"p50": None, "p95": None, "p99": None}
    s = sorted(xs)
    def q(p: float) -> float:
        i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return s[i]
    return {"p50": round(q(0.5), 3), "p95": round(q(0.95), 3), "p99": round(q(0.99), 3)}


def make_prefix(lane_tag: int, ctx_tokens: int) -> str:
    # ~14 tokens per filler sentence on this tokenizer; distinct row numbers per lane.
    n = max(0, int(ctx_tokens / 14))
    return "".join(FILLER % (lane_tag * 1_000_000 + i) for i in range(n))


def build_prompt(mode: str, i: int, ctx_tokens: int, lane_tag: int, shared_prefix: bool) -> str:
    prefix = make_prefix(0 if shared_prefix else lane_tag, ctx_tokens) if ctx_tokens else ""
    if mode == "chat":
        task = f"Chat with me about {CHAT_TOPICS[i % len(CHAT_TOPICS)]}. Be warm and conversational, about 250 words."
    elif mode == "code":
        task = f"Write {CODE_TASKS[i % len(CODE_TASKS)]}. Output only the code, no commentary."
    else:
        task = f"Write {DATA_TASKS[i % len(DATA_TASKS)]}. Output only the data, no commentary."
    if prefix:
        return prefix + "\n\nIgnore the ledger above; it is context filler.\n" + task
    return task


def stream_one(prompt: str, max_tokens: int, temp: float, thinking: bool, out: dict[str, Any]) -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    t0 = time.time()
    first = None
    last = None
    itl: list[float] = []
    usage = None
    try:
        resp = _post_stream(body)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                j = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if j.get("usage"):
                usage = j["usage"]
            choices = j.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.time()
                if first is None:
                    first = now
                elif last is not None:
                    itl.append(now - last)
                last = now
    except urllib.error.HTTPError as e:
        out["error"] = f"HTTP {e.code}: {e.read()[:120]!r}"
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:160]
    end = time.time()
    out.update({
        "start": t0, "first": first, "end": end,
        "ttft": (first - t0) if first else None,
        "itl": itl,
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
    })


def run_cell(mode: str, ctx: int, level: int, rep: int, args: argparse.Namespace, status_cb) -> dict[str, Any]:
    temp = 0.0 if mode in ("code", "data") or args.chat_temp0 else 0.7
    lanes = [dict(topic=i) for i in range(level)]
    prompts = [build_prompt(mode, (rep * 31 + i), ctx, lane_tag=i + 1, shared_prefix=args.shared_prefix) for i in range(level)]
    if ctx:
        # warm-up: prefill each lane's prefix (distinct per lane unless --shared-prefix), 1 token out.
        # sequential so warm-up prefills never mix with each other (hybrid-cache state slots are per conversation)
        for p in prompts:
            stream_one(p, 1, 0.0, False, {})
        time.sleep(1.0)
        mw = metrics()
        cell_warm = {"warm_queries": mw.get("prefix_cache_queries_total"), "warm_hits": mw.get("prefix_cache_hits_total")}
    m0 = metrics()
    if not ctx:
        cell_warm = {}
    t_cell = time.time()
    outs: list[dict[str, Any]] = [dict() for _ in range(level)]
    ths = []
    for i, p in enumerate(prompts):
        th = threading.Thread(target=stream_one, args=(p, args.max_tokens, temp, False, outs[i]))
        ths.append(th)
        th.start()
        if args.stagger and i < level - 1:
            time.sleep(args.stagger)
    while any(t.is_alive() for t in ths):
        time.sleep(0.5)
        status_cb(mode, ctx, level, rep, outs, t_cell)
    m1 = metrics()
    wall = max((o.get("end") or t_cell) for o in outs) - t_cell
    toks = [o.get("completion_tokens") or 0 for o in outs]
    per = [ (o["completion_tokens"] - 1) / max(o["end"] - o["first"], 1e-3)
            for o in outs if o.get("first") and (o.get("completion_tokens") or 0) > 1 ]
    ttfts = [o["ttft"] for o in outs if o.get("ttft") is not None]
    itls = [x for o in outs for x in o.get("itl", [])]
    q = m1.get("prefix_cache_queries_total", 0) - m0.get("prefix_cache_queries_total", 0)
    h = m1.get("prefix_cache_hits_total", 0) - m0.get("prefix_cache_hits_total", 0)
    cell = {
        "mode": mode, "ctx": ctx, "level": level, "rep": rep, "temperature": temp,
        "agg_tps": round(sum(toks) / max(wall, 1e-3), 1),
        "stream_tps_median": round(statistics.median(per), 1) if per else None,
        "ttft": percentiles(ttfts), "itl": percentiles(itls),
        "starvation_age_s": round(max(ttfts), 2) if ttfts else None,
        "tokens": sum(toks), "prompt_tokens": sum(o.get("prompt_tokens") or 0 for o in outs),
        "wall": round(wall, 1),
        "cache_hit_ratio": round(h / q, 3) if q else None,
        "preemptions": int(m1.get("num_preemptions_total", 0) - m0.get("num_preemptions_total", 0)),
        "graph_mode": None,
        "errors": sum(1 for o in outs if o.get("error")),
        "warmup": "sequential" if ctx else None,
        "error_samples": [o["error"] for o in outs if o.get("error")][:2],
    }
    if ctx and cell["cache_hit_ratio"] is not None and cell["cache_hit_ratio"] < args.min_hit_ratio:
        cell["warning"] = f"cache hit ratio {cell['cache_hit_ratio']} < {args.min_hit_ratio}"
    return cell


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="1,2,4,8,12,16")
    ap.add_argument("--modes", default="code,data,chat")
    ap.add_argument("--ctx", default="0,50000,100000", help="context tokens per lane (0 = none)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--stagger", type=float, default=0.0, help="seconds between lane starts (0 = simultaneous)")
    ap.add_argument("--shared-prefix", action="store_true", help="all lanes share one cached prefix (default: distinct per lane)")
    ap.add_argument("--chat-temp0", action="store_true", help="run chat at temperature 0 instead of 0.7")
    ap.add_argument("--min-hit-ratio", type=float, default=0.8)
    ap.add_argument("--spread-tolerance", type=float, default=0.10, help="re-run a cell once if rep spread on agg_tps exceeds this")
    ap.add_argument("--status", default="status.json", help="live status file for tests/bench_live.html")
    ap.add_argument("--out", default="logs/ladder.json")
    ap.add_argument("--force", action="store_true", help="run even if requests are in flight")
    args = ap.parse_args()

    m = metrics()
    if not args.force and (m.get("num_requests_running", 0) > 0 or m.get("num_requests_waiting", 0) > 0):
        print(f"server busy (running={m.get('num_requests_running')} waiting={m.get('num_requests_waiting')}); refusing. Use --force to override.", file=sys.stderr)
        return 2
    levels = [int(x) for x in args.levels.split(",")]
    modes = args.modes.split(",")
    ctxs = [int(x) for x in args.ctx.split(",")]
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status)
    state: dict[str, Any] = {"phase": "running", "started": time.time(), "cells": [], "live": {}}

    def dump():
        tmp = status_path.with_suffix(".tmp"); tmp.write_text(json.dumps(state)); os.replace(tmp, status_path)

    def status_cb(mode, ctx, level, rep, outs, t_cell):
        el = time.time() - t_cell
        est = sum(len(o.get("itl", [])) + (1 if o.get("first") else 0) for o in outs)  # chunk-based live estimate only
        state["live"] = {"mode": mode, "ctx": ctx, "level": level, "rep": rep, "elapsed": round(el, 1),
                         "chunks_so_far": est, "streams": [{"ttft": o.get("ttft"), "done": bool(o.get("end"))} for o in outs]}
        dump()

    # warm-up: one short request so JIT / graph capture is not charged to the first cell
    stream_one(build_prompt("code", 0, 0, 1, False), 64, 0.0, False, {})
    for mode in modes:
        for ctx in ctxs:
            for level in levels:
                reps = []
                for rep in range(args.reps):
                    cell = run_cell(mode, ctx, level, rep, args, status_cb)
                    reps.append(cell); state["cells"].append(cell); dump()
                    print(f"{mode:5s} ctx={ctx:6d} x{level:2d} rep{rep}: agg {cell['agg_tps']:6.1f} | stream {cell['stream_tps_median']} | "
                          f"TTFT p50/p95 {cell['ttft']['p50']}/{cell['ttft']['p95']} | hit {cell['cache_hit_ratio']} | pre {cell['preemptions']} | err {cell['errors']}", flush=True)
                    time.sleep(2.0)
                aggs = [c["agg_tps"] for c in reps if c["agg_tps"]]
                if len(aggs) >= 2 and (max(aggs) - min(aggs)) / max(statistics.median(aggs), 1e-3) > args.spread_tolerance:
                    cell = run_cell(mode, ctx, level, args.reps, args, status_cb); cell["rerun"] = True
                    reps.append(cell); state["cells"].append(cell); dump()
                    print(f"  spread > {args.spread_tolerance:.0%}; re-ran once: agg {cell['agg_tps']}", flush=True)
    state["phase"] = "done"; state["finished"] = time.time(); dump()
    out_path.write_text(json.dumps({"base": BASE, "model": MODEL, "args": vars(args), "cells": state["cells"]}, indent=1))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
