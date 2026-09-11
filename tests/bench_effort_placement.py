#!/usr/bin/env python3
"""Where should the `Reasoning Effort:` directive live? HEAD vs TAIL vs PRE vs NONE.

Runs against a live server. Nothing is persisted server-side and no template
is changed: the server renders the prompt once (via /tokenize + /detokenize),
the directive is moved by string surgery, each arm is re-tokenized and sent as
token ids through /v1/completions at temperature 0, so every byte of every arm
is controlled and the arms share one code path.

  head  directive at char ~39, right after [gMASK]<sop>   (template before this change)
  pre   directive immediately before the LAST <|user|>    (template after this change)
  tail  directive after the last user turn, just before <|assistant|>  (rejected)
  none  no directive at all (thinking on, effort unspecified)

Phase 0  cache proof: effort low -> high -> low on a long prompt (~16.5k tokens by
         default; --phase0-tokens 16500,42000,128000 for more). `head` drops
         cached_tokens to 0 on every change; `pre`/`tail` keep the pages.
         Needs --enable-prompt-tokens-details on the server for usage.cached_tokens
         (EXTRA_ARGS="--enable-prompt-tokens-details" in .env; the launcher does not add it).
Phase 1  obedience + quality: fixtures x tiers x arms x reps, thinking ON.
         Reasoning tokens are counted from the <think> block; answers are checked
         by execution (code), parsing (json) or exact value (math), never by
         token count alone. A `</think>` inside the answer channel is the
         malformed-output signature that sank tail placement.

  python3 tests/bench_effort_placement.py --reps 4
  python3 tests/bench_effort_placement.py --phase0-tokens 16500,42000,128000 --arms head,pre --fixtures tool
  python3 tests/bench_effort_placement.py --fixtures code --arms head,pre --tiers high --reps 10 --skip-phase0
  python3 tests/bench_effort_placement.py --summary-only --out results/<file>.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "http://127.0.0.1:8888"
MODEL = "GLM-5.3-Flash-EXL3"
API_KEY_ENV = "API_KEY"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

EFFORT_RE = re.compile(r"<\|system\|>Reasoning Effort: (Low|High|Max)")
GEN_TAIL = "<|assistant|><think>"

_cfg = {"base": BASE, "model": MODEL, "api_key": ""}


def post(path, body, timeout=900):
    req = urllib.request.Request(_cfg["base"] + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if _cfg["api_key"]:
        req.add_header("Authorization", f"Bearer {_cfg['api_key']}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize_text(text):
    return post("/tokenize", {"model": _cfg["model"], "prompt": text, "add_special_tokens": False})["tokens"]


def server_tokens(messages, tier, tools=None):
    body = {"model": _cfg["model"], "messages": messages, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": tier}}
    if tools:
        body["tools"] = tools
    return post("/tokenize", body)["tokens"]


def render_server(messages, tier, tools=None):
    """The server's own rendering as text, whatever placement its template uses."""
    text = post("/detokenize", {"model": _cfg["model"], "tokens": server_tokens(messages, tier, tools)})["prompt"]
    assert len(EFFORT_RE.findall(text)) == 1, "expected exactly one effort directive: " + text[:120]
    assert text.endswith(GEN_TAIL), "unexpected tail: " + text[-40:]
    return text


def make_variants(server_text):
    """Return {arm: text} for head / tail / pre / none, all from one rendering."""
    m = EFFORT_RE.search(server_text)
    line = m.group(0)
    stripped = server_text.replace(line, "", 1)
    assert not EFFORT_RE.search(stripped)
    assert stripped.startswith("[gMASK]<sop>")
    head = "[gMASK]<sop>" + line + stripped[len("[gMASK]<sop>"):]
    tail = stripped[: -len(GEN_TAIL)] + line + GEN_TAIL
    k = stripped.rfind("<|user|>")
    pre = stripped[:k] + line + stripped[k:]
    variants = {"head": head, "tail": tail, "none": stripped, "pre": pre}
    # which arm is the server's own placement? (round-trip guard uses it)
    server_arm = next(a for a, t in variants.items() if t == server_text)
    return variants, server_arm


def complete(ids, max_tokens, seed=0):
    t0 = time.time()
    r = post("/v1/completions", {"model": _cfg["model"], "prompt": ids, "max_tokens": max_tokens,
                                 "temperature": 0, "seed": seed, "stream": False})
    dt = time.time() - t0
    ch = r["choices"][0]
    return ch["text"], ch.get("finish_reason"), r.get("usage", {}), dt


def split_think(text):
    # the prompt already ends in <think>, so generation begins inside the block
    if "</think>" in text:
        think, ans = text.split("</think>", 1)
        return think, ans, True
    return text, "", False


# ---------------- fixtures (synthetic, checkable) ----------------
SYS = "You are a precise assistant. Follow the output format exactly."


def check_code(ans):
    fences = re.findall(r"```(?:python)?\n(.*?)```", ans, re.S)
    if len(fences) != 1:
        return False, f"fences={len(fences)}"
    ns = {}
    try:
        exec(fences[0], ns)  # synthetic fixture, model-written function under test
        f = ns["is_balanced"]
        ok = (f("([]{})") is True and f("([)]") is False and f("") is True
              and f("((") is False and f("{[()()]}") is True and f("]") is False)
        return ok, "asserts" if ok else "wrong output"
    except Exception as e:  # noqa: BLE001 - any failure is a failed fixture
        return False, type(e).__name__


def check_json(ans):
    s = ans.strip()
    if s.startswith("```"):
        return False, "fenced"
    try:
        d = json.loads(s)
    except Exception:  # noqa: BLE001
        return False, "not raw json"
    ok = (set(d) == {"city", "population_millions", "founded_year", "is_capital"}
          and d.get("is_capital") is True and d.get("city") == "Tokyo")
    return ok, "keys/values" if ok else f"bad content {list(d)[:6]}"


def check_math(ans):
    m = re.findall(r"ANSWER:\s*([0-9,]+)", ans)
    if not m:
        return False, "no ANSWER line"
    return m[-1].replace(",", "") == "201003", m[-1]


def check_tool(ans):
    calls = re.findall(r"<tool_call>(\w+)(.*?)</tool_call>", ans, re.S)
    if len(calls) != 1:
        return False, f"tool_calls={len(calls)}"
    name, args = calls[0]
    if name != "multiply":
        return False, f"wrong tool {name}"
    vals = dict(re.findall(r"<arg_key>(\w+)</arg_key><arg_value>([^<]*)</arg_value>", args))
    ok = vals.get("a", "").strip() == "8" and vals.get("b", "").strip() == "9"
    return ok, "args a=8 b=9" if ok else f"bad args {vals}"


TOOLS = [{"type": "function", "function": {"name": "multiply", "description": "Multiply two integers.",
          "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                         "required": ["a", "b"]}}}]

# fixture = (messages, checker[, tools]); the tool fixture renders through the same template path
# with a tool block, one prior tool call + result, and a follow-up user turn.
FIXTURES = {
    "code": (
        [{"role": "system", "content": SYS},
         {"role": "user", "content": "Write a Python function is_balanced(s) that returns True if the brackets ()[]{} in s are balanced and properly nested, else False. Respond with exactly one fenced python code block and nothing else."}],
        check_code),
    "json": (
        [{"role": "system", "content": SYS},
         {"role": "user", "content": "Return ONLY a raw JSON object, no code fence, no prose, with keys city, population_millions, founded_year, is_capital, describing Tokyo, Japan."}],
        check_json),
    "math": (
        [{"role": "system", "content": SYS},
         {"role": "user", "content": "What is the sum of all integers from 1 to 1000 inclusive that are divisible by 3 or by 5 but not by 15? Finish with a line of the form ANSWER: <integer>."}],
        check_math),
    "tool": (
        [{"role": "system", "content": SYS + " Always use the multiply tool for arithmetic."},
         {"role": "user", "content": "Use the multiply tool to compute 6 times 7."},
         {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
             "function": {"name": "multiply", "arguments": "{\"a\": 6, \"b\": 7}"}}]},
         {"role": "tool", "tool_call_id": "call_1", "content": "42"},
         {"role": "user", "content": "Thanks. Now use the multiply tool for 8 times 9."}],
        check_tool, TOOLS),
}
TIERS = ["low", "high", "max"]
ARMS = ["head", "tail", "none", "pre"]


def _record(i):
    return f"Record {i}: the quick brown fox jumps over the lazy dog number {i}."


def phase0(out, sizes):
    """Cache proof with a long shared prefix (> 2 x 3584-token page alignment), one pass per prompt size."""
    per_record = len(tokenize_text(" ".join(_record(i) for i in range(1, 101)))) / 100
    res = {}
    for target in sizes:
        n = max(2, int(target / per_record))
        filler = " ".join(_record(i) for i in range(1, n))
        msgs = [{"role": "system", "content": SYS + "\n\n" + filler},
                {"role": "user", "content": "Reply with the single word OK."}]
        v_low, _ = make_variants(render_server(msgs, "low"))
        v_high, _ = make_variants(render_server(msgs, "high"))
        for arm in [a for a in ARMS if a != "none"]:
            seq = []
            for tier, v in (("low", v_low), ("high", v_high), ("low", v_low)):
                ids = tokenize_text(v[arm])
                _, _, usage, dt = complete(ids, 4)
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                seq.append({"tier": tier, "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached, "ttft_s": round(dt, 2)})
                print(f"phase0 ~{target} {arm:4s} {tier:4s} prompt={usage.get('prompt_tokens')} cached={cached} t={dt:.2f}s", flush=True)
            res[f"{target}/{arm}"] = seq
        Path(out["_path"]).write_text(json.dumps({**out, "phase0": res}, indent=1))
    out["phase0"] = res
    if all(s["cached_tokens"] is None for seq in res.values() for s in seq):
        print("phase0: usage.prompt_tokens_details is absent; the server is not started with "
              "--enable-prompt-tokens-details (EXTRA_ARGS=\"--enable-prompt-tokens-details\" in .env). "
              "TTFT still tells the story; cached_tokens will read null.", flush=True)


def run_one(name, tier, arm, rep, ids, checker, max_tokens):
    text, fin, usage, dt = complete(ids, max_tokens, seed=rep)
    think, ans, closed = split_think(text)
    think_toks = len(tokenize_text(think)) if think else 0
    ans_toks = len(tokenize_text(ans)) if ans else 0
    ok, why = checker(ans) if closed else (False, "think never closed")
    malformed = closed and "</think>" in ans
    rec = {"fixture": name, "tier": tier, "arm": arm, "rep": rep, "think_tokens": think_toks,
           "answer_tokens": ans_toks, "completion_tokens": usage.get("completion_tokens"),
           "finish": fin, "closed": closed, "correct": ok, "malformed": malformed, "why": why,
           "secs": round(dt, 1), "answer_head": ans.strip()[:160], "answer_full": ans, "think_full": think}
    print(f"{name:4s} {tier:4s} {arm:4s} r{rep} think={think_toks:5d} ans={ans_toks:4d} ok={ok!s:5s} "
          f"{'MALFORMED ' if malformed else ''}{why:14s} {dt:6.1f}s fin={fin}", flush=True)
    return rec


def phase1(out, reps, max_tokens, conc):
    jobs = []
    for name, fx in FIXTURES.items():
        msgs, checker, tools = (fx + (None,))[:3]
        for tier in TIERS:
            variants, server_arm = make_variants(render_server(msgs, tier, tools))
            for arm in ARMS:
                ids = tokenize_text(variants[arm])
                if arm == server_arm:
                    # round-trip guard: our re-tokenized text must equal the server's own tokens
                    assert ids == server_tokens(msgs, tier, tools), f"round-trip mismatch {name}/{tier}"
                for rep in range(reps):
                    jobs.append((name, tier, arm, rep, ids, checker))
    out["runs"] = []
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(run_one, *j, max_tokens) for j in jobs]
        for f in cf.as_completed(futs):
            out["runs"].append(f.result())
            Path(out["_path"]).write_text(json.dumps(out, indent=1))


def summarize(out):
    runs = out.get("runs", [])
    print("\n=== reasoning tokens (mean), correctness, malformed answers, by fixture x arm x tier ===")
    for name in FIXTURES:
        for arm in ARMS:
            row = []
            for tier in TIERS:
                rs = [r for r in runs if r["fixture"] == name and r["arm"] == arm and r["tier"] == tier]
                if not rs:
                    row.append(f"{tier}: -")
                    continue
                mean = sum(r["think_tokens"] for r in rs) / len(rs)
                okc = sum(1 for r in rs if r["correct"])
                bad = sum(1 for r in rs if r.get("malformed") or "</think>" in r.get("answer_full", ""))
                secs = sum(r["secs"] for r in rs) / len(rs)
                row.append(f"{tier}: {mean:6.0f} tok {okc}/{len(rs)} ok {secs:5.1f}s" + (f" {bad} malformed" if bad else ""))
            print(f"{name:4s} {arm:4s} | " + " | ".join(row))


def main() -> int:
    global FIXTURES, ARMS, TIERS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--api-key-env", default=API_KEY_ENV)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--conc", type=int, default=2, help="in-flight completions (batched decode)")
    ap.add_argument("--out", help="results json (default results/glm53-exl3-effort-placement-<utc>.json)")
    ap.add_argument("--skip-phase0", action="store_true")
    ap.add_argument("--phase0-tokens", default="16500",
                    help="comma list of approximate prompt sizes for the cache proof (e.g. 16500,42000,128000)")
    ap.add_argument("--summary-only", action="store_true", help="re-print the table from --out")
    ap.add_argument("--fixtures", default=",".join(FIXTURES), help="comma list subset")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--tiers", default=",".join(TIERS))
    a = ap.parse_args()
    _cfg.update(base=a.base_url.rstrip("/").removesuffix("/v1"), model=a.model,
                api_key=os.environ.get(a.api_key_env, ""))
    FIXTURES = {k: v for k, v in FIXTURES.items() if k in a.fixtures.split(",")}
    ARMS = [x for x in ARMS if x in a.arms.split(",")]
    TIERS = [x for x in TIERS if x in a.tiers.split(",")]
    if a.summary_only:
        summarize(json.loads(Path(a.out).read_text()))
        return 0
    out_path = Path(a.out) if a.out else RESULTS_DIR / \
        f"glm53-exl3-effort-placement-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {"_path": str(out_path), "started": datetime.now(timezone.utc).isoformat(), "model": a.model,
           "base_url": a.base_url, "reps": a.reps, "max_tokens": a.max_tokens, "conc": a.conc}
    if not a.skip_phase0:
        phase0(out, [int(x) for x in a.phase0_tokens.split(",") if x])
    phase1(out, a.reps, a.max_tokens, a.conc)
    out["finished"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(json.dumps(out, indent=1))
    summarize(out)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
