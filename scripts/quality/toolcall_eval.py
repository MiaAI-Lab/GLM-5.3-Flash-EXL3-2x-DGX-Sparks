#!/usr/bin/env python3
"""Tool-calling and verbatim-recall probe through the served chat endpoint at the served
sampling defaults (thinking on, temperature 1.0 / top_p 0.95). Every case has a checker:
right tool, well-formed arguments, no leaked <tool_call>/<arg_key> text, and — for the
long-context cases — old_string copied EXACTLY from a 20k-token file (what an Edit tool
needs) and verbatim recall of a numbered line at 2k / 9k / 18k tokens depth.

usage: toolcall_eval.py OUT.json [--samples 3] [--workers 3] [--url URL] [--no-think] [--temperature T]
"""
import argparse, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "GLM-5.3-Flash-EXL3"
LEAK = re.compile(r"</?tool_call>|</?arg_key>|</?arg_value>")


def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req}}}


TOOLS = [
    fn("Read", "Read a file from the local filesystem.", {"file_path": {"type": "string", "description": "Absolute or repo-relative path"}}, ["file_path"]),
    fn("Bash", "Run a shell command and return its output.", {"command": {"type": "string"}}, ["command"]),
    fn("Grep", "Search file contents with a regular expression.", {"pattern": {"type": "string"}, "path": {"type": "string", "description": "Directory or file to search"}}, ["pattern"]),
    fn("Edit", "Replace old_string with new_string in file_path. old_string must match the file contents exactly (including whitespace) or the edit fails.",
       {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, ["file_path", "old_string", "new_string"]),
    fn("Write", "Create or overwrite a file with content.", {"file_path": {"type": "string"}, "content": {"type": "string"}}, ["file_path", "content"]),
    fn("get_weather", "Get the current weather for a city.", {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    fn("calculate", "Evaluate an arithmetic expression.", {"expression": {"type": "string"}}, ["expression"]),
]

SYS = "You are a coding agent working in a git repository. Use the provided tools whenever they are needed; call a tool directly instead of describing what you would do."

BUGGY = "def last_n(items, n):\n    \"\"\"Return the last n items of the list.\"\"\"\n    return items[-n+1:]\n\n\ndef total(xs):\n    return sum(xs)\n"


def call(name, args, cid="call_1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def cases():
    src_lines = open(os.path.join(ROOT, "overlay/exl3.py")).read().splitlines()
    numbered = "\n".join(f"{i+1:>6}\t{l}" for i, l in enumerate(src_lines))
    file_text = "\n".join(src_lines)
    cs = []

    def add(cid, msgs, check, tools=TOOLS, expect_tool=True, tags=()):
        cs.append({"id": cid, "messages": msgs, "check": check, "tools": tools, "expect_tool": expect_tool, "tags": list(tags)})

    def want(tool, pred=lambda a: True):
        return lambda tc, content, reasoning: (tc is not None and tc["name"] == tool and pred(tc["args"]), f"want {tool}")

    def no_tool():
        return lambda tc, content, reasoning: (tc is None and bool((content or "").strip()), "want plain answer")

    u = lambda s: [{"role": "system", "content": SYS}, {"role": "user", "content": s}]
    add("read_main", u("Open src/main.py and show me what's in it."), want("Read", lambda a: str(a.get("file_path", "")).endswith("main.py")))
    add("run_tests", u("Run the test suite with pytest -q and tell me if anything fails."), want("Bash", lambda a: "pytest" in str(a.get("command", ""))))
    add("grep_todo", u("Find every TODO comment under the src directory."), want("Grep", lambda a: "todo" in str(a.get("pattern", "")).lower()))
    add("grep_path", u("Search for the pattern 'def parse' but only inside the utils directory."), want("Grep", lambda a: "def parse" in str(a.get("pattern", "")) and "utils" in str(a.get("path", ""))))
    # reading the file before editing it is legitimate agent behaviour: accept Read(config.py) too
    add("edit_debug", u("In config.py, change `DEBUG = True` to `DEBUG = False`."),
        lambda tc, content, reasoning: (tc is not None and ((tc["name"] == "Edit" and "DEBUG = True" in str(tc["args"].get("old_string", "")) and "DEBUG = False" in str(tc["args"].get("new_string", "")))
                                                             or (tc["name"] == "Read" and "config.py" in str(tc["args"].get("file_path", "")))), "want Edit or Read(config.py)"))
    add("write_notes", u("Create a file called NOTES.md containing exactly the single line 'hello world'."), want("Write", lambda a: str(a.get("file_path", "")).endswith("NOTES.md") and "hello world" in str(a.get("content", ""))))
    add("git_status", u("Check the git status of the repo."), want("Bash", lambda a: "git status" in str(a.get("command", ""))))
    add("weather", u("What's the weather like in Paris right now, in celsius?"), want("get_weather", lambda a: "paris" in str(a.get("city", "")).lower() and str(a.get("unit", "celsius")).lower() == "celsius"))
    add("calc", u("Use the calculator to compute 17*23+5."), want("calculate", lambda a: "17" in str(a.get("expression", "")) and "23" in str(a.get("expression", ""))))
    add("no_tool", u("Just say hello in one short sentence. Do not use any tools."), no_tool(), expect_tool=False)
    # multi-turn: tool result fed back, then a fix via Edit whose old_string must be a substring of the file
    mt = u("Read utils.py and fix the bug in last_n.")
    mt += [{"role": "assistant", "content": "", "tool_calls": [call("Read", {"file_path": "utils.py"})]},
           {"role": "tool", "tool_call_id": "call_1", "content": BUGGY}]
    add("fix_after_read", mt, want("Edit", lambda a: str(a.get("old_string", "")) in BUGGY and "last_n" in BUGGY and str(a.get("old_string", "")).strip() != ""))
    # multi-turn: after a passing test run the model should answer, not call again
    at = u("Run the tests with pytest -q.")
    at += [{"role": "assistant", "content": "", "tool_calls": [call("Bash", {"command": "pytest -q"})]},
           {"role": "tool", "tool_call_id": "call_1", "content": "...                                                    [100%]\n3 passed in 0.12s"}]
    add("answer_after_result", at, no_tool(), expect_tool=False)
    # long-context: exact recall of a numbered line at three depths, and an Edit deep into the file
    dump = f"Here is the full content of overlay/exl3.py with line numbers:\n\n{numbered}\n\n"
    for depth_name, ln in (("2k", 160), ("9k", 800), ("18k", 1600)):
        target = src_lines[ln - 1]
        add(f"recall_{depth_name}", [{"role": "user", "content": dump + f"Reply with exactly the content of line {ln} (without the line number), and nothing else."}],
            (lambda t: (lambda tc, content, reasoning: (LEAKSTRIP(content) == t.strip(), f"want {t.strip()[:60]!r}")))(target), tools=[], expect_tool=False, tags=("long",))
    ln = 1500
    target = src_lines[ln - 1]
    add("edit_deep_18k", [{"role": "system", "content": SYS}, {"role": "user", "content": dump + f"Use the Edit tool to append the comment '# reviewed' to the end of line {ln}. Copy old_string exactly from the file."}],
        want("Edit", lambda a: str(a.get("old_string", "")).strip() != "" and str(a.get("old_string", "")) in file_text and target.strip() in str(a.get("old_string", ""))), tags=("long",))
    add("bash_after_long", [{"role": "system", "content": SYS}, {"role": "user", "content": dump + "Now run this module's unit tests with pytest -q."}],
        want("Bash"), tags=("long",))  # any shell call: the probe is tool syntax after 24k tokens, not test discovery
    return cs


def LEAKSTRIP(s):
    s = (s or "").strip()
    m = re.search(r"```[a-z]*\n(.*?)```", s, flags=re.S)
    if m:
        s = m.group(1).strip()
    return s.strip("`").strip()


def post(url, body, timeout=1800):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def run_case(url, c, sample, a):
    body = {"model": MODEL, "messages": c["messages"], "max_tokens": a.max_tokens}
    if c["tools"]:
        body["tools"] = c["tools"]
    if a.temperature is not None:
        body["temperature"] = a.temperature
    if a.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    try:
        d = post(url, body)
    except Exception as e:  # noqa: BLE001
        return {"id": c["id"], "sample": sample, "ok": False, "why": f"http: {e}", "tags": c["tags"]}
    ch = d["choices"][0]; m = ch["message"]
    content = m.get("content") or ""; reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    tcs = m.get("tool_calls") or []
    tc = None; bad_json = False
    if tcs:
        f = tcs[0]["function"]
        try:
            args = json.loads(f.get("arguments") or "{}")
        except Exception:  # noqa: BLE001
            args, bad_json = {}, True
        tc = {"name": f.get("name"), "args": args}
    ok, why = c["check"](tc, content, reasoning)
    leak = bool(LEAK.search(content))
    rec = {"id": c["id"], "sample": sample, "tags": c["tags"], "ok": bool(ok and not leak and not bad_json), "why": why,
           "tool": tc["name"] if tc else None, "n_tool_calls": len(tcs), "bad_json": bad_json, "leak": leak,
           "finish_reason": ch.get("finish_reason"), "completion_tokens": d["usage"]["completion_tokens"], "prompt_tokens": d["usage"]["prompt_tokens"],
           "reasoning_chars": len(reasoning), "secs": round(time.time() - t0, 1),
           "args": json.dumps(tc["args"])[:600] if tc else None, "content": content[:400]}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--only", default=None, help="comma list of case ids")
    ap.add_argument("--rescore", default=None, help="re-apply the current checkers to a stored results JSON and rewrite its summary")
    a = ap.parse_args()
    cs = cases()
    if a.rescore:
        byid = {c["id"]: c for c in cs}
        results = json.load(open(a.rescore))["results"]
        for r in results:
            c = byid.get(r["id"])
            if c is None or r.get("why", "").startswith("http"):
                continue
            tc = {"name": r["tool"], "args": json.loads(r["args"])} if r.get("tool") and r.get("args") else None
            ok, why = c["check"](tc, r.get("content", ""), "")
            r["ok"] = bool(ok and not r.get("leak") and not r.get("bad_json")); r["why"] = why
        summarize_and_write(results, a.out, a); return
    if a.only:
        keep = set(a.only.split(",")); cs = [c for c in cs if c["id"] in keep]
    jobs = [(c, s) for s in range(a.samples) for c in cs]  # sample-major so long prompts share prefix-cache hits within a round
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda j: run_case(a.url, j[0], j[1], a), jobs))
    summarize_and_write(results, a.out, a, t0)


def summarize_and_write(results, out, a, t0=None):
    by = {}
    for r in results:
        by.setdefault(r["id"], []).append(r)
    summary = {"cases": {}, "secs": round(time.time() - t0) if t0 else None, "args": vars(a)}
    tot = ok = 0; long_tot = long_ok = 0; leaks = badjson = 0
    for cid, rs in by.items():
        k = sum(r["ok"] for r in rs); tot += len(rs); ok += k
        leaks += sum(r.get("leak", False) for r in rs); badjson += sum(r.get("bad_json", False) for r in rs)
        if "long" in rs[0]["tags"]:
            long_tot += len(rs); long_ok += k
        summary["cases"][cid] = {"ok": k, "n": len(rs), "tools": [r.get("tool") for r in rs], "why": [r["why"] for r in rs if not r["ok"]][:3]}
    summary["total"] = {"ok": ok, "n": tot, "rate": round(ok / max(1, tot), 4), "leaks": leaks, "bad_json": badjson,
                        "long_ok": long_ok, "long_n": long_tot, "long_rate": round(long_ok / max(1, long_tot), 4)}
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=1)
    for cid, s in summary["cases"].items():
        print(f"{cid:20s} {s['ok']}/{s['n']}  tools={s['tools']}  {('; '.join(s['why']))[:120]}")
    print(json.dumps(summary["total"]))


if __name__ == "__main__":
    main()
