#!/usr/bin/env python3
"""Tool-calling and verbatim-recall probe through the served chat endpoint at the served
sampling defaults (thinking on, temperature 1.0 / top_p 0.95). Every case has a checker:
right tool, object-shaped arguments (an absent, empty or non-object — null/array/number —
payload is recorded as a failed sample, see decode_args), no leaked <tool_call>/<arg_key>
text, and — for the two long-context Edit cases — the requested relative file_path and a
deterministic reference outcome: the candidate's old_string -> new_string is applied literally
to the file content the case supplied and the RESULT must match the expected edit (see
applied_edit; candidate code is never executed). The long cases also require verbatim recall
of a numbered line at 2k / 9k / 18k tokens depth. Scoring is pure (score_completion), so
fixture tests can drive it without the endpoint.

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
# the corrected reference the fix_after_read edit must produce when applied to BUGGY
FIXED = BUGGY.replace("items[-n+1:]", "items[-n:]")


def call(name, args, cid="call_1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def same_rel_path(path, target):
    """True when file_path names exactly the intended relative target; a leading ./ is harmless.

    The two Edit cases are a deterministic reference rubric against the file the prompt named, so a
    sibling basename (notutils.py), another directory or an unknown absolute root does not qualify.
    """
    if not isinstance(path, str):
        return False
    p = path.strip()
    while p.startswith("./"):
        p = p[2:]
    return p == target


def applied_edit(content, old, new):
    """Result of applying the candidate's old_string -> new_string literally to the reference content.

    None when the call is unusable: old_string and new_string must be real strings, and old_string must
    be non-empty and occur exactly once in the reference content, so the edit is unambiguous. Only the
    RESULT of the replacement is scored; candidate code is never executed, and an Edit call that merely
    mentions the expected tokens proves nothing.
    """
    if not (isinstance(old, str) and isinstance(new, str)):
        return None
    if not old.strip() or content.count(old) != 1:
        return None
    return content.replace(old, new, 1)


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
    # multi-turn: tool result fed back, then the fix via Edit: the requested utils.py and a replacement
    # whose application to that content yields the corrected reference — not a call that merely mentions
    # the buggy and fixed tokens, and not any other file
    def fixes_last_n(a):
        if not same_rel_path(a.get("file_path"), "utils.py"):
            return False
        return applied_edit(BUGGY, a.get("old_string"), a.get("new_string")) == FIXED

    mt = u("Read utils.py and fix the bug in last_n.")
    mt += [{"role": "assistant", "content": "", "tool_calls": [call("Read", {"file_path": "utils.py"})]},
           {"role": "tool", "tool_call_id": "call_1", "content": BUGGY}]
    add("fix_after_read", mt, want("Edit", fixes_last_n))
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
    # the requested edit: applying the replacement to the dumped content must leave every other line
    # untouched and end THAT line with '# reviewed' — a comment placed elsewhere, a changed neighbouring
    # line or a shortened file does not score
    def appends_reviewed(a):
        if not same_rel_path(a.get("file_path"), "overlay/exl3.py"):
            return False
        got = applied_edit(file_text, a.get("old_string"), a.get("new_string"))
        if got is None:
            return False
        want = re.compile(re.escape(target.rstrip()) + r"[ \t]*# reviewed[ \t]*$")
        lines = got.split("\n")
        return (len(lines) == len(src_lines) and lines[:ln - 1] == src_lines[:ln - 1]
                and want.match(lines[ln - 1]) is not None and lines[ln:] == src_lines[ln:])

    add("edit_deep_18k", [{"role": "system", "content": SYS}, {"role": "user", "content": dump + f"Use the Edit tool to append the comment '# reviewed' to the end of line {ln}. Copy old_string exactly from the file."}],
        want("Edit", appends_reviewed), tags=("long",))
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


def decode_args(raw):
    """Decode a tool-call arguments payload; every checker reads named fields off an object.

    The payload must be present and decode to a JSON object: an absent, empty or non-object payload is a
    recorded invalid-argument failure, while an explicit object string (including "{}") is valid — a
    payload is never replaced by an empty object. Returns (args, invalid): args is {} when the payload
    is unusable, and invalid marks the recorded failure.
    """
    if not isinstance(raw, str) or not raw.strip():
        return {}, True
    try:
        args = json.loads(raw)
    except Exception:  # noqa: BLE001
        return {}, True
    return (args, False) if isinstance(args, dict) else ({}, True)


def stored_tool_call(r):
    """Rebuild a recorded tool call for --rescore; invalid marks a stored payload that is missing or no longer decodes to an object.

    A record with no tool is a legitimate no-tool sample; a tool recorded with a missing, empty or
    non-object payload is an invalid-argument failure rather than a silent no-tool. Records keep at most
    600 characters of the payload, so one truncated mid-object is reported as invalid as well.
    """
    if not r.get("tool"):
        return None, False
    args, invalid = decode_args(r.get("args"))
    return {"name": r["tool"], "args": args}, invalid


def score_completion(c, d):
    """Score one parsed /v1/chat/completions response. Pure: fixture tests drive it without the endpoint."""
    ch = d["choices"][0]; m = ch["message"]
    content = m.get("content") or ""; reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    tcs = m.get("tool_calls") or []
    tc = None; evidence = None; bad_json = False
    if tcs:
        f = tcs[0]["function"]
        raw = f.get("arguments")
        args, bad_json = decode_args(raw)
        tc = {"name": f.get("name"), "args": args}
        evidence = raw if isinstance(raw, str) else json.dumps(raw)  # keep the offending payload as evidence
    if bad_json:
        # bad_json: the arguments payload was absent, unparseable or not a JSON object, so no field
        # predicate can apply and the sample is a failure rather than a checker crash.
        ok, why = False, "tool arguments are not a JSON object"
    else:
        ok, why = c["check"](tc, content, reasoning)
    leak = bool(LEAK.search(content))
    return {"ok": bool(ok and not leak and not bad_json), "why": why,
            "tool": tc["name"] if tc else None, "n_tool_calls": len(tcs), "bad_json": bad_json, "leak": leak,
            "finish_reason": ch.get("finish_reason"),
            "completion_tokens": d["usage"]["completion_tokens"], "prompt_tokens": d["usage"]["prompt_tokens"],
            "reasoning_chars": len(reasoning), "args": evidence[:600] if evidence is not None else None,
            "content": content[:400]}


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
    rec = score_completion(c, d)
    rec.update({"id": c["id"], "sample": sample, "tags": c["tags"], "secs": round(time.time() - t0, 1)})
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
            tc, invalid = stored_tool_call(r)
            r["bad_json"] = bool(r.get("bad_json")) or invalid  # a recorded failure is never cleared here
            ok, why = (False, "tool arguments are not a JSON object") if invalid else c["check"](tc, r.get("content", ""), "")
            r["ok"] = bool(ok and not r.get("leak") and not r["bad_json"]); r["why"] = why
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
