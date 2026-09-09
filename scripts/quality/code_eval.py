#!/usr/bin/env python3
"""HumanEval + MBPP(sanitized) pass@1 through the served chat endpoint, with the SERVED
sampling defaults (temperature 1.0 / top_p 0.95 from generation_config, thinking on) unless
overridden — i.e. what users actually get. Fixed seeded subset so arms are comparable.

usage: code_eval.py OUT.json --data DIR [--n-he 60] [--n-mbpp 60] [--workers 3] [--seed 1]
                    [--max-tokens 6144] [--no-think] [--temperature T] [--url URL]
"""
import argparse, json, os, random, re, subprocess, sys, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = "GLM-5.3-Flash-EXL3"


def post(url, body, timeout=1800):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", flags=re.S)
    return (blocks[-1] if blocks else (text or "")).rstrip() + "\n"


def run_program(src, timeout=15):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "prog.py")
        open(p, "w").write(src)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=td, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1][:200]


def load_tasks(data, n_he, n_mbpp, seed):
    rnd = random.Random(seed)
    he = [json.loads(l) for l in open(os.path.join(data, "humaneval.jsonl"))]
    mb = [json.loads(l) for l in open(os.path.join(data, "mbpp_sanitized_test.jsonl"))]
    rnd.shuffle(he); rnd.shuffle(mb)
    tasks = []
    for t in he[:n_he]:
        prompt = ("Complete the following Python function. Reply with the complete function "
                  "(signature, imports and body) in a single ```python code block.\n\n```python\n" + t["prompt"] + "\n```")
        tasks.append({"id": t["task_id"], "set": "humaneval", "prompt": prompt, "he": t})
    for t in mb[:n_mbpp]:
        tests = "\n".join(t["test_list"])
        prompt = (t["prompt"] + "\nThe function must pass these tests:\n```python\n" + tests +
                  "\n```\nReply with the complete solution in a single ```python code block.")
        tasks.append({"id": f"mbpp/{t['task_id']}", "set": "mbpp", "prompt": prompt, "mb": t})
    return tasks


def build_program(task, code):
    if task["set"] == "humaneval":
        t = task["he"]
        if f"def {t['entry_point']}" not in code:
            code = t["prompt"] + "\n" + code
        return code + "\n\n" + t["test"] + f"\n\ncheck({t['entry_point']})\n"
    t = task["mb"]
    return "\n".join(t.get("test_imports") or []) + "\n" + code + "\n\n" + "\n".join(t["test_list"]) + "\n"


def one(url, task, a):
    body = {"model": MODEL, "messages": [{"role": "user", "content": task["prompt"]}], "max_tokens": a.max_tokens}
    if a.temperature is not None:
        body["temperature"] = a.temperature
    if a.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    try:
        d = post(url, body)
    except Exception as e:  # noqa: BLE001
        return {"id": task["id"], "set": task["set"], "passed": False, "error": f"http: {e}"}
    m = d["choices"][0]["message"]
    content = m.get("content") or ""
    reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    code = extract_code(content)
    ok, err = run_program(build_program(task, code))
    return {"id": task["id"], "set": task["set"], "passed": ok, "error": err if not ok else "",
            "finish_reason": d["choices"][0].get("finish_reason"), "completion_tokens": d["usage"]["completion_tokens"],
            "reasoning_chars": len(reasoning), "content_chars": len(content), "has_code_block": "```" in content,
            "secs": round(time.time() - t0, 1), "code": code[:4000]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-he", type=int, default=60)
    ap.add_argument("--n-mbpp", type=int, default=60)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    a = ap.parse_args()
    tasks = load_tasks(a.data, a.n_he, a.n_mbpp, a.seed)
    t0 = time.time()
    # Submit in synchronized rounds of `workers`: this kit's GLM53_MIXED_PREFILL_CHUNK=skip policy
    # defers a new request's prefill while any peer is decoding, so a rolling pool serializes.
    # Every finished task is appended to OUT.partial.jsonl so a killed run keeps its results.
    partial = a.out + ".partial.jsonl"
    done = {}
    if os.path.exists(partial):
        for l in open(partial):
            r = json.loads(l); done[r["id"]] = r
    todo = [t for t in tasks if t["id"] not in done]
    print(f"{len(done)} tasks already done, {len(todo)} to run in rounds of {a.workers}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex, open(partial, "a") as fh:
        for i in range(0, len(todo), a.workers):
            for r in ex.map(lambda t: one(a.url, t, a), todo[i:i + a.workers]):
                done[r["id"]] = r; fh.write(json.dumps(r) + "\n"); fh.flush()
            print(f"round {i // a.workers + 1}: {sum(r['passed'] for r in done.values())}/{len(done)} passed, {round(time.time() - t0)}s", flush=True)
    results = [done[t["id"]] for t in tasks if t["id"] in done]
    summary = {}
    for s in ("humaneval", "mbpp"):
        rs = [r for r in results if r["set"] == s]
        if not rs:
            continue
        summary[s] = {"n": len(rs), "pass": sum(r["passed"] for r in rs), "pass_rate": round(sum(r["passed"] for r in rs) / len(rs), 4),
                      "truncated": sum(r.get("finish_reason") == "length" for r in rs), "no_code_block": sum(not r.get("has_code_block", True) for r in rs),
                      "mean_completion_tokens": round(sum(r.get("completion_tokens", 0) for r in rs) / len(rs)),
                      "errors": sum(bool(r.get("error", "").startswith("http")) for r in rs)}
    summary["secs"] = round(time.time() - t0)
    summary["args"] = vars(a)
    json.dump({"summary": summary, "results": results}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
