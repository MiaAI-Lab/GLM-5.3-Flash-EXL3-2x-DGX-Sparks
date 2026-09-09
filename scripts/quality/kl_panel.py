#!/usr/bin/env python3
"""Long-context logprob panel for the served model (numerics gate, prefill only).

Captures prompt logprobs (top-K per position) for fixed long texts and two tool-calling
transcripts, then compares two captures by context depth: KL(A||B) on the top-K union
support, argmax agreement, and mean NLL of the actual token. Prefill-only, so it isolates
target numerics (FP8 / ABLIT / kernels) from the decode policy (adaptive-k does not touch it).

usage: kl_panel.py capture OUT.json [--k 20] [--url http://127.0.0.1:8888]
       kl_panel.py compare A.json B.json
"""
import argparse, json, math, os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "GLM-5.3-Flash-EXL3"
BINS = [(0, 2000), (2000, 4000), (4000, 8000), (8000, 16000), (16000, 32000)]

TOOLS = [
    {"type": "function", "function": {"name": "Read", "description": "Read a file from the filesystem.",
     "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "Bash", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "Edit", "description": "Replace old_string with new_string in file_path. old_string must match the file exactly.",
     "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["file_path", "old_string", "new_string"]}}},
]


def post(url, path, body, timeout=1800):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def read(rel):
    with open(os.path.join(ROOT, rel)) as fh:
        return fh.read()


def tool_transcript(long: bool):
    """A realistic agentic exchange ending in an Edit call whose old_string is copied from the file."""
    src = read("overlay/exl3.py").splitlines()
    n = 1500 if long else 120
    shown = "\n".join(f"{i+1:>6}\t{l}" for i, l in enumerate(src[:n]))
    target_line = src[n - 40]
    msgs = [
        {"role": "system", "content": "You are a coding agent. Use the tools to inspect and change the repository."},
        {"role": "user", "content": "Read overlay/exl3.py, then rename nothing; just add a comment '# reviewed' at the end of the line I point to next."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": json.dumps({"file_path": "overlay/exl3.py"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": shown},
        {"role": "user", "content": f"Append the comment to line {n-39}."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "Edit", "arguments": json.dumps({"file_path": "overlay/exl3.py", "old_string": target_line, "new_string": target_line + "  # reviewed"})}}]},
    ]
    return msgs


def items():
    out = {}
    out["code_exl3"] = {"kind": "text", "text": read("overlay/exl3.py")}
    out["code_start_sh"] = {"kind": "text", "text": read("start.sh")}
    out["prose_docs"] = {"kind": "text", "text": read("docs/astra-new-decode.md")}
    out["tools_short"] = {"kind": "chat", "messages": tool_transcript(False)}
    out["tools_long"] = {"kind": "chat", "messages": tool_transcript(True)}
    return out


def capture(url, out, k, only=None):
    res = {"meta": {"url": url, "k": k, "time": time.strftime("%FT%TZ", time.gmtime())}, "items": {}}
    for name, it in items().items():
        if only and name not in only:
            continue
        t0 = time.time()
        if it["kind"] == "text":
            ids = post(url, "/tokenize", {"model": MODEL, "prompt": it["text"]})["tokens"]
            d = post(url, "/v1/completions", {"model": MODEL, "prompt": it["text"], "max_tokens": 1, "temperature": 0, "prompt_logprobs": k})
            pl = d.get("prompt_logprobs") or d["choices"][0].get("prompt_logprobs") or []
            tail_start = tail_end = 0
        else:
            # continue_final_message drops the final assistant turn's tool_calls, so render the
            # transcript with a trailing tool ack + generation prompt and score the span of the
            # final tool-call turn: [tail_start, tail_end).
            msgs = it["messages"] + [{"role": "tool", "tool_call_id": "call_2", "content": "ok"}]
            body = {"model": MODEL, "messages": msgs, "tools": TOOLS}
            ids = post(url, "/tokenize", dict(body, add_generation_prompt=True))["tokens"]
            head = post(url, "/tokenize", dict(body, messages=it["messages"][:-1], add_generation_prompt=True))["tokens"]
            upto = post(url, "/tokenize", dict(body, messages=it["messages"], add_generation_prompt=False))["tokens"]
            tail_start, tail_end = len(head), len(upto)
            assert ids[:tail_end] == upto, "tokenization of the tool-call turn is not a prefix"
            d = post(url, "/v1/chat/completions", dict(body, max_tokens=1, temperature=0, prompt_logprobs=k))
            pl = d.get("prompt_logprobs") or []
        if len(pl) != len(ids):
            print(f"WARN {name}: {len(pl)} logprob positions vs {len(ids)} tokens", file=sys.stderr)
        # compact: per position -> {tok: logprob} (top-k plus the actual token), and the actual token id
        pos = []
        for i, p in enumerate(pl):
            if not p:
                pos.append(None)
                continue
            pos.append({str(t): v["logprob"] for t, v in p.items()})
        res["items"][name] = {"ids": ids, "pos": pos, "tail_start": tail_start, "tail_end": tail_end}
        nll = [-pos[i][str(ids[i])] for i in range(1, min(len(pos), len(ids))) if pos[i] and str(ids[i]) in pos[i]]
        print(f"{name:14s} tokens={len(ids):6d} tail={tail_start}-{tail_end} mean NLL={sum(nll)/max(1,len(nll)):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh)
    print("wrote", out)


def compare(a_path, b_path):
    A = json.load(open(a_path)); B = json.load(open(b_path))
    print(f"A={a_path}\nB={b_path}")
    print(f"{'item':14s} {'depth':>12s} {'n':>6s} {'NLL_A':>7s} {'NLL_B':>7s} {'dNLL':>7s} {'KL(A||B)':>9s} {'argmax=':>8s} {'act.dlp>1':>9s}")
    for name in A["items"]:
        if name not in B["items"]:
            continue
        ia, ib = A["items"][name], B["items"][name]
        ids = ia["ids"]
        if ib["ids"] != ids:
            print(f"{name}: token ids differ between captures — skipping"); continue
        # Chat items: score ONLY the final assistant tool-call turn. User/tool turns are loss-masked in
        # chat-tuned models (never trained to predict them), so their prompt-NLL is meaningless noise.
        ranges = [(ia["tail_start"], ia.get("tail_end") or len(ids))] if ia.get("tail_start") else list(BINS)
        for lo, hi in ranges:
            kls = []; nlla = []; nllb = []; agree = 0; n = 0; big = 0
            for i in range(max(lo, 1), min(hi, len(ids), len(ia["pos"]), len(ib["pos"]))):
                x, y = ia["pos"][i], ib["pos"][i]
                if not x or not y:
                    continue
                keys = set(x) & set(y)
                if not keys:
                    continue
                px = {k: math.exp(x[k]) for k in keys}; py = {k: math.exp(y[k]) for k in keys}
                zx = sum(px.values()); zy = sum(py.values())
                kls.append(sum((px[k]/zx) * math.log((px[k]/zx) / (py[k]/zy)) for k in keys))
                agree += max(x, key=x.get) == max(y, key=y.get); n += 1
                t = str(ids[i])
                if t in x and t in y:
                    nlla.append(-x[t]); nllb.append(-y[t]); big += abs(x[t] - y[t]) > 1.0
            if n == 0:
                continue
            label = f"{lo}-{hi}" if (lo, hi) in BINS else f"tail@{lo}"
            print(f"{name:14s} {label:>12s} {n:6d} {sum(nlla)/max(1,len(nlla)):7.4f} {sum(nllb)/max(1,len(nllb)):7.4f} {sum(nllb)/max(1,len(nllb))-sum(nlla)/max(1,len(nlla)):+7.4f} {sum(kls)/len(kls):9.5f} {agree/n:8.4f} {big/max(1,len(nlla)):9.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "compare"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--only", default=None, help="comma list of item names to capture")
    a = ap.parse_args()
    if a.mode == "capture":
        capture(a.url, a.paths[0], a.k, set(a.only.split(",")) if a.only else None)
    else:
        compare(a.paths[0], a.paths[1])
