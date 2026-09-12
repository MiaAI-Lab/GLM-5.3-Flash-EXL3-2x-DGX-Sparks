#!/usr/bin/env python3
"""Replay a deep-swe (mini-swe-agent) trajectory prefix against /v1/responses.

For selected API-call indices, rebuild the exact request `input` list the
harness would send and report the server-reported cached_tokens. This isolates
"prompt forked client-side" from "server evicted the prefix under load".
"""
import argparse
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8888"
MODEL = "GLM-5.3-Flash-EXL3"
BASH_TOOL = {
    "type": "function",
    "name": "bash",
    "description": "Execute a bash command",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
        "required": ["command"],
    },
}


def prepare(messages):
    """Mirror minisweagent LitellmResponseModel._prepare_messages_for_api."""
    out = []
    for msg in messages:
        if msg.get("object") == "response":
            for item in msg.get("output", []):
                out.append({k: v for k, v in item.items() if k != "extra"})
        else:
            out.append({k: v for k, v in msg.items() if k != "extra"})
    return out


def response_indices(messages):
    return [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("object") == "response"]


def call(idx: int, input_items: list, timeout: float = 900) -> dict:
    body = {
        "model": MODEL,
        "input": input_items,
        "tools": [BASH_TOOL],
        "reasoning": {"effort": "high"},
        "temperature": 1.0,
        "top_p": 0.95,
        "max_output_tokens": 1,
    }
    req = urllib.request.Request(
        BASE + "/v1/responses", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode())
    u = obj.get("usage") or {}
    det = u.get("input_tokens_details") or {}
    return {
        "input_tokens": u.get("input_tokens"),
        "cached_tokens": det.get("cached_tokens"),
        "ttft_wall_s": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--calls", default="0,1,2,3,4,5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = json.load(open(args.trajectory))
    messages = d["messages"]
    ridx = response_indices(messages)
    out = {"trajectory": args.trajectory, "calls": []}
    for c in (int(x) for x in args.calls.split(",")):
        prefix = prepare(messages[: ridx[c]])
        r = call(c, prefix)
        r["call"] = c
        print(json.dumps(r), flush=True)
        out["calls"].append(r)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
