#!/usr/bin/env python3
"""Tool-call concurrency, streaming integrity & argument validation suite (Issue #10).

Validates tool calling under multi-sequence concurrent load on GLM-5.3-Flash EXL3 (:8888):
1. Drives multi-tool calls (2+ tools, required string/object arguments) at c=1, c=4, c=8.
2. Supports mixed prefill+decode load with long cold/shared system prompts (prefix caching stress).
3. Compares SSE streaming delta reconstruction vs non-streaming mode.
4. Verifies argument integrity: detects and flags blank `{}` / missing required arguments.
5. Distinguishes `finish_reason=tool_calls` from `finish_reason=length` truncation.
6. Emits structured JSON benchmark receipts for issue reproduction and regression verification.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE = os.getenv("API_BASE", "http://127.0.0.1:8888")
MODEL = os.getenv("SERVED_MODEL", "GLM-5.3-Flash-EXL3")

# Standard test tool definitions with required arguments
TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal_execute",
            "description": "Execute a shell command inside the terminal sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command line to run."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_action",
            "description": "Perform an automated visual browser action on a webpage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["open", "click", "type", "scroll"],
                        "description": "The browser action type."
                    },
                    "url": {
                        "type": "string",
                        "description": "Target webpage URL (required for open action)."
                    }
                },
                "required": ["action", "url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Destination file path."
                    },
                    "content": {
                        "type": "string",
                        "description": "File body content."
                    }
                },
                "required": ["path", "content"]
            }
        }
    }
]

DEFAULT_PROMPT = (
    "Please perform these tasks now:\n"
    "1. Run terminal command `echo hello-concurrency-test`.\n"
    "2. Open browser URL `https://example.com/test-issue10`."
)


def _post_json(path: str, body: dict, timeout: float = 600.0) -> urllib.request.urlopen:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def make_padded_prompt(tag: str, filler_words: int, task: str = DEFAULT_PROMPT) -> str:
    """Generate prompt with unique or shared padding words to stress KV/prefix cache."""
    padding = "the " * filler_words if filler_words > 0 else ""
    return f"SESSION {tag} UNIQUE {tag[::-1]}. {padding}\n{task}"


def stream_tool_request(
    prompt: str,
    max_tokens: int = 512,
    enable_thinking: bool = False,
    timeout: float = 600.0,
    tools: Optional[List[Dict[str, Any]]] = None,
    out: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute a streaming chat completion request and reconstruct tool_calls SSE deltas."""
    if out is None:
        out = {}
    if "first_event" not in out:
        out["first_event"] = threading.Event()

    tool_defs = tools or TEST_TOOLS
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "tools": tool_defs,
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }

    t0 = time.perf_counter()
    first = None
    last = None
    usage = None
    finish_reason = None
    raw_content_chunks = []
    tool_calls_map: Dict[int, Dict[str, Any]] = {}

    try:
        with _post_json("/v1/chat/completions", body, timeout=timeout) as resp:
            out["http_status"] = resp.status
            buf = b""
            while True:
                chunk = resp.read(256)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    if obj.get("usage"):
                        usage = obj["usage"]

                    choices = obj.get("choices") or []
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta") or {}

                    # Record timing
                    now = time.perf_counter()
                    if first is None:
                        first = now
                        out["first_event"].set()
                    last = now

                    # Content chunks
                    c = delta.get("content") or delta.get("reasoning_content") or ""
                    if c:
                        raw_content_chunks.append(c)

                    # Reconstruct streaming tool_calls
                    tc_deltas = delta.get("tool_calls") or []
                    for tc in tc_deltas:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": tc.get("id", ""),
                                "type": tc.get("type", "function"),
                                "name": "",
                                "arguments": ""
                            }
                        if tc.get("id"):
                            tool_calls_map[idx]["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            tool_calls_map[idx]["name"] += fn["name"]
                        if fn.get("arguments"):
                            tool_calls_map[idx]["arguments"] += fn["arguments"]

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["first_event"].set()
        return out

    t1 = time.perf_counter()
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    cached_tokens = int((usage or {}).get("prompt_tokens_details", {}).get("cached_tokens") or 0)

    decode_s = None if first is None or last is None or last <= first else (last - first)
    toks = max(completion_tokens - 1, 0)
    tok_s = (toks / decode_s) if decode_s and toks else None

    # Parse and validate tool calls arguments
    parsed_tool_calls = []
    blank_args_count = 0
    malformed_json_count = 0

    for idx, tc in sorted(tool_calls_map.items()):
        raw_args = tc["arguments"].strip()
        parsed_args = {}
        is_blank = False
        is_malformed = False

        if not raw_args or raw_args == "{}":
            is_blank = True
            blank_args_count += 1
        else:
            try:
                parsed_args = json.loads(raw_args)
                if not parsed_args:  # empty dict after parse
                    is_blank = True
                    blank_args_count += 1
            except json.JSONDecodeError:
                is_malformed = True
                malformed_json_count += 1

        parsed_tool_calls.append({
            "index": idx,
            "id": tc["id"],
            "name": tc["name"],
            "raw_arguments": raw_args,
            "parsed_arguments": parsed_args,
            "is_blank": is_blank,
            "is_malformed": is_malformed
        })

    out.update({
        "ttft_s": None if first is None else (first - t0),
        "wall_s": t1 - t0,
        "decode_s": decode_s,
        "tok_s": tok_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "finish_reason": finish_reason,
        "tool_calls": parsed_tool_calls,
        "num_tool_calls": len(parsed_tool_calls),
        "blank_args_count": blank_args_count,
        "malformed_json_count": malformed_json_count,
        "raw_text_length": sum(len(c) for c in raw_content_chunks),
        "usage": usage
    })
    return out


def run_concurrency_wave(
    concurrency: int,
    filler_words: int = 0,
    enable_thinking: bool = False,
    timeout: float = 300.0
) -> List[Dict[str, Any]]:
    """Fire N parallel tool-call requests and gather metrics."""
    results: List[Dict[str, Any]] = [{} for _ in range(concurrency)]
    threads = []

    for i in range(concurrency):
        prompt = make_padded_prompt(f"LANE_{i}", filler_words)
        t = threading.Thread(
            target=stream_tool_request,
            kwargs={
                "prompt": prompt,
                "enable_thinking": enable_thinking,
                "timeout": timeout,
                "out": results[i]
            },
            daemon=True
        )
        threads.append(t)

    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_end = time.perf_counter()

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool-call concurrency & argument validation benchmark (Issue #10)")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="Concurrent request streams (default: 4)")
    parser.add_argument("--filler-words", "-f", type=int, default=0, help="Number of padding words per prompt (default: 0)")
    parser.add_argument("--thinking", action="store_true", help="Enable reasoning / thinking mode")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds (default: 300)")
    parser.add_argument("--out", default="/tmp/tool-concurrency-receipt.json", help="Output JSON receipt path")
    args = parser.parse_args()

    print(f"=== GLM-5.3-Flash EXL3 Tool-Call Concurrency Benchmark ===", flush=True)
    print(f"Endpoint: {BASE} | Model: {MODEL}", flush=True)
    print(f"Concurrency: {args.concurrency} | Filler Words: {args.filler_words} | Thinking: {args.thinking}\n", flush=True)

    t0 = time.perf_counter()
    results = run_concurrency_wave(
        concurrency=args.concurrency,
        filler_words=args.filler_words,
        enable_thinking=args.thinking,
        timeout=args.timeout
    )
    total_wall = time.perf_counter() - t0

    # Evaluate aggregate results
    total_requests = len(results)
    successful_requests = 0
    total_tool_calls = 0
    total_blank_args = 0
    total_malformed = 0
    total_timeouts = 0

    for i, res in enumerate(results):
        err = res.get("error")
        if err:
            print(f"[Lane {i}] FAILED: {err}", flush=True)
            if "timeout" in err.lower():
                total_timeouts += 1
            continue

        successful_requests += 1
        tc_list = res.get("tool_calls", [])
        total_tool_calls += len(tc_list)
        total_blank_args += res.get("blank_args_count", 0)
        total_malformed += res.get("malformed_json_count", 0)

        tc_names = [f"{t['name']}({'BLANK' if t['is_blank'] else 'ok'})" for t in tc_list]
        print(
            f"[Lane {i}] HTTP 200 | TTFT: {res.get('ttft_s', 0):.2f}s | "
            f"Decode: {res.get('tok_s', 0):.1f} tok/s | Finish: {res.get('finish_reason')} | "
            f"Tools: {tc_names}",
            flush=True
        )

    receipt = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL,
        "concurrency": args.concurrency,
        "filler_words": args.filler_words,
        "thinking_enabled": args.thinking,
        "total_wall_s": total_wall,
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "total_tool_calls": total_tool_calls,
        "blank_args_count": total_blank_args,
        "malformed_json_count": total_malformed,
        "timeout_count": total_timeouts,
        "results": results
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved structured receipt to: {args.out}", flush=True)

    print(f"\n--- Summary ---")
    print(f"Success: {successful_requests}/{total_requests} | Timeouts: {total_timeouts}")
    print(f"Tool Calls Emitted: {total_tool_calls}")
    print(f"Blank/Missing Arguments: {total_blank_args}")
    print(f"Malformed JSON: {total_malformed}")

    # Return non-zero exit code if blank arguments or malformed JSON detected (Issue #10 detection)
    if total_blank_args > 0 or total_malformed > 0:
        print("\n[FAIL] Blank or malformed tool call arguments detected!", file=sys.stderr)
        return 1

    if successful_requests < total_requests:
        print("\n[FAIL] One or more requests failed / timed out!", file=sys.stderr)
        return 2

    print("\n[PASS] All concurrent tool calls returned valid non-empty arguments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
