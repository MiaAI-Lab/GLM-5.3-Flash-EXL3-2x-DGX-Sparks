#!/usr/bin/env python3
"""Opt-in live probe: repeat image history across tool turns and measure RPC bytes.

Run on the API head with Pillow installed. Requires --base-url and --peer for
each remote worker's advertised address. VLLM_API_KEY supplies authentication.
Emits JSONL; never prints the key or image payloads. Sends real inference work.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import threading
import time
import urllib.request


OUTPUT_LOCK = threading.Lock()


def socket_counters(peers):
    lines = subprocess.check_output(["ss", "-tinH"], text=True).splitlines()
    rows = []
    for index, line in enumerate(lines):
        match = re.search(r"bytes_acked:(\d+)", line)
        if match and index:
            fields = lines[index - 1].split()
            if len(fields) >= 5 and fields[4].rsplit(":", 1)[0] in peers:
                rows.append({"local": fields[3], "peer": fields[4],
                             "send_q": int(fields[2]), "acked": int(match[1])})
    return rows


def emit(event, **kwargs):
    with OUTPUT_LOCK:
        print(json.dumps({"event": event, "at": time.time(), **kwargs}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--peer", action="append", required=True)
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument("--images", type=int, default=12)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    if not 1 <= args.images <= 48 or not 2 <= args.turns <= 10:
        parser.error("use 1..48 images and 2..10 turns")
    from PIL import Image, ImageDraw

    content = []
    for number in range(args.images):
        # Distinct images avoid accidentally benchmarking one memoized tensor.
        image = Image.new("RGB", (1536, 1024), (32 + number * 3, 150, 60))
        draw = ImageDraw.Draw(image)
        for x in range(0, 1536, 96):
            draw.rectangle((x, 0, x + 32, 1023), fill=(20, 40 + number * 4, 200))
        draw.text((50, 50), f"Network validation image {number}", fill="white")
        output = io.BytesIO()
        image.save(output, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": "Call record_check with value 42."})
    messages = [{"role": "user", "content": content}]
    tool = {"type": "function", "function": {"name": "record_check",
            "description": "Record the requested integer.", "parameters": {
                "type": "object", "properties": {"value": {"type": "integer"}},
                "required": ["value"]}}}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
    emit("fixture", phase=args.phase, images=args.images,
         image_history_sha256=hashlib.sha256(json.dumps(content).encode()).hexdigest())
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                emit("sockets", phase=args.phase, sockets=socket_counters(args.peer))
            except Exception as error:
                emit("monitor_error", error=str(error))
            stop.wait(1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for turn in range(args.turns):
            body = {"model": args.model, "messages": messages, "tools": [tool],
                    "tool_choice": "auto", "temperature": 0, "max_tokens": 128,
                    "stream": True, "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False}}
            encoded = json.dumps(body).encode()
            emit("request", phase=args.phase, turn=turn,
                 request_bytes=len(encoded), sockets=socket_counters(args.peer))
            started = time.monotonic()
            first = None
            text = ""
            calls = {}
            usage = None
            request = urllib.request.Request(args.base_url.rstrip("/") +
                      "/v1/chat/completions", encoded, headers)
            with urllib.request.urlopen(request, timeout=600) as response:
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise RuntimeError(event["error"])
                    usage = event.get("usage") or usage
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        if first is None and (delta.get("content") or
                                              delta.get("tool_calls")):
                            first = time.monotonic() - started
                        text += delta.get("content") or ""
                        for part in delta.get("tool_calls", []):
                            call = calls.setdefault(part["index"], {"id": "",
                                   "type": "function", "function": {
                                       "name": "", "arguments": ""}})
                            call["id"] += part.get("id") or ""
                            for key in ("name", "arguments"):
                                call["function"][key] += part.get("function", {}).get(key) or ""
            calls = [calls[index] for index in sorted(calls)]
            assert len(calls) == 1, calls
            assert calls[0]["function"]["name"] == "record_check", calls
            assert json.loads(calls[0]["function"]["arguments"]) == {"value": 42}, calls
            emit("result", phase=args.phase, turn=turn, ttft_s=first,
                 total_s=time.monotonic() - started, usage=usage,
                 tool_correct=True, sockets=socket_counters(args.peer))
            messages.extend([
                {"role": "assistant", "content": text or None, "tool_calls": calls},
                {"role": "tool", "tool_call_id": calls[0]["id"], "content": "Recorded 42."},
                {"role": "user", "content": "Call record_check with value 42 again."},
            ])
    finally:
        stop.set()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
