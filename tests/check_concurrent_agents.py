#!/usr/bin/env python3
"""One live concurrency canary: B must answer while A is still generating.

No sweep or automatic retry. Run after approving the pinned deployment recipe.
Uses only stdlib; OPENAI_API_KEY is optional and is never included in receipts.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit
import uuid


class Request(threading.Thread):
    def __init__(self, base_url, body, timeout):
        super().__init__(daemon=True)
        self.url, self.body, self.timeout = urlsplit(base_url), body, timeout
        self.first, self.done, self.cancelled = threading.Event(), threading.Event(), threading.Event()
        self.sock = None
        self.started_at = self.first_at = self.last_at = self.finished_at = None
        self.status = self.error = self.finish_reason = self.usage = None
        self.text = ""

    def cancel(self):
        self.cancelled.set()
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def run(self):
        self.started_at = time.monotonic()
        cls = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
        conn = cls(self.url.hostname, self.url.port, timeout=min(self.timeout, 30))
        try:
            conn.connect()
            self.sock = conn.sock
            self.sock.settimeout(self.timeout)
            if self.cancelled.is_set():
                return
            headers = {"Content-Type": "application/json"}
            key = os.environ.get("OPENAI_API_KEY")
            if key:
                headers["Authorization"] = "Bearer " + key
            conn.request("POST", self.url.path.rstrip("/") + "/chat/completions",
                         json.dumps(self.body), headers)
            response = conn.getresponse()
            self.status = response.status
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            while not self.cancelled.is_set():
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("error"):
                    raise RuntimeError("provider returned an SSE error")
                if chunk.get("usage"):
                    self.usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        now = time.monotonic()
                        if self.first_at is None:
                            self.first_at = now
                            self.first.set()
                        self.last_at = now
                        self.text += content
                    if choice.get("finish_reason"):
                        self.finish_reason = choice["finish_reason"]
        except Exception as exc:
            if not self.cancelled.is_set():
                # Exception type suffices; don't echo a provider body or secrets.
                self.error = type(exc).__name__
        finally:
            conn.close()
            self.finished_at = time.monotonic()
            self.done.set()

    def receipt(self):
        def elapsed(t):
            return None if t is None or self.started_at is None else round(t - self.started_at, 3)
        return {"http": self.status, "ttft_s": elapsed(self.first_at),
                "wall_s": elapsed(self.finished_at), "finish_reason": self.finish_reason,
                "usage": self.usage, "error": self.error, "cancelled": self.cancelled.is_set()}


def wait_for_first(req, deadline):
    while time.monotonic() < deadline:
        if req.first.is_set():
            return True
        if req.done.is_set():
            return False
        req.first.wait(min(0.1, max(0, deadline - time.monotonic())))
    return False


def assess(a, b):
    if a.error or b.error or a.cancelled.is_set() or b.cancelled.is_set():
        return False, "a request failed, timed out, or was cancelled"
    if a.status != 200 or b.status != 200 or not a.finish_reason or not b.finish_reason:
        return False, "both requests must complete valid streams"
    if b.text.strip() != "OK":
        return False, "short request did not return the expected OK"
    if not a.text.lstrip().startswith("1"):
        return False, "long request did not begin the requested count"
    if a.first_at is None or a.last_at is None or b.started_at is None or b.first_at is None:
        return False, "missing content timing"
    if not a.first_at <= b.started_at <= b.first_at < a.last_at:
        return False, "B must start after A's first token and answer before A's last token"
    return True, "B answered and A continued generating afterward"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument("--filler-words", type=int, default=8000)
    parser.add_argument("--long-max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.query or url.fragment:
        parser.error("base URL must be HTTP(S) without credentials, query, or fragment")
    if args.filler_words < 1 or args.long_max_tokens < 128 or args.timeout <= 0:
        parser.error("positive filler/timeout and at least 128 long-request tokens required")
    common = {"model": args.model, "stream": True, "stream_options": {"include_usage": True},
              "temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": False}}
    unique = uuid.uuid4().hex
    a_body = {**common, "max_tokens": args.long_max_tokens, "messages": [{"role": "user", "content":
        f"Run {unique}-A. Count from 1 to 100000. Output only the numbers separated by spaces. Do not stop early."}]}
    b_body = {**common, "max_tokens": 16, "messages": [{"role": "user", "content":
        f"Run {unique}-B. Context: " + "the " * args.filler_words + "\nReply with exactly OK."}]}
    a, b = Request(args.base_url, a_body, args.timeout), Request(args.base_url, b_body, args.timeout)
    deadline = time.monotonic() + args.timeout
    reason = None
    try:
        a.start()
        if not wait_for_first(a, deadline):
            reason = "A failed to produce content; B was not submitted"
        else:
            print("A is generating; starting B with a distinct prompt.", flush=True)
            b.start()
            while not (a.done.is_set() and b.done.is_set()) and time.monotonic() < deadline:
                if a.error or b.error or (b.first_at is None and a.done.is_set()):
                    reason = "request failure or A finished before B produced content"
                    break
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            if not (a.done.is_set() and b.done.is_set()) and reason is None:
                reason = "canary deadline exceeded"
    except KeyboardInterrupt:
        reason = "canary interrupted"
    finally:
        for req in (a, b):
            if req.ident is not None and not req.done.is_set():
                req.cancel()
        for req in (a, b):
            if req.ident is not None:
                req.join(timeout=2)
    passed, assessed_reason = assess(a, b)
    receipt = {"passed": passed and reason is None, "reason": reason or assessed_reason,
               "model": args.model, "filler_words": args.filler_words,
               "long_max_tokens": args.long_max_tokens, "a": a.receipt(), "b": b.receipt(),
               "b_first_before_a_last_s": None if b.first_at is None or a.last_at is None
               else round(a.last_at - b.first_at, 3)}
    print(json.dumps(receipt, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
