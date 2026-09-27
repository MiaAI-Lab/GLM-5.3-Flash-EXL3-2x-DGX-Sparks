#!/usr/bin/env python3
"""CPU-only regression coverage for #270: the NaN check must not fire on ordinary words."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("bench_decode.py")
SPEC = importlib.util.spec_from_file_location("bench_decode", MODULE_PATH)
assert SPEC and SPEC.loader
bench_decode = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench_decode)

WORDS = ["banana", "nano", "finance", "Nancy", "resonance", "The keys are apple, banana and grape."]
NANS = ["nan", "NaN", "NaN, NaN", "value: nan.", "x=nan)", "nannannan", "NaNNaNNaN", "locklocklock"]


@contextmanager
def local_api(content: str):
    """Serve `content` for every chat request, streamed or not."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(b"ok", "text/plain")
            elif self.path == "/metrics":
                self.reply(b'vllm:spec_decode_num_drafts_total{model="fixture"} 4\n', "text/plain")
            else:
                self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            usage = {"prompt_tokens": 3, "completion_tokens": 2}
            if body.get("stream"):
                chunks = [
                    {"choices": [{"delta": {"content": content}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": usage},
                ]
                payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                self.reply(payload.encode(), "text/event-stream")
            else:
                payload = {"choices": [{"message": {"content": content}}], "usage": usage}
                self.reply(json.dumps(payload).encode(), "application/json")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1"}, clear=True), patch.object(
            bench_decode, "BASE", f"http://127.0.0.1:{server.server_port}"
        ):
            yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class NanDetection(unittest.TestCase):
    def test_regex(self):
        for text in WORDS:
            with self.subTest(text=text):
                self.assertIsNone(bench_decode.NAN_RE.search(text))
        for text in NANS:
            with self.subTest(text=text):
                self.assertIsNotNone(bench_decode.NAN_RE.search(text))

    def test_stream_bench_flag(self):
        for text, expected in [(t, False) for t in WORDS] + [(t, True) for t in NANS]:
            with self.subTest(text=text), local_api(text):
                self.assertIs(bench_decode.stream_bench(2)["nan"], expected)

    def test_coherence_sky(self):
        for text, expected in [(t, True) for t in WORDS] + [(t, False) for t in NANS]:
            with self.subTest(text=text), local_api(text):
                self.assertIs(bench_decode.coherence()["sky"]["ok"], expected)

    def test_concurrent_cli_report(self):
        cases = [("The keys are apple, banana and grape.", False), ("NaNNaNNaN", True)]
        for text, expected in cases:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp, local_api(text):
                report = Path(tmp) / "decode.json"
                argv = [
                    "bench_decode.py", "--phase", "dec-x8", "--out", str(report),
                    "--runs", "1", "--max-tokens", "2", "--warmup-tokens", "2",
                    "--concurrency", "8", "--skip-coherence",
                ]
                with patch.object(bench_decode.sys, "argv", argv), redirect_stdout(io.StringIO()):
                    self.assertEqual(bench_decode.main(), 0)
                recorded = json.loads(report.read_text())
                self.assertEqual(len(recorded["runs"][0]["parts"]), 8)
                self.assertIs(recorded["any_nan"], expected)
                self.assertIs(recorded["runs"][0]["nan"], expected)
                for part in recorded["runs"][0]["parts"]:
                    self.assertEqual(part["text"], text)
                    self.assertIs(part["nan"], expected)


if __name__ == "__main__":
    unittest.main()
