#!/usr/bin/env python3
"""CPU-only bearer-auth regression coverage using an ephemeral localhost API."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import tempfile
import threading
import traceback
import unittest
import urllib.error
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("bench_decode.py")
SPEC = importlib.util.spec_from_file_location("bench_decode", MODULE_PATH)
assert SPEC and SPEC.loader
bench_decode = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench_decode)

# Synthetic credentials only; the fixture never reads the caller's credentials.
API_KEY = "fixture-api-key"
VLLM_KEY = "fixture-vllm-key"


@contextmanager
def local_api(env: dict[str, str], key: str | None, status: int = 200):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Never log request headers or credentials.

        def reply(self, code, body, content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # These endpoints must stay keyless, even for an authenticated run.
            if self.headers.get("Authorization") is not None:
                self.reply(400, b'{"error":"unexpected authorization"}')
            elif self.path == "/health":
                self.reply(200, b"ok", "text/plain")
            elif self.path == "/metrics":
                self.reply(200, b'vllm:spec_decode_num_drafts_total{model="fixture"} 4\n', "text/plain")
            else:
                self.reply(404, b'{"error":"not found"}')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != "/v1/chat/completions":
                self.reply(404, b'{"error":"not found"}')
                return
            expected = f"Bearer {key}" if key else None
            if self.headers.get_all("Authorization", []) != ([] if expected is None else [expected]):
                self.reply(401, b'{"error":"unauthorized"}')
                return
            if status != 200:
                self.reply(status, b'{"error":"request rejected"}')
                return
            usage = {"prompt_tokens": 3, "completion_tokens": 2}
            if body.get("stream"):
                chunks = [
                    {"choices": [{"delta": {"content": "fixture completion"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": usage},
                ]
                payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                self.reply(200, (payload + "data: [DONE]\n\n").encode(), "text/event-stream")
            else:
                payload = {"choices": [{"message": {"content": "fixture completion"}}], "usage": usage}
                self.reply(200, json.dumps(payload).encode())

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # Clear inherited credentials and proxies; all traffic stays on localhost.
            with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1", **env}, clear=True), patch.object(
                bench_decode, "BASE", f"http://127.0.0.1:{server.server_port}"
            ):
                yield
        finally:
            server.shutdown()
            thread.join()


class DecodeAuthTests(unittest.TestCase):
    def test_key_precedence_and_empty_values_over_http(self):
        cases = [
            ("unset", {}, None),
            ("api-empty", {"API_KEY": ""}, None),
            ("vllm-empty", {"VLLM_API_KEY": ""}, None),
            ("both-empty", {"API_KEY": "", "VLLM_API_KEY": ""}, None),
            ("api-only", {"API_KEY": API_KEY}, API_KEY),
            ("vllm-only", {"VLLM_API_KEY": VLLM_KEY}, VLLM_KEY),
            ("api-wins", {"API_KEY": API_KEY, "VLLM_API_KEY": VLLM_KEY}, API_KEY),
            ("empty-api-falls-back", {"API_KEY": "", "VLLM_API_KEY": VLLM_KEY}, VLLM_KEY),
            ("empty-vllm-keeps-api", {"API_KEY": API_KEY, "VLLM_API_KEY": ""}, API_KEY),
        ]
        for name, env, key in cases:
            with self.subTest(case=name), local_api(env, key):
                response = bench_decode.chat_nonstream("fixture prompt")
                self.assertEqual(response["choices"][0]["message"]["content"], "fixture completion")
                stream = bench_decode.stream_bench(2)
                self.assertEqual(stream["text"], "fixture completion")
                self.assertEqual(stream["completion_tokens"], 2)
                self.assertEqual(stream["finish_reason"], "stop")

    def test_health_and_metrics_remain_keyless(self):
        with local_api({"API_KEY": API_KEY, "VLLM_API_KEY": VLLM_KEY}, API_KEY):
            self.assertEqual(bench_decode.health(), (200, "ok"))
            self.assertEqual(bench_decode.spec_snapshot(), {"vllm:spec_decode_num_drafts_total": 4.0})

    def test_http_errors_propagate_without_credentials(self):
        cases = [
            ("missing", {}, 200, 401),
            ("wrong-key", {"API_KEY": API_KEY}, 200, 401),
            ("forbidden", {"VLLM_API_KEY": VLLM_KEY}, 403, 403),
            ("unavailable", {"VLLM_API_KEY": VLLM_KEY}, 503, 503),
        ]
        for name, env, status, expected in cases:
            with self.subTest(case=name), local_api(env, VLLM_KEY, status):
                for call in (lambda: bench_decode.chat_nonstream("fixture prompt"), bench_decode.stream_bench):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        call()
                    with caught.exception as error:
                        self.assertEqual(error.code, expected)
                        self.assertNotIn(API_KEY, str(error))
                        self.assertNotIn(VLLM_KEY, str(error))

    def test_cli_output_and_report_do_not_disclose_credentials(self):
        with tempfile.TemporaryDirectory() as tmp, local_api(
            {"API_KEY": API_KEY, "VLLM_API_KEY": VLLM_KEY}, API_KEY
        ):
            report = Path(tmp) / "result.json"
            output = io.StringIO()
            argv = ["bench_decode.py", "--phase", "fixture", "--out", str(report),
                    "--runs", "1", "--max-tokens", "2", "--skip-coherence"]
            with patch.object(bench_decode.sys, "argv", argv), redirect_stdout(output), redirect_stderr(output):
                self.assertEqual(bench_decode.main(), 0)
            recorded = report.read_text()
            self.assertEqual(json.loads(recorded)["runs"][0]["text"], "fixture completion")
            for key in (API_KEY, VLLM_KEY):
                self.assertNotIn(key, output.getvalue() + recorded)

    def test_cli_auth_failure_does_not_disclose_credentials(self):
        with tempfile.TemporaryDirectory() as tmp, local_api(
            {"API_KEY": API_KEY, "VLLM_API_KEY": VLLM_KEY}, VLLM_KEY
        ):
            report = Path(tmp) / "result.json"
            output = io.StringIO()
            argv = ["bench_decode.py", "--phase", "fixture", "--out", str(report),
                    "--runs", "1", "--skip-coherence"]
            with patch.object(bench_decode.sys, "argv", argv), redirect_stdout(output), redirect_stderr(output):
                try:
                    bench_decode.main()
                except urllib.error.HTTPError as error:
                    with error:
                        self.assertEqual(error.code, 401)
                        traceback.print_exc()
                else:
                    self.fail("unauthorized benchmark succeeded")
            self.assertFalse(report.exists())
            for key in (API_KEY, VLLM_KEY):
                self.assertNotIn(key, output.getvalue())


if __name__ == "__main__":
    unittest.main()
