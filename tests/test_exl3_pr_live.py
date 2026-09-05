"""CPU-only protocol/failure tests. No service, GPU, or network is required."""
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("pr_live", Path(__file__).with_name("bench_exl3_pr_live.py"))
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class Response(io.BytesIO):
    status = 200


def sse(tokens=8192, completion=8, content="OK", finish="stop"):
    events = [
        {"choices": [{"delta": {"content": content}, "finish_reason": finish}]},
        {"usage": {"prompt_tokens": tokens, "completion_tokens": completion}, "choices": []},
    ]
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "results.json"
        self.panel = bench.Panel("http://unused.invalid", self.out)
        # Any accidental I/O outside explicit mocks fails instead of using a network.
        no_network = patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden"))
        no_network.start()
        self.addCleanup(no_network.stop)

    def test_capture_preserves_utf8_across_read_boundaries(self):
        text = 'data: ' + json.dumps({"text": "résumé → 缓存"}, ensure_ascii=False) + '\n\n'
        rec = {"raw_sse": ""}
        with bench.Capture(Response(text.encode("utf-8")), rec) as capture:
            while capture.read(1):
                pass
        self.assertEqual(rec["raw_sse"], text)

    def saved(self):
        return json.loads(self.out.read_text())

    def prepare(self, rec):
        return bench.cold.chat_messages("seed"), 8192, 8192, {}, 1200

    def test_stream_failure_keeps_partial_events_and_restores_helper(self):
        class Broken(Response):
            def read(self, size=-1):
                if self.tell():
                    raise OSError("lost connection")
                return super().read(size)
        post = lambda *a, **k: Broken(sse())
        with patch.object(bench.cold, "http_post", post), patch.object(
                bench.cold, "metrics_snapshot", return_value={"local_cache_hit": 0}):
            with self.assertRaisesRegex(RuntimeError, "stream"):
                self.panel.sample("broken", 0, "cold", self.prepare)
            self.assertIs(bench.cold.http_post, post)
        sample = self.saved()["samples"][0]
        self.assertEqual(sample["status"], "failed")
        self.assertIn("lost connection", sample["error"])
        self.assertEqual(sample["prompt_tokens"], 8192)
        self.assertEqual(sample["gen"], "OK")
        self.assertEqual(len(sample["events"]), 2)
        self.assertIn("[DONE]", sample["raw_sse"])

    def test_malformed_json_is_saved_not_silently_skipped(self):
        raw = sse() + b'data: {"usage":\n'
        with patch.object(bench.cold, "http_post", return_value=Response(raw)), patch.object(
                bench.cold, "metrics_snapshot", return_value={"local_cache_hit": 0}):
            with self.assertRaises(RuntimeError):
                self.panel.sample("partial_json", 0, "cold", self.prepare)
        sample = self.saved()["samples"][0]
        self.assertEqual(sample["malformed_sse"], ['{"usage":'])
        self.assertEqual(sample["completion_tokens"], 8)
        self.assertEqual(sample["finish_reason"], "stop")
        self.assertIn('data: {"usage":', sample["raw_sse"])

    def test_metrics_failure_does_not_lose_completed_generation(self):
        with patch.object(bench.cold, "http_post", return_value=Response(sse())), patch.object(
                bench.cold, "metrics_snapshot", side_effect=[{"local_cache_hit": 0}, OSError("metrics down")]):
            with self.assertRaisesRegex(OSError, "metrics down"):
                self.panel.sample("metrics_failure", 0, "cold", self.prepare)
        sample = self.saved()["samples"][0]
        self.assertEqual(sample["status"], "failed")
        self.assertEqual(sample["prompt_tokens"], 8192)
        self.assertEqual(sample["gen"], "OK")
        self.assertEqual(len(sample["events"]), 2)

    def test_decode_exception_keeps_partial_json(self):
        class Broken(Response):
            def read(self, size=-1):
                if self.tell():
                    raise OSError("decode disconnected")
                return super().read(size)
        with patch.object(bench.decode, "_post", return_value=Broken(b'data: {"choices":\n')), patch.object(
                bench.cold, "metrics_snapshot", return_value={"local_cache_hit": 0}), patch.object(
                bench.decode, "spec_snapshot", return_value={}):
            with self.assertRaisesRegex(OSError, "decode disconnected"):
                self.panel.sample("decode_failure", 0, "decode", lambda r: (
                    bench.cold.chat_messages("count"), None, None, {"max_tokens": 400}, 900))
        sample = self.saved()["samples"][0]
        self.assertEqual(sample["http"], 200)
        self.assertEqual(sample["malformed_sse"], ['{"choices":'])
        self.assertEqual(sample["request"]["max_tokens"], 400)

    def test_invalid_rates_are_preserved_and_never_filtered(self):
        for value in (None, float("nan"), float("inf"), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bench.strict_median([1, value, 3])
        self.assertEqual(bench.strict_median([5, 1, 4, 2, 3]), 3)
        self.panel.doc["samples"] = [{"tok_s": float("nan")}, {"tok_s": None}]
        self.panel.save()
        self.assertEqual(self.saved()["samples"], [{"tok_s": "nan"}, {"tok_s": None}])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.panel.doc["samples"] = [{"case": "readme_prose", "kind": "decode", "measured": True,
                                          "status": "ok", "tok_s": 100}]
            self.panel.aggregate()

    def test_zero_draft_diagnostic_allowed_but_measured_sample_rejected(self):
        counters = {"vllm:spec_decode_num_drafts_total": 0,
                    "vllm:spec_decode_num_draft_tokens_total": 0,
                    "vllm:spec_decode_num_accepted_tokens_total": 0,
                    **{f"pos:{i}": 0 for i in range(7)}}
        for measured in (False, True):
            with self.subTest(measured=measured), patch.object(
                    bench.decode, "_post", return_value=Response(sse(tokens=30))), patch.object(
                    bench.cold, "metrics_snapshot", return_value={"local_cache_hit": 0}), patch.object(
                    bench.decode, "spec_snapshot", return_value=counters):
                prepare = lambda r: (bench.cold.chat_messages("probe"), None, None,
                                     {"max_tokens": 400 if measured else 8}, 900)
                if measured:
                    with self.assertRaisesRegex(RuntimeError, "invalid speculative acceptance"):
                        self.panel.sample("measured_decode", 0, "decode", prepare)
                else:
                    result = self.panel.sample("short_after", 0, "decode", prepare, measured=False)
                    self.assertEqual(result["status"], "ok")
                    self.assertGreater(result["ttft_s"], 0)
                    self.assertGreater(result["tok_s"], 0)
            sample = self.saved()["samples"][-1]
            self.assertEqual(sample["status"], "failed" if measured else "ok")
            self.assertEqual(sample["spec_before"], counters)
            self.assertEqual(sample["spec_after"], counters)
            self.assertEqual(sample["spec"]["drafts"], 0)
            self.assertIsNone(sample["spec"]["accept_ratio"])
            self.assertIsNone(sample["spec"]["accepted_per_step"])
            self.assertEqual(sample["finish_reason"], "stop")
            self.assertEqual(sample["completion_tokens"], 8)
            self.assertTrue(sample["raw_sse"])

    def test_zero_draft_diagnostic_still_checks_output_timing_and_counters(self):
        rec = {"http": 200, "prompt_tokens": 30, "completion_tokens": 8,
               "ttft_s": 0.1, "tok_s": 70, "cache_hits": 0, "kind": "decode",
               "measured": False, "finish_reason": "stop", "request": {"max_tokens": 8},
               "spec": {"drafts": 0, "draft_tokens": 0, "accepted": 0,
                        "accept_ratio": None, "accepted_per_step": None, "pos": [0] * 7}}
        bench.Panel.validate(rec)
        for invalid in ({"nan": True}, {"ttft_s": None}, {"tok_s": float("nan")},
                        {"finish_reason": None}, {"completion_tokens": 9},
                        {"spec": {**rec["spec"], "accepted": 1}}):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                bench.Panel.validate({**rec, **invalid})

    def test_nan_sample_is_checkpointed_before_validation_raises(self):
        result = {"http": 200, "prompt_tokens": 8192, "completion_tokens": 8,
                  "ttft_s": 1, "prefill_tok_s": float("nan"), "gen": "OK",
                  "gen_ok": True, "finish_reason": "stop"}
        with patch.object(bench.cold, "stream_chat", return_value=result), patch.object(
                bench.cold, "metrics_snapshot", return_value={"local_cache_hit": 0}):
            with self.assertRaisesRegex(RuntimeError, "invalid tok_s"):
                self.panel.sample("nan_rate", 0, "cold", self.prepare)
        sample = self.saved()["samples"][0]
        self.assertEqual(sample["status"], "failed")
        self.assertEqual(sample["tok_s"], "nan")
        self.assertEqual(sample["completion_tokens"], 8)

    def test_guard_threshold_and_invalid_cache_evidence(self):
        rec = {"http": 200, "prompt_tokens": 10000, "completion_tokens": 8, "ttft_s": 1,
               "tok_s": 10000, "cache_hits": 8000, "kind": "apc", "gen_ok": True, "finish_reason": "stop"}
        bench.Panel.validate(rec)
        self.assertEqual(rec["hit_ratio"], 0.8)
        for hits in (7999, float("nan"), -1):
            with self.subTest(hits=hits), self.assertRaises(RuntimeError):
                bench.Panel.validate({**rec, "cache_hits": hits})
        with self.assertRaisesRegex(RuntimeError, "reused"):
            bench.Panel.validate({**rec, "kind": "cold", "target_tokens": 10000, "cache_hits": 1})

    def test_full_mocked_run_request_counts_salts_provenance_and_aggregation(self):
        calls = []
        counters = {"local_cache_hit": 0.0}
        spec = {"vllm:spec_decode_num_drafts_total": 0,
                "vllm:spec_decode_num_draft_tokens_total": 0,
                "vllm:spec_decode_num_accepted_tokens_total": 0,
                **{f"pos:{i}": 0 for i in range(7)}}
        clock = iter(i / 100 for i in range(10000))

        def post(path, body, **kwargs):
            self.assertEqual(path, "/v1/chat/completions")
            calls.append(body)
            match = re.search(r"TARGET (\d+)", body["messages"][0]["content"])
            tokens = int(match[1]) if match else 30
            if len(body["messages"]) == 3:
                tokens += 10
                counters["local_cache_hit"] += tokens * 0.85
            spec["vllm:spec_decode_num_drafts_total"] += 1
            spec["vllm:spec_decode_num_draft_tokens_total"] += 7
            spec["vllm:spec_decode_num_accepted_tokens_total"] += 4
            for i in range(4):
                spec[f"pos:{i}"] += 1
            return Response(sse(tokens, body["max_tokens"], finish="stop" if match else "length"))

        def fit(seed, target):
            return seed + f"TARGET {target}", target

        with patch.object(bench, "fit_prompt", side_effect=fit), patch.object(
                bench.cold, "calibrate", side_effect=lambda target, salt, **k: (target, target, salt + f" TARGET {target}")), patch.object(
                bench.cold, "http_get", side_effect=lambda path: (200, '{"data":[{"id":"mock-model","max_model_len":1000000}]}' if path == "/v1/models" else "ok")), patch.object(
                bench.cold, "http_post", side_effect=post), patch.object(bench.decode, "_post", side_effect=post), patch.object(
                bench.cold, "metrics_snapshot", side_effect=lambda: counters.copy()), patch.object(
                bench.decode, "spec_snapshot", side_effect=lambda: spec.copy()), patch.object(
                bench.time, "perf_counter", side_effect=lambda: next(clock)):
            result = self.panel.run()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(calls), 44)  # 1 warmup + 3*(4 panel+2 holdout+2 APC) + 19 decode/probes
        self.assertEqual(result["protocol"]["expected_generation_requests"], 44)
        self.assertEqual(result["served_model"], "mock-model")
        self.assertTrue(result["provenance"]["local_source_sha256"])
        self.assertFalse(result["samples"][0]["measured"])
        self.assertTrue(calls[0]["messages"][0]["content"].startswith("warmup="))
        self.assertIn(bench.PROSE, calls[0]["messages"][0]["content"])
        self.assertIn(bench.CODE, calls[2]["messages"][0]["content"])
        salts = [c["cache_salt"] for c in calls if "cache_salt" in c]
        self.assertEqual(len(salts), 13)
        self.assertEqual(len(set(salts)), 13)
        for i in (8, 16, 24):
            self.assertEqual(calls[i]["messages"][0], calls[i - 1]["messages"][0])
            self.assertEqual(calls[i]["messages"][-1]["content"], "Confirm with OK.")
            self.assertNotIn("cache_salt", calls[i])
        for name, prompt in (("readme_structured", bench.decode.STRUCTURED_PROMPT),
                             ("readme_prose", bench.decode.BENCH_PROMPT), ("lru_code_guard", bench.LRU)):
            samples = [s for s in result["samples"] if s["case"] == name]
            self.assertEqual(len(samples), 5)
            self.assertTrue(all(s["request"]["max_tokens"] == 400 for s in samples))
            self.assertTrue(all(s["request"]["messages"][0]["content"] == prompt for s in samples))
            self.assertTrue(all("cache_salt" not in s["request"] for s in samples))
            summary = result["summary"][name]
            self.assertEqual(summary["tok_s_median"], bench.strict_median([s["tok_s"] for s in samples]))
            self.assertEqual(summary["accept_per_position_median"], [1, 1, 1, 1, 0, 0, 0])
            self.assertEqual(summary["completion_tokens_median"], 400)
        self.assertEqual(result["summary"]["prose_8k"]["sample_count"], 3)
        self.assertEqual(self.saved()["status"], "ok")

    def test_fitting_uses_historical_seed_and_calibration(self):
        seen = []
        def tokenize(messages, **kwargs):
            text = messages[0]["content"]
            seen.append(text)
            return len(text)
        seed = "session=abc\nRead the material, then reply exactly OK.\n" + bench.CODE
        with patch.object(bench.cold, "tokenize_messages", side_effect=tokenize):
            text, count = bench.fit_prompt(seed, 8192)
        self.assertEqual(seen[0], seed)
        self.assertTrue(text.startswith(seed))
        self.assertLessEqual(abs(count - 8192), 16)
        self.assertEqual(bench.PROSE, "The maintenance team measured inference latency under changing request loads. Each observation records the node, sequence length, checksum, and result. Operators compare the record with the previous run before changing the service.\n")

    def test_cli_defaults_and_health_failure_receipt(self):
        with patch.object(bench.cold, "http_get", return_value=(503, "not ready")):
            with self.assertRaisesRegex(RuntimeError, "health returned 503"):
                bench.main(["--base-url", "http://unused.invalid", "--out", str(self.out)])
        result = self.saved()
        self.assertEqual(result["protocol"]["prefill_runs"], 3)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["health"]["http"], 503)
        self.assertEqual(result["samples"], [])


if __name__ == "__main__":
    unittest.main()
