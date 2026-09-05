#!/usr/bin/env python3
"""Sequential PR prefill/holdout/APC panel and README decode protocol.

Example: python3 tests/bench_exl3_pr_live.py --base-url http://localhost:8888 \
    --out /tmp/pr-live.json --runs 3

--runs repeats the four cold prose/code cases, two long holdouts and APC pair.
Decode always uses five 400-token requests per case, independent of --runs.
Structured has the bench_decode 32-token warmup; prose does not. Both retain
its final unmeasured 8-token probe. LRU is a distinct guard (32 + 5x400), NOT
README/sparkDash Code. One mandatory unmeasured cold 8k warmup precedes all.
No profiling, parallel clients, cache flushes, or service changes. An otherwise
idle endpoint is required for counter deltas; this is not a concurrency test.
Every attempted sample and raw stream is checkpointed before validation raises.
Local source hashes are NOT proof of the running server's source/configuration;
use --provenance JSON to attach independently collected deployment evidence.
"""
from __future__ import annotations

import argparse
import codecs
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time
import urllib.error
import uuid

ROOT = Path(__file__).resolve().parents[1]


def load_helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cold = load_helper("pr_cold", "_run_cold_prefill.py")
decode = load_helper("pr_decode", "bench_decode.py")
PROSE = (
    "The maintenance team measured inference latency under changing request loads. "
    "Each observation records the node, sequence length, checksum, and result. "
    "Operators compare the record with the previous run before changing the service.\n"
)
CODE = (
    "def update_cache(items, limit):\n"
    "    result = {key: value for key, value in items if value is not None}\n"
    "    return sorted(result.items(), key=lambda pair: pair[0])[:limit]\n\n"
)
LRU = "Write a Python LRU cache class using a dictionary and a doubly linked list. Output code only."
CASES = [("prose_8k", PROSE, 8192), ("code_8k", CODE, 8192),
         ("prose_64k", PROSE, 65536), ("code_64k", CODE, 65536)]


def fit_prompt(seed, target):
    """Historical bench_live fit algorithm, including the exact repeated seed."""
    def count(text):
        value = cold.tokenize_messages(cold.chat_messages(text), timeout=180)
        if value <= 0:
            raise RuntimeError("tokenizer returned no tokens")
        return value
    n = count(seed)
    repeats = max(1, math.ceil(target / n))
    text = (seed * repeats)[:max(len(seed), int(len(seed) * repeats * target / n))]
    for _ in range(4):
        n = count(text)
        if abs(n - target) <= max(8, target // 500):
            return text, n
        length = max(1, round(len(text) * target / n))
        text = (text + seed * (math.ceil(length / len(seed)) + 1))[:length]
    return text, count(text)


def strict_median(values):
    """Do not silently improve a median by removing failed/nonfinite samples."""
    if not values or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                         or v < 0 for v in values):
        raise ValueError(f"invalid median inputs: {values!r}")
    return statistics.median(values)


def json_safe(value):
    # Strict JSON, while preserving the distinction between NaN/Inf and missing.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class Capture:
    """Tap helper I/O without changing its timing/parser or losing partial SSE."""
    def __init__(self, response, sample):
        self.response, self.sample = response, sample
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.flushed = False
        sample["http"] = response.status

    def __enter__(self):
        self.response.__enter__()
        return self

    def __exit__(self, *args):
        if not self.flushed:
            self.sample["raw_sse"] += self.decoder.decode(b"", final=True)
            self.flushed = True
        return self.response.__exit__(*args)

    @property
    def status(self):
        return self.response.status

    def read(self, size=-1):
        try:
            piece = self.response.read(size)
        except Exception as exc:
            partial = getattr(exc, "partial", b"")
            if isinstance(partial, bytes):
                self.sample["raw_sse"] += self.decoder.decode(partial)
            raise
        self.sample["raw_sse"] += self.decoder.decode(piece, final=not piece)
        self.flushed = not piece
        return piece


def inspect_stream(sample):
    """Preserve parsed partial events and malformed payloads the helpers skip."""
    sample["events"], sample["malformed_sse"] = [], []
    for line in sample.get("raw_sse", "").splitlines():
        if not line.strip().startswith("data:"):
            continue
        payload = line.strip()[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
            sample["events"].append(event)
        except json.JSONDecodeError:
            sample["malformed_sse"].append(payload)
    # A helper may throw before returning its accumulated result. Recover fields
    # that actually arrived, without inventing timing for a partial stream.
    text = []
    for event in sample["events"]:
        if not isinstance(event, dict):
            continue
        usage = event.get("usage")
        if isinstance(usage, dict):
            sample.setdefault("usage", usage)
            for key in ("prompt_tokens", "completion_tokens"):
                if sample.get(key) is None:
                    sample[key] = usage.get(key)
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
            content = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
            if isinstance(content, str):
                text.append(content)
            if sample.get("finish_reason") is None and choice.get("finish_reason"):
                sample["finish_reason"] = choice["finish_reason"]
    sample["captured_text"] = "".join(text)


class Panel:
    def __init__(self, base_url, out, runs=3, provenance=None):
        cold.BASE = decode.BASE = base_url.rstrip("/")
        self.out = Path(out)
        files = [Path(__file__), ROOT / "tests/_run_cold_prefill.py",
                 ROOT / "tests/bench_decode.py", ROOT / "README.md"]
        self.doc = {
            "status": "running", "base_url": cold.BASE, "started": time.time(),
            "protocol": {"sequential": True, "concurrency_benchmark": False,
                         "prefill_runs": runs, "decode_runs": 5, "decode_max_tokens": 400,
                         "prefill_max_tokens": 8, "temperature": 0, "thinking": False,
                         "apc_min_hit_ratio": 0.8, "profiling": False,
                         "expected_generation_requests": 1 + 8 * runs + 19,
                         "decode_timing": "bench_decode: (completion_tokens-1)/(EOF-first_token)",
                         "prefill_timing": "_run_cold_prefill: full prompt_tokens/first content TTFT",
                         "cache_policy": "live: unique session text + cache_salt; holdout: unique text salt; APC: same history; decode: unsalted README protocol"},
            "provenance": {"local_source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                "historical_source_sha256": {
                    ".auto/bench_live.py": "10a8ab4f1af4f0ecd7836e4e6dd25cf8301e41782912f68a8f7e244523cd2cae",
                    ".auto/bench_holdout.py": "439288a6cd496ed3d6c3a85219c542566329fc01196b1378a23616bb8e58729b"},
                "deployment": provenance,
                "deployment_verified_by_harness": False},
            "samples": [], "summary": {},
        }
        self.save()

    def save(self):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.out.with_name(self.out.name + ".tmp")
        temporary.write_text(json.dumps(json_safe(self.doc), indent=2, allow_nan=False) + "\n")
        os.replace(temporary, self.out)

    def sample(self, name, repeat, kind, prepare, measured=True):
        rec = {"case": name, "repeat": repeat, "kind": kind, "measured": measured,
               "status": "started", "raw_sse": "", "prompt_tokens": None,
               "completion_tokens": None, "ttft_s": None, "tok_s": None,
               "finish_reason": None, "cache_hits": None}
        self.doc["samples"].append(rec)
        self.save()
        module, attr = (decode, "_post") if kind == "decode" else (cold, "http_post")
        original = getattr(module, attr)
        try:
            messages, estimated, target, extras, timeout = prepare(rec)
            rec.update(estimated_tokens=estimated, target_tokens=target)
            rec["metrics_before"] = cold.metrics_snapshot()
            if kind == "decode":
                rec["spec_before"] = decode.spec_snapshot()

            def post(path, body, timeout, **kwargs):
                body = {**body, **extras}
                rec["request"] = body
                # No disk checkpoint inside the helper's timed request interval.
                # The sample's finally block preserves this even on failure.
                try:
                    return Capture(original(path, body, timeout=timeout, **kwargs), rec)
                except urllib.error.HTTPError as exc:
                    rec["http"] = exc.code
                    rec["http_error_body"] = exc.read().decode("utf-8", "replace")
                    raise

            setattr(module, attr, post)
            if kind == "decode":
                rec.update(decode.stream_bench(extras["max_tokens"], prompt=messages[0]["content"]))
            else:
                rec.update(cold.stream_chat(messages, timeout=timeout))
                rec["tok_s"] = rec.get("prefill_tok_s")
            # Save generation even if the following metrics request fails.
            self.save()
            rec["metrics_after"] = cold.metrics_snapshot()
            rec["metrics_delta"] = cold.delta_metrics(rec["metrics_before"], rec["metrics_after"])
            if any("local_cache_hit" not in rec[k] for k in ("metrics_before", "metrics_after")):
                raise RuntimeError("local_cache_hit counter missing; cannot verify cache guard")
            rec["cache_hits"] = rec["metrics_delta"]["local_cache_hit"]
            if kind == "decode":
                rec["spec_after"] = decode.spec_snapshot()
                required = {"vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
                            "vllm:spec_decode_num_accepted_tokens_total", *(f"pos:{i}" for i in range(7))}
                if any(not required.issubset(rec[k]) for k in ("spec_before", "spec_after")):
                    raise RuntimeError("missing speculative decode counters/per-position evidence")
                rec["spec"] = decode.spec_delta(rec["spec_before"], rec["spec_after"])
            inspect_stream(rec)
            self.validate(rec)
            rec["status"] = "ok"
            return rec
        except Exception as exc:
            rec["status"] = "failed"
            rec["failure"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            setattr(module, attr, original)
            inspect_stream(rec)
            self.save()

    @staticmethod
    def validate(rec):
        if rec.get("malformed_sse") or rec.get("error") or rec.get("http") != 200:
            raise RuntimeError("HTTP, stream or JSON failure")
        for key in ("prompt_tokens", "completion_tokens", "ttft_s", "tok_s"):
            value = rec.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise RuntimeError(f"invalid {key}: {value!r}")
        hits = rec["cache_hits"]
        if not math.isfinite(hits) or hits < 0:
            raise RuntimeError("invalid/reset cache counter")
        if rec["kind"] == "decode":
            if rec.get("nan") or rec["finish_reason"] not in ("stop", "length"):
                raise RuntimeError("invalid decode output/finish")
            if rec["completion_tokens"] > rec["request"]["max_tokens"]:
                raise RuntimeError("decode completion exceeds request budget")
            spec = rec["spec"]
            if rec.get("measured") is False and spec["drafts"] == 0:
                # A short/EOS diagnostic can finish without a speculative step.
                # Keep the helper's zero counters and undefined ratios as evidence;
                # its per-position zeros also have no denominator in this case.
                if (spec["draft_tokens"] != 0 or spec["accepted"] != 0
                        or spec["accept_ratio"] is not None
                        or spec["accepted_per_step"] is not None
                        or any(v != 0 for v in spec["pos"])):
                    raise RuntimeError("inconsistent zero-draft diagnostic counters")
            elif spec["drafts"] <= 0 or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                                          or not 0 <= v <= 1 for v in [spec["accept_ratio"], *spec["pos"]]):
                raise RuntimeError("invalid speculative acceptance")
        else:
            if not rec.get("gen_ok") or rec["finish_reason"] != "stop":
                raise RuntimeError("invalid OK generation/finish")
            if rec["completion_tokens"] > 8:
                raise RuntimeError("prefill completion exceeds 8-token budget")
            if rec["kind"] == "apc":
                rec["hit_ratio"] = hits / rec["prompt_tokens"]
                if rec["hit_ratio"] < 0.80:
                    raise RuntimeError("APC hit ratio below 80%")
            else:
                if hits != 0:
                    raise RuntimeError("cold request reused cached tokens")
                if abs(rec["prompt_tokens"] - rec["target_tokens"]) / rec["target_tokens"] > 0.02:
                    raise RuntimeError("cold prompt outside 2% of target")

    def live(self, name, seed, target, repeat, measured=True):
        def prepare(rec):
            salt = uuid.uuid4().hex
            prefix = "session" if measured else "warmup"
            rec["text_salt"] = salt
            text, estimated = fit_prompt(
                f"{prefix}={salt}\nRead the material, then reply exactly OK.\n" + seed, target)
            return cold.chat_messages(text), estimated, target, {"top_p": 1, "cache_salt": uuid.uuid4().hex}, 1200
        return self.sample(name, repeat, "cold", prepare, measured)

    def holdout(self, name, target, repeat, measured=True):
        def prepare(rec):
            rec["text_salt"] = cold.unique_salt()
            filler, estimated, text = cold.calibrate(target, rec["text_salt"], timeout=180)
            rec["filler_count"] = filler
            return cold.chat_messages(text), estimated, target, {}, 1800 if target == 300000 else 1200
        return self.sample(name, repeat, "cold", prepare, measured)

    def aggregate(self):
        groups = {}
        for rec in self.doc["samples"]:
            if rec["measured"]:
                groups.setdefault(rec["case"], []).append(rec)
        summary = self.doc["summary"]
        for name, samples in groups.items():
            expected = 5 if samples[0]["kind"] == "decode" else self.doc["protocol"]["prefill_runs"]
            if len(samples) != expected or any(s["status"] != "ok" for s in samples):
                raise ValueError(f"incomplete/failed group {name}")
            result = {"sample_count": len(samples)}
            for field in ("tok_s", "ttft_s", "prompt_tokens", "completion_tokens", "cache_hits"):
                result[field + "_median"] = strict_median([s[field] for s in samples])
            if samples[0]["kind"] == "decode":
                for field in ("accept_ratio", "accepted_per_step"):
                    result[field + "_median"] = strict_median([s["spec"][field] for s in samples])
                result["accept_per_position_median"] = [strict_median([s["spec"]["pos"][i] for s in samples]) for i in range(7)]
            summary[name] = result
        for label, names in (("prefill_score_tok_s", [c[0] for c in CASES]),
                             ("long_prefill_score_tok_s", ["holdout_100k", "holdout_300k"])):
            # Retain historical geometric score per repeat; then median across repeats.
            scores = [math.exp(statistics.mean(math.log(groups[n][i]["tok_s"]) for n in names))
                      for i in range(self.doc["protocol"]["prefill_runs"])]
            summary[label] = {"samples": scores, "median": strict_median(scores)}

    def run(self):
        try:
            status, raw = cold.http_get("/health")
            self.doc["health"] = {"http": status, "body": raw}
            if status != 200:
                raise RuntimeError(f"health returned {status}")
            status, raw = cold.http_get("/v1/models")
            self.doc["models"] = {"http": status, "raw": raw}
            if status != 200:
                raise RuntimeError(f"models returned {status}")
            models = json.loads(raw)
            cold.SERVED = decode.MODEL = models["data"][0]["id"]
            self.doc["served_model"] = cold.SERVED
            self.live("warmup_8k", PROSE, 8192, 0, measured=False)
            for repeat in range(self.doc["protocol"]["prefill_runs"]):
                for name, seed, target in CASES:
                    self.live(name, seed, target, repeat)
                self.holdout("holdout_100k", 100000, repeat)
                self.holdout("holdout_300k", 300000, repeat)
                history = self.holdout("apc_cold_8k", 8192, repeat)
                messages = history["request"]["messages"] + [
                    {"role": "assistant", "content": history["gen"]},
                    {"role": "user", "content": "Confirm with OK."}]
                self.sample("apc_follow_8k", repeat, "apc", lambda rec: (messages, None, None, {}, 1200))
            for name, prompt in (("readme_structured", decode.STRUCTURED_PROMPT),
                                 ("readme_prose", decode.BENCH_PROMPT), ("lru_code_guard", LRU)):
                def request(tokens, text=prompt):
                    return lambda rec: (cold.chat_messages(text), None, None, {"max_tokens": tokens}, 900)
                if name != "readme_prose":
                    self.sample(name + "_warmup", 0, "decode", request(32), measured=False)
                for repeat in range(5):
                    self.sample(name, repeat, "decode", request(400))
                if name != "lru_code_guard":
                    status, raw = cold.http_get("/health")
                    self.doc.setdefault("health_after_decode", {})[name] = {"http": status, "body": raw}
                    if status != 200:
                        raise RuntimeError("health after decode failed")
                    self.sample(name + "_short_after", 0, "decode", request(8, decode.BENCH_PROMPT), measured=False)
            self.aggregate()
            self.doc["status"] = "ok"
        except Exception as exc:
            self.doc["status"] = "failed"
            self.doc["failure"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.doc["ended"] = time.time()
            self.save()
        return self.doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=3, help="prefill/holdout/APC repetitions; decode is always 5x400")
    parser.add_argument("--provenance", type=Path, help="optional independently collected server source/config JSON")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be positive")
    provenance = json.loads(args.provenance.read_text()) if args.provenance else None
    result = Panel(args.base_url, args.out, args.runs, provenance).run()
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
