#!/usr/bin/env python3
"""Bounded synthetic 128K APC edit/branch comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "glm-5.3-flash-exl3"
BASE = "http://127.0.0.1:32105"
POSITIONS = (1, 5, 9)
ORIGINAL = {1: "PINE-1017", 5: "EMBER-5051", 9: "QUARTZ-9097"}
EDITED = {1: "CEDAR-1183", 5: "FLAME-5279", 9: "ONYX-9311"}


def digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def request(path: str, payload: dict | None = None, timeout: float = 1800):
    data = json.dumps(payload, separators=(",", ":")).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET")
    started = time.perf_counter()
    try:
        return urllib.request.urlopen(req, timeout=timeout), started
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {body[:500]}") from exc


def tokenize(messages: list[dict]) -> list[int]:
    response, _ = request("/tokenize", {
        "model": MODEL, "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False}})
    with response:
        body = json.loads(response.read())
    return body["tokens"]


def metric_snapshot() -> dict[str, float]:
    response, _ = request("/metrics")
    with response:
        source = response.read().decode("utf-8", "replace")
    patterns = {
        "local_compute": re.compile(r'^vllm:prompt_tokens_by_source_total\{[^}]*source="local_compute"[^}]*\}\s+(\S+)$'),
        "local_cache_hit": re.compile(r'^vllm:prompt_tokens_by_source_total\{[^}]*source="local_cache_hit"[^}]*\}\s+(\S+)$'),
        "preemptions": re.compile(r'^vllm:num_preemptions_total\{[^}]*\}\s+(\S+)$'),
        "running": re.compile(r'^vllm:num_requests_running\{[^}]*\}\s+(\S+)$'),
        "waiting": re.compile(r'^vllm:num_requests_waiting\{[^}]*\}\s+(\S+)$'),
    }
    result: dict[str, float] = {}
    for line in source.splitlines():
        for key, pattern in patterns.items():
            match = pattern.match(line)
            if match:
                result[key] = result.get(key, 0.0) + float(match.group(1))
    missing = sorted(set(patterns) - set(result))
    if missing:
        raise RuntimeError(f"required metrics absent: {missing}")
    return result


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: round(after[key] - before[key], 3) for key in before}


def reset_cache() -> dict:
    idle = metric_snapshot()
    if idle["running"] != 0 or idle["waiting"] != 0:
        raise RuntimeError(f"refusing reset with active requests: {idle}")
    response, started = request("/reset_prefix_cache", {})
    with response:
        body_raw = response.read().decode("utf-8", "replace")
        status = response.status
    try:
        body = json.loads(body_raw)
    except json.JSONDecodeError:
        body = body_raw
    after = metric_snapshot()
    if status != 200 or not isinstance(body, dict) or body.get("success") is not True:
        raise RuntimeError(f"cache reset failed: HTTP {status}: {body!r}")
    if after["running"] != 0 or after["waiting"] != 0:
        raise RuntimeError(f"requests appeared during reset: {after}")
    return {"http": status, "body": body,
            "seconds": round(time.perf_counter() - started, 3), "idle": idle}


def segment(index: int, repeats: int) -> dict:
    key = ORIGINAL.get(index, f"AUX-{index:02d}42")
    content = (
        f"Synthetic ledger segment {index:02d}. Checkpoint S{index:02d} has value {key}. "
        "All checkpoint assignments are authoritative and independent."
        + " alpha" * repeats)
    return {"role": "user", "content": content}


def build_base(target: int = 128000) -> list[dict]:
    messages: list[dict] = []
    for index in range(10):
        messages.append(segment(index, 12700))
        messages.append({"role": "assistant", "content": f"Recorded synthetic segment {index:02d}."})
    messages.append({"role": "user", "content": "Acknowledge the complete synthetic ledger in three words."})
    for _ in range(8):
        tokens = tokenize(messages)
        gap = target - len(tokens)
        if gap == 0:
            return messages
        if 12700 + gap < 1:
            raise RuntimeError(f"target adjustment invalid: gap={gap}")
        messages[18] = segment(9, 12700 + gap)
    raise RuntimeError(f"could not construct exact {target} tokens; got {len(tokenize(messages))}")


def identity(messages: list[dict]) -> dict:
    tokens = tokenize(messages)
    return {
        "count": len(tokens), "messages_sha256": digest(messages),
        "token_ids_sha256": digest(tokens), "first_token_ids": tokens[:8],
        "last_token_ids": tokens[-8:], "tokens": tokens,
    }


def lcp(left: list[int], right: list[int]) -> int:
    for index, pair in enumerate(zip(left, right)):
        if pair[0] != pair[1]:
            return index
    return min(len(left), len(right))


def stream_chat(messages: list[dict], max_tokens: int) -> dict:
    payload = {
        "model": MODEL, "messages": messages, "temperature": 0, "seed": 7391,
        "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    before = metric_snapshot()
    response, started = request("/v1/chat/completions", payload)
    first = None
    chunks: list[str] = []
    usage: dict = {}
    finish_reason = None
    with response:
        status = response.status
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            obj = json.loads(line[5:].strip())
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            finish_reason = choices[0].get("finish_reason") or finish_reason
            part = choices[0].get("delta") or {}
            value = part.get("content") or part.get("reasoning") or part.get("reasoning_content") or ""
            if value:
                first = first or time.perf_counter()
                chunks.append(value)
    ended = time.perf_counter()
    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(prompt_tokens, int):
        raise RuntimeError(f"stream usage omitted prompt_tokens: {usage}")
    after = metric_snapshot()
    for _ in range(20):
        metrics = delta(before, after)
        if round(metrics["local_compute"] + metrics["local_cache_hit"]) >= prompt_tokens:
            break
        time.sleep(0.1)
        after = metric_snapshot()
    metrics = delta(before, after)
    accounted = round(metrics["local_compute"] + metrics["local_cache_hit"])
    if accounted != prompt_tokens:
        raise RuntimeError(
            f"prompt accounting mismatch: compute+hit={accounted}, usage={prompt_tokens}")
    return {
        "ok": status == 200 and first is not None, "http": status,
        "ttft_s": round(first - started, 3) if first else None,
        "wall_s": round(ended - started, 3), "usage": usage,
        "metric_delta": metrics, "finish_reason": finish_reason,
        "content": "".join(chunks),
    }


def cases(base: list[dict], prime_answer: str) -> list[tuple[str, list[dict], str]]:
    suffix = [{"role": "assistant", "content": prime_answer}]
    rows = [("append_control", base + suffix + [{"role": "user", "content":
        "Append control: reply with exactly APPEND-OK."}], "APPEND-OK")]
    for position in POSITIONS:
        edited = json.loads(json.dumps(base))
        edited[2 * position]["content"] = edited[2 * position]["content"].replace(
            ORIGINAL[position], EDITED[position])
        rows.append((f"edit_{position * 10}pct", edited + suffix + [{"role": "user", "content":
            f"What is the current value of checkpoint S{position:02d}? Reply with the value only."}],
            EDITED[position]))
    for position in POSITIONS:
        branch = json.loads(json.dumps(base[:2 * position]))
        marker = f"Checkpoint S{position:02d} has value {ORIGINAL[position]}."
        source = base[2 * position]["content"]
        marker_end = source.index(marker) + len(marker)
        branch.append({"role": "user", "content": source[:marker_end]})
        branch.append({"role": "assistant", "content": f"Recorded synthetic segment {position:02d}."})
        branch.append({"role": "user", "content":
            f"From the retained branch, give checkpoint S{position:02d}'s value only."})
        rows.append((f"branch_{position * 10}pct", branch, ORIGINAL[position]))
    return rows


def persist(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def self_test() -> None:
    assert lcp([1, 2, 3], [1, 2, 4]) == 2
    assert lcp([1, 2], [1, 2, 3]) == 2
    toy = [{"role": "user", "content": "x"}]
    assert json.loads(json.dumps(toy)) == toy
    assert set(ORIGINAL) == set(EDITED) == set(POSITIONS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime", required=True, choices=("new", "old"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--reference-prepared")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("self-test passed")
        return 0
    output = Path(args.output)
    base = build_base()
    base_id = identity(base)
    prepared = {"runtime": args.runtime, "base_identity": dict(base_id), "cases": []}
    prepared["base_identity"].pop("tokens")
    synthetic_cases = cases(base, "Ledger acknowledged.")
    for label, messages, expected in synthetic_cases:
        item = identity(messages)
        tokens = item.pop("tokens")
        prepared["cases"].append({"label": label, "expected": expected,
            "identity": item, "lcp_tokens": lcp(base_id["tokens"], tokens),
            "lcp_fraction": round(lcp(base_id["tokens"], tokens) / len(base_id["tokens"]), 6)})
    if args.reference_prepared:
        reference = json.loads(Path(args.reference_prepared).read_text())
        reference_shape = (
            reference["base_identity"]["count"],
            reference["base_identity"]["token_ids_sha256"],
            [(row["label"], row["identity"]["count"], row["identity"]["token_ids_sha256"])
             for row in reference["cases"]],
        )
        current_shape = (
            prepared["base_identity"]["count"],
            prepared["base_identity"]["token_ids_sha256"],
            [(row["label"], row["identity"]["count"], row["identity"]["token_ids_sha256"])
             for row in prepared["cases"]],
        )
        if current_shape != reference_shape:
            raise RuntimeError("old/new tokenized inputs differ")
        prepared["reference_inputs_match"] = True
    if args.prepare_only:
        persist(output, prepared)
        print(json.dumps(prepared, indent=2, sort_keys=True))
        return 0

    result = {"started_unix": time.time(), **prepared, "rows": []}
    persist(output, result)
    for case_index in range(7):
        reset = reset_cache()
        prime = stream_chat(base, 8)
        if prime["metric_delta"]["local_cache_hit"] != 0:
            raise RuntimeError(f"cold prime had cached tokens: {prime['metric_delta']}")
        if prime["metric_delta"]["local_compute"] != 128000:
            raise RuntimeError(f"cold prime compute was not 128000: {prime['metric_delta']}")
        actual_cases = cases(base, "Ledger acknowledged.")
        label, messages, expected = actual_cases[case_index]
        probe_id = identity(messages)
        probe_tokens = probe_id.pop("tokens")
        probe = stream_chat(messages, 24)
        content = probe["content"]
        row = {
            "label": label, "reset": reset, "prime": prime,
            "probe_identity": probe_id,
            "lcp_tokens": lcp(base_id["tokens"], probe_tokens),
            "lcp_fraction": round(lcp(base_id["tokens"], probe_tokens) / len(base_id["tokens"]), 6),
            "expected": expected, "expected_found": expected.lower() in content.lower(),
            "stale_values_found": [value for value in ORIGINAL.values()
                if value != expected and value.lower() in content.lower()],
            "probe": probe,
        }
        result["rows"].append(row)
        persist(output, result)
    result["finished_unix"] = time.time()
    result["elapsed_s"] = round(result["finished_unix"] - result["started_unix"], 3)
    result["pass"] = all(r["prime"]["ok"] and r["probe"]["ok"] and
        r["expected_found"] and not r["stale_values_found"] and
        r["prime"]["metric_delta"]["local_cache_hit"] == 0 and
        r["probe"]["metric_delta"]["preemptions"] == 0 for r in result["rows"])
    persist(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
