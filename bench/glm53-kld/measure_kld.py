#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KL(BF16 teacher || captured student) over sealed logit captures.

Chunks positions through a float64 log-softmax on the chosen device, records
the per-token vector as the primary artifact, and seals a receipt with
per-window / per-domain summaries and top-1 agreement. Accepts any capture
role on the student side (our engine capture, or a published offline capture)
as long as both sides name the same panel windows and token hashes.

``--self-canary`` compares a capture with itself: every per-token value must be
*exactly* zero. That is the protocol's R0 canary for pipeline determinism and
logit alignment; a non-zero canary invalidates the session.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from kld_core import (
    LOGITS_FIELD,
    load_capture_receipt,
    prepare_empty_destination,
    seal,
    sha256_bytes,
    summarize,
    token_kld,
    write_json,
)

SCHEMA = "cmp170hx.glm53-kld-report.v1"
REQUIRED_FIELDS = ("document_id", "domain", "role", "token_ids_sha256",
                   "attention_mask_sha256", "prediction_positions")


def _slice(path: Path, start: int, stop: int, device: str) -> np.ndarray:
    from safetensors import safe_open

    with safe_open(path, framework="np", device="cpu") as handle:
        rows = handle.get_slice(LOGITS_FIELD)[start:stop]
        return np.asarray(rows, dtype=np.float32)


def _validate(teacher: dict, student: dict) -> dict[str, tuple[dict, dict]]:
    if teacher.get("vocab_size") != student.get("vocab_size"):
        raise ValueError("teacher and student vocabularies differ")
    if teacher.get("logits_dtype") != student.get("logits_dtype"):
        raise ValueError("teacher and student logits dtypes differ")
    teacher_rows = {row["window_id"]: row for row in teacher.get("logit_files", ())}
    student_rows = {row["window_id"]: row for row in student.get("logit_files", ())}
    if set(teacher_rows) != set(student_rows) or not teacher_rows:
        raise ValueError("teacher and student window sets differ")
    for window_id, left in teacher_rows.items():
        right = student_rows[window_id]
        if any(left.get(field) != right.get(field) for field in REQUIRED_FIELDS):
            raise ValueError(f"student capture relabels or resegments sealed window {window_id}")
        if left["prediction_positions"] <= 0:
            raise ValueError(f"window {window_id} has no prediction positions")
    return {window_id: (teacher_rows[window_id], student_rows[window_id]) for window_id in sorted(teacher_rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-receipt", type=Path, required=True)
    parser.add_argument("--student-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-positions", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="cpu is exact and fast enough; cuda:N also works")
    parser.add_argument("--self-canary", action="store_true",
                        help="compare a capture with itself; every KLD value must be exactly 0")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.chunk_positions < 1:
        raise ValueError("--chunk-positions must be positive")

    teacher = load_capture_receipt(args.teacher_receipt)
    student = load_capture_receipt(args.student_receipt)
    pairs = _validate(teacher, student)
    plan = {
        "schema": SCHEMA,
        "teacher_capture_sha256": sha256_bytes(_canonical(teacher)),
        "student_capture_sha256": sha256_bytes(_canonical(student)),
        "student_label": student.get("student_label"),
        "kld_direction": "teacher_to_student",
        "windows": len(pairs),
        "prediction_positions": sum(left["prediction_positions"] for left, _ in pairs.values()),
        "compute_device": args.device,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    output = prepare_empty_destination(args.output)
    started = time.monotonic()
    per_window: dict[str, np.ndarray] = {}
    top1_matches = 0
    total_positions = 0
    for window_id, (teacher_row, student_row) in pairs.items():
        count = int(teacher_row["prediction_positions"])
        values = np.empty(count, dtype=np.float64)
        for start in range(0, count, args.chunk_positions):
            stop = min(start + args.chunk_positions, count)
            teacher_logits = _slice(Path(teacher_row["path"]), start, stop, args.device)
            student_logits = _slice(Path(student_row["path"]), start, stop, args.device)
            values[start:stop] = token_kld(teacher_logits, student_logits)
            top1_matches += int(np.count_nonzero(
                teacher_logits.argmax(axis=-1) == student_logits.argmax(axis=-1)))
        per_window[window_id] = values
        total_positions += count
        print(f"[glm53-kld] {window_id}: mean={values.mean():.6f} p99={np.quantile(values, 0.99):.6f}",
              flush=True)

    flat = np.concatenate([per_window[window_id] for window_id in sorted(per_window)])
    if args.self_canary and not np.all(flat == 0.0):
        raise RuntimeError(f"R0 canary failed: max={flat.max()} (pipeline is not deterministic or misaligned)")

    domains: dict[str, list[float]] = defaultdict(list)
    for window_id, (teacher_row, _) in pairs.items():
        domains[teacher_row["domain"]].extend(per_window[window_id].tolist())

    report = {
        "schema": SCHEMA,
        "student_label": student.get("student_label"),
        "student_capture_role": student.get("capture_role"),
        "teacher_capture_role": teacher.get("capture_role"),
        "kld_direction": "teacher_to_student",
        "self_canary": bool(args.self_canary),
        "engine": student.get("engine"),
        "engine_flags": student.get("engine_flags"),
        "aggregate": summarize(flat),
        "top1_agreement": top1_matches / total_positions,
        "prediction_positions": int(total_positions),
        "vocab_size": int(teacher["vocab_size"]),
        "per_window": {window_id: summarize(values) for window_id, values in sorted(per_window.items())},
        "per_domain": {domain: summarize(np.asarray(values)) for domain, values in sorted(domains.items())},
        "compute_device": args.device,
        "elapsed_seconds": time.monotonic() - started,
    }
    tokenwise_path = output / "tokenwise-kld.npy"
    np.save(tokenwise_path, flat)
    report["tokenwise_kld_sha256"] = sha256_bytes(tokenwise_path.read_bytes())
    seal(report, "report_sha256")
    write_json(output / "kld-report.json", report)
    print(json.dumps({"ok": True, "mean_kld": report["aggregate"]["mean"],
                      "p99": report["aggregate"]["p99"],
                      "cvar95": report["aggregate"]["cvar95"],
                      "top1_agreement": report["top1_agreement"],
                      "report_sha256": report["report_sha256"]}, sort_keys=True), flush=True)
    return 0


def _canonical(value: dict) -> bytes:
    from kld_core import canonical_json

    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return canonical_json(body)


if __name__ == "__main__":
    raise SystemExit(main())
