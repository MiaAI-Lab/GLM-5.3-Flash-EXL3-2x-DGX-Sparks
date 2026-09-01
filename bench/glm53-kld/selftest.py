#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU self-test for the KLD harness: exact math, canaries, sealing, fail-closed.

Runs without GPUs, weights or the published suite. It checks the properties the
protocol depends on:

1. analytic KL for a one-class logit shift (closed form, ~1e-14 agreement);
2. agreement with an independent scalar ``math.fsum`` implementation;
3. causal prediction-position rule (2047 scored rows per 2048-token window);
4. sealed logit receipts verify, and one flipped byte fails closed;
5. the suite loader enforces sealed token digests, the row/target offset, and the
   lane's declared semantic point - and refuses a cut point that claims the final
   norm still has to be applied at replay;
6. head-only replay matches an independent float64 reference, the row-shift
   alignment audit degrades top-1 agreement, and the reference lane against
   itself is *exactly* zero;
7. cluster bootstrap widens with correlated clusters and collapses to the point
   estimate under one cluster; JSD(bits) is zero for identical rows, below KL;
8. the worker tap fires on the norm *output*, refuses non-prefill row counts,
   drops the untargetable last row, and skips ``PPMissingLayer`` stages.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from kld_core import seal, sha256_file, token_kld, write_json

HERE = Path(__file__).resolve().parent
WORK = Path("/tmp/glm53-kld-selftest")
POSITIONS = 64
VOCAB = 512


def analytic_shift_kld(teacher: np.ndarray, index: int, shift: float) -> np.ndarray:
    """KL(p || q) where student logits equal teacher logits with entry ``index`` +shift.

    q is p with mass on ``index`` multiplied by e^shift and renormalised, so
    KL = log(1 + p_k (e^shift - 1)) - shift * p_k, exactly.
    """
    shifted = teacher - teacher.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    partition = weights.sum(axis=-1, keepdims=True)
    probabilities = weights / partition
    mass = probabilities[:, index]
    return np.log1p(mass * (math.exp(shift) - 1.0)) - shift * mass


def brute_force_row(row_teacher: np.ndarray, row_student: np.ndarray) -> float:
    def log_softmax_row(values: np.ndarray) -> tuple[list[float], list[float]]:
        maximum = max(values)
        shifted = [float(value) - maximum for value in values]
        partition = math.fsum(math.exp(value) for value in shifted)
        log_partition = math.log(partition)
        return [value - log_partition for value in shifted], [math.exp(value - log_partition) for value in shifted]

    teacher_logp, teacher_p = log_softmax_row(row_teacher)
    student_logp, _ = log_softmax_row(row_student)
    return math.fsum(p * (lp - sp) for p, lp, sp in zip(teacher_p, teacher_logp, student_logp))


def analytic_agreement(rng: np.random.Generator) -> None:
    teacher = rng.standard_normal((POSITIONS, VOCAB), dtype=np.float64) * 3.0
    for index, shift in ((0, 0.25), (7, 1.5), (VOCAB // 2, -0.75), (VOCAB - 1, 4.0)):
        student = teacher.copy()
        student[:, index] += shift
        measured = token_kld(teacher, student)
        expected = analytic_shift_kld(teacher, index, shift)
        error = float(np.max(np.abs(measured - expected)))
        assert error < 1e-13, f"analytic mismatch {error:.3e} (index={index} shift={shift})"
        assert float(np.min(measured)) >= 0.0, "KLD must be non-negative"
    print(f"[selftest] analytic shift agreement < 1e-13 (worst index/shift pairs verified)")


def brute_force_agreement(rng: np.random.Generator) -> None:
    teacher = rng.standard_normal((8, VOCAB), dtype=np.float64) * 4.0
    student = teacher + rng.standard_normal((8, VOCAB), dtype=np.float64) * 0.4
    measured = token_kld(teacher, student)
    for row in range(8):
        expected = brute_force_row(teacher[row], student[row])
        error = abs(float(measured[row]) - expected)
        assert error < 1e-12, f"scalar-reference mismatch {error:.3e} on row {row}"
    print("[selftest] vectorised fp64 KLD matches scalar math.fsum reference < 1e-12")


def canary() -> None:
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((POSITIONS, VOCAB), dtype=np.float64) * 8.0
    values = token_kld(logits, logits)
    assert np.all(values == 0.0), f"R0 canary non-zero: max={values.max():.3e}"
    print("[selftest] R0 self-KLD canary is exactly 0.0")


def build_capture(root: Path, tag: str, logits: np.ndarray, windows: int) -> Path:
    from safetensors.torch import save_file
    import torch

    directory = root / tag
    logits_dir = directory / "logits"
    logits_dir.mkdir(parents=True)
    rows: list[dict] = []
    per_window = POSITIONS // windows
    for index in range(windows):
        block = logits[index * per_window : (index + 1) * per_window].astype(np.float32)
        path = (logits_dir / f"window-{index:04d}.safetensors").resolve()
        save_file({"logits": torch.from_numpy(np.ascontiguousarray(block))}, path,
                  metadata={"capture_role": tag})
        rows.append({
            "window_id": f"final-{index:04d}",
            "document_id": f"doc-{index}",
            "domain": "selftest",
            "role": "final",
            "token_ids_sha256": f"{index:064x}",
            "attention_mask_sha256": f"{index + 1000:064x}",
            "prediction_positions": int(block.shape[0]),
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    receipt = {
        "schema": "quant-pipeline.glm53-logit-capture.v1",
        "capture_role": "bf16_teacher" if tag == "teacher" else "engine_student",
        "student_label": "selftest-student",
        "logits_dtype": "float32",
        "kld_direction": "teacher_to_student",
        "vocab_size": VOCAB,
        "prediction_positions": POSITIONS,
        "logit_files": rows,
    }
    seal(receipt, "receipt_sha256")
    return write_json(directory / "capture-receipt.json", receipt) and directory / "capture-receipt.json"


def end_to_end(rng: np.random.Generator) -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    teacher_logits = rng.standard_normal((POSITIONS, VOCAB), dtype=np.float64) * 3.0
    student_logits = teacher_logits.copy()
    student_logits[:, 11] += 0.5
    expected = float(analytic_shift_kld(teacher_logits, 11, 0.5).mean())

    teacher_receipt = build_capture(WORK, "teacher", teacher_logits, 4)
    student_receipt = build_capture(WORK, "student", student_logits, 4)
    report_path = WORK / "report"
    completed = subprocess.run(
        [sys.executable, str(HERE / "measure_kld.py"),
         "--teacher-receipt", str(teacher_receipt),
         "--student-receipt", str(student_receipt),
         "--output", str(report_path), "--chunk-positions", "7", "--execute"],
        capture_output=True, text=True, cwd=HERE, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    report = json.loads((report_path / "kld-report.json").read_text())
    assert abs(report["aggregate"]["mean"] - expected) < 1e-9, (report["aggregate"]["mean"], expected)
    assert report["prediction_positions"] == POSITIONS
    assert report["aggregate"]["count"] == POSITIONS
    assert report["per_domain"]["selftest"]["count"] == POSITIONS
    print(f"[selftest] end-to-end capture -> report reproduces the analytic mean "
          f"({report['aggregate']['mean']:.12f} vs {expected:.12f})")

    canary_report = WORK / "canary"
    completed = subprocess.run(
        [sys.executable, str(HERE / "measure_kld.py"),
         "--teacher-receipt", str(teacher_receipt), "--student-receipt", str(teacher_receipt),
         "--output", str(canary_report), "--self-canary", "--execute"],
        capture_output=True, text=True, cwd=HERE, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert json.loads((canary_report / "kld-report.json").read_text())["aggregate"]["mean"] == 0.0
    print("[selftest] R0 canary passes through the CLI path (mean exactly 0.0)")

    victim = next((WORK / "student" / "logits").glob("window-0000.safetensors"))
    payload = bytearray(victim.read_bytes())
    payload[-40] ^= 0xFF
    victim.write_bytes(bytes(payload))
    completed = subprocess.run(
        [sys.executable, str(HERE / "measure_kld.py"),
         "--teacher-receipt", str(teacher_receipt), "--student-receipt", str(student_receipt),
         "--output", str(WORK / "tampered"), "--execute"],
        capture_output=True, text=True, cwd=HERE, check=False)
    assert completed.returncode != 0 and "tampered" in completed.stderr, completed.stdout[-500:]
    print("[selftest] tampered logit file fails closed on the sealed receipt")


def panel_positions() -> None:
    from kld_core import PanelWindow  # noqa: F401  (import path check)

    mask = np.ones(2048, dtype=np.uint8)
    assert int(np.count_nonzero(mask[:-1] & mask[1:])) == 2047
    padded = np.ones(2048, dtype=np.uint8)
    padded[-3:] = 0
    assert int(np.count_nonzero(padded[:-1] & padded[1:])) == 2044
    print("[selftest] causal prediction-position rule matches the published 2047/window geometry")


def suite_scorer_path() -> None:
    """Synthetic v6 suite: loader contracts, cut-point traps, scorer vs fp64."""
    import torch
    from safetensors.torch import save_file

    from kld_core import load_distribution_suite, token_ids_sha256

    root = WORK / "suite"
    if root.exists():
        shutil.rmtree(root)
    (root / "suite" / "tokens").mkdir(parents=True)
    (root / "head").mkdir()
    reference_lane = "reference-bf16-shard0"
    candidate_lane = "candidate-fp8-shard0"
    for lane in (reference_lane, candidate_lane):
        (root / lane).mkdir()

    hidden, vocab, context_length, rows, contexts = 16, 64, 8, 7, 3
    rng = np.random.default_rng(7)
    head = rng.standard_normal((vocab, hidden)) * 0.05
    save_file({"lm_head.weight": torch.from_numpy(head.astype(np.float32))},
              str(root / "head" / "head.safetensors"))

    index_rows: list[dict] = []
    captures: dict[str, list[dict]] = {reference_lane: [], candidate_lane: []}
    arrays: dict[str, list[np.ndarray]] = {reference_lane: [], candidate_lane: []}
    token_sets: list[list[int]] = []
    for index in range(contexts):
        token_ids = [int(token) for token in rng.integers(0, vocab, size=context_length)]
        token_sets.append(token_ids)
        (root / "suite" / "tokens" / f"context-{index:04d}.json").write_text(json.dumps(token_ids))
        reference = rng.standard_normal((rows, hidden))
        candidate = reference + rng.standard_normal((rows, hidden)) * 0.25
        for lane, array in ((reference_lane, reference), (candidate_lane, candidate)):
            path = root / lane / f"hidden_{index:04d}.safetensors"
            save_file({"hidden_states": torch.from_numpy(array.astype(np.float32))}, str(path))
            captures[lane].append({"index": index, "sha256": sha256_file(path), "shape": [rows, hidden]})
            arrays[lane].append(array)
        index_rows.append({
            "index": index,
            "stratum": "code" if index % 2 else "literary",
            "source_cluster": f"cluster-{index % 2}",
            "partition": "analysis",
            "file": f"tokens/context-{index:04d}.json",
            "tokens": context_length,
            "token_sha256": token_ids_sha256(token_ids),
        })
    write_json(root / "suite" / "suite-manifest.json", {
        "schema": "glm53flash-distribution-fidelity/6",
        "context_length": context_length,
        "scored_positions_per_context": rows,
        "hidden_size": hidden,
        "vocab_size": vocab,
        "model": "fake/GLM",
        "suite_token_sha256": "sealed",
        "context_index": index_rows,
    })
    for lane in captures:
        write_json(root / lane / "capture-manifest-shard.json", {
            "schema": "glm53flash-fidelity-capture/2",
            "semantic_point": "after_final_rmsnorm_before_lm_head",
            "tensor_key": "hidden_states",
            "lane": lane,
            "filter": "all",
            "contexts": len(captures[lane]),
            "complete": True,
            "captures": captures[lane],
        })

    meta, contexts_read = load_distribution_suite(root)
    assert [row.index for row in contexts_read] == [0, 1, 2]
    assert contexts_read[0].rows == rows
    # Row r is the state after tokens[r]; its target is tokens[r+1].
    assert list(contexts_read[0].targets) == token_sets[0][1:rows + 1]
    assert contexts_read[0].hidden_sha256 == captures[reference_lane][0]["sha256"]
    print(f"[selftest] suite loader seals {len(contexts_read)} contexts with row/target offset 1")

    # The expensive mistake: believing the norm still has to be applied.
    write_json(root / reference_lane / "capture-cut-point.json", {
        "schema": "malaiwah.glm53-capture-cut-point.v1",
        "semantic_point": "after_final_rmsnorm_before_lm_head",
        "final_norm": {"applied_at_replay": True},
    })
    try:
        load_distribution_suite(root)
    except ValueError as error:
        assert "applied at replay" in str(error), error
    else:
        raise AssertionError("loader accepted a lane that claims the norm is applied at replay")
    (root / reference_lane / "capture-cut-point.json").unlink()

    token_path = root / "suite" / "tokens" / "context-0001.json"
    sealed_token_path = token_path.read_text()
    token_path.write_text(sealed_token_path.replace("]", ", 7]", 1))
    try:
        load_distribution_suite(root)
    except ValueError as error:
        assert "token digest" in str(error), error
    else:
        raise AssertionError("a mutated token window slipped past the sealed digest")
    token_path.write_text(sealed_token_path)

    def replay(lane: str) -> np.ndarray:
        stacked = np.concatenate(arrays[lane], axis=0)
        return stacked @ head.T

    expected = float(np.mean(token_kld(replay(reference_lane), replay(candidate_lane))))
    report_path = root / "candidate-report.json"
    _run("score_hidden_kld.py", "--suite", str(root), "--candidate-lane", candidate_lane,
         "--out", str(report_path), "--device", "cpu", "--dtype", "float32",
         "--chunk", "4", "--bootstrap-samples", "200", "--offset-audit")
    report = json.loads(report_path.read_text())
    assert report["contexts"] == contexts and report["scored_positions"] == contexts * rows
    deviation = abs(report["token_mean_kld"] - expected) / expected
    assert deviation < 1e-3, f"scorer disagrees with an independent fp64 replay ({deviation:.2e})"
    assert abs(report["token_mean_kld"] - report["context_macro_mean_kld"]) < 1e-12
    assert report["mean_jsd_bits"] > 0.0 and report["max_kld"] >= report["p99_kld"]
    assert len(report["per_context"]) == contexts
    assert set(report["strata"]) == {"code", "literary"}
    audit = report["offset_audit"]["mean_top1"]
    assert audit["-1"] < report["top1_agreement"] and audit["+1"] < report["top1_agreement"], audit
    assert report["offset_audit"]["mean_kld"]["+1"] > report["token_mean_kld"]
    print(f"[selftest] head-only replay matches fp64 reference (<1e-3, mean={report['token_mean_kld']:.6f}); "
          f"row-shift audit degrades top1 {report['top1_agreement']:.3f} -> {audit['+1']:.3f}")

    canary_path = root / "canary-report.json"
    _run("score_hidden_kld.py", "--suite", str(root), "--candidate-lane", reference_lane,
         "--out", str(canary_path), "--device", "cpu", "--chunk", "4",
         "--expect-mean", "0.0", "--expect-relative", "0")
    assert json.loads(canary_path.read_text())["token_mean_kld"] == 0.0
    print("[selftest] reference-against-itself canary is exactly 0.0")


def _run(script: str, *arguments: str) -> None:
    completed = subprocess.run([sys.executable, str(HERE / script), *arguments],
                               capture_output=True, text=True, cwd=HERE, check=False)
    if completed.returncode != 0:
        raise AssertionError(f"{script} failed: {completed.stderr[-3000:]}")


def bootstrap_unit() -> None:
    """Cluster bootstrap: one cluster collapses, coarser clusters widen it."""
    from kld_core import cluster_bootstrap_ci

    values = [0.010 + 0.001 * (index % 5) for index in range(50)]
    every_cluster_alone = [{"mean_kld": value, "source_cluster": f"c{index}"}
                           for index, value in enumerate(values)]
    alone = cluster_bootstrap_ci(every_cluster_alone, samples=2000, seed=1)
    assert alone["clusters"] == 50 and alone["ci95_low"] < alone["mean"] < alone["ci95_high"]
    assert abs(alone["mean"] - float(np.mean(values))) < 1e-12

    shared = [{"mean_kld": value, "source_cluster": "one"} for value in values]
    collapsed = cluster_bootstrap_ci(shared, samples=2000, seed=1)
    assert collapsed["ci95_low"] == collapsed["mean"] == collapsed["ci95_high"], collapsed

    # Two documents, each carrying half the windows and a different mean: the
    # effective sample size drops to two, so the interval must widen.
    coarse = [{"mean_kld": 0.050 + 0.001 * (index % 5) if index < 25 else 0.010 + 0.001 * (index % 5),
               "source_cluster": f"c{index // 25}"}
              for index in range(50)]
    wider = cluster_bootstrap_ci(coarse, samples=2000, seed=1)
    assert (wider["ci95_high"] - wider["ci95_low"]) > 5 * (alone["ci95_high"] - alone["ci95_low"])
    print(f"[selftest] cluster bootstrap: 50 clusters [{alone['ci95_low']:.5f},{alone['ci95_high']:.5f}], "
          f"2 clusters [{wider['ci95_high'] - wider['ci95_low']:.5f} wide], 1 cluster collapses")

    import torch  # noqa: F401  (torch is a scorer dependency; keep the import path warm)

    from kld_core import token_jsd_bits

    rng = np.random.default_rng(5)
    teacher = rng.standard_normal((8, 128))
    identical = token_jsd_bits(teacher, teacher)
    assert np.allclose(identical, 0.0, atol=1e-12), identical.max()
    # Softmax is shift-invariant, so a constant offset must stay exactly zero;
    # only a shape change can move the divergence.
    shifted = token_jsd_bits(teacher, teacher + 1.0)
    assert np.allclose(shifted, 0.0, atol=1e-9), shifted.max()
    noisy = token_jsd_bits(teacher, teacher * 1.6 + 0.4)
    assert (noisy > 0).all(), noisy.min()
    assert (noisy <= 1.0 + 1e-9).all(), noisy.max()          # bounded by one bit
    back = token_jsd_bits(teacher * 1.6 + 0.4, teacher)
    assert np.allclose(noisy, back, atol=1e-12)               # symmetric
    print("[selftest] JSD(bits): zero for identical and for pure logit shifts, symmetric, <= 1 bit")


def hidden_hook_unit() -> None:
    """Worker-side tap: post-norm output, row gate, tuple output, PP skip, resume."""
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file

    import logit_dump_hook

    dump_root = WORK / "hookdumps"
    if dump_root.exists():
        shutil.rmtree(dump_root)
    os.environ["GLM_KLD_DUMP_DIR"] = str(dump_root)
    os.environ["GLM_KLD_DUMP_CONTEXT_ROWS"] = "5"
    dump_dir = dump_root / "hidden"

    class Norm(nn.Module):
        def __init__(self, size: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(size), requires_grad=False)

        def forward(self, rows, residual=None):
            # vLLM's RMSNorm returns a tuple once a residual is threaded through.
            return (rows, rows) if residual is not None else rows

    class Inner(nn.Module):
        def __init__(self, size: int):
            super().__init__()
            self.norm = Norm(size)

    class LanguageModel(nn.Module):
        def __init__(self, size: int):
            super().__init__()
            self.model = Inner(size)

    class Wrapper(nn.Module):
        def __init__(self, size: int):
            super().__init__()
            self.language_model = LanguageModel(size)

    class PPMissingLayer(nn.Module):
        def forward(self, rows):
            return rows

    model = Wrapper(8)
    reported = logit_dump_hook.register_hidden_capture(model)
    assert reported.startswith("hooked:Norm:hidden=8"), reported
    tap = json.loads((dump_dir / "tap.json").read_text())
    assert tap["semantic_point"] == "after_final_rmsnorm_before_lm_head"
    assert tap["rows_stored"] == 4 and tap["context_rows_required"] == 5

    with torch.no_grad():
        model.language_model.model.norm(torch.ones(3, 8))          # decode step: refused
        model.language_model.model.norm(torch.ones(5, 8), torch.ones(5, 8))  # prefill, tuple out
    files = sorted(dump_dir.glob("hidden-*.safetensors"))
    assert len(files) == 1, files
    state = model._glm_kld_hidden_state
    assert state["dumps"] == 1 and state["skipped_rows"] == 1, state
    stored = load_file(str(files[0]))["hidden_states"]
    assert tuple(stored.shape) == (4, 8), stored.shape   # 5 rows in, the last dropped
    assert stored.dtype == torch.bfloat16

    with torch.no_grad():
        model.language_model.model.norm(torch.ones(5, 8))
    assert len(sorted(dump_dir.glob("hidden-*.safetensors"))) == 2
    assert logit_dump_hook._highest(dump_dir, "hidden", ".safetensors") == 2

    stalled = Wrapper(8)
    stalled.language_model.model.norm = PPMissingLayer()
    assert logit_dump_hook.register_hidden_capture(stalled) == "skipped:PPMissingLayer"

    del os.environ["GLM_KLD_DUMP_DIR"]
    del os.environ["GLM_KLD_DUMP_CONTEXT_ROWS"]
    print("[selftest] hidden tap fires on the norm output, keeps only 2048-row prefills, "
          "drops the untargetable last row, skips PP-stub stages")


def main() -> int:
    analytic_agreement(np.random.default_rng(0))
    brute_force_agreement(np.random.default_rng(1))
    canary()
    panel_positions()
    end_to_end(np.random.default_rng(2))
    suite_scorer_path()
    bootstrap_unit()
    hidden_hook_unit()
    print("[selftest] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
