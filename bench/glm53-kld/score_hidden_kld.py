#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score a hidden-state capture against the sealed BF16 reference lane.

Both lanes are post-final-RMSNorm tensors of shape [2047, 4096] per context. The
only thing between them and the vocabulary is the shared ``lm_head``, which is
byte-identical in both published lanes (``reports/head-equality-fp8.json``) - so
replay is a single matmul:

    logits = hidden.to(dtype) @ lm_head.T          # no RMSNorm here, on purpose

Re-applying ``head/final_norm.safetensors`` on top of the published lane is the
classic way to produce a confident, wrong number: it rescales every logit row,
which reads like a temperature change and inflates KLD. The scorer refuses the
file rather than accept it, and ``--expect-mean`` re-checks the finished number
against the published row before anything is written.

Run the FP8 lane first - it needs no GPU capture and pins whether this scorer
agrees with the published protocol:

    python score_hidden_kld.py --candidate-lane as-served-fp8-shard0 \\
        --expect-mean 0.028103897727130314 --expect-relative 2e-3 --out fp8-anchor.json

With that green, ``--candidate-lane`` points at a lane we captured ourselves.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from kld_core import (
    PUBLISHED_ENGINE_NOISE_FLOOR_NATS,
    SEMANTIC_POINT,
    cluster_bootstrap_ci,
    load_distribution_suite,
    load_head,
    load_lane_tensor,
    load_replay_head,
    seal,
    sha256_file,
    write_json,
)

SCHEMA = "cmp170hx.glm53-hidden-kld.v2"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="/srv/models/fidelity-suites/GLM-5.3-Flash-fidelity-suite-v1")
    parser.add_argument("--reference-lane", default="reference-bf16-shard0")
    parser.add_argument("--candidate-lane", required=True)
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--indices", default=None, help="comma list of context indices")
    parser.add_argument("--partitions", default=None, help="e.g. analysis,qualification")
    parser.add_argument("--strata", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=256, help="rows per matmul")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                        help="matmul precision; float32 costs 2.5 GB for the head, bf16 1.27 GB")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offset-audit", action="store_true",
                        help="also score the candidate shifted by one row; alignment proof")
    parser.add_argument("--expect-mean", type=float, default=None,
                        help="published mean to reproduce; fails closed on mismatch")
    parser.add_argument("--expect-relative", type=float, default=2e-3)
    parser.add_argument("--compare-report", default=None,
                        help="published report with per_context[] to compare index by index")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def verify_lane_file(row) -> None:
    """Re-verify the sealed digest of a lane file before scoring it.

    A silently corrupted download (see fetch-fidelity-suite.sh - it has happened)
    biases KLD upward and looks exactly like a worse quantization, so the digest
    the lane manifest publishes is checked again at the moment of use.
    """
    if row.hidden_sha256 and sha256_file(row.hidden_path) != row.hidden_sha256:
        raise ValueError(
            f"{row.hidden_path}: digest does not match the lane manifest "
            f"({sha256_file(row.hidden_path)[:12]} != {row.hidden_sha256[:12]})")


def load_lane_rows(path: Path, expected_rows: int):
    rows = load_lane_tensor(path)
    if rows.shape[0] != expected_rows:
        raise ValueError(f"{path}: {rows.shape[0]} rows, suite scores {expected_rows}")
    return rows


@torch.no_grad()
def score_context(reference, candidate, head, *, chunk: int, dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-position KL(ref||cand) in nats, JSD in bits, top-1 agreement flag."""
    import torch

    rows = reference.shape[0]
    kl = np.empty(rows, dtype=np.float64)
    jsd = np.empty(rows, dtype=np.float64)
    agree = np.empty(rows, dtype=bool)
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        ref_logits = reference[start:stop].to(head.device, dtype) @ head.T
        cand_logits = candidate[start:stop].to(head.device, dtype) @ head.T
        ref_logp = torch.log_softmax(ref_logits.float(), dim=-1)
        cand_logp = torch.log_softmax(cand_logits.float(), dim=-1)
        p = ref_logp.exp()
        kl[start:stop] = (p * (ref_logp - cand_logp)).sum(dim=-1).cpu().numpy()
        mixture = torch.logaddexp(ref_logp, cand_logp) - math.log(2.0)
        left = (p * (ref_logp - mixture)).sum(dim=-1)
        right = (cand_logp.exp() * (cand_logp - mixture)).sum(dim=-1)
        jsd[start:stop] = ((left + right) * 0.5 / math.log(2.0)).cpu().numpy()
        agree[start:stop] = (ref_logits.argmax(dim=-1) == cand_logits.argmax(dim=-1)).cpu().numpy()
        del ref_logits, cand_logits, ref_logp, cand_logp, p, mixture, left, right
    if not np.isfinite(kl).all() or not np.isfinite(jsd).all():
        raise ValueError(f"non-finite divergence at rows {np.flatnonzero(~np.isfinite(kl))[:5]}")
    return kl, jsd, agree


def context_record(row) -> dict:
    return {
        "index": int(row.index),
        "stratum": row.stratum,
        "partition": row.partition,
        "source_cluster": row.source_cluster,
        "token_sha256": row.token_sha256,
    }


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    indices = None if not args.indices else [int(x) for x in args.indices.split(",") if x]
    partitions = None if not args.partitions else args.partitions.split(",")
    strata = None if not args.strata else args.strata.split(",")

    head_path, head_meta = load_replay_head(args.suite)
    print(f"head {head_path} sha256={head_meta['sha256'][:16]}…", flush=True)
    head = load_head(head_path).to(device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    # The published head is BF16. float32 replay casts it: the tensor-core bf16
    # matmul carries ~0.1 nats of its own bias (measured), which is larger than
    # the whole repeat-noise floor and would be blamed on the quantization.
    head = head.to(dtype)

    # Both lanes are enumerated completely, then intersected: while the download
    # is in progress the lanes cover different prefixes, and scoring anything but
    # the intersection would pair contexts with the wrong partner. The scored
    # scope is recorded in the report so a partial row can never read as the
    # published one.
    reference_meta, references = load_distribution_suite(
        args.suite, lane=args.reference_lane, indices=indices, partitions=partitions,
        strata=strata,
    )
    candidate_meta, candidates = load_distribution_suite(
        args.suite, lane=args.candidate_lane, indices=indices, partitions=partitions,
        strata=strata,
    )
    by_candidate = {row.index: row for row in candidates}
    pairs = [(row, by_candidate[row.index]) for row in references if row.index in by_candidate]
    if not pairs:
        raise ValueError("the two lanes share no contexts")
    if args.limit:
        pairs = pairs[:args.limit]
    references = [pair[0] for pair in pairs]
    candidates = [pair[1] for pair in pairs]
    scope = {
        "contexts": len(references),
        "index_min": references[0].index,
        "index_max": references[-1].index,
        "reference_lane_files": len(references),
        "candidate_lane_files": len(by_candidate),
        "sealed_contexts_total": reference_meta["sealed_contexts_total"],
        "complete_scope": len(references) == reference_meta["sealed_contexts_total"],
    }
    rows_per_context = reference_meta["scored_positions_per_context"]
    print(f"contexts={len(references)} (indices {references[0].index}-{references[-1].index}; "
          f"reference files={len(pairs)}/{len(references)}, candidate files present={len(by_candidate)}) "
          f"rows/context={rows_per_context} vocab={reference_meta['vocab_size']} device={device}", flush=True)

    per_context: list[dict] = []
    per_context_offsets: dict[int, dict] = {}
    all_kl: list[np.ndarray] = []
    started = time.time()
    for position, (reference, candidate) in enumerate(zip(references, candidates), start=1):
        if reference.token_sha256 != candidate.token_sha256:
            raise ValueError(f"context {reference.index}: token digest differs between lanes")
        verify_lane_file(reference)
        verify_lane_file(candidate)
        ref_rows = load_lane_rows(reference.hidden_path, rows_per_context)
        cand_rows = load_lane_rows(candidate.hidden_path, rows_per_context)
        ref = torch.as_tensor(ref_rows)
        cand = torch.as_tensor(cand_rows)
        kl, jsd, agree = score_context(ref, cand, head, chunk=args.chunk, dtype=dtype)
        record = context_record(reference)
        record.update({
            "mean_kld": float(kl.mean()),
            "median_kld": float(np.median(kl)),
            "max_kld": float(kl.max()),
            "mean_jsd_bits": float(jsd.mean()),
            "top1_agreement": float(agree.mean()),
        })
        per_context.append(record)
        all_kl.append(kl)
        if args.offset_audit:
            entry = {}
            for offset in (-1, 1):
                shifted = torch.roll(cand, shifts=offset, dims=0)
                kl_shift, _, agree_shift = score_context(
                    ref, shifted, head, chunk=args.chunk, dtype=dtype)
                entry[f"{offset:+d}"] = {
                    "mean_kld": float(kl_shift.mean()),
                    "top1_agreement": float(agree_shift.mean()),
                }
            per_context_offsets[reference.index] = entry
        del ref, cand, ref_rows, cand_rows
        if position % 32 == 0 or position == len(references):
            rate = position / (time.time() - started)
            print(f"  {position}/{len(references)} contexts "
                  f"({rate:.1f}/s, eta {(len(references) - position) / max(rate, 1e-9):.0f}s)", flush=True)

    flat = np.concatenate(all_kl)
    means = np.array([row["mean_kld"] for row in per_context])
    token_mean = float(flat.mean())
    report = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "semantic_point": SEMANTIC_POINT,
        "reference_lane": reference_meta["lane_dir"],
        "candidate_lane": candidate_meta["lane_dir"],
        "candidate_label": args.candidate_label or Path(candidate_meta["lane_dir"]).name,
        "reference_identity": reference_meta["model_identity"],
        "candidate_identity": candidate_meta["model_identity"],
        "head": head_meta,
        "suite_token_sha256": reference_meta["suite_token_sha256"],
        "hidden_size": reference_meta["hidden_size"],
        "vocab_size": reference_meta["vocab_size"],
        "dtype": args.dtype,
        "device": str(device),
        "scope": scope,
        "contexts": len(per_context),
        "scored_positions": int(flat.size),
        "scored_position_window": {
            "score_from": 0,
            "positions_per_context": rows_per_context,
            "policy": "every scored position of every context in the lane",
        },
        "token_mean_kld": token_mean,
        "context_macro_mean_kld": float(means.mean()),
        "token_median_kld": float(np.median(flat)),
        "p95_kld": float(np.quantile(flat, 0.95)),
        "p99_kld": float(np.quantile(flat, 0.99)),
        "p999_kld": float(np.quantile(flat, 0.999)),
        "max_kld": float(flat.max()),
        "mean_jsd_bits": float(np.mean([row["mean_jsd_bits"] for row in per_context])),
        "top1_agreement": float(np.mean([row["top1_agreement"] for row in per_context])),
        "context_bootstrap": cluster_bootstrap_ci(
            per_context, samples=args.bootstrap_samples),
        "engine_noise_floor_nats": PUBLISHED_ENGINE_NOISE_FLOOR_NATS,
        "strata": {
            stratum: {
                "contexts": len(group),
                "mean_kld": float(np.mean([row["mean_kld"] for row in group])),
            }
            for stratum, group in _by(per_context, "stratum").items()
        },
        "partitions": {
            partition: {
                "contexts": len(group),
                "mean_kld": float(np.mean([row["mean_kld"] for row in group])),
            }
            for partition, group in _by(per_context, "partition").items()
        },
        "worst_contexts": sorted(per_context, key=lambda row: -row["mean_kld"])[:10],
        "per_context": per_context,
    }
    if per_context_offsets:
        report["offset_audit"] = {
            "description": "candidate rows shifted by one position; agreement must collapse",
            "contexts": per_context_offsets,
            "mean_top1": {
                offset: float(np.mean([entry[offset]["top1_agreement"] for entry in per_context_offsets.values()]))
                for offset in ("-1", "+1")
            },
            "mean_kld": {
                offset: float(np.mean([entry[offset]["mean_kld"] for entry in per_context_offsets.values()]))
                for offset in ("-1", "+1")
            },
        }

    print(f"\ntoken_mean_kld   = {token_mean:.9f} nats")
    print(f"context macro    = {report['context_macro_mean_kld']:.9f}")
    print(f"bootstrap 95%    = [{report['context_bootstrap']['ci95_low']:.6f}, "
          f"{report['context_bootstrap']['ci95_high']:.6f}] "
          f"({report['context_bootstrap']['clusters']} clusters)")
    print(f"mean_jsd_bits    = {report['mean_jsd_bits']:.9f}")
    print(f"top1_agreement   = {report['top1_agreement']:.6f}")
    print(f"positions        = {flat.size}")

    failures: list[str] = []
    if args.expect_mean is not None:
        # An exactly-zero anchor (reference lane against itself) has no relative
        # scale, so the check falls back to absolute error there.
        if args.expect_mean == 0.0:
            drift = abs(token_mean)
        else:
            drift = abs(token_mean - args.expect_mean) / abs(args.expect_mean)
        verdict = "PASS" if drift <= args.expect_relative else "FAIL"
        print(f"anchor {args.expect_mean:.9f} -> drift {drift:.2e} (tol {args.expect_relative:.0e}) {verdict}")
        if verdict == "FAIL":
            failures.append(f"mean KLD off by {drift:.2e} > {args.expect_relative:.0e}")
        report["anchor_check"] = {
            "expected_mean": args.expect_mean,
            "relative_tolerance": args.expect_relative,
            "drift": drift,
            "passed": verdict == "PASS",
        }
    if args.compare_report:
        report["per_context_comparison"] = compare_per_context(
            args.compare_report, per_context, failures)

    report = seal(report, "report_sha256")
    if args.out:
        digest = write_json(args.out, report)
        print(f"wrote {args.out} report_sha256={digest}")
    if failures:
        print("\nGATE FAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


def _by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def compare_per_context(path: str, ours: list[dict], failures: list[str]) -> dict:
    """Index-by-index agreement with the published per-context numbers.

    This is the gate that works at partial scope. The published *global* mean is
    computed over all 5,120 contexts, so comparing our mean over a 16-context
    smoke run against it only measures sampling noise; comparing the same context
    indices one by one measures whether our replay *is* the suite's replay.
    """
    published = json.loads(Path(path).read_text())
    theirs = {int(row["index"]): row for row in published.get("per_context", [])}
    deltas = []
    overlapping_theirs = []
    for row in ours:
        other = theirs.get(row["index"])
        if other is None:
            continue
        deltas.append((abs(row["mean_kld"] - other["mean_kld"]) / max(other["mean_kld"], 1e-12), row["index"]))
        overlapping_theirs.append(float(other["mean_kld"]))
    if not deltas:
        failures.append(f"{path} has no per_context rows overlapping this lane")
        return {"compared": 0}
    deltas.sort(reverse=True)
    worst = deltas[:5]
    relative = float(np.median([delta for delta, _ in deltas]))
    our_scope_mean = float(np.mean([row["mean_kld"] for row in ours]))
    their_scope_mean = float(np.mean(overlapping_theirs))
    scope_drift = abs(our_scope_mean - their_scope_mean) / their_scope_mean
    print(f"per-context vs published: compared={len(deltas)} median relative delta {relative:.3e}")
    print(f"  scope mean ours {our_scope_mean:.9f} vs published {their_scope_mean:.9f} "
          f"(drift {scope_drift:.2e})")
    print("  worst: " + ", ".join(f"#{index}:{delta:.1e}" for delta, index in worst))
    if relative > 1e-3:
        failures.append(
            f"per-context numbers disagree with the published report (median relative delta {relative:.2e}); "
            "the replay or the row alignment is not the suite's")
    if scope_drift > 1e-3:
        failures.append(
            f"scope mean drifts {scope_drift:.2e} from the published values for these same indices")
    return {
        "reference_report": str(path),
        "reference_report_sha256": sha256_file(path),
        "compared": len(deltas),
        "median_relative_delta": relative,
        "our_scope_mean_kld": our_scope_mean,
        "published_scope_mean_kld": their_scope_mean,
        "scope_relative_drift": scope_drift,
        "worst": [{"index": index, "relative_delta": delta} for delta, index in worst],
    }


if __name__ == "__main__":
    raise SystemExit(main())
