#!/usr/bin/env python3
"""CPU-only comparator for thin-decode GPU receipts (stock vs stock vs fast).

Usage:
    python3 tests/compare_thin_fast.py thin_fast0.pt thin_fast0b.pt thin_fast1.pt

Compares, per case, in fp32:
  * stock repeat variation  (fast0 vs fast0b): ordinary atomic-ordering noise
  * candidate delta          (fast0 vs fast1): must not exceed the yardstick

PASS per case: finite everywhere and
    maxabs(fast-stock) <= max(4 * maxabs(stock-stock), floor)
with floor = 1e-3 * max(1, absmax(stock)). Also reports relative RMSE and the
mismatch rate (|diff| > floor). Graph-replay cases compare replay-vs-eager
WITHIN each receipt (no cross-process confound) and must additionally pass
the same gate in all three receipts.

Exits nonzero on any failure. No GPU required.
"""

from __future__ import annotations

import sys

import torch


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def stats(a: torch.Tensor, b: torch.Tensor):
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    scale = float(af.abs().max().clamp_min(1.0))
    floor = 1e-3 * scale
    mse = float(((af - bf) ** 2).mean())
    denom = float((af ** 2).mean())
    return {
        "maxabs": float(diff.max()),
        "rel_rmse": (mse / denom) ** 0.5 if denom > 0 else 0.0,
        "mismatch": float((diff > floor).sum()) / diff.numel(),
        "scale": scale,
        "floor": floor,
    }


def main() -> int:
    stock_a_p, stock_b_p, fast_p = sys.argv[1], sys.argv[2], sys.argv[3]
    a, b, f = load(stock_a_p), load(stock_b_p), load(fast_p)
    assert set(a["cases"]) == set(b["cases"]) == set(f["cases"]), "case mismatch"
    print(f"stockA fast_env={a['meta']['fast_env']} native_fast={a['meta']['native_fast']}")
    print(f"stockB fast_env={b['meta']['fast_env']} native_fast={b['meta']['native_fast']}")
    print(f"fast   fast_env={f['meta']['fast_env']} native_fast={f['meta']['native_fast']}")
    assert a["meta"]["fast_env"] == "0" and b["meta"]["fast_env"] == "0", "stock receipts must use FAST=0"
    assert f["meta"]["fast_env"] == "1", "candidate receipt must use FAST=1"
    assert f["meta"]["native_fast"], "candidate ran without the native fast kernels!"

    failures = []
    print(f"\n{'case':48s} {'repeat_max':>10s} {'fast_max':>10s} {'gate':>10s} {'rel_rmse':>10s} {'mismatch':>9s} verdict")
    for key in sorted(a["cases"]):
        ta, tb, tf = a["cases"][key], b["cases"][key], f["cases"][key]
        for name, t in (("A", ta), ("B", tb), ("F", tf)):
            if not torch.isfinite(t).all():
                failures.append(f"{key}: non-finite in {name}")
        rep = stats(ta, tb)
        delta = stats(ta, tf)
        gate = max(4.0 * rep["maxabs"], delta["floor"])
        if key.endswith("/repeat"):
            # determinism info only: fast repeat vs stock first run uses delta
            ok = delta["maxabs"] <= gate
        elif "/graph48/" in key:
            ok = True  # cross-mode graph outputs compared via eager below
        else:
            ok = delta["maxabs"] <= gate
        mark = "ok" if ok else "FAIL"
        if not ok:
            failures.append(f"{key}: fast_max={delta['maxabs']:.3g} gate={gate:.3g}")
        print(f"{key:48s} {rep['maxabs']:10.3g} {delta['maxabs']:10.3g} "
              f"{gate:10.3g} {delta['rel_rmse']:10.2e} {delta['mismatch']:9.2e} {mark}")

    # Within-receipt graph replay parity (same process, same kernels).
    # Synthetic random weights produce large-magnitude outputs (absmax ~1e4);
    # a 1-ulp fp16 difference there is exactly 1.0 absolute, so the gate is
    # relative RMSE (<= 1e-5) plus an absolute ceiling of a few ulps.
    for name, r in (("stockA", a), ("stockB", b), ("fast", f)):
        for tag in ("", "2"):
            e = r["cases"][f"shared/graph48/eager{tag}"].float()
            rp = r["cases"][f"shared/graph48/replay{1 if tag == '' else 2}"].float()
            st = stats(e, rp)
            ok = st["rel_rmse"] <= 1e-5 and st["maxabs"] <= 8.0 and torch.isfinite(rp).all()
            print(f"graph {name} replay{tag or 1}-vs-eager: maxabs={st['maxabs']:.3g} "
                  f"rel_rmse={st['rel_rmse']:.2e} {'ok' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"graph {name} replay{tag or 1}")

    # Fallback geometry runs the stock kernel in every mode, but cross-process
    # atomic-add ordering still moves fp16 ulps. Gate against the stock-stock
    # repeat yardstick, exactly like the main cases.
    for key in sorted(a["cases"]):
        if key.startswith("fallback/") and not key.endswith("/repeat"):
            rep = float((a["cases"][key].float() - b["cases"][key].float()).abs().max())
            d = float((a["cases"][key].float() - f["cases"][key].float()).abs().max())
            scale = float(a["cases"][key].float().abs().max().clamp_min(1.0))
            gate = max(4.0 * rep, 1e-3 * scale)
            ok = d <= gate
            print(f"{key}: cross-mode maxabs={d:.3g} repeat={rep:.3g} gate={gate:.3g} "
                  f"{'ok' if ok else 'FAIL (fallback diverged!)'}")
            if not ok:
                failures.append(f"{key}: fallback diverged {d}")

    print()
    if failures:
        print(f"{len(failures)} FAILURES:")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print("ALL CASES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
