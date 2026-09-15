# Atlas EXL3 SM121 Kernel Lab

This lab measures and tunes the existing ExLlamaV3 packed EXL3 CUDA path. It
does not define a second weight format, materialize a full weight during direct
timing, or put CUDA implementation code in the Atlas profiler.

The workload interface is model-agnostic. Model IDs and operator names are
receipt provenance; arbitrary future models provide their actual `M/N/K`
shapes without being added to a registry.

This lab is development-only source in this repository. The production
Dockerfile neither copies nor installs `kernel_lab/`, so the lab is absent
from the served image and no production launch path imports it. Nothing here
is a speedup claim for the production pin: the SM121 tactic sweep has not yet
run on target hardware.

## Portable correctness

The NumPy oracle supports the upstream MCG trellis at every integer bitrate
K1-K8. It compares a materialized reference with a tile-streaming matmul that
never constructs the complete decoded weight.

```bash
python3 -m kernel_lab.exl3.benchmark \
  --backend cpu-oracle --bits 1 2 3 4 5 6 7 8
python3 -m pytest tests/test_exl3_kernel_lab.py -q
```

## Arbitrary workload input

Pass shapes directly as `OPERATOR:K:N`:

```bash
python3 -m kernel_lab.exl3.benchmark \
  --backend sm121 \
  --model-id publisher/new-model \
  --shape experts.gate:7168:2304 \
  --shape experts.down:2304:7168 \
  --tp 4 --bits 2 3 4 --m 1 2 4 8
```

Or pass a JSON file exported from a profiler/model census:

```json
{
  "workloads": [
    {
      "model_id": "publisher/new-model",
      "operator": "layers.0.mlp.experts.gate",
      "phase": "decode",
      "k": 7168,
      "n": 2304,
      "tp_world_size": 4,
      "grouped_moe": false
    }
  ]
}
```

`m` belongs on the command line because one operator is normally swept over
several routed-row counts. Workload-file objects therefore contain `K/N` and
provenance, while `--m` controls the tuning sweep.

## SM121 tactic sweep

For every exact shape, the runner measures:

- ExLlamaV3's automatic dispatch;
- every compatible built-in GEMM tile shape;
- quarter, half, and full-device SM quotas;
- dense FP16, reconstruction-only, and reconstruct-then-GEMM diagnostics;
- correctness, direct latency, effective bandwidth/TFLOP/s, reconstruction
  overhead, and a bottleneck classification.

Direct timings call `exl3_gemm` on packed trellis tensors. Dense and
reconstruction baselines are outside the CUDA event range. The output is an
`atlas.kernel-catalog/v1` document containing exact
`atlas.kernel-benchmark/v1` receipts:

```bash
python3 -m kernel_lab.exl3.benchmark \
  --backend sm121 \
  --shape linear.square:4096:4096 \
  --bits 3 --m 1 2 4 8 \
  --warmup 10 --iterations 50 \
  --output /tmp/atlas-exl3-sm121.json
```

Use `--no-tactic-sweep` to time only automatic dispatch. The optional legacy
fixtures under `--fixture` reproduce early GLM/Hy4 measurements; they are not
the supported-model set.

## Upstream comparison

The production image remains pinned to ExLlamaV3 `c5d9c657` (0.0.43). Building
and self-checking an alternate immutable revision stays an isolated experiment
in a source checkout of this repository: the production Dockerfile carries no
benchmark build argument, commit stamp or copy of `kernel_lab/` for it, and
this branch does not silently replace the production pin.

Measured receipts take their runtime-source provenance from `--runtime-commit`
or from the Git identity of the clean checkout that produced them. The
`RUNTIME_SOURCE_COMMIT` environment override is honored only when the caller
set it to the exact source commit that is loaded; it is a declaration, not
attestation. If no full SHA can be resolved, the runner fails instead of
emitting a receipt with unknown provenance.

The current upstream candidate at
`0c49587a7c235e6303a6bbedc8b665272ad3a2ea` adds the QTIP-style small-M GEMV
and fused mul1 int8-activation GEMV. A developer-provided backend must set
`EXLLAMAV3_COMMIT` to its actual commit before the metadata module is imported;
the pinned fallback describes only the production pin and cannot prove an
arbitrary installed extension's identity. Unknown or mismatched backend/source
identity is not measurement evidence.

Grouped MoE, prefill, mixed EXL3/NVFP4 dispatch, vLLM/SGLang-wide integration,
attention, KV, scheduling, and networking remain outside this first measured
decode gate.
