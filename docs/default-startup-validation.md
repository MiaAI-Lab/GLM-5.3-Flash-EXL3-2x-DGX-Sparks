# Default-startup memory-budget qualification

**Default startup qualified on the tested 2x GB10 cluster:** four fresh TP2
boots passed with the updated 5120-token defaults, graph estimation enabled,
and no inference overrides. Both baseline and candidate passed with fresh and
warm compilation caches. This resolves the startup-budget checklist item for
the revised defaults, not the separate semantic-canary caveat.

## Diagnosis

The original 850000-context / 0.85-utilization / 7168-token prefill defaults
failed KV-budget initialization with `CG_ESTIMATE=1` on this cluster. The first
fatal error was insufficient planned KV capacity, not the subsequent NCCL
broken-pipe shutdown warnings. Both upstream and the FP32 E3 candidate failed.

Instrumented dry capture measured ~1.41 GiB device-memory growth, comprising
~170 MiB in the private PyTorch graph pool, ~576 MiB ordinary PyTorch pool growth,
and memory outside that allocator. A repeated profile lowered estimated graph
cost by about 1.1 GiB but raised retained non-KV memory by about the same amount.
This did not establish double-counting or justify discounting the reservation.
Diagnostic images did not replace the budget calculation.

## Default correction

Set the launcher fallback and `.env.example` prefill batch to **5120**, preserving:

- `MAX_MODEL_LEN=850000`
- `GPU_MEM_UTIL=0.85`
- `CG_ESTIMATE=1` and CUDA graphs
- E3/cap32, maxseq4, DFlash2 k7, vision/image100

This is an explicit default change that lowers batch-related activation,
scratch, and KV reservations. It is not a claim that the old defaults passed,
an automatic fallback, a lowered advertised context length, or a disabled guard.
It applies to the general launcher batch default; other tiers/concurrency levels
need their own throughput qualification. Existing `.env` files with an explicit
7168 setting are not silently rewritten.

## Sizing controls (one boot each)

| Prefill batch | Compilation cache | Startup | Notes |
|---|---|---|---|
| 7168 | warm and cold controls | failed | Baseline and candidate KV-budget failures |
| 4096 | existing isolated cache | passed | 14.36 GiB available KV; 4.38 GiB minimum sampled head MemAvailable |
| 5120 | fresh isolated cache | passed | 13.09 GiB available KV after 2.17 GiB graph estimate; 4.36 GiB minimum sampled head MemAvailable |
| 6144 | fresh isolated cache | failed | 12.40 GiB available versus 12.65 GiB required KV |

Both successful controls passed all 20 boot-warmup requests, vision, and the
25-fixture prefill replay. Each retained the same three original-code arithmetic
canary failures seen upstream; their runner exit2 reflects this semantic caveat.

Relative to the earlier **7168/CG0 candidate** measurements, the 5120/CG1 control
was 0.73% lower on the size-ladder geometric mean; docs-short was 10.99% lower,
code-medium 3.39% lower, and mixed-long 3.30% lower. These compare configurations,
not just the kernel, and are not a paired confidence estimate. The 4096 control
was 5.29% lower on the size ladder and 13.95% lower on mixed-long. Lower batches
can add prefill chunks for prompts just over the chunk boundary.

## Completed cold/warm qualification

The actual updated `start.sh` and `.env.example` were used without inference
overrides: baseline/candidate A/B/A/B, first boot per variant with a fresh isolated
compilation cache and second with that cache warm. Only site-specific NFS/fabric,
container names, cache paths, and pinned-image selection were customized.

| Arm | Cache | Startup | Available KV GiB | Prefill size-geomean tok/s | Canaries |
|---|---|---|---:|---:|---:|
| A1 baseline | fresh | pass | 13.05 | 1476.10 | 22/25 |
| B1 candidate | fresh | pass | 12.76 | 1586.14 | 22/25 |
| A2 baseline | warm | pass | 13.70 | 1537.43 | 23/25 |
| B2 candidate | warm | pass | 13.61 | 1578.95 | 23/25 |

The warm-pair size-ladder throughput gain was **2.70%**; the cold pair was 7.46%.
The combined 5.05% is descriptive, not a stronger kernel claim: cold-baseline
64K/holdout timing was substantially slower than its warm repeat. The cold-pair
docs-short holdout regressed 0.86%; warm-pair holdouts improved 1.02–2.03%.
No confidence interval is claimed from two boots per variant.

All 100 fixture requests completed with identical usage across arms and zero
prefix-hit deltas. Both ranks had zero restarts and no OOM kills. Minimum sampled
system MemAvailable was **3.31 GiB head / 6.78 GiB worker**, above the 1.5 GiB guard.
Vision passed in every arm. Failures were confined to the existing original-code
arithmetic question (three failures per cold arm, two per warm arm); expected
answers were unchanged and the runner correctly exited 2. Startup is qualified;
fully-green semantic behavior is not.

Test ranks stopped, worker staging assets were restored, and observability was
left running. [Raw rows and default/source receipts](default-startup-results.json)
preserve both the passing startup evidence and failed semantic checks.

A successful startup is not a populated-850K-context, arbitrary-concurrency,
100-image workload, or universal host-memory certification. Vision maximum-size
profiling remains skipped by the existing launcher; this limitation must not be
hidden by the new startup setting.

The [earlier E3 activation measurements](e3-activation-validation.md) used
7168/CG0 and remain historical evidence for that configuration, not measurements
of the revised default.
