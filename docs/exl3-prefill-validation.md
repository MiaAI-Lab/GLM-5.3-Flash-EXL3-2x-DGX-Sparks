# GB10 prefill validation

## Scope and status

This change retains both K-loop barriers. It does not change weights,
quantization, KV precision, context limits, allocator defaults, or sampling.
The direct/scatter panel tests the M64 launch-bound and async-staging changes;
it is not a full-model benchmark or a measurement of every change in the PR.

Tests ran on two NVIDIA GB10 GPUs, SM121 and driver 580.173.02, on 2026-09-05.
The reference is upstream `3021f24c88a0904c768c46ff22a508407e31360a`;
the candidate production source is `fd44e0c0fffebf51769d1f84314c507686d12736`.
Later documentation and test additions do not change those production files.
The image used PyTorch 2.13.0+cu130, vLLM `0.1.dev20051+g487ecf187` and
ExLlamaV3 package version 0.0.43. Image IDs and production-file hashes are in
[the GPU receipt](benchmarks/gb10-prefill/gpu-summary.json).

- Full image GPU self-checks passed on both GPUs.
- The extended stream/graph suite passed all three sanitizers on both GPUs,
  with the explicit barrier-tracking capacity described below.
- The CPU suite passed 108 checks and omitted ten integration checks, listed in
  [cpu-summary.json](benchmarks/gb10-prefill/cpu-summary.json). Skips are not passes.
- Stock workspace/spinwait startup failed the same KV-capacity check in both
  images. This is an existing configuration problem on this kit, not a pass.
- Candidate startup and one full live panel passed with the existing kit opt-ins,
  automatic KV profiling, and **no allocator cap**.
- The fixed-capacity upstream control hit the memory guard during its first cold
  8k warmup. No measured baseline samples completed. Further model runs were
  stopped; the PR remains a draft without an end-to-end speedup claim.

The follow-up [memory-budget audit](gb10-memory-budget-audit.md) uses retained
records only. It confirms that fat scratch was already initialized before the
failed request, separates the allocation budgets, and documents the missing
telemetry. It does not establish a safe replacement control configuration.

## Fresh isolated results

Each device ran the panel in reference/candidate/candidate/reference/reference/
candidate order. Values below include every panel; the summary uses their median.

| Device | Reference panel times, ms | Candidate panel times, ms | Median reduction |
|---|---|---|---:|
| Head | 9.453312, 9.594144, 9.243808 | 7.410272, 7.157696, 7.115552 | 24.3% |
| Worker | 9.205216, 9.243744, 9.102656 | 7.019040, 7.087360, 6.927968 | 23.7% |

The real indexer-loop benchmark reproduced the same allocation result on both
GPUs: the largest case fell from **1045.256348 to 538.246582 MiB**, saving
**507.009766 MiB of peak temporary allocation**. The sum across three separate
cases fell from 1620.467773 to 841.833008 MiB; those peaks are not simultaneous.
Loop times were effectively unchanged: reference/candidate 82.59/82.76 ms on the
head and 81.90/81.52 ms on the worker.

The reference loop differs only by retaining `logits`. The benchmark verifies
causal validity, selection cardinality and output equality. Differing top-k
selections are accepted only after proving exactly tied real-MQA boundary scores.
Diagnostics and copied baseline outputs are outside measured allocation peaks.

These results do not establish a general end-to-end speedup, lower host RSS,
or a larger automatically profiled KV pool. Do not add the kernel and memory
percentages. Earlier allocator-capped experiments, unstable long-context controls
and rejected decode regressions are not substituted for fresh PR validation.

## Reproduce image and kernel checks

Build each checkout separately using its Dockerfile and the same model-independent
build arguments. Do not copy model weights into either image. Use distinct tags,
then run the following on each GPU with the model service stopped:

```sh
docker build -t glm53-exl3-validation:candidate .
docker run --rm --gpus all --entrypoint python3 \
  glm53-exl3-validation:candidate /opt/glm53/test_exl3_overlay.py
```

The public wrapper imports the installed image self-check, runs its four fat
checks, then checks direct/pair/scatter on a nondefault stream and graph replay
at `(M,K,N)=(1,16,128),(65,80,256),(173,4096,1024)`. Inputs change between replays;
graph results must match eager results exactly. Single/odd K tiles, M tails,
production dimensions, activation edge cases and rejected API inputs are covered.

```sh
docker run --rm --gpus all --entrypoint python3 \
  -v "$PWD/tests:/validation:ro" glm53-exl3-validation:candidate \
  /validation/check_exl3_fat_cuda.py
```

Compute Sanitizer 2025.3.1.0 was already installed on the hosts. It was mounted
read-only; nothing was installed on the hosts. Adjust the host tool path if needed:

```bash
for tool in memcheck racecheck synccheck; do
  extra=()
  if [ "$tool" = synccheck ]; then extra=(--num-cuda-barriers 64); fi
  docker run --rm --gpus all \
    -v /usr/local/cuda-13.0/compute-sanitizer:/opt/cuda-sanitizer:ro \
    -v "$PWD/tests:/validation:ro" \
    --entrypoint /opt/cuda-sanitizer/compute-sanitizer \
    glm53-exl3-validation:candidate \
    --tool "$tool" "${extra[@]}" --error-exitcode 86 \
    python3 /validation/check_exl3_fat_cuda.py
done
```

Initial automatic-capacity synccheck runs failed on both GPUs: the tool warned of
tracked `cuda::barrier` overflow, followed by a launch failure in upstream
Hadamard. Those failures remain in the receipts. The unchanged original suite,
then the extended public suite, passed with `--num-cuda-barriers 64`. Final
racecheck reported zero errors **and zero warnings** on both GPUs.

For the fixed historical panels, run the same public script in both images:

```sh
docker run --rm --gpus all --entrypoint python3 \
  -v "$PWD/tests:/validation:ro" glm53-exl3-validation:reference \
  /validation/bench_exl3_prefill_kernels.py --validation upstream \
  --projection-arm copies
docker run --rm --gpus all --entrypoint python3 \
  -v "$PWD/tests:/validation:ro" glm53-exl3-validation:candidate \
  /validation/bench_exl3_prefill_kernels.py --validation candidate \
  --projection-arm both
docker run --rm --gpus all --entrypoint python3 \
  -v "$PWD/tests:/validation:ro" glm53-exl3-validation:candidate \
  /validation/bench_indexer_logits_memory.py --require-patch
```

The panel uses M=129/256/512/1024/2048/4096/7168, direct 4096×2048 and scatter
1024×4096, three warmups and median seven CUDA-event timings per case. Pair/copy
projection measurements are reported separately. Do not replace these shapes or
omit a slow case when comparing implementations. `--help` does not require Torch.

CPU checks require CPU PyTorch, NumPy and Jinja2 in an isolated environment:

```sh
python3 tests/run_cpu_validation.py --out /tmp/glm53-cpu-summary.json
```

The runner does not install dependencies, use a GPU, or contact a server. It
explicitly omits installed-vLLM and live integration checks. An initial missing
NumPy failure was resolved in a temporary dependency directory before the passed
rerun; it was not a production code failure.

## Live protocol and configuration

The tested kit uses TP2, DFlash k7, MNBT 7168, max sequences 4, 1M context,
FP8 MLA KV, GPU utilization .87, and the same model snapshots in both arms.
`GLM53_INDEXER_WORKSPACE=rightsize` and `GLM53_SPINWAIT_MS=16` are existing kit
opt-ins. The shipped values are `stock`; the two configurations are not equivalent.
Neither `MALLOC_ARENA_MAX` nor `MALLOC_TRIM_THRESHOLD_` is set in these tests.

With stock workspace and spinwait, candidate/reference automatic profiling
reported **10.47/10.40 GiB** available KV memory, below the **14.52 GiB** required
for 1M context. Both startups failed before serving. Context and utilization were
not lowered to obtain a pass.

With the existing kit opt-ins, the uncapped candidate started with **15.67 GiB**
automatic KV allocation and **1,057,491 KV tokens**. One full panel passed:

| Measurement | Result |
|---|---:|
| Cold 8k/64k geometric score | 1271.21 tok/s |
| Cold 100k/300k geometric score | 1403.64 tok/s |
| Structured decode, median five | 63.64 tok/s |
| Prose decode, median five | 26.46 tok/s |
| LRU-code guard, median five | 51.43 tok/s |

Every cold request had zero cached tokens; APC reused 7,168 tokens. All fifteen
measured decode requests produced 400 tokens. This is a deployment/workload check,
not a throughput comparison against upstream. One pass does not establish long-run
stability. [The live receipt](benchmarks/gb10-prefill/live-summary.json) includes
TTFT and speculative acceptance, including each draft position. During this panel,
the sampled available-memory minima were 1628.13 MiB on the head and 4516.57 MiB
on the worker. Worker full-memory PSI avg10 reached 15.26%; the head maximum was
zero. The receipt includes both memory records, not only the successful request data.

The attempted A/B/A control specified **17,448,304,640 KV bytes per rank**,
16.25 GiB. The upstream startup reported **1,118,466 KV tokens**. This is above
the automatically profiled pool, not a capacity cut. Explicit bytes bypass
profiling and were intended to hold capacity identical in both arms. They are
an experimental control, not a new launcher default.

The head crossed the memory floor during the first cold 8k warmup. Available
memory fell from 978.71 to **112.04 MiB** between one-second samples. The resident
guard stopped the head model; the controller stopped the worker. Both hosts
recovered. The attempted request and failure remain in the receipt. No measured
baseline samples completed, and no fixed-capacity candidate or return control
ran. A one-second guard did not prevent the abrupt drop into very low memory.

Do not retry this configuration unchanged. Completing the controlled throughput
comparison requires a separately reviewed safe configuration. We did not lower
KV capacity or introduce an arena cap to turn this failure into a pass.

Before loading either model, run a resident guard on each host in a separate
terminal. It samples once per second and stops only the named model container
below 768 MiB available or 512 MiB free. Keep its output and stop the peer rank if
it exits nonzero. This guard cannot guarantee host survival under every failure.

```sh
python3 scripts/watch_model_memory.py --container glm53-exl3-head > head-memory.jsonl
# On the worker, use --container glm53-exl3-worker instead.
```

On an otherwise idle endpoint, run:

```sh
python3 tests/bench_exl3_pr_live.py --base-url http://HEAD:8888 \
  --runs 3 --provenance /path/to/server-provenance.json --out /tmp/live.json
```

Each fresh deployment receives an unmeasured cold 8k warmup. Every repetition
contains prose/code at 8k/64k, the unchanged cold 100k/300k holdouts, and a cold/APC
pair requiring at least 80% reuse. Decode follows the README structured/prose
five-by-400 protocol, plus a separately labelled LRU-code guard. Requests are
sequential; this does not validate concurrent or mixed prefill/decode service.

The harness saves every attempted request, full outputs, raw SSE, metrics and
failures before aggregation. It rejects invalid samples rather than filtering
rates. Measured decode requires valid speculative counters; short unmeasured
probes may legitimately produce no draft step. Prose/code can vary at temperature
zero, so compare acceptance and output lengths as well as timing. A material decode
regression or safety abort prevents promotion even if prefill is faster.

The first automatic-KV panel predates a capture-only UTF-8 boundary fix. That fix
preserves multibyte characters split across reads; it changes neither requests nor
the timing helpers. Subsequent comparisons use the corrected harness.

## Receipts and remaining gates

[Raw receipts](benchmarks/gb10-prefill/) retain all isolated panel samples,
initial sanitizer failures, final passes and CPU skips. Startup logs and full
live JSON are gzip-compressed; inspect them with `gzip -dc FILE.gz`.
Large profiler traces and old exploratory runs are not bundled into this PR.

Before calling the combined change production-ready:

- Establish a safe unchanged-capacity comparison configuration before any further
  full-model controls. Retain the aborted upstream attempt rather than replacing it.
- Compare median-five decode and per-position acceptance against valid controls.
  The passed automatic-KV panel alone does not prove absence of a decode regression.
- Leave the stock-default checklist item open: both images failed that configuration
  on this kit. A successful opt-in configuration is not a stock-default pass.
- Do not claim concurrency coverage or normal current-upstream vLLM pytest from
  local import-stub tests. Those are separate validation tasks.
