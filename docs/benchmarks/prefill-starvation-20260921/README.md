# Fair-prefill allocation starvation and TP2 tuning — 2026-09-21

## Scope and root cause

Root cause: a KV-blocked WAITING request can consume the fair-prefill selection window before RUNNING traversal. Allocation fails only after an admitted runnable prefill has already been passed. Resetting selection repeats the same ordering and can starve that resident while a decoder remains active.

Evidence: three regression tests fail against upstream base 775a58b704924b13cf5c38de97559b3630f67bbf and pass with this overlay. They cover chunk windows of one and eight, more blocked waiters than the window, and a blocked waiter preceding an allocatable waiter.

The fix carries a passed runnable candidate into the next bounded selection after later allocation failure. An unselected resident's zero-progress callback is not treated as a failed admission; a genuinely failed selected admission advances round-robin rank. It preserves decode priority, the allocator, credit/debt accounting and chunk/time limits. Carry state is removed with departed requests and contention epochs.

No launcher defaults change. In particular, fair share remains 0.30 upstream. The separate local 0.50 tuning result below is evidence, not a proposed universal default.

## Draft integration notes

The deployed overlay uses the v6 marker. Open #80 and #180 independently use v6 for another helper. These implementations are NOT interchangeable: this installer rejects mismatched v6 helper bytes. Maintainers must reconcile versioning and migration before merging. #221 changes priority ranking in the same file and also needs integration review.

This delta authenticates migration of the exact current v5 helper. Existing v1–v4 fixture tests are synthetic structural checks, not proof of authenticated historical producers. Their inherited migration behavior does not replace the stricter migration work in #80/#180. This draft is not a request to bypass those reviews.

## Runtime and fixed controls

Two DGX Sparks / GB10, one GLM-5.3-Flash-EXL3 TP2 deployment. Runtime recipe base ca8557665bffa6529758f2c330ba8fb44c1e801a; publication base is newer and received CPU/source checks, not a new full image build.

- Image digest: sha256:147531595e8c2e26e4c79e8735b44e875f58d0c244c196c312820d356cc8811a
- Overlay SHA-256: 0870fab903a4e6a7323037834e5c4843d1abe1503b1b7b2dcc5f85de54a524df
- Context 262144; batch tokens 7168; long-prefill threshold 3584; GPU utilization 0.85.
- Fair policy; probe 256; max interval/step 2000 ms; max chunks 1.
- Adaptive DFlash EMA, k=2/4/7; graph sizes 1,2,3,4,5,6,8,9,10,12,15,16,18,20,21,24,25,30,32,35,40,48,56,64.
- FAST MoE off; dense/KDA FP8 off; KDA large-M not enabled; ABLIT=0.
- Synthetic measurements: temperature 0, top_p 1, thinking disabled. Production High reasoning default unchanged.

Both ranks were checked for exact environment differences and identical image/overlay. Measurement arms bound the API to loopback, excluding normal traffic. Counter deltas accounted for all 66 measured requests.

## Share 0.30 versus 0.50

Three repetitions each. A distinct cold approximately 8.25K-token coding prompt produces 1024 tokens. After its first content arrives, a separate cold approximately 16.7K-token marker-retrieval prompt starts. All runs verified overlap and retrieval correctness. Between arms, prompt lengths differ by at most 4 tokens for A and 24 for B.

| Metric | Share 0.30 | Share 0.50 |
|---|---:|---:|
| Newcomer TTFT median | 37.616 s | 23.708 s |
| Incumbent whole-request decode median | 21.053 tok/s | 22.584 tok/s |
| Largest measured content-event gap | 1.952 s | 1.663 s |

TTFT fell 36.97%; whole-request decode rose 7.27%. **Both arms use the repaired scheduler: this is a share comparison, not an isolated speedup attributable to the bug fix.** The streaming rate uses completion count and first/last content timing; SSE events can contain multiple tokens.

## Four versus eight active sequences

At share 0.50, each arm received identical three-mixed-run preparation followed by two eight-request bursts. Each request: distinct cold approximately 8.28K-token prompt, 256 output tokens. Matched prompt differences are at most five tokens. Values are medians of two burst summaries.

| Metric | Four slots | Eight slots |
|---|---:|---:|
| Median TTFT | 65.672 s | 50.018 s |
| Burst p95 (effectively slowest of eight) | 85.512 s | 79.157 s |
| Burst completion time | 99.676 s | 91.126 s |
| Aggregate throughput including prefill | 20.552 tok/s | 22.506 tok/s |

Four slots increased median wait 31.3%. Eight were retained for this workload. These two small bursts do not establish a stable production p95 or prove that eight is best for every context/load mix.

## Correctness, thermals and limitations

All measured windows had zero additional errors, aborts or preemptions. Each arm and restored production passed five bounded tool/JSON/code/image cases. These are smoke tests, not a comprehensive reasoning-quality evaluation. Recorded thermal samples reached 74 C, with positive hardware temperature headroom and no thermal-throttling flag; sampling was not continuous.

Production was restored with eight slots and share 0.50. Both ranks, public health HTTP 200, five quality cases and restored liveness monitoring were verified. The public branch overlay is byte-identical to that deployed overlay; the test harness additionally accepts already-installed v6 source for the image build's test phase.

## Evidence and reproduction

[evidence.zip](evidence.zip) contains 50 sanitized files: full synthetic request payloads, generated responses/content-event timestamps, token usage, raw before/after metrics, summaries, quality traces and sanitized identity/thermal receipts. SHA256SUMS.json inside hashes every payload. No credentials, user workload, account directories, private node addresses, environment snapshots or restart controller are included.

The accompanying measurement runners use Python's standard library and localhost:8888. They submit load but never restart services. Run only inside an operator-controlled isolated comparison window:
1. Configure the chosen arm with the supported launcher and verify idle state and exact settings.
2. Run `python quality_gate.py --out <durable-output>/quality.json`.
3. Run `python share_bench.py --arm share30 --out <durable-output>/share30 --reps 3` (use share50 for the other arm).
4. Run `python lane_bench.py --out <durable-output>/share30-burst`; repeat for share50-burst. The special lane4-burst name automatically runs its matching three mixed preparation repetitions.
5. Compare matched input/output lengths, raw traces and before/after counters before comparing summaries. Do not mix live-user traffic into these measurements.

These runners preserve the original measurement logic with portable sibling imports and a local counter helper. They do not provide a deployment/rollback orchestrator. New UUID prefixes vary token counts slightly; enforce the matching checks above.

CPU validation: set GLM53_SCHEDULER_PY_SRC to the pinned installed v5 source or this candidate's installed v6 source, then run tests/test_scheduler_decode_floor.py. Run tests/test_scheduler_decode_floor_restart.py for restart/adaptive-overlay ordering. Both source-input paths passed 33/33; restart tests and Python compilation passed. See [validation.json](validation.json). No full Docker rebuild or new GPU trial was performed solely for publication.

Built on MiaAI-Lab's serving recipe and existing fair-prefill policy, vLLM, EXL3/TR3 weights credited by the upstream README, and DFlash2. This contribution is the bounded starvation correction, regression coverage and the reported operator experiments.
