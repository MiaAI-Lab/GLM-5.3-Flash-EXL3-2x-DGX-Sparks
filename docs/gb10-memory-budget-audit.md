# GB10 memory-budget audit for PR131

This audit reads the retained startup logs, memory samples and source. It does not
run the model. The failed fixed-KV control remains a failure. No safe replacement
configuration has been established, and full-model runs remain paused.

## Findings that change the next step

**Fat scratch was already allocated before the failed request.** The reference
startup logged maximum expert rows 7168, one 432,017,408-byte scratch allocation,
and prior direct/scatter calls at 13:24:07. The fixed-KV accounting message follows
at 13:24:08. It skipped automatic budget profiling, not the fat-kernel warmup.
The automatic candidate also initialized the same scratch before readiness.

**The abort occurred during an attempted generation stream, not client prompt
fitting.** The saved request has an estimated token count, pre-request metrics,
HTTP 200 and an `IncompleteRead` stream error. The final connection-refused
exception came from the subsequent `/metrics` request after shutdown. Empty SSE
and default zero token fields do not prove that no GPU work occurred. The exact
server-side allocation or execution phase is unknown.

**The reference had too little observed headroom to justify dispatch.** The last
head sample before panel start showed 982.49 MiB available. The next sample showed
978.71 MiB, followed by 112.04 MiB about 1.05 seconds later. That last decline is
866.67 MiB. The watchdog's available-memory floor triggered; its free-memory floor
did not. A 768 MiB stop threshold at one-second intervals was not a sufficient
admission policy for this run.

The successful candidate is not a matched safety control. Its automatic KV budget
was smaller, the image differed, and the panel followed server startup by roughly
23 minutes rather than 31 seconds. Both had already completed 20 boot requests.
Those intervals include the boot sweep and do not establish that the remaining
time was idle. The captured client versions also differ by the documented UTF-8
capture fix. None of these differences identifies the cause of the abort.

## Allocation ledger

All values below are per rank. They describe different accounting boundaries and
**must not be summed into a host-RAM total**. GB10 host and device allocations share
physical memory. Tensor sizes, allocator peaks, loader deltas and host RSS are not
independent pools.

| Item | Size or reported value | Interpretation |
|---|---:|---|
| Model-loading usage | 82.06 GiB in both logs | Rounded loader accounting, not a complete resident-memory inventory |
| Routed-expert packed tensors | 71.22 GiB at H4096/I1024/K4/E288/L42 | Conditional source-derived subset of model storage, not extra to the loader total |
| Fat scratch | 432,017,408 bytes, 412.00390625 MiB | Persistent shared cache, already initialized in both runs |
| Fused thin/decode temps | 15,728,640 bytes, 15 MiB | Already initialized; shared across matching layers |
| Stock indexer gather request, including radix scratch | 5,281,048,576 bytes, 5036.40 MiB | Requested persistent workspace geometry at 1M context |
| Rightsized gather request, including radix scratch | 133,049,632 bytes, 126.89 MiB | Existing opt-in at four sequences and kpool4, not a PR131 allocation change |
| Candidate automatic KV budget | 15.67 GiB; 1,057,491 tokens | Rounded automatic budget |
| Reference fixed KV budget | 17,448,304,640 bytes; 1,118,466 tokens | 16.25 GiB; explicit sizing ignores utilization-based admission |
| Graph capture usage | Candidate 0.46 GiB; reference 0.19 GiB | Reported deltas from different allocation paths, not an isolated regression measurement |

The fat cache is keyed by device and tensor geometry, not by layer or expert. Do
not multiply its 412 MiB by 42 layers or 288 experts. The same-geometry cache has
8192-row capacity after startup, enough for the configured 7168-token chunk.
Normal serving at that ceiling does not require first-warmup growth of this cache.

The candidate still allocates 188.00390625 MiB of buffers unused by its pair/fused
fast path. Lazy allocation remains deferred; PR131 does not deliver that saving.

The gather calculation uses `max_model_len * 40` entries in stock mode. Rightsize
uses the scheduler-bound compressed length. Here it requests 1,000,008 entries,
not the four-million-entry example for 16 sequences in the older design document.
Each FP8 entry and scale requires 132 bytes; both requests include 1 MiB radix
scratch. Their requested-size difference is about 4.79 GiB. A shared workspace
manager can retain the largest simultaneous request, so this calculation does not
inventory every engine arena or prove the same host-resident reduction.

The new logits-lifetime change is separate. Its measured 507.009766 MiB saving is
a temporary allocation-peak reduction in the isolated repeated-chunk benchmark.
That benchmark excludes the gather arena, weights, KV, graph pools and normal
serving allocations. It cannot be subtracted from the failed 8k request as a
promised host-memory safety margin.

Other prefill transients include FP32 MoE output, converted hidden states,
routing/sort buffers and indexer logits. At T7168/H4096, FP32 output alone is
112 MiB and an FP16 hidden conversion is 56 MiB. Their overlap with caller-owned
activations, allocator reservations and other layers was not traced in the failed
request. No static total here explains the observed host-memory drop.

## Timeline and pressure

Times below interpret server log times as UTC on 2026-09-05, consistent with the
panel epochs. Clock offsets between the client and hosts were not recorded.
Sample minima and PSI maxima can occur at different moments.

| Observation | Candidate automatic KV | Reference fixed KV |
|---|---|---|
| Fat initialization logged | 12:36:25 | 13:24:07 |
| Engine initialization completed | 12:37:15 | 13:24:56 |
| Server-start log | 12:37:36 | 13:25:12 |
| Panel start | 13:00:37.994 | 13:25:42.542 |
| Last sampled head MemAvailable before panel | 4571.59 MiB | 982.49 MiB |
| Live-panel head minimum | 1628.13 MiB | 112.04 MiB, safety abort |
| Live-panel worker minimum | 4516.57 MiB | 4432.37 MiB |
| Startup-window maximum full PSI avg10, head/worker | 45.60% / 19.08% | 24.21% / 36.43% |
| Live-panel maximum full PSI avg10, head/worker | 0% / 15.26% | 1.05% / 0.72% |

The candidate passed all 28 live samples, including the warmup and diagnostics.
Its substantial startup pressure must not be hidden by reporting only the later
zero head PSI. The reference completed its boot sweep, then failed the first live
warmup. There are no measured fixed-reference samples and no fixed-candidate or
return control.

The head log stops at the trigger. It does not itself measure recovery; subsequent
SSH checks confirmed recovery separately. The worker log shows reclamation after
the controller stopped the peer model. This is evidence of a watchdog stop, not
an allocator-attributed CUDA OOM or a diagnosed kernel bug.

## What is still missing

The receipts do not provide synchronized process RSS/PSS, anonymous/file/shared
breakdowns, glibc arena statistics, CUDA allocated/reserved snapshots or a trace
of allocation ownership around the failed request. They also lack timestamped
request-stage events and complete per-rank workspace backing-allocation records.
There is no evidence that the CPU arena cap alone would make this run safe.

Before proposing another full-model control:

1. Get maintainer agreement on whether the pre-existing stock-startup failure may
   remain outside this PR. A successful opt-in boot does not close that checkbox.
2. Design bounded diagnostic instrumentation for host, process and CUDA accounting,
   recording each category separately. Record the real request-stage timeline and
   both ranks, not just the final `/metrics` error.
3. Establish an admission policy with measured transient headroom and stop-latency
   allowance before sending a request. The published watchdog reproduces the old
   monitor; it is not proof of safe admission. Faster polling alone does not bound
   one large allocation, and 866.67 MiB is an observed decline, not a worst-case cap.
4. Keep weights, context and KV capacity unchanged. Any allocator experiment needs
   explicit configuration and a separate safety justification. No arena cap or
   longer idle delay has been validated as a remedy by this audit.

Until then, retain the draft's isolated claims and do not retry the failed
configuration. This audit does not mark the PR production-ready.

## Evidence and source references

The existing [receipts directory](benchmarks/gb10-prefill/) contains the originals.
Use `gzip -dc` for compressed files. Source and log references below use the tested
`f5a784d` review tree; LF-delimited line counts retain carriage-return progress bars.

- `fixed-reference-a1-start.log`, lines 303-307: initialized scratch, then fixed
  budget and capacity. Lines 320-392: graph capture and boot sweep.
- `candidate-kit-auto-start.log`, lines 302-332: scratch, automatic accounting and
  graph capture. Lines 381-401: boot sweep.
- `head-memory.jsonl`, lines 6029-6033: preceding sample, decline and stop marker.
- `fixed-reference-a1-live.json`: request, stream error and failed metrics receipt.
- `overlay/exl3.py`, lines 717-792: scratch ownership and capacity; lines 925-986:
  shared thin temps; lines 1109-1120 and 1291: serving transients.
- `overlay/patch_indexer_workspace.py`, lines 179-217: existing workspace formula.
- `tests/bench_indexer_logits_memory.py`: isolated-loop measurement boundary.
