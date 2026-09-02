# Sparse-indexer prefill workspace right-sizing

`GLM53_INDEXER_WORKSPACE=rightsize` · `overlay/patch_indexer_workspace.py` ·
`tests/test_indexer_workspace.py`

Condensed from the 2026-08-31 read-only inspection of the live fork
(`0.1.dev20051+g487ecf187` in `glm53-exl3-head`, head Spark 10.0.22.1), with the
corrections from Codex's review folded in and the 2026-09-01 workspace receipt
attached. `file:line` citations are against
`/usr/local/lib/python3.12/dist-packages/vllm/` inside that container.

---

## 1. The receipt

Stage A boot, 2026-09-01, `VLLM_DEBUG_WORKSPACE=1`:

```
[WORKSPACE DEBUG] Resized workspace from
  'sparse_attn_indexer_kpool.py:284:sparse_attn_indexer_kpool':
  0.00 MB -> 5036.40 MB (ubatch 0)
```

`sparse_attn_indexer_kpool.py:284` is the profiling-run
`current_workspace_manager().get_simultaneous(values_spec, scales_spec, radix)`
call, so the line covers the K-gather workspace plus the 1 MiB radix top-k
scratch. It reconciles to the byte:

```
values   40,000,000 x 128 B  = 5,120,000,000
scales   40,000,000 x   4 B  =   160,000,000
radix                          =     1,048,576
                                 -------------
                                 5,281,048,576 B = 5036.40 MiB
```

That is `get_max_prefill_buffer_size` = `max_model_len * 40` = 40,000,000
entries at `max_model_len = 1,000,000`
(`v1/attention/backends/mla/indexer.py:636-646`). The allocation happens during
the memory profile, so it is **subtracted from the KV pool**, and it is locked
for the life of the process (`v1/worker/gpu_model_runner.py` ->
`v1/worker/workspace.py`). It never shrinks.

**This closes UNKNOWN U3 of the research draft: the 4.92 GiB is measured, not
computed.** It also settles Codex item 6's "132 bytes depends on the active
FP8/FP4 layout" — the FP8 layout is what this deployment actually locked, and
the measured total matches 132 B/entry exactly.

## 2. Why it is over-sized

The workspace is sized in **tokens** and indexed in **pools**.

`split_indexer_prefill_chunks` (`indexer.py:81-127`) is the only consumer of
the size. The metadata builder hands it `seq_lens_cpu // self.compress_ratio`
(`indexer.py:1100-1104`), and for GLM-5.3-Flash `compress_ratio == index_kpool
== 4`: `models/glm5next/nvidia/attention.py:137` replaces the indexer kv-cache
spec with `compress_ratio=index_kpool`. `sparse_attn_indexer_kpool.py:495-496`
then slices the workspace to `chunk.total_seq_lens`, which is a sum of those
compressed lengths.

Upstream already knows: `models/deepseek_v4/attention.py:776-778` divides
`get_max_prefill_buffer_size(vllm_config)` by `compress_ratio` at its call
site. `models/glm5next/nvidia/attention.py:302` does not — it passes the raw
token-granular value straight into `SparseAttnIndexerKpool` as
`total_seq_lens`. The 4x is the whole of the gap between the two call sites.

## 3. What `rightsize` computes

Not a tuned request count — the **legal maximum a single step can present**:

```
entries = min(max_num_seqs, max_num_batched_tokens)
          * cdiv(max_model_len + num_speculative_tokens, compress_ratio)
```

capped at the stock `max_model_len * 40` so the patch can only ever narrow.

| Term | Why |
|---|---|
| `min(max_num_seqs, MNBT)` | the scheduler admits at most `max_num_seqs` sequences, and every prefill row costs at least one query token out of the `max_num_batched_tokens` budget — so this bounds the prefill requests in one step |
| `+ num_speculative_tokens` | the splitter is fed `common_attn_metadata.seq_lens_cpu_upper_bound`, which the builder documents at `indexer.py:1095-1099` as "an upper bound for async-spec extend rows" |
| `cdiv`, not floor | the consumer floors; cdiv is the safe side |
| clamp to stock | a config whose legal maximum exceeds stock is one where **stock is already under-sized**. That is an upstream latent risk this overlay does not inherit and does not pretend to fix: it returns stock unchanged and logs a warning |

At the live config (M=1e6, kpool=4, `max_num_seqs=16`, MNBT=2048, k=7):
`cdiv(1_000_007, 4) = 250,002` per request x 16 = **4,000,032 entries =
528,004,224 B = 503.5 MiB**, against 5035.4 MiB. At the recipe default
`MAX_NUM_SEQS=4` it is **1,000,008 entries = 125.9 MiB**.

**The hard floor** is one whole max-length request. `indexer.py:115-119` admits
a single request that exceeds the budget and sub-chunks it on the query
dimension M only, never on N, so a workspace below `cdiv(max_model_len, kpool)`
would overrun. The formula's `max(1, ...)` request count keeps that floor even
for degenerate configs (`max_num_seqs=0`, `MNBT=0`), and
`test_prefill_request_count_is_mnbt_bounded` asserts it.

### 3.1 Why the chunk list does not change

Because the size is >= the largest total the splitter can ever be asked to
pack, the `new_n <= workspace_size` constraint at `indexer.py:109` cannot bind
anywhere the stock 40x value did not already bind. Every batch the scheduler
can legally form produces the **same chunk list**, so the same
`cp_gather_indexer_k_quant_cache` calls, the same `fp8_fp4_mqa_logits` shapes,
and the same top-k inputs.

This is not asserted, it is checked: `test_chunking_is_identical_to_stock`
replays a host replica of `split_indexer_prefill_chunks` over 400 randomized
legal batches plus the extremes and compares chunk lists, and
`test_chunking_test_has_power` proves the comparison can fail by starving the
workspace deliberately.

## 4. Decisions on Codex's review

| Codex item | Decision |
|---|---|
| **2 — the 128-byte alignment is unsupported by the SM120 paged-MQA kernel** | **Dropped the change entirely.** The draft's Anchor A (decode logits pitch) is not in this PR. The alignment contract is unresolved, so the honest move is to not ship an unproven pitch, not to ship it with a caveat. §6 keeps it as a separate, separately-gated item |
| **3 — Anchor A's safety proof is incomplete** | Moot: Anchor A is out of scope |
| **4 — "real need ~33 MB" understates the legal requirement; 264 MB is a throughput tradeoff** | **Accepted, and the draft's `GLM53_INDEXER_WORKSPACE_REQS=8` default is rejected.** There is no request-count knob. The default is the legal maximum derived from `max_num_seqs`/MNBT, which is what Codex asked for, and it also removes the chunking regression surface (§3.1) rather than "accepting and measuring" it. Spec tokens are in the span, per the same item |
| **5 — "fails closed" needs runtime proof** | **Strengthened from a claim to a check.** The lock's `AssertionError` is still the backstop, but the patch no longer relies on it: the builder raises at `__init__`, on every TP rank, before any step, if the workspace is below the legal maximum. See §5 |
| **6 — 4.67 GiB is plausible but not demonstrated** | The stock half is now **measured** (§1). The patched half is still an estimate and is gated on the live receipt in §7 |
| **7 — the capacity projection is approximate** | **Accepted.** The acceptance criterion is the allocator's own `GPU KV cache size` / `Available KV cache memory` startup lines, not GiB arithmetic. §6's percentages are labelled estimates |
| **11 — Anchor A and Anchor B read different config sources** | **Accepted, with a check.** Sizing runs in `get_max_prefill_buffer_size`, which only receives `vllm_config`, so it reads `hf_text_config.index_kpool`; the runtime divides by `kv_cache_spec.compress_ratio`. They are the same number by construction (`models/glm5next/nvidia/attention.py:137`), but the patch now **refuses to serve** if they disagree |
| **12 — split the rollout gates; this is a memory patch** | **Accepted, structurally.** This PR is Anchor B only, default off, with memory receipts. Mixed-step latency (research §6, Codex 8-10) is not a success criterion here and is not investigated by this PR |
| **1 — the refutation of the original top-k hypothesis is too absolute** | Noted; nothing in this PR depends on it. The claim carried forward is only the narrow one the receipt supports |

## 5. Failure modes

* **Wrong compression ratio.** `DeepseekV32IndexerMetadataBuilder.__init__` now
  raises after `self.compress_ratio` is resolved if
  `hf_text_config.index_kpool` disagrees with it, naming both values and the
  knob to unset. Boot fails; no step runs. The comparison is **unconditional**
  whenever `rightsize` is active and raises in **both** directions. It is not
  guarded on `self.compress_ratio > 1`: that guard would skip the one
  combination that actually under-sizes — `index_kpool=4` with
  `compress_ratio=1` sizes for ~4M compressed entries while the runtime may
  present ~16M token entries (Codex PR-8 blocker). The mirror direction sizes
  at stock and cannot under-run, but it still means the two config sources
  disagree about the model, which is not a state to serve `rightsize` from.
  Both directions are covered by
  `tests/test_indexer_workspace.py::test_builder_ratio_mismatch_raises_both_directions`,
  and `::test_builder_guard_is_inert_in_stock_mode` pins that stock mode runs
  no cross-check at all.
* **Workspace below the legal maximum.** Same site, same raise. This can only
  fire if the formula and the clamp disagree, which is exactly the condition
  worth crashing on.
* **Anything else.** The workspace is still locked after warmup, so
  `_ensure_workspace_size` still raises an `AssertionError` naming the calling
  site and the requested size (`v1/worker/workspace.py`) rather than
  overrunning. That is the backstop, not the plan.
* **Bad knob value.** `_glm53_workspace_mode()` raises on anything but exactly
  `stock`/`rightsize`, and `start.sh`'s numeric guard rejects it earlier still,
  turning a container boot failure into a launcher error. The two enums are
  literally identical: the default applies **only** to an unset var, and the
  value is then compared as-is, so `""`, `" rightsize "` and `RIGHTSIZE` are
  errors on both sides. The launcher's caller capture uses the
  `${GLM53_INDEXER_WORKSPACE+1}` setness probe, so an explicitly empty caller
  value reaches the guard instead of being replaced by a `.env` value.
  `tests/test_indexer_workspace.py::test_mode_enum_matches_launcher_enum` runs
  both implementations over the same values and requires the verdicts to match.
* **TP-rank divergence.** Both quantities derive from `model_config`,
  `scheduler_config` and `speculative_config` plus one env var, all shipped
  identically to head and worker by `start.sh`. No device tensor, no scheduler
  state, no `.item()`. Each rank logs its computed sizing at boot, so a
  mismatch is visible in the receipts rather than inferred.
* **Default off.** With the knob unset or `stock`,
  `get_max_prefill_buffer_size` returns `max_model_len * 40` and logs nothing.
  The image can carry the overlay and serve exactly stock.

## 6. Expected gain — and what this PR does not claim

**Memory (the claim).** At `MAX_NUM_SEQS=16`, 4,751,995,776 B = **4.43 GiB**
returns to the KV pool; at the recipe default `MAX_NUM_SEQS=4`, **4.79 GiB**.
Against the 16.93 GiB available-KV figure from the inspected boot that is an
estimated **+26%** and **+28%** respectively. Per Codex item 7 these are
estimates: the acceptance number is whatever the allocator prints (§7).

(The research draft's +27.6% assumed the rejected `REQS=8` sizing. The
difference between 4.43 and 4.67 GiB is the price of removing the chunking
regression, and it is worth paying.)

**Latency (not a claim).** The research train's arithmetic put the indexer's
entire context-proportional decode cost at ~1 ms/step at 100K KV, ~0.5% of the
measured 200 ms long-context delta. Codex item 8 correctly notes that roofline
arithmetic cannot *prove* the indexer is not a contributor — so this PR simply
does not make a latency claim in either direction. The pass bar is **no receipt
worse than -3%**. The plan's original "+10% on 100K-KV decode or mixed-step"
gate does not apply to this PR and would fail a correct memory change.

**Out of scope, deliberately:**

* *The decode logits pitch.* `sparse_attn_indexer_kpool.py:788-797` passes
  `max_model_len` as the paged-MQA logits row pitch
  (`utils/deep_gemm.py:620,625-626`: output is `[B*next_n, max_model_len]`
  float32) while `decode_metadata.seq_lens` is pool-granular — a flat 4x
  over-stride on a transient allocation. It is a genuine bug and an
  upstreamable one, but the required pitch alignment is not documented in the
  shipped SM120 headers (Codex item 2) and the op is captured inside the FULL
  cudagraph, so it needs its own kernel-ABI and bitwise-equivalence gate. Not
  here.
* *The mixed-step tax.* Codex items 8-10. Separate investigation.

**Upstreamability.** The magic 40 has a stated rationale (`indexer.py:638-645`)
tied to the flashmla_sparse workspace, which is not the backend here
(`FLASHINFER_MLA_SPARSE_SM120`). An upstream fix should make the factor
backend- and compression-aware rather than env-gated — most simply, by dividing
at the glm5next call site the way `deepseek_v4/attention.py:776-778` already
does. Keep this as a recipe overlay; file the upstream issue separately.

## 7. Test plan

**Host (green before merge — `tests/test_indexer_workspace.py`, 16 tests):**

* the sizing formula across Codex's edge cases: MNBT-bounded request count,
  `max_num_seqs`-bounded request count, `k=7` spec tokens (and monotonicity in
  k), the true per-step maximum, the one-request floor at degenerate configs,
  `index_kpool <= 1` / missing / junk => stock, and the never-exceeds-stock
  clamp over a `M x kpool x max_num_seqs x MNBT` sweep;
* the stock sizing reproducing the 5036.40 MB receipt to the byte;
* chunk-list equivalence against stock by exhaustion, plus a power check;
* the knob enum, including that a typo raises rather than selecting a mode;
* apply / idempotence / `--preflight`-writes-nothing / three-way anchor drift /
  half-patched refusal, against a fixture assembled from verbatim excerpts of
  the live container's `indexer.py`, and against the live file itself
  (`GLM53_INDEXER_BACKEND_PY_SRC=…`, or `GLM53_INDEXER_LIVE_SSH=1` to pull it
  read-only from the running head container);
* the shipped dispatch, exec'd out of the patched file text by AST, so the
  mode dispatch and the stock-clamp warning are covered and not just the
  formula;
* launcher / Dockerfile / README wiring.

**Live (the gate on the memory claim).** Boot A stock, boot B
`GLM53_INDEXER_WORKSPACE=rightsize`, everything else identical:

1. **Workspace receipt.** `VLLM_DEBUG_WORKSPACE=1` on both. Expect
   `5036.40 MB` -> `~504.5 MB` at `MAX_NUM_SEQS=16` (`~126.9 MB` at 4), and the
   `[glm53-indexer-workspace] rightsize:` / `builder:` lines on **both** TP
   ranks with equal values.
2. **Capacity receipt — the headline.** `Available KV cache memory` and
   `GPU KV cache size: N tokens` / `Maximum concurrency` from both startup
   logs. This is the acceptance number, per Codex item 7; the GiB arithmetic in
   §6 is only the prediction.
3. **Temp-0 logprob equivalence.** Same prompts at 8K / 50K / 100K,
   `temperature=0`, top-5 logprobs compared token-by-token against the stock
   boot. §3.1 says the chunk list is identical, so this must be **exactly**
   equal; any drift is a bug, not a tolerance.
4. **100K-prompt prefill smoke.** One cold 100K prefill to completion, then a
   concurrent long-context prefill at the configured `MAX_NUM_SEQS` — the case
   the rejected `REQS=8` sizing would have re-chunked. Expect no
   `_ensure_workspace_size` assertion, no extra chunk count, and prefill
   timings within the -3% bar.

Only after 1 and 2 land as receipts does the +26% become a claim rather than a
prediction. The default stays `stock` until then, and flipping it is a separate
change.

## 8. Anchors

Three pinned sites in `v1/attention/backends/mla/indexer.py`, all preflighted
before any is written:

| Site | Location | Change |
|---|---|---|
| `os import` | module header | `import os` (stock has none at module scope) |
| `workspace sizing` | `get_max_prefill_buffer_size` | helper functions + mode dispatch; stock path returns the unchanged expression |
| `builder cross-check` | `DeepseekV32IndexerMetadataBuilder.__init__`, after `compress_ratio` resolution | fail-closed ratio/size check + per-rank boot log |

Drift in any anchor exits nonzero and leaves the file untouched.
`--preflight` validates without writing, so CI catches drift while the knob is
off.
