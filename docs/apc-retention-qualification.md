# Opt-in APC retention: qualification and tradeoffs

## Scope

The implementation extends [PR #83](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/83)
by nood-co1 / Blockbrain Labs. Its per-group retention and launcher tests are
included with their commit attribution. The additional behavior is:

- Explicit SWA `0` gives draft-only cached blocks lower eviction priority and
  frees draft managers before target managers. Unset SWA inherits the global
  retention value without opting into that priority policy.
- Sparse Mamba retention preserves the prior checkpoint needed for replay.
- DFlash current-boundary reuse requires a successful EAGLE-pop lookup at the
  reconciled hit and a complete, non-null visible window. Otherwise the hit
  backs up until the fresh suffix can rebuild the draft window. Failed EAGLE
  lookups cannot mark a group verified on a subsequent reconciliation pass.
- Overlay migration, composition and rank wiring are checked before use.
  Launcher changes also enforce an immutable DFlash snapshot, preserve caller
  overrides, and validate mounted artifacts before a restart stops the ranks.

This is a draft contribution for maintainer discussion, not a recommendation
to make all-zero retention the default. Global and SWA retention remain
unset by default. Neither the 390-block qualification budget nor the tested
262,144 context / 2,048 MNBT settings are new launcher defaults. The fine-grained
lookup proposals in #84 and #125 are not included.

## Measured runtime pair

Tests used one 2× DGX Spark GB10 pair over CX7, stock EXL3/TR3 4bpw, target TP2,
DFlash2 k=7 / draft TP2, target FP8 (`fp8_ds_mla`) and draft auto/BF16 KV.
The target revision was `61e26e1484e16d7a603f77040cda9b43cc4a31d6` and the draft
revision was `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`.

| Field | Old | Candidate |
|---|---|---|
| Recipe source | `eb0469fbb2b49fd7c025f594a3339a121e58f7a9` | `3d01c7b7fa85a4b1419621a2b0657a2cde48a27e` |
| Image ID | `sha256:277f348d5afe2a48c025c0b270afb89c4f0ca67e6dab192ccef9f3e223328004` | `sha256:29bb99dc2c9ffb34e216743b5714bbfd2dddd25216289654e581af3e6d2f4901` |
| Retention | Original dense/default policy | Explicit global/SWA `0/0`; vector `[0,0,0,0,0,0,0]`; low-priority group `[6]` |
| Physical blocks / nominal KV tokens | 390 / 491,520 | 390 / 491,520 |
| Context / MNBT / max sequences | 262,144 / 2,048 / 4 | Same |
| GPU utilization / indexer workspace | 0.82 / rightsize | Same |
| Execution | E2 fat kernel, mixed-prefill skip, CUDA graphs `1,2,4,8,16,24,32` | Same |

The candidate image was built from `cfcceedbecd5778d7acd6109be2c22f305567ab4`.
Later host/patcher changes render byte-identical model/cache runtime files.
The image IDs identify retained local images, not publicly pullable registry
digests. The tested runtime hashes are:

| File | SHA-256 |
|---|---|
| `kv_cache_coordinator.py` | `507685d5a135f3777b2b3157fbbebcace5c4d81f67969db624756bffaa832056` |
| `block_pool.py` | `176f6abe48471e617af8ac538ac12b578a51f864fc8637c9b640336bf6858472` |
| `single_type_kv_cache_manager.py` | `143c8a822ebf100f3507ee2a648b10e9b0e2d4ac43e4fbd9345d6162a45131e8` |

The old runtime was temporarily capped at 390 blocks to match the candidate,
rather than reproducing its historical automatic allocation. These are
runtime-plus-policy comparisons, not isolation of a single changed line.

## 128K edits and branches: matched comparison

The [archived harness](qualification/apc-2026-09-05/branch_matrix.py) constructs
ten synthetic ledger segments with known values and exactly 128,000 tokens.
For every case, it resets an idle cache, primes that same complete history,
then submits one probe. An edit changes a value while preserving subsequent
history; a branch truncates at the selected point and adds a question. The
assistant continuation is fixed text, not a generated old/new prime answer.
Each old/new probe has identical message and token hashes and token-level LCP.

One trial per case, sequential requests, candidate first and old second;
temperature 0, seed 7391, thinking off, prime maximum 8 output tokens and probe
maximum 24. TTFT is measured to first non-empty streamed text. Required local
compute/cache-hit counters must sum to both tokenized and server-reported
prompt counts; absent counters fail the harness. Every prime computed exactly
128,000 tokens with zero hits.

| Probe | Prompt tokens | LCP tokens | Old hit / compute | Candidate hit / compute | Old TTFT (s) | Candidate TTFT (s) |
|---|---:|---:|---|---|---:|---:|
| Unchanged append | 128,019 | 128,000 | 125,440 / 2,579 | 125,440 / 2,579 | 2.801 | 2.796 |
| Edit at 10% | 128,025 | 12,761 | 0 / 128,025 | 0 / 128,025 | 108.178 | 109.982 |
| Edit at 50% | 128,024 | 63,713 | 0 / 128,024 | 0 / 128,024 | 107.855 | 111.755 |
| Edit at 90% | 128,024 | 114,666 | 111,104 / 16,920 | 0 / 128,024 | 14.700 | 112.491 |
| Branch at 10% | 12,794 | 12,767 | 0 / 12,794 | 0 / 12,794 | 11.409 | 11.016 |
| Branch at 50% | 63,747 | 63,720 | 0 / 63,747 | 0 / 63,747 | 53.857 | 56.312 |
| Branch at 90% | 114,701 | 114,674 | 111,104 / 3,597 | 0 / 114,701 | 3.351 | 99.885 |

All 14 probes returned the expected value with no stale value; preemptions
were zero. **The late edit and branch are regressions:** the candidate loses
111,104 tokens of reuse and takes approximately 7.65× / 29.81× as long. Early
and middle changes missed on both runtimes; unchanged append was equivalent.

Seven cold-prime TTFT medians were 108.074 s old and 111.262 s candidate
(~2.95% longer). The fixed run order and lack of repeated alternation prevent
a general kernel-performance conclusion. Short answers do not qualify decode
throughput. These measurements do not qualify SWA-only sparse / dense-target
retention or other context sizes, hardware, or concurrency levels.

### Reproduction boundary

The harness is preserved byte-for-byte as measurement evidence, not a generic
production benchmark. It targets loopback port 32105 and served ID
`glm-5.3-flash-exl3`; inspect those constants before adapting it. It resets the
whole prefix cache and requires exclusive use of a disposable test deployment.
The pinned engine exposes that reset through developer mode, which enables
other developer routes too. The test server must bind loopback and be
unreachable through public or gateway routes. An unloaded gateway entry is
not isolation if it can start the model on demand. The harness does not set up
isolation, start monitors, launch models, or restore production for you.

On an already isolated, monitored test deployment, `--prepare-only` records
input identities without inference; `--reference-prepared` checks the second
runtime's token identities. Equivalent invocations for the measured matrix are:

```bash
python3 docs/qualification/apc-2026-09-05/branch_matrix.py --runtime new --prepare-only --output new-prepared.json
python3 docs/qualification/apc-2026-09-05/branch_matrix.py --runtime new --output new.json
# Switch the isolated test deployment to the matched old runtime before continuing.
python3 docs/qualification/apc-2026-09-05/branch_matrix.py --runtime old --prepare-only --reference-prepared new-prepared.json --output old-prepared.json
python3 docs/qualification/apc-2026-09-05/branch_matrix.py --runtime old --reference-prepared new-prepared.json --output old.json
```

Both measured boots had per-node memory/rank/swap monitoring before launch.
Minimum available memory was 7.915 / 11.247 GiB head/worker for the candidate
and 7.692 / 11.430 GiB for old. No OOM, restart, swap use, or intervention was
observed. A discarded setup boot had a 4m48s monitoring gap before any test
traffic; it is not included in those measurements. Developer mode and all
temporary settings were removed after testing.

## Long-history retention and DFlash checks

Separate candidate qualification used four independent exact 210,000-token
histories, filled sequentially, then revisited oldest-first without clearing
between them. Cold times were 212.923 / 209.013 / 207.078 / 207.040 s, each
0 cached / 210,000 computed. Revisits took 2.506 / 2.453 / 2.475 / 2.586 s,
each 207,872 cached / 2,128 computed, with zero preemptions. This demonstrates
four retained histories, **not four simultaneous full-context streams**.

DFlash suffix checks exercised 0, 1, 64, 2,047 and 2,048 token extensions.
Warm/cold draft-acceptance ratios were 1.1771 / 1.24307 / 1.02396 / 1.0 / 1.0.
Three warm throughput rates were 54.864 / 54.510 / 54.872 tok/s against a
cold median of 54.783 tok/s. These are warm/cold comparisons on the candidate,
not old/new decode comparisons. Valid-current suffix-64 reused 21,504 tokens
and computed 65; forced missing-draft suffix-64 backed up to 17,920 cached /
3,649 computed. API, cancellation, tools, thinking and 120K key+Vision checks
passed for stock and abliterated profiles; the four-history test was stock.

The [pinned deployment receipt](https://github.com/ratulsarna/spark-serve/blob/3c295a723aaa7750faef95cee758da2f76896dc1/services/glm-5.3-flash-exl3/qualification/2026-09-05-apc-global0-dflash-replay-v2.md)
records exact-image migration, both-order overlay composition, runtime hashes,
acceptance, boot facts and artifact hashes. Its host-local raw paths are not
public downloads. This PR preserves the 128K comparison artifacts below.

## Limits and maintainer decision

Cache-hit correctness tests do not establish universal cold/cached output
equality: both cold/cached and repeated-run output variation remain recorded.
Manager release ordering also does not guarantee that every pre-existing
unhashed free block precedes every draft-cache block in the global queue.
The exact cause of a consecutive valid-then-fallback draft eviction is not
proven. TP=4 sparse SWA is rejected, not qualified. A full boot using the
maintainer's default 1M/7168 geometry is not part of this qualification.

The useful decision is whether to accept the reusable replay/priority work
as an opt-in extension to #83, and whether to split the launcher/pin work into
a separate change. All-zero retention should not become the default on the
strength of the four-history result; edited and branching workloads need the
tradeoff above. No ideal checkpoint interval is claimed.

## Public evidence identities

| Artifact | SHA-256 |
|---|---|
| [Harness](qualification/apc-2026-09-05/branch_matrix.py) | `7c41494bb6011ef33ec0851d52ae656731921b1d10edf669d63f3fa046db6656` |
| [Candidate raw matrix](qualification/apc-2026-09-05/new.json) | `82078a09861d9ee9ea07fc7d0976f1c56ad075e118c20a6d701987422070766b` |
| [Old raw matrix](qualification/apc-2026-09-05/old.json) | `bdbedaccd02c7be608329e8aed8a02456e4acbc17ab80da3ceee21fa1767c52a` |
