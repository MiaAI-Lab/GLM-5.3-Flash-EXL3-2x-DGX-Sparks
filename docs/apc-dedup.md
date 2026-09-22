# Duplicate prefix-cache pages

Repeated prefixes can leave several idle pages carrying the same cache key.
Those copies consume eviction slots and push out unrelated conversation history.
The cleanup runs when a page loses its final reference. It clears that page's
cache metadata only if every primary and secondary key has another idle copy
in the same cache group. The existing free queue then reuses that page first.
Active pages, copy pins, unique aliases, and the remaining pages' queue order
keep their existing behavior. Model weights, tensor precision, and context
limits are unchanged.

The overlay follows per-group retention in the image build and TP2/TP3/TP4
launchers. It builds on [#130](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/130)
and the retention work by nood-co1 / BlockbrainLabs in
[#83](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/83).

## CPU reproduction

Run against the pinned image's vLLM source. The test patches source in memory
and executes its actual pool, queue, and hash-map classes without GPU imports.

```sh
python3 tests/test_apc_free_duplicates.py --source-dir /path/to/vllm/v1/core -v
```

Seven tests cover active and copy references, batch release, unique secondary
aliases, separate groups, draft priority, cache events, metrics, and patch drift.
The pressure case seeds three distinct history keys and one repeated key into
seven usable pages, then allocates and releases three repeated copies per round.
With unchanged code, all three history keys are evicted in round two. With the
cleanup, all three remain through twelve rounds. This isolates the pool policy;
it does not simulate model execution or prove a particular token capacity.

On 20 September 2026, the seven tests passed on the Dockerfile's pinned base
image, both before and after composing the hybrid, retention, dedup, and no-store
overlays. Reapplying all four overlays also passed. The existing no-store suite
passed all 183 checks, including 99 checks through real vLLM cache managers.

## Two-Spark serving observation

The TP2 candidate at [f7b84e7](https://github.com/ratulsarna/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/commit/f7b84e7fbd381b5d6f1b3188e00fb2726234c604)
ran three distinct 229,000-token histories with a 232,000-token context limit,
three maximum sequences, 2,048 batched tokens, FP8 target KV, and BF16 draft KV.
It also included draft-page packing and checkpoint-boundary fixes. Its pool had
412 usable pages; global retention was 57,344 tokens and draft retention was zero.

| History | Cold first content | Return first content | Reused tokens on return |
| --- | ---: | ---: | ---: |
| 0 | 146.901 s | 3.451 s | 225,792 |
| 1 | 145.763 s | 3.396 s | 225,792 |
| 2 | 146.440 s | 3.837 s | 225,792 |

These are one cold/return round with synthetic fact prompts. All six fact answers
matched their expected values, with no reported preemptions. The retained result
artifact has SHA-256 `be1b4c4cda0e465442f8d42c5559c598aaaef5f70caf17135e6499737fbf9067`.
The measurements describe that combined candidate, not an isolated speedup from
this PR or a GPU qualification of this PR on current main. TP3/TP4 GPU behavior
was not measured.
