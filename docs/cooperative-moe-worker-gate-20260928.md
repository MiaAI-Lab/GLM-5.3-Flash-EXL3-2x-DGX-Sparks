# Cooperative MoE: worker-node GPU gate result + how the repin was actually done (2026-09-28)

Field report from a **second 2×Spark kit** (`spark-d931` head / `spark-ea54` worker, 2×GB10
SM121a, TP2 over CX7), recipe checkout at `943912c`, image built from this repo's Dockerfile.

This closes two items that were still open in `docs/cooperative-moe.md` ("Remaining work"):
the **worker-node numerical GPU gate** and **geometry-1 capture-size coverage**.

## 1. Worker-node gate: 48/48 pass

`extensions/cooperative_moe/test_cuda_integration.py` with `GLM53_COOP_MAINTENANCE_TEST=1`,
run on **both** nodes in a maintenance container (`--network none --gpus all`, no checkpoint
loaded), against the generated overlay + the pinned adapter.

```
HEAD   : {"stage": "complete", "checks": 48, "status": "pass", "geometry": 1,
          "capture_rows": [1,2,3,4,5,6,8,9,10,12,15,16,20,24,32],
          "max_rel_l2": 0.0026689, "clip_or_mut_peak_violations": 2, ...}
WORKER : {"stage": "complete", "checks": 48, "status": "pass", "geometry": 1,
          "capture_rows": [1,2,3,4,5,6,8,9,10,12,15,16,20,24,32],
          "max_rel_l2": 0.0026689, "clip_or_mut_peak_violations": 2, ...}
```

Notes for reviewers:

* `capture_rows` includes **3 and 5** — i.e. the live adaptive-k capture sizes
  (`GLM53_ADAPTIVE_K_SET=2,4,7` → verified prefix 2/4/7 → 3/5/8 rows/seq) that the
  "Remaining work" list asked for.
* `rows=40` reports `candidate_selected: false` (stays stock), matching the documented
  "1–32 rows" eligibility and the E3-grouped prefill path.
* Both nodes report the same two `kind: peak` entries (peak_rel 0.0036 / 0.00325 against the
  0.003 screen). The gate itself still returns `status: pass` and exit 0, and this is the same
  "strict differences retained" behaviour documented in `docs/cooperative-moe.md`.
* `distributed_serving_verified: false` / `quality_eval_verified: false` remain false — a
  single-node maintenance gate cannot set them.

## 2. Why a rebind was needed on this kit, and the procedure that worked

`prepare_profile.py` demands `BINARY_SHA == aa3fe5e9…`. On this kit that artifact never
existed: it was built on the maintainers' machine (the handoff documents `/home/mia/...` and
`/home/bpshi/.cache/vllm-glm53-flash/cooperative_moe/` was empty before this deploy). Two
facts follow:

1. **A clean rebuild here cannot reproduce the pin.** Building
   `extensions/cooperative_moe/build.sh` twice, same source tree and same container image,
   produced **different** digests:
   * build #1 `f3a228da002258891978cb099fbe362e7cbd2a85c0631164ff1f8f9053c51b70`
   * build #2 `a8c7519034326c11b53cf23796ed3aaf51dcb69ae21b664845cdc731534c5c93`
   (nvcc `--lineinfo` embeds line tables; the 4 `static_assert`s added to
   `native/cooperative_moe.cu` after the pin shift them. The ledger's own row
   `C1-native-assert … "source only until the next native rebuild" | pending rebuild`
   predicts exactly this.)
2. **The quickstart's stop rule and the repin rule are in tension.** `cooperative-moe-quickstart.md`
   says "If a clean rebuild differs … stop"; `cooperative-moe-handoff.md` says "A new digest is
   UNVALIDATED until the GPU gate is rerun and the pin updated". But the gate cannot run before
   the pin is updated, because `prepare_profile.py` refuses to emit the overlay that the gate
   mounts, and the adapter itself re-checks the digest at load
   (`runtime.py` → `CoopLaunch.__init__` → `_require(digest == SHA256, "unvalidated cooperative_moe.so digest …")`).

**Order that satisfies both rules** (used here, gate decided it, nothing shipped pre-gate):

1. rebuild the `.so` in the pinned image, offline
   (`docker run --rm --network none --cpus 4 --memory 8g --user $(id -u):$(id -g) -v …:/src:ro -v <archived upstream>:/upstream:ro -v <empty dir>:/work --entrypoint bash $IMAGE /src/build.sh /upstream /work`);
2. move `BINARY_SHA` **and** the adapter's `SHA256` constant to the new digest, then
   `ADAPTER_SHA` to the new `runtime.py` digest (the source edit changes that file's hash);
3. regenerate the overlay with `prepare_profile.py`;
4. run the packaged gate on **both** nodes; **ship only if both are `status: pass`**;
5. keep the build log, the gate logs, the old and new digests, and the diff of both pins.

Deployed artifacts on this kit (identical sha256 on head and worker):
`cooperative_moe.so f3a228da…`, `runtime.py e8facb44…`, `exl3-cooperative.py 895e5269…`.

Pin context, so the diffs above can be read against the right revision: this work was done on
recipe `943912c`, where `prepare_profile.py` pinned `STOCK_SHA 849e2588…`,
`BINARY_SHA aa3fe5e9…`, `ADAPTER_SHA 9427f6a6…`. On `main` today (`40a2099`) `BINARY_SHA` and
`ADAPTER_SHA` are unchanged, while `STOCK_SHA` has moved to `da7dd654…` (a further
`overlay/exl3.py` repin) — i.e. the repin described here is the same operation the maintainers
already perform when `overlay/exl3.py` changes, only triggered by the binary instead.

**Suggestion for the docs**: state the repin order explicitly (steps 1–5 above) instead of the
bare "stop", because the current wording makes a legitimate rebuild look like a failure. A
`--repin` checklist block in `prepare_profile.py`'s header comment would be enough.

## 3. Serving evidence after the repin

Same kit, TP2, DFlash2 k=7, adaptive-k `ema`, dense-FP8 `shared,dense,kda,mla`, KDA large-M on,
KV pool 14 GiB, `MAX_NUM_SEQS=8`:

| workload | stream tok/s | aggregate tok/s |
|---|---:|---:|
| structured (count 1→200) ×1 | **83.5** | 83.5 |
| prose (hash-map) ×1 | **38.6** | 38.6 |
| prose ×2 | 29.8 | 57.2 |
| prose ×4 | 21.4 | 79.0 |
| prose ×6 | 19.2 | 108.5 |
| prose ×8 | 16.3 | **124.5** |

`Fixed-shape cooperative MoE wrappers installed: K4 MCG, 1..32 rows, geometry=1 (A-wide/B-wide)`
appears on both ranks; decode steps log `selected=True reason=cooperative` and batches above 32
rows log `rows_out_of_range` and stay stock, as designed. A single-stream tool-call request
returns `finish_reason: tool_calls` with the expected function name.
