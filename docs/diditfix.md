# Did v5 fair mixed-prefill fix the reporter hang?

**Independent retest:** 2026-09-15 ~12:26–12:36 local, this session.  
**Serve already on v5** (`glm53-exl3-head`, EngineCore pid 2019); no extra restart. `/health` 200.  
**Overlay:** `# [glm53-decode-floor:v5]`, `GLM53_MIXED_PREFILL_CHUNK=fair`, chunk 256, share 0.20, interval 2000 ms, max step 1000 ms, max chunks 1, `LONG_PREFILL_TOKEN_THRESHOLD=3584`.  
**Harness:** `logs/fair-v5-20260915/overlap_arms.py`  
**Receipts:** `logs/fair-v5-retest-20260915/` and `/tmp/mixed-prefill-fair-v5-retest.json`  
**Prior pass (other agent):** `logs/fair-v5-20260915/` — numbers agree in direction; this file’s tables are the independent rerun.  
**Design:** [astra-fix.md](astra-fix.md)

## Verdict

**Yes, for the reporter freeze, when `fair` v5 is live.** A 2k unique newcomer submitted 10 s into a thinking-on essay (`max_tokens=4000`) got its first token in **11.6 s, while the essay was still streaming**. Under `skip` that request waits for the whole essay (~163 s here; 221 s / 20+ min in the original report).

A ~30k unique newcomer on the same recipe also finished **during** the essay (**163.6 s TTFT**, 16 s before A’s last token), with **all 29.4k prompt tokens** in mixed steps. That duration is the 20 % share budget, not Waiting starvation.

Incumbent decode did **not** collapse to the reporter’s 0.8–2.8 tok/s under `CHUNK=0`. On the essay, overlap rate stayed **~20.6–20.9 tok/s** vs ~24.7 solo (about **1.18–1.20×**). In-run, vs the 10 s of decode before B: **1.06×** (2k B) and **1.21×** (30k B).

`start.sh` and `.env.example` default `GLM53_MIXED_PREFILL_CHUNK` to **`fair`**. TP=3 stays **`0`**; TP=4 stays **`skip`**. Opt in on those launchers with `fair`.

| Claim | `skip` | v4 `fair` (prior) | **v5 `fair` (this retest)** |
|---|---|---|---|
| 2k newcomer 10 s into thinking essay | waits the whole decode | 21.9 s, during A | **11.6 s, during A** |
| ~30k newcomer 10 s into thinking essay | waits the whole decode, then ~23 s solo prefill | 186 s, after A | **163.6 s, during A**; 29 365 mixed tokens, 0 solo |
| `CHUNK=0` drops A ~10–36× | N/A | A held | A held: essay **1.18–1.21×**; unique-800 **1.21–1.34×** whole-window |

## Live confirm

```
patched scheduler.py (# [glm53-decode-floor:v5])
[glm53-decode-floor] fair v5 probe_chunk=256 ladder=128..2048 share=0.2 interval_s=2.0 max_step_s=1.0 max_chunks=1
```

Container env matched the knobs above. GPU was idle (0 %) before the retest.

## What was run

Same arms as the other agent, thinking/temp as specified, **fresh unique-word prefixes** (not cache-cheap `the`). C1 = A solo; C2 = A then B. Reporter B is submitted **10 s after A’s first token**.

| Arm | A | B | B delay |
|---|---|---|---|
| cold2k | unique ~1.9k, thinking off, max 800 | unique ~1.9k, thinking off, max 8 | at A first token |
| cold30k | unique ~29–30k, thinking off, max 800 | unique ~29–30k | at A first token |
| rep2k | thinking-on essay, max 4000 | unique ~1.9k | +10 s |
| rep30k | same A type (C1 reused) | unique ~29.4k | +10 s |

## Independent results

Overlap tok/s is stream-event rate scaled by `(completion_tokens − 1) / (events − 1)`. Gaps are A’s inter-event gaps in the overlap window (until B’s first token, or A’s last if B is later).

| Arm | C1 tok/s | A overlap tok/s (vs C1; vs in-run before B) | A gaps p50 / p95 / max | B TTFT | B during A? | B mixed progress |
|---|---:|---|---|---:|---|---|
| **rep2k** | 24.7 | **20.9** (1.18×; **1.06×** vs 22.2 before; 25.0 after) | 0.09 / 0.19 / **1.39 s** | **11.6 s** | **Yes** (−141 s vs A last) | 4 mixed steps, **1892 / 1892**, chunks 100 / 256 / 768 |
| **rep30k** | 24.7 | **20.6** (1.20×; **1.21×** vs 24.8 before; 24.3 after) | 0.09 / 0.19 / **1.44 s** | **163.6 s** | **Yes** (−16 s vs A last) | 39 mixed steps, **29 365 / 29 365**, chunks 117 / 768 / 832 |
| cold2k | 65.4 | 41.7 until B first (whole-window A **54.1**, 1.21×) | 0.12 / 0.40 / 1.12 s | **8.0 s** | **Yes** | 3 mixed steps, **1883 / 1883**, chunks 91 / 768 / 1024 |
| cold30k | 67.7 | **50.3** (1.34×) | 0.11 / 0.23 / 0.82 s | 37.2 s | **No** (A decode only 15.8 s) | 5 × **768 = 3840** mixed, then 25 600 solo |

Whole-window A on rep2k stayed **24.5 tok/s** (B left after 12 s of a 163 s decode). Whole-window A on rep30k was **21.1 tok/s** over a stretched **190 s** decode because B occupied almost the entire essay.

## Reading

- **The 20-minute / 221 s Waiting lockout is gone** on this v5 process. Scheduler logs during overlap are mixed `completed_step` rows with 768–1024-class chunks, not `skip` zeros.
- **v5 is using the step budget.** This retest’s mixed sizes are 768 / 832 / 1024, not v4’s stall at 128. That is why 2k TTFT dropped from ~16–22 s (v4) to **8–12 s**.
- **30k during an 800-token A still cannot finish in 16 s.** It moved **3840 tokens** in that window (v4 moved ~1k), then finished solo. Same capacity story as [astra-fix.md](astra-fix.md): at 20 % share, 30k needs on the order of **~2–3 minutes** of continuous mixed service. The reporter essay is long enough; a short 800-token decode is not.
- **Max gap ~1.1–1.4 s** is one mixed step against `MAX_STEP_MS=1000`. p95 stays on the normal decode cadence (~0.19 s on the essay).
- **cold2k overlap 41.7 tok/s** is the short window that contains the 1024-token step; after B’s first token A recovered to **70 tok/s**. Do not treat 41.7 as the whole-request decode rate.

## vs the other agent’s first v5 pass

Same process, new unique prefixes. Direction matches. This retest’s reporter 2k was **11.6 s** vs their **9.0 s**; reporter 30k **163.6 s** vs their **146.4 s**. Chunk ladder and “B during A” are the same. Single-run scatter of that size is expected.

## Still open

- Single runs only (no three-repeat matrix, no multi-newcomer).
- TP3 `0` and TP4 `skip` not measured.
- First mixed chunk is sometimes clipped (91–117 tokens); later rungs recover.
- Host busy-time proxy; one mixed step ran ~1.4 s against a 1.0 s budget.

PR [186](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/186) at v2 contains neither v4 nor v5.
