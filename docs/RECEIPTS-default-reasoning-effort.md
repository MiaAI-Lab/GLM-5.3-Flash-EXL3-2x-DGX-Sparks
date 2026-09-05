# `GLM53_DEFAULT_REASONING_EFFORT` — receipts

Server-side default reasoning effort for the **GLM-5.3-Flash EXL3** 2× DGX Spark serve, via
vLLM's `--default-chat-template-kwargs`. The launcher default is **empty** — this PR changes
nothing until an operator sets the knob.

## Why the knob exists

`files/chat_template.jinja` line 7:

```jinja
{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}
```

The value maps to itself only for `low` and `high`. Everything else — including **undefined** —
becomes `max`. So every client that omits `chat_template_kwargs` (OpenCode, plain `curl`, most
SDK defaults) silently gets the most expensive setting, and no serve flag says so.

That also fixes the enum: **`low | high | max` only**. `medium` is rejected at launch, because
the template does not recognize it and would render `Max` while the operator believed otherwise.

## A/B — unset(max) vs `high` (2026-09-01)

One frozen agentic build brief, run twice per arm on the live serve. Arm A reached the server
through a proxy injecting `chat_template_kwargs.reasoning_effort=high`; arm B hit the server
directly, i.e. **the server default, which on this template is `max`**. Server flags were not
changed and the server was not restarted. Grader is a frozen 80-point rubric.

| Run | Arm | Wall (s) | Grader | Turns | Tool calls | Compactions | Prompt tok | Completion tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A-1 | high | 498 | **80/80** | 29 | 30 | 0 | 562,519 | 13,856 |
| A-2 | high | 438 | **80/80** | 26 | 32 | 0 | 447,190 | 11,621 |
| B-1 | unset → max | 1,955 | **80/80** | 14 | 21 | 2 | 303,872 | 54,410 |
| B-2 | unset → max | 2,365 | **80/80** | 20 | 31 | 2 | 399,109 | 66,915 |

Medians (n = 2 per arm):

| Metric | high | unset → max | max ÷ high |
|---|---:|---:|---:|
| Grader score | 80.0 / 80 | 80.0 / 80 | 1.00 |
| Wall time | **468 s** | **2,160 s** | **4.61×** |
| Completion tokens | **12,739** | **60,663** | **4.76×** |
| Prompt tokens | 504,855 | 351,491 | 0.70× |
| Turns | 27.5 | 17 | 0.62× |
| Tool calls | 31 | 26 | 0.84× |
| Compactions | 0 | 2 | — |

Effective decode rate was comparable across arms (A-1 ≈ 27.8 completion tok/s, B-2 ≈ 28.3). That is
consistent with most of the wall-time gap being **extra generated reasoning**; it does not isolate
prompt-processing or tool latency. Every run was graded 80/80 and every run's own test suite passed, so
the normal tool-calling protocol was exercised at both settings (guided JSON / `response_format` was not).

**Recommendation: `high` for agentic coding.** Not `low` — a genuine `high`-vs-`low` quality A/B
has not been run, so `low` is not evidenced here. Arm B was **not** `low`; it was `max`.

## A/B v2 — `high` vs `low` (2026-09-01)

Same task, grader, harness and vLLM process as above; arm B now went through a `low`-injecting proxy.
Four runs, alternating A,B,A,B, one at a time, server never reconfigured.

| Run | Arm | Wall (s) | Grader | Turns | Tool calls (errors) | Compactions | Prompt tok | Completion tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A-1 | high | 486 | **80/80** | 37 | 36 (0) | 0 | 639,301 | 13,054 |
| B-1 | low | 297 | **79/80** | 19 | 21 (1, self-recovered) | 0 | 285,585 | 8,478 |
| A-2 | high | 700 | **80/80** | 27 | 35 (0) | 0 | 733,450 | 20,028 |
| B-2 | low | 208 | **80/80** | 11 | 15 (0) | 0 | 146,586 | 6,004 |

Medians: wall 593 s vs 252.5 s, completion tokens 16,541 vs 7,241, grader 80.0 vs 79.5 (the lost point was a
41-line README against a 40-line limit). A tool-call canary (`opencode run` with a `read` tool) passed at both efforts.
Across v1 + v2 the observed wall-time medians ordered low 252 s < high 468–593 s < max 2,160 s, with quality
at or near the grader ceiling in every arm.

Caveats that apply to both A/Bs: n = 2 per arm, one task, and a grader at or near its ceiling — these small runs
primarily measure **cost**, not reasoning quality. `low` visibly does less exploration (half the turns and tool calls). That is why the
recommendation stays `high`, and why `low` is documented as legal but not recommended.

## Live receipts (captured 2026-09-01)

Captured read-only on the live 2× DGX Spark serve — image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`
(`sha256:ad0cdd86…`), vLLM `0.1.dev20051+g487ecf187`, launched with `GLM53_DEFAULT_REASONING_EFFORT=high` in `.env`.
No restart, no chat completions.

### 1. The flag exists in the target image

`vllm serve --help` (short page) hides it; `--help=all` lists it:

```
$ docker exec glm53-exl3-head vllm serve --help=all 2>&1 | grep -n -A2 default-chat-template-kwargs
80:  --default-chat-template-kwargs DEFAULT_CHAT_TEMPLATE_KWARGS
81-                        Should either be a valid JSON string or JSON keys
82-                        passed individually. (default: None)
```

Source in this vLLM: `vllm/entrypoints/openai/cli_args.py` and `vllm/entrypoints/openai/serving.py`, where server
defaults are merged first and **request** `chat_template_kwargs` win.

- [x] captured

### 2. Both ranks carry the flag

The containers' `Cmd` is `bash /start.sh`, so `docker inspect` does not show the serve line; `vllm serve` is pid 1 in
each container and `/proc/1/cmdline` (NUL-split) is the parsed argv. Element index, next element, occurrence count:

```
== HEAD ==
host=spark-deb8 container=glm53-exl3-head image=ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3
vllm serve pid=1
42:--default-chat-template-kwargs
43-{"reasoning_effort":"high"}
occurrences=1
argc=58
== WORKER ==
host=spark-d9d5 container=glm53-exl3-worker image=ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3
vllm serve pid=1
43:--default-chat-template-kwargs
44-{"reasoning_effort":"high"}
occurrences=1
argc=59
```

Head API-server parsed config (the headless rank has no API server and prints no such line):

```
$ docker logs glm53-exl3-head 2>&1 | grep -i default_chat_template_kwargs
(APIServer pid=1) INFO 09-01 01:03:44 [api_utils.py:273] non-default args: {… 'chat_template': '/opt/glm53/chat_template.jinja', 'default_chat_template_kwargs': {'reasoning_effort': 'high'}, 'enable_auto_tool_choice': True, 'tool_call_parser': 'glm47', … 'reasoning_parser': 'glm45', … 'nnodes': 2, 'tensor_parallel_size': 2, …}
```

- [x] head captured
- [x] worker captured

### 3. Render boundary — the default reaches Jinja, and a request overrides it

`/tokenize` renders the template; because `token_strs` are tokenizer pieces, each id list was passed through
`/detokenize` for the literal prompt. Served template md5 `37639425…` == `files/chat_template.jinja`.

```
# A: no chat_template_kwargs -> server default
RESP: {"count":13,…,"token_strs":["[gMASK]","<sop>","<|system|>","Reason","ing","ĠEff","ort",":","ĠHigh","<|user|>","hi","<|assistant|>","<think>"]}
DETOK: {"prompt":"[gMASK]<sop><|system|>Reasoning Effort: High<|user|>hi<|assistant|><think>"}

# B: "chat_template_kwargs":{"reasoning_effort":"low"} -> request override wins
DETOK: {"prompt":"[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>hi<|assistant|><think>"}

# C: "chat_template_kwargs":{"reasoning_effort":"max"} -> request override wins
DETOK: {"prompt":"[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>"}

# D: "chat_template_kwargs":{"reasoning_effort":"medium"} -> no such level in the template, renders Max
token_strs: [… "ĠMax" …]
```

Only token position 9 differs between A/B/C: `5124` (`ĠHigh`), `12035` (`ĠLow`), `7487` (`ĠMax`).

**Before state (knob empty).** Not captured live — it needs a serve launched without the knob, i.e. a restart, and the
box was in use. Substitute: the served template file rendered offline inside the same container with jinja2
(`jinja2.ext.loopcontrols`, as `tests/test_chat_template.py` does), which is the branch a no-flag serve takes:

```
 <no kwargs> -> '[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>'
         low -> '[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>hi<|assistant|><think>'
        high -> '[gMASK]<sop><|system|>Reasoning Effort: High<|user|>hi<|assistant|><think>'
         max -> '[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>'
      medium -> '[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>'
```

The A/B v1 "direct" arm was measured against this very serve before the knob existed and behaved as `max`.

- [ ] before (knob empty): request A renders `Max` — **not captured live** (offline render above instead)
- [x] after (knob `high`): request A renders `High`
- [x] after (knob `high`): request B renders `Low` — request override wins (`max` override also shown)

### 4. Canaries at the chosen effort

Effort changes template text only, not `<think>` or tool-call grammar. Evidence from the A/B runs (all through the
live serve, all with `reasoning_content` streamed before content):

- [x] reasoning extraction: every A/B run streamed `reasoning` deltas ahead of content (this is why TTFT-to-content is
      longer at `high`, see the v2 results' caveat 5)
- [x] tool calling: 71/71 tool calls succeeded at `high` across v2; canary `read` tool call completed at `high` and `low`
- [ ] JSON validity under `response_format` / guided-JSON at the chosen effort: **not run** — nothing in the A/B used
      guided decoding. The effort hint is prompt text and does not touch the grammar path, but this is stated, not shown.

## Host tests (no hardware)

```
tests/test_default_reasoning_effort.sh    enum guard + both serve-arg sites (27 checks)
tests/test_numeric_config.py              unchanged, still green
tests/test_start_overrides.py             + setness-aware caller override
```
