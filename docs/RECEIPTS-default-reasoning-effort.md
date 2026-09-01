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

Effective decode rate was comparable across arms (A-1 ≈ 27.8 completion tok/s, B-2 ≈ 28.3), so
essentially the whole wall-time gap is **extra generated reasoning**, not slower serving.
Every run was graded 80/80 and every run's own test suite passed, so tool-calling and structured
output were exercised at both settings.

**Recommendation: `high` for agentic coding.** Not `low` — a genuine `high`-vs-`low` quality A/B
has not been run, so `low` is not evidenced here. Arm B was **not** `low`; it was `max`.

## Live receipts to capture on the box

Run these against the target image / running serve and paste the output under each item.

### 1. The flag exists in the target image

```bash
docker exec glm53-exl3-head vllm serve --help | grep -A2 default-chat-template-kwargs
```

Expect the option to be listed. Source in this vLLM: `vllm/entrypoints/openai/cli_args.py`
(:93, :167, :200-201) and `vllm/entrypoints/openai/serving.py` (:131, :146), where server
defaults are merged first and **request** `chat_template_kwargs` win.

- [ ] captured

### 2. Both ranks carry the flag

```bash
docker inspect --format '{{.Args}}' glm53-exl3-head
ssh "$WORKER_SSH" "docker inspect --format '{{.Args}}' glm53-exl3-worker"
```

Both must show `--default-chat-template-kwargs {"reasoning_effort":"high"}` exactly once, as a
single argv element. (The launcher builds the JSON space-free precisely so it survives the
word-split `${worker_nccl}` env path to the worker.)

- [ ] head captured
- [ ] worker captured

### 3. Render boundary — the default reaches Jinja, and a request overrides it

The claim is about the rendered prompt, not the flag, so probe the renderer. `/tokenize` with
`return_token_strs` renders the chat template and hands back the text:

```bash
# A: no chat_template_kwargs -> must render "Reasoning Effort: High" (the server default)
curl -s localhost:8888/tokenize -H 'Content-Type: application/json' -d '{
  "model":"GLM-5.3-Flash-EXL3",
  "messages":[{"role":"user","content":"hi"}],
  "return_token_strs":true}' | python3 -c 'import json,sys; print("".join(json.load(sys.stdin)["token_strs"])[:200])'

# B: request override -> must render "Reasoning Effort: Low"
curl -s localhost:8888/tokenize -H 'Content-Type: application/json' -d '{
  "model":"GLM-5.3-Flash-EXL3",
  "messages":[{"role":"user","content":"hi"}],
  "chat_template_kwargs":{"reasoning_effort":"low"},
  "return_token_strs":true}' | python3 -c 'import json,sys; print("".join(json.load(sys.stdin)["token_strs"])[:200])'
```

Also record the **before** state (server launched with the knob empty): request A must render
`Reasoning Effort: Max`. That before/after pair is the whole proof — it shows both that the
default lands and that it was `max` beforehand.

If `/tokenize` does not surface the rendered text on this build, fall back to a
logprobs-of-prefix probe: send `max_tokens=1, prompt_logprobs=0` on the chat endpoint and read
the echoed prompt tokens.

- [ ] before (knob empty): request A renders `Max`
- [ ] after (knob `high`): request A renders `High`
- [ ] after (knob `high`): request B renders `Low` — request override wins

### 4. Canaries at the chosen effort

Per the Codex advisory, effort changes template text only, not `<think>` or tool-call grammar —
but confirm rather than assume, on the running serve with the default in place:

- [ ] reasoning extraction: a normal chat request still returns a populated `reasoning_content`
- [ ] tool calling: a request with `tools` still emits a well-formed `tool_calls` entry
- [ ] JSON validity: a `response_format` / guided-JSON request still parses

## Host tests (no hardware)

```
tests/test_default_reasoning_effort.sh    enum guard + both serve-arg sites
tests/test_numeric_config.py              unchanged, still green
tests/test_start_overrides.py             + setness-aware caller override
```
