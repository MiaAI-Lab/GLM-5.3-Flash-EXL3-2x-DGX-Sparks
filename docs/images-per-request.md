# Images per request vs. stateless chat clients — why the 5th image of a session fails, and two fixes

**Date:** 2026-09-05 · **Scope:** the `LIMIT_MM` knob (`--limit-mm-per-prompt`) on the GLM-5.3-Flash EXL3 2× DGX Spark
serve, as seen from an OpenAI-compatible chat client that resends the conversation on every turn (OpenCode, Open WebUI,
LibreChat, most agent harnesses). Reported from a stock deployment of this kit (2× DGX Spark, `.env` defaults plus
`LANGUAGE_MODEL_ONLY=0`).

---

## 1. The symptom

With the kit's default `LIMIT_MM='{"image":4,"video":1}'`, a chat session works for the first four images, then every
later turn — including text-only ones — fails with:

    At most 4 image(s) may be provided in one prompt. (parameter=image)

The client shows it as an error bubble; the session never recovers on its own.

## 2. Why

- Chat completions are stateless: the client sends the whole message history on every request, image parts included.
- vLLM enforces `--limit-mm-per-prompt` **per request**, not per session. Once the history holds five images, the
  history itself exceeds the limit, so the fifth image poisons the rest of the session.
- Prefix caching makes the resend cheap for the engine (unchanged image tokens hit the KV cache; the multimodal
  processor cache skips re-encoding a hashed image), but the limit is checked on the request as sent.

Checked straight against the head (`gx10a:8888`) with a synthetic six-image conversation: the request is rejected
with the message above. It is not a client bug.

## 3. Fix A — raise the limit in the recipe (measured)

`LIMIT_MM` is the knob. `.env`:

    LIMIT_MM='{"image":16,"video":1}'

then `./start.sh restart`. With the README's `SKIP_MM_PROFILING=1` (the default: a max-size image+video dummy profile
OOMs this UMA) the limit is not a boot-time reservation. Measured on this kit (vision on, `GPU_MEM_UTIL=0.875`,
`MAX_NUM_BATCHED_TOKENS=2048`, 1M):

| | `image: 4` (default) | `image: 16` |
|---|---:|---:|
| Available KV cache memory | 12.93 GiB | 12.89 GiB |
| GPU KV cache size | 1,234,299 tokens | 1,229,468 tokens |

One request carrying 16 distinct 1080×1920 screenshots: HTTP 200 in 63 s, all 16 read in order, `prompt_tokens`
43,136 (≈2.7k tokens per screenshot), head alive, no OOM. The head's host-side free memory sat at ~1 GB during that
request (it is 1–2 GB at rest with the vision tower on at util 0.875). That headroom is the real limit — see §6.

## 3b. What the default 4 actually encodes

The kit runs `MAX_NUM_SEQS=4`. Four requests at the default limit is 4 × 4 = **16 images in flight**, and 16 in one
request is exactly what the stress above put through with ~1 GB to spare. So the quantity this UMA is sized for is
images in flight across concurrent requests, not images per request: `MAX_NUM_SEQS × per-request limit ≈ 16` at
these settings. Raising `LIMIT_MM` to 16 with four sequences allows 64 in flight on paper; a runtime OOM there kills
the TP worker (no rejection, the kernel does it). Raise the limit for a single-user serve, or pair it with a
client-side budget (§4).

## 4. Fix B — trim at the client or a gateway (no restart, any limit)

If the serve sits behind a small OpenAI-compatible proxy (we run one for tool-call streaming repairs), cap images per
request there: keep the newest N image parts, replace the older ones with a short text note so the model knows an
image was attached and its own earlier reply about it still stands. Reference implementation (Python, FastAPI proxy;
~20 lines) — `_limita_imagenes` in
https://github.com/Capicua25x/apexia-toolstream-proxy/blob/master/apexia_toolstream_proxy.py — with per-model limits
in a JSON file so only models that enforce a limit get trimmed.

Behavior we measured through the proxy with N=4: a six-image conversation answers "I can still see four images (3–6)"
and reads the last one; the proxy logs `6 en la conversación, 2 omitidas`.

The same proxy also enforces the §3b invariant as an **images-in-flight budget** per upstream (default 16): a request
whose images would exceed the budget waits until an earlier one finishes, so colliding image turns cost seconds of
latency instead of a dead head. Measured: three concurrent ten-image conversations, trimmed to 8 each, the third
waited (`16 + 8 > 16 — esperando`), all three returned 200, head alive. The note text matters — with a terse
placeholder the model apologized for "missing" images; a note that says the image was attached, that only the newest
N are forwarded, and to ask for a resend only if needed, ends that.

## 5. Protocol to reproduce / verify (one small PNG, any content)

    B64=$(base64 -w0 test.png)
    python3 - "$B64" <<'PY' > six.json
    import json, sys
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + sys.argv[1]}}
    msgs = []
    for i in range(6):
        msgs += [{"role": "user", "content": [{"type": "text", "text": f"image {i+1}"}, img]},
                 {"role": "assistant", "content": f"Noted image {i+1}."}]
    msgs.append({"role": "user", "content": [{"type": "text", "text": "How many images can you see in this conversation? A number only."}]})
    json.dump({"model": "GLM-5.3-Flash-EXL3", "max_tokens": 600, "messages": msgs}, sys.stdout)
    PY
    curl -s http://<head>:8888/v1/chat/completions -H 'Content-Type: application/json' --data-binary @six.json

Default `.env`: the error. `LIMIT_MM` ≥ 6: the model answers `6`. Through a trimming proxy with N=4: `4` (asked to
count, it may still say `6` when the omitted slots carry a note that an image was attached — that is the note doing its job).
(`max_tokens` needs room for the thinking tokens; with 40 the answer comes back empty.)

## 6. Headroom note

With the vision tower on at `GPU_MEM_UTIL=0.875` this kit runs with 1–2 GB of host-side free memory (unified memory:
what the engine reserves, the host loses). Every image in flight borrows encoder activations from that slack, and
when it runs out the kernel kills `VLLM::Worker_TP` — we hit exactly that at util 0.90. The §3b rule and the §4
budget are the two ways to stay inside it without giving KV back.
