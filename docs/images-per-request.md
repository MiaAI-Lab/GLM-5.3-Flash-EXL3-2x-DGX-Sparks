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

## 3. Fix A — raise the limit in the recipe

`LIMIT_MM` is the knob. `.env`:

    LIMIT_MM='{"image":16,"video":1}'

then `./start.sh restart`. With the README's `SKIP_MM_PROFILING=1` (the default: a max-size image+video dummy profile
OOMs this UMA), the limit is not a boot-time reservation, so the KV pool line (`GPU KV cache size: …`) should read the
same as before; the encoder cost is paid per request for the images actually sent. Verify after the restart with the
six-image protocol in §5. Pick the number from your clients' habits: a coding agent that attaches screenshots all day
will reach 16 in an afternoon.

## 4. Fix B — trim at the client or a gateway (no restart, any limit)

If the serve sits behind a small OpenAI-compatible proxy (we run one for tool-call streaming repairs), cap images per
request there: keep the newest N image parts, replace the older ones with a short text note so the model knows an
image was attached and its own earlier reply about it still stands. Reference implementation (Python, FastAPI proxy;
~20 lines) — `_limita_imagenes` in
https://github.com/Capicua25x/apexia-toolstream-proxy/blob/master/apexia_toolstream_proxy.py — with per-model limits
in a JSON file so only models that enforce a limit get trimmed.

Behavior we measured through the proxy with N=4: a six-image conversation answers "I can still see four images (3–6)"
and reads the last one; the proxy logs `6 en la conversación, 2 omitidas`. The note text matters — with a terse
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

Default `.env`: the error. `LIMIT_MM` ≥ 6: the model answers `6`. Through a trimming proxy with N=4: `4`.
(`max_tokens` needs room for the thinking tokens; with 40 the answer comes back empty.)
