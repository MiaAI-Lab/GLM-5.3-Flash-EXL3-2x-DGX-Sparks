#!/usr/bin/env python3
"""Codex / ChatGPT responses API compatibility patch for GLM-5.3-Flash EXL3.

Applied at build (Dockerfile RUN) and at container start (mount to /opt/glm53/).
Idempotent: checks for patch markers before applying.

Three fixes:
1. api_server.py — ASGI middleware that normalizes Codex-private item types
   BEFORE pydantic validation (agent_message, custom_tool_call, web_search_call,
   compaction, encrypted_content, etc.). The middleware _receive function
   correctly returns http.disconnect after the request body to avoid a
   busy-loop deadlock.
2. resp_utils.py — _construct_message_from_response_item handles dict-form
   items (agent_message → assistant, web_search_call → text, etc.) and
   ensures every message has a 'role' field.
3. resp_protocol.py — widen ResponsesRequest.input union to accept dict
   items alongside typed ResponseInputOutputItem.
"""
import ast, json, os, shutil, sys, time

VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"

# ── Fix 1: api_server.py ASGI middleware ───────────────────────────────
API_SERVER = os.path.join(VLLM, "entrypoints/openai/api_server.py")
src = open(API_SERVER, encoding="utf-8").read()

if "_CodexCompatMiddleware" not in src:
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = API_SERVER + ".bak-codex-compat-" + ts
    shutil.copy2(API_SERVER, bak)
    print("api_server backup ->", bak)

    # The middleware class + register before ScalingMiddleware
    mw_code = """

import json as _cc_json

class _CodexCompatMiddleware:
    '''Normalize Codex-private /v1/responses input items BEFORE pydantic.'''

    OUT_CAP = 40000

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope.get("type") == "http" and scope.get("method") == "POST"
                and scope.get("path", "").rstrip("/").endswith("/v1/responses")):
            body = b""
            while True:
                msg = await receive()
                if msg["type"] == "http.request":
                    body += msg.get("body", b"")
                    if not msg.get("more_body"):
                        break
                elif msg["type"] == "http.disconnect":
                    return
                else:
                    break

            new_body = body
            try:
                data = _cc_json.loads(body.decode("utf-8"))
                changed = False
                if isinstance(data, dict):
                    items = data.get("input")
                    if isinstance(items, list):
                        new_items = []
                        n_agent = 0; n_enc = 0; n_flat = 0
                        for it in items:
                            if not isinstance(it, dict):
                                new_items.append(it); continue
                            t = it.get("type")
                            if t == "agent_message":
                                n_agent += 1
                                txt = "\\n".join(
                                    (c.get("text") or "") for c in (it.get("content") or []) if isinstance(c, dict))
                                new_items.append({"type": "message", "role": "assistant",
                                                  "content": [{"type": "output_text", "text": txt}]})
                                changed = True
                            elif t == "custom_tool_call":
                                new_items.append({"type": "function_call", "id": it.get("id"),
                                                  "call_id": it.get("call_id"),
                                                  "name": it.get("name") or "tool",
                                                  "arguments": _cc_json.dumps(it.get("input") or {})})
                                changed = True
                            elif t == "custom_tool_call_output":
                                out = it.get("output")
                                if not isinstance(out, str): out = _cc_json.dumps(out, ensure_ascii=False)
                                if len(out) > self.OUT_CAP: out = out[:self.OUT_CAP] + " ...[truncated %d chars]" % (len(out) - self.OUT_CAP)
                                new_items.append({"type": "function_call_output", "call_id": it.get("call_id"), "output": out})
                                changed = True
                            elif t == "web_search_call":
                                act = it.get("action") if isinstance(it.get("action"), dict) else {}
                                q = str(act.get("query") or "")[:200]
                                new_items.append({"type": "message", "role": "user",
                                                  "content": [{"type": "input_text", "text": "(web search: %s)" % ("'" + q + "'" if q else "")}]})
                                changed = True
                            elif t in ("compaction", "item_reference"):
                                new_items.append({"type": "message", "role": "user",
                                                  "content": [{"type": "input_text", "text": "(history archived/compacted; continue from here)"}]})
                                changed = True
                            else:
                                if isinstance(it.get("encrypted_content"), str):
                                    it.pop("encrypted_content", None); n_enc += 1; changed = True
                                new_items.append(it)
                            n_flat += 1
                        if changed:
                            data["input"] = new_items
                    inc = data.get("include")
                    if isinstance(inc, list):
                        data["include"] = [x for x in inc if "encrypted_content" not in str(x)]
                        if not data["include"]: data.pop("include", None)
                        changed = True
                    if not data.get("max_output_tokens"):
                        data["max_output_tokens"] = 32768; changed = True
                    if changed:
                        new_body = _cc_json.dumps(data, ensure_ascii=False).encode("utf-8")
            except Exception:
                new_body = body

            headers = [(k, v) for (k, v) in scope.get("headers", []) if k.lower() != b"content-length"]
            headers.append((b"content-length", str(len(new_body)).encode("ascii")))
            scope = dict(scope)
            scope["headers"] = headers

            _receive_sent = False
            async def _receive():
                nonlocal _receive_sent
                if not _receive_sent:
                    _receive_sent = True
                    return {"type": "http.request", "body": new_body, "more_body": False}
                return {"type": "http.disconnect"}

            await self.app(scope, _receive, send)
            return
        await self.app(scope, receive, send)
"""

    anchors = ["def build_app(", "app.add_middleware(ScalingMiddleware)"]
    for a in anchors:
        if a not in src:
            print("!! anchor not found:", a); sys.exit(1)

    src = src.replace("def build_app(", mw_code.lstrip("\n") + "\n\ndef build_app(", 1)
    src = src.replace("app.add_middleware(ScalingMiddleware)",
                      "app.add_middleware(ScalingMiddleware)\n    app.add_middleware(_CodexCompatMiddleware)", 1)
    ast.parse(src)
    open(API_SERVER, "w", encoding="utf-8").write(src)
    print("api_server: patched OK")
else:
    print("api_server: already patched")

# ── Fix 2: resp_utils.py converter ─────────────────────────────────────
UTILS = os.path.join(VLLM, "entrypoints/openai/responses/utils.py")
src = open(UTILS, encoding="utf-8").read()
if "__codex_compat_patched" not in src:
    # encrypted_content: strip instead of raise
    old_enc = '        if item.encrypted_content:\n            raise ValueError("Encrypted content is not supported.")\n        elif item.content'
    new_enc = '        if item.encrypted_content:\n            pass\n        if item.content'
    if old_enc in src:
        src = src.replace(old_enc, new_enc)
        print("resp_utils: encrypted_content stripped")
    # agent_message + unknown dict fallback (before final return item)
    old_fallback = "    return item  # type: ignore[arg-type]"
    new_fallback = """    if isinstance(item, dict):
        t = item.get("type")
        if t == "agent_message":
            parts = item.get("content", [])
            if isinstance(parts, list):
                txt = "\\n".join((p.get("text") or p.get("content") or "") for p in parts if isinstance(p, dict))
            else: txt = str(parts)
            return {"role": "assistant", "content": txt}
        if t in ("web_search_call", "computer_call", "function_call", "custom_tool_call"):
            args = item.get("arguments") or item.get("input") or item.get("action") or ""
            nm = item.get("name") or item.get("tool_name") or ""
            return {"role": "assistant", "content": "[Tool call: %s(%s)]" % (nm, str(args)[:200])}
        if t in ("function_call_output", "custom_tool_call_output", "web_search_output"):
            out = item.get("output") or str(item.get("content", ""))
            if isinstance(out, (dict, list)): out = str(out)
            return {"role": "tool", "content": str(out)[:4000], "tool_call_id": item.get("call_id", "call_unknown")}
        if "role" not in item:
            return {"role": "user", "content": str(item.get("content", json.dumps(item))[:2000])}
    return item  # __codex_compat_patched"""
    if old_fallback in src:
        src = src.replace(old_fallback, new_fallback)
        print("resp_utils: fallback patched")
    if "import json" not in src.splitlines()[0]:
        src = "import json\n" + src
    ast.parse(src)
    open(UTILS, "w", encoding="utf-8").write(src)
    print("resp_utils: patched OK")
else:
    print("resp_utils: already patched")

# ── Fix 3: resp_protocol.py widen input union ──────────────────────────
PROTO = os.path.join(VLLM, "entrypoints/openai/responses/protocol.py")
psrc = open(PROTO, encoding="utf-8").read()
if "__codex_compat_patched" not in psrc:
    old_input = "    input: str | list[ResponseInputOutputItem]"
    new_input = "    input: str | list[ResponseInputOutputItem | dict]  # __codex_compat_patched"
    if old_input in psrc:
        psrc = psrc.replace(old_input, new_input)
        ast.parse(psrc)
        open(PROTO, "w", encoding="utf-8").write(psrc)
        print("resp_protocol: patched OK")
    else:
        print("resp_protocol: already patched?")
else:
    print("resp_protocol: already patched")

print("CODEX_COMPAT_PATCH DONE")
