#!/usr/bin/env python3
"""Bounded GLM quality gate: never executes generated tools or code."""
import argparse, base64, copy, json, struct, time, urllib.request, zlib

URL = "http://127.0.0.1:8888/v1/chat/completions"
TIMEOUT = 90
TRACES = []
TOOLS = [
 {"type":"function","function":{"name":"weather","description":"weather lookup","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"unit":{"type":"string"}},"required":["city","days","unit"]}}},
 {"type":"function","function":{"name":"reminder","description":"save a reminder","parameters":{"type":"object","properties":{"title":{"type":"string"},"when":{"type":"string"},"urgent":{"type":"boolean"}},"required":["title","when","urgent"]}}},
 {"type":"function","function":{"name":"add","description":"add integers","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}},
]
CASES = [
 ("weather", {"city":"Paris","days":3,"unit":"celsius"}, {"city":"Paris","forecast":"sunny"}, ["paris","sunny"], 0.0),
 ("reminder", {"title":"Call dentist","when":"2026-10-01T09:30:00","urgent":False}, {"id":"r-100","title":"Call dentist"}, ["r-100","dentist"], 0.0),
 ("add", {"a":17,"b":25}, {"sum":42}, ["42"], 0.6),
]

def png():
    row = b"\0" + (b"\xff\0\0" * 64) + (b"\0\0\xff" * 64)
    raw = row * 64
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 128, 64, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

def request(payload):
    payload.setdefault("chat_template_kwargs", {"enable_thinking":False})
    trace = {"request":copy.deepcopy(payload)}
    try:
        req = urllib.request.Request(URL, json.dumps(payload).encode(), {"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            body = json.loads(response.read())
        trace["response"] = body
        return body
    except Exception as exc:
        trace["error"] = repr(exc)
        raise
    finally:
        TRACES.append(trace)

def same(actual, expected):
    if type(actual) is not type(expected): return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(same(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list): return len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    return actual == expected

def message(response):
    choices = response.get("choices") or []
    if not choices or choices[0].get("finish_reason") == "length": raise ValueError("empty or truncated completion")
    value = choices[0].get("message")
    if not isinstance(value, dict): raise ValueError("missing message")
    return value

def object_only(text):
    text = (text or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict): raise ValueError("expected JSON object only")
    return value

def tool_case(name, args, result, required_words, temperature):
    prompt = "Call the required tool with exactly these JSON arguments; no extra fields: " + json.dumps(args, separators=(",", ":"))
    first = request({"model":"GLM-5.3-Flash-EXL3","messages":[{"role":"user","content":prompt}],"tools":TOOLS,"tool_choice":{"type":"function","function":{"name":name}},"temperature":temperature,"max_tokens":300})
    assistant = message(first); calls = assistant.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != name: raise ValueError("wrong tool name/count")
    if not isinstance(calls[0].get("id"), str) or not calls[0]["id"]: raise ValueError("missing tool call id")
    actual = json.loads(calls[0]["function"].get("arguments", ""))
    if not same(actual, args): raise ValueError("tool JSON keys/types/values mismatch")
    replay = [{"role":"user","content":prompt}, assistant, {"role":"tool","tool_call_id":calls[0].get("id"),"content":json.dumps(result)}, {"role":"user","content":"Summarize the synthetic tool result in one sentence, naming every returned field."}]
    second = request({"model":"GLM-5.3-Flash-EXL3","messages":replay,"temperature":temperature,"max_tokens":300})
    final = message(second); text = final.get("content") or ""
    if final.get("tool_calls") or not text.strip() or any(word not in text.lower() for word in required_words): raise ValueError("invalid replay summary")
    return {"name":name,"ok":True}

def clamp_case():
    prompt = "Python task: clamp each input [-5, 7, 18] to the inclusive range [0, 10]. Return only a JSON object with a results array."
    response = request({"model":"GLM-5.3-Flash-EXL3","messages":[{"role":"user","content":prompt}],"temperature":0,"max_tokens":300})
    if not same(object_only(message(response).get("content")), {"results":[0,7,10]}): raise ValueError("clamp correctness mismatch")
    return {"name":"code_clamp","ok":True}

def image_case():
    image = "data:image/png;base64," + base64.b64encode(png()).decode()
    prompt = "Identify the dominant colors in the left and right halves. Return JSON with keys left and right, lowercase color names."
    content = [{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":image}}]
    response = request({"model":"GLM-5.3-Flash-EXL3","messages":[{"role":"user","content":content}],"temperature":0,"max_tokens":300})
    if not same(object_only(message(response).get("content")), {"left":"red","right":"blue"}): raise ValueError("image quadrant mismatch")
    return {"name":"image_quadrants","ok":True}

def selfcheck():
    assert png().startswith(b"\x89PNG") and len(png()) > 100
    assert same({"b":2,"a":1}, {"a":1,"b":2})
    assert same({"x":False}, {"x":False}) and not same({"x":0}, {"x":False})
    assert object_only('```json\n{"x":1}\n```') == {"x":1}
    try: object_only('answer: {"x":1}')
    except json.JSONDecodeError: pass
    else: raise AssertionError("arbitrary-prefix JSON accepted")
    print("offline self-check passed")

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out"); parser.add_argument("--selfcheck", action="store_true"); ns = parser.parse_args()
    if ns.selfcheck: selfcheck(); return
    if not ns.out: parser.error("--out is required unless --selfcheck")
    started, results = time.time(), []
    for case in CASES:
        try: results.append(tool_case(*case))
        except Exception as exc: results.append({"name":case[0],"ok":False,"error":repr(exc)})
    for fn in (clamp_case, image_case):
        try: results.append(fn())
        except Exception as exc: results.append({"name":fn.__name__,"ok":False,"error":repr(exc)})
    report = {"url":URL,"timeout_seconds":TIMEOUT,"elapsed_seconds":round(time.time()-started,2),"results":results,"traces":TRACES,"ok":all(x["ok"] for x in results)}
    with open(ns.out, "w", encoding="utf-8") as handle: json.dump(report, handle, indent=2)
    print(json.dumps({"ok":report["ok"],"out":ns.out})); raise SystemExit(0 if report["ok"] else 1)

if __name__ == "__main__": main()