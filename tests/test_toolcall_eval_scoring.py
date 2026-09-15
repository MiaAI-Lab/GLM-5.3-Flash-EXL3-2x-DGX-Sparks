#!/usr/bin/env python3
"""CPU-only fixture tests for scripts/quality/toolcall_eval.py scoring.

Completions are built by hand and scored through the pure scorer: no endpoint, no model call and no
network. They pin the two multi-turn Edit cases to the intended relative file_path and to the result
of applying the candidate's old_string -> new_string literally to the file content the case supplied
(any Edit call whose old_string appeared in the file used to be enough), and pin absent, empty and
non-object argument payloads to recorded invalid-argument failures instead of a predicate crash.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QUALITY = ROOT / "scripts" / "quality"
TOOLCALL = QUALITY / "toolcall_eval.py"

SPEC = importlib.util.spec_from_file_location("toolcall_eval", TOOLCALL)
te = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(te)

CASES = {c["id"]: c for c in te.cases()}
SRC = (ROOT / "overlay" / "exl3.py").read_text().splitlines()
# the line the edit_deep_18k prompt points at, read from the case itself so the fixture cannot drift
DEEP_LN = int(re.search(r"line (\d+)", CASES["edit_deep_18k"]["messages"][-1]["content"]).group(1))
DEEP_LINE = SRC[DEEP_LN - 1]


def completion(tool=None, arguments=None, content="", finish_reason="tool_calls"):
    """A /v1/chat/completions body as the endpoint returns it; arguments is the raw payload string."""
    tcs = [] if tool is None else [{"id": "call_1", "type": "function", "function": {"name": tool, "arguments": arguments}}]
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tcs}, "finish_reason": finish_reason}],
            "usage": {"completion_tokens": 12, "prompt_tokens": 4096}}


def scored(case_id, **kw):
    return te.score_completion(CASES[case_id], completion(**kw))


def check_edit_fixtures():
    """Intended relative file_path and the reference result, for both multi-turn Edit cases."""
    fix = {"file_path": "utils.py", "old_string": "    return items[-n+1:]", "new_string": "    return items[-n:]"}
    r = scored("fix_after_read", tool="Edit", arguments=json.dumps(fix))
    assert r["ok"] and not r["bad_json"], r
    # a full-function replacement that produces the reference correction is still a valid edit
    r = scored("fix_after_read", tool="Edit", arguments=json.dumps(dict(fix, old_string=te.BUGGY, new_string=te.FIXED)))
    assert r["ok"] and not r["bad_json"], r
    # a leading ./ is harmless normalization; nothing else names the requested file
    r = scored("fix_after_read", tool="Edit", arguments=json.dumps(dict(fix, file_path="./utils.py")))
    assert r["ok"] and not r["bad_json"], r
    for path in ("other.py", "notutils.py", "sub/utils.py", "/repo/utils.py"):
        r = scored("fix_after_read", tool="Edit", arguments=json.dumps(dict(fix, file_path=path)))
        assert not r["ok"], (path, r)
    r = scored("fix_after_read", tool="Edit", arguments=json.dumps(dict(fix, new_string="    return items[-n+2:]")))
    assert not r["ok"], r
    # a replacement that destroys the buggy line and only mentions the fixed token is not the correction
    r = scored("fix_after_read", tool="Edit", arguments=json.dumps(dict(fix, new_string="    # items[-n:]")))
    assert not r["ok"], r
    # an unrelated no-op edit of the same file used to score ok: old_string only had to be a substring
    noop = {"file_path": "utils.py", "old_string": '    """Return the last n items of the list."""',
            "new_string": '    """Return the last n items of the list."""'}
    r = scored("fix_after_read", tool="Edit", arguments=json.dumps(noop))
    assert not r["ok"], r
    # an old_string matching two places cannot be applied unambiguously
    assert te.applied_edit("return x\nreturn x\n", "return x", "return y") is None

    deep = {"file_path": "overlay/exl3.py", "old_string": DEEP_LINE, "new_string": DEEP_LINE + "  # reviewed"}
    r = scored("edit_deep_18k", tool="Edit", arguments=json.dumps(deep))
    assert r["ok"] and not r["bad_json"], r
    # the same edit under ./ and as one full-file context both produce that reference result
    r = scored("edit_deep_18k", tool="Edit", arguments=json.dumps(dict(deep, file_path="./overlay/exl3.py")))
    assert r["ok"] and not r["bad_json"], r
    whole = list(SRC); whole[DEEP_LN - 1] = DEEP_LINE + "  # reviewed"
    r = scored("edit_deep_18k", tool="Edit", arguments=json.dumps({"file_path": "overlay/exl3.py", "old_string": "\n".join(SRC),
                                                                   "new_string": "\n".join(whole)}))
    assert r["ok"] and not r["bad_json"], r
    for path in ("overlay/other.py", "overlay/notexl3.py", "notexl3.py", "/repo/overlay/exl3.py"):
        r = scored("edit_deep_18k", tool="Edit", arguments=json.dumps(dict(deep, file_path=path)))
        assert not r["ok"], (path, r)
    r = scored("edit_deep_18k", tool="Edit", arguments=json.dumps(dict(deep, new_string=DEEP_LINE)))
    assert not r["ok"], r
    # appending the comment to the next line instead of the requested one used to score ok
    nxt = SRC[DEEP_LN]
    r = scored("edit_deep_18k", tool="Edit",
               arguments=json.dumps({"file_path": "overlay/exl3.py", "old_string": f"{DEEP_LINE}\n{nxt}",
                                     "new_string": f"{DEEP_LINE}\n{nxt}  # reviewed"}))
    assert not r["ok"], r
    # dropping the neighbouring line and keeping only the commented one destroys content the edit preserves
    r = scored("edit_deep_18k", tool="Edit",
               arguments=json.dumps({"file_path": "overlay/exl3.py", "old_string": f"{DEEP_LINE}\n{nxt}",
                                     "new_string": DEEP_LINE + "  # reviewed"}))
    assert not r["ok"], r
    print("edit fixtures OK (intended relative file_path and reference result; other files/paths/replacements fail)")


def check_invalid_arguments():
    """Absent, empty and non-object payloads are recorded failures; no checker runs and none raises."""
    r = scored("read_main", tool="Read", arguments="null")
    assert r["ok"] is False and r["bad_json"] is True and r["args"] == "null", r
    # an absent or empty payload is not an empty object: it is an invalid-argument failure
    for raw in (None, "", "   ", "[]", "[1, 2]", "17", '"src/main.py"', "{not json"):
        r = scored("read_main", tool="Read", arguments=raw)
        assert r["ok"] is False and r["bad_json"] is True, (raw, r)
    # an explicit JSON object payload stays valid, the empty object included
    r = scored("bash_after_long", tool="Bash", arguments="{}")
    assert r["ok"] is True and r["bad_json"] is False, r
    r = scored("read_main", tool="Read", arguments='{"file_path": "src/main.py"}')
    assert r["ok"] is True and r["bad_json"] is False, r
    # a checker that ignores the arguments (any Bash call) must not score an invalid payload ok either
    for raw in ("null", "[]", None, ""):
        r = scored("bash_after_long", tool="Bash", arguments=raw)
        assert r["ok"] is False and r["bad_json"] is True, (raw, r)
    print("invalid arguments OK (absent/empty/non-object recorded as failed samples; object payloads valid)")


def check_rescore_record():
    """--rescore rewrites from the recorded payload: invalid payloads stay failed samples, a no-tool record stays legitimate."""
    with tempfile.TemporaryDirectory() as d:
        stored = Path(d) / "stored.json"
        out = Path(d) / "out.json"
        stored.write_text(json.dumps({"results": [
            {"id": "read_main", "sample": 0, "tags": [], "ok": True, "why": "want Read", "tool": "Read",
             "n_tool_calls": 1, "bad_json": False, "leak": False, "args": "null", "content": ""},
            {"id": "read_main", "sample": 1, "tags": [], "ok": True, "why": "want Read", "tool": "Read",
             "n_tool_calls": 1, "bad_json": False, "leak": False, "args": json.dumps({"file_path": "src/main.py"}),
             "content": ""},
            # a recorded tool call with no payload is not a no-tool sample: it is an invalid-argument failure
            {"id": "read_main", "sample": 2, "tags": [], "ok": True, "why": "want Read", "tool": "Read",
             "n_tool_calls": 1, "bad_json": False, "leak": False, "args": None, "content": ""},
            {"id": "bash_after_long", "sample": 0, "tags": ["long"], "ok": True, "why": "want Bash", "tool": "Bash",
             "n_tool_calls": 1, "bad_json": False, "leak": False, "args": "", "content": ""},
            {"id": "no_tool", "sample": 0, "tags": [], "ok": True, "why": "want plain answer", "tool": None,
             "n_tool_calls": 0, "bad_json": False, "leak": False, "args": None, "content": "Hello."},
        ]}))
        p = subprocess.run([sys.executable, str(TOOLCALL), str(out), "--rescore", str(stored)],
                           capture_output=True, text=True)
        assert p.returncode == 0, (p.returncode, p.stdout, p.stderr)
        written = json.loads(out.read_text())
        assert [r["ok"] for r in written["results"]] == [False, True, False, False, True], written["results"]
        assert [r["bad_json"] for r in written["results"]] == [True, False, True, True, False], written["results"]
        assert written["results"][2]["why"] == "tool arguments are not a JSON object", written["results"][2]
        assert written["summary"]["total"]["bad_json"] == 3, written["summary"]["total"]
    print("rescore OK (invalid payloads stay failed samples; a no-tool record stays legitimate)")


def main() -> int:
    check_edit_fixtures()
    check_invalid_arguments()
    check_rescore_record()
    print("toolcall_eval scoring fixtures OK (reference edit results and invalid argument payloads)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
