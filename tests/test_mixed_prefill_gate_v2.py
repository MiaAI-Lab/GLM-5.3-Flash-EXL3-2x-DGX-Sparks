#!/usr/bin/env python3
"""Host unit test for the v2 mixed-prefill gate (warm bypass + wall-clock deadline).

Extracts _glm53_mixed_prefill_policy/_glm53_mixed_prefill_gate from the overlay's HELPER text and exercises them with
stub requests. No vLLM import needed. Also applies the overlay to a copy of scheduler.py when GLM53_SCHEDULER_PY_SRC
points at one (container/CI), asserting both gate sites use the v2 helper.
"""
from __future__ import annotations
import os, re, shutil, subprocess, sys, tempfile, time, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = next(p for p in (HERE / "patch_scheduler_decode_floor.py", HERE.parent / "overlay" / "patch_scheduler_decode_floor.py") if p.is_file())

def load_helpers():
    text = PATCH.read_text()
    m = re.search(r"HELPER = '''(.*?)'''", text, re.S)
    assert m, "HELPER block not found"
    ns: dict = {"os": os}
    exec(m.group(1), ns)
    gate = ns["_glm53_mixed_prefill_gate"]

    def fresh_gate(*a, **k):  # knobs are parsed once per process; reset the cache so each case sees its env
        ns["_GLM53_GATE_CFG"] = None
        return gate(*a, **k)
    return fresh_gate, ns["_glm53_mixed_prefill_policy"]

class R(types.SimpleNamespace):
    pass

def req(rid, prompt, computed, arrival=None):
    r = R(request_id=rid, num_prompt_tokens=prompt, num_tokens=prompt, num_computed_tokens=computed)
    if arrival is not None:
        r._glm53_gate_first_seen = time.monotonic() - (time.time() - arrival)   # "first seen that long ago"
    return r

def main() -> int:
    gate, policy = load_helpers()
    decoding_peer = req("peer", 1000, 1000)          # a lane that finished prefill = decoding
    decoding_peer.num_tokens = 1200                   # it has generated 200 tokens
    env = os.environ
    env["GLM53_MIXED_PREFILL_CHUNK"] = "skip"; env.pop("GLM53_MIXED_PREFILL_WARM_TOKENS", None)
    env["GLM53_MIXED_PREFILL_MAX_WAIT_MS"] = "1500"; env["GLM53_MIXED_PREFILL_LATE_CAP"] = "512"
    # 1. no decoding peer -> no policy
    assert gate([], req("a", 50000, 0), 0) is None
    # 2. cold read behind a decoder, fresh arrival -> skip (0)
    assert gate([decoding_peer], req("a", 50000, 0), 0) == 0
    # 3. warm: remainder <= 3584 -> admit (None)
    assert gate([decoding_peer], req("a", 50000, 47000), 47000) is None
    assert gate([decoding_peer], req("a", 50000, 46416), 46416) is None      # exactly 3584 remaining
    assert gate([decoding_peer], req("a", 50000, 46415), 46415) == 0         # 3585 remaining -> still gated
    # 4. fully computed -> None
    assert gate([decoding_peer], req("a", 50000, 50000), 50000) is None
    # 5. late: first seen 2 s ago -> min(late cap, remaining); a first call stamps first_seen
    fresh = req("a", 50000, 0); assert gate([decoding_peer], fresh, 0) == 0 and hasattr(fresh, "_glm53_gate_first_seen")
    assert gate([decoding_peer], req("a", 50000, 0, arrival=time.time() - 2.0), 0) == 512
    os.environ["GLM53_MIXED_PREFILL_WARM_TOKENS"] = "100"
    assert gate([decoding_peer], req("a", 50000, 49700, arrival=time.time() - 2.0), 49700) == 300
    os.environ.pop("GLM53_MIXED_PREFILL_WARM_TOKENS")
    # 6. cap mode (N>0) unchanged and not overridden by late (cap already > 0)
    env["GLM53_MIXED_PREFILL_CHUNK"] = "256"
    assert gate([decoding_peer], req("a", 50000, 0), 0) == 256
    assert gate([decoding_peer], req("a", 50000, 0, arrival=time.time() - 5.0), 0) == 256
    # 7. policy off
    env["GLM53_MIXED_PREFILL_CHUNK"] = "0"
    assert gate([decoding_peer], req("a", 50000, 0), 0) is None
    # 8. deadline disabled (0) -> never late
    env["GLM53_MIXED_PREFILL_CHUNK"] = "skip"; env["GLM53_MIXED_PREFILL_MAX_WAIT_MS"] = "0"
    assert gate([decoding_peer], req("a", 50000, 0, arrival=time.time() - 60), 0) == 0
    # 9. warm threshold override
    env["GLM53_MIXED_PREFILL_MAX_WAIT_MS"] = "1500"; env["GLM53_MIXED_PREFILL_WARM_TOKENS"] = "100"
    assert gate([decoding_peer], req("a", 50000, 49950), 49950) is None
    assert gate([decoding_peer], req("a", 50000, 49800), 49800) == 0
    env.pop("GLM53_MIXED_PREFILL_WARM_TOKENS")
    # 10. self is excluded from peers
    me = req("a", 50000, 0)
    assert gate([me], me, 0) is None
    # 11. resumed (preempted) request: prompt cached, 4000 generated tokens to replay -> remainder 4000 > warm -> gated
    res = req("r", 50000, 50000); res.num_tokens = 54000
    assert gate([decoding_peer], res, 50000) == 0
    # 12. bad env values fall back to defaults; LATE_CAP below 64 rejected
    env["GLM53_MIXED_PREFILL_MAX_WAIT_MS"] = "abc"; env["GLM53_MIXED_PREFILL_LATE_CAP"] = "0"
    assert gate([decoding_peer], req("a", 50000, 0, arrival=time.time() - 2.0), 0) == 512
    env["GLM53_MIXED_PREFILL_MAX_WAIT_MS"] = "1500"; env["GLM53_MIXED_PREFILL_LATE_CAP"] = "512"
    # 13. Request that rejects attributes -> weak-keyed fallback, still becomes late
    class Slotted:
        __slots__ = ("request_id", "num_prompt_tokens", "num_tokens", "num_computed_tokens", "__weakref__")
    sl = Slotted(); sl.request_id = "s"; sl.num_prompt_tokens = 50000; sl.num_tokens = 50000; sl.num_computed_tokens = 0
    assert gate([decoding_peer], sl, 0) == 0
    time.sleep(0.01)
    os.environ["GLM53_MIXED_PREFILL_MAX_WAIT_MS"] = "5"; time.sleep(0.01)
    assert gate([decoding_peer], sl, 0) == 512
    os.environ["GLM53_MIXED_PREFILL_MAX_WAIT_MS"] = "1500"
    print("gate v2 helper logic OK (13 cases)")

    src = os.environ.get("GLM53_SCHEDULER_PY_SRC", "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")
    if Path(src).is_file():
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "scheduler.py"; shutil.copyfile(src, dst)
            e = os.environ.copy(); e["GLM53_SCHEDULER_PY"] = str(dst)
            subprocess.check_call([sys.executable, str(PATCH)], env=e)
            t = dst.read_text()
            assert t.count("[glm53-decode-floor-v2]") >= 2, t.count("[glm53-decode-floor-v2]")
            assert "def _glm53_mixed_prefill_gate(" in t
            assert "_glm53_mixed_prefill_gate(self.running, request, request.num_computed_tokens)" in t
            assert "_glm53_mixed_prefill_gate(self.running, request, num_computed_tokens)" in t
            subprocess.check_call([sys.executable, str(PATCH)], env=e)   # idempotent
            assert dst.read_text() == t
        print("scheduler.py apply/idempotence OK")
    else:
        print("scheduler.py source not present; apply test skipped")
    return 0

if __name__ == "__main__":
    sys.exit(main())
