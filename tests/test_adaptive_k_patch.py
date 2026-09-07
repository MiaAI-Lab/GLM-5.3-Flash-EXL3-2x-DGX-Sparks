#!/usr/bin/env python3
"""Apply overlay/patch_adaptive_k.py to copies of scheduler.py / cudagraph_utils.py
and unit-test the EMA policy (default off, batch minimum, structured, ratchet escape)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = HERE.parent / "overlay" / "patch_adaptive_k.py"
SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SCHED_SRC = Path(os.environ.get("GLM53_SCHEDULER_PY_SRC", SITE / "v1/core/sched/scheduler.py"))
CG_SRC = Path(os.environ.get("GLM53_CUDAGRAPH_UTILS_PY_SRC", SITE / "v1/worker/gpu/cudagraph_utils.py"))


class _Req:
    def __init__(self, rid, k=7):
        self.request_id = rid
        self.spec_token_ids = [-1] * k


def policy_tests(helper_src: str) -> None:
    import math  # noqa: F401

    def make(env):
        ns = {"os": os}
        old = dict(os.environ)
        os.environ.update(env)
        try:
            exec(helper_src, ns)
            inst = ns["_Glm53AdaptiveK"]()
        finally:
            os.environ.clear()
            os.environ.update(old)
        return inst

    # default off: never trims
    p = make({"GLM53_ADAPTIVE_K": "off"})
    r = _Req("a")
    for _ in range(10):
        p.observe("a", 7, 0)
    p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 7 and not p.enabled

    # on: prose-like (1 accepted) -> trims to 2 after min_steps; structured stays 7
    p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0"})
    r = _Req("a")
    for i in range(3):
        p.observe("a", 7, 1)
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
        assert len(r.spec_token_ids) == 7, "min_steps guard"
    for i in range(12):
        p.observe("a", len(r.spec_token_ids), 1)
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 2, r.spec_token_ids
    s = _Req("s")
    r.spec_token_ids = [-1] * 7
    p.apply([(r, False), (s, True)], {"a": r, "s": s})
    assert len(r.spec_token_ids) == 7 and len(s.spec_token_ids) == 7, "structured forces batch to 7"

    # ratchet escape: saturation at n=2 feeds the full length, EMA can climb back to 7
    r.spec_token_ids = [-1] * 2
    for _ in range(20):
        p.observe("a", len(r.spec_token_ids), len(r.spec_token_ids))
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 7, r.spec_token_ids

    # saturate=n reproduces the ratchet (documented behaviour)
    p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_SATURATE": "n", "GLM53_ADAPTIVE_K_HIST": "0"})
    r = _Req("a")
    for _ in range(15):
        p.observe("a", 7, 1)
    r.spec_token_ids = [-1] * 7
    p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 2
    for _ in range(30):
        p.observe("a", 2, 2)
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 2, "saturate=n never climbs"

    # runtime override file: mode off -> no trimming; set clamped to boot set
    import json, tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        f = Path(td) / "ak.json"
        p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0", "GLM53_ADAPTIVE_K_FILE": str(f)})
        r = _Req("a")
        for _ in range(15):
            p.observe("a", 7, 1)
        p.apply([(r, False)], {"a": r})
        assert len(r.spec_token_ids) == 2
        f.write_text(json.dumps({"mode": "off"}))
        os.utime(f, (1, 1))
        p.steps = 0  # force the mtime check
        for _ in range(15):
            p.observe("a", 7, 1)
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
        assert len(r.spec_token_ids) == 7 and not p.enabled, "file mode=off must disable trimming"
        f.write_text(json.dumps({"mode": "ema", "set": [3, 5, 9]}))
        os.utime(f, (2, 2))
        p.steps = 0
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})  # steps % 50 == 0 -> reload
        assert p.enabled and p.k_set == [2, 4, 7], (p.enabled, p.k_set)  # no overlap with the boot set -> falls back
    with _tf.TemporaryDirectory() as td:
        f = Path(td) / "ak.json"
        p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_SET": "2,3,4,5,7", "GLM53_ADAPTIVE_K_HIST": "0", "GLM53_ADAPTIVE_K_FILE": str(f)})
        f.write_text(json.dumps({"mode": "ema", "set": "3,5,7", "margin": 1.5}))
        p.steps = 0
        p._reload()
        assert p.k_set == [3, 5, 7] and p.margin == 1.5, (p.k_set, p.margin)
        r = _Req("a")
        for _ in range(15):
            p.observe("a", 7, 1)
        p.apply([(r, False)], {"a": r})
        assert len(r.spec_token_ids) == 3, r.spec_token_ids

    # schedule-time hook (async scheduler path)
    class _SReq:
        def __init__(self, rid, structured=False, prefill=False):
            self.request_id = rid; self.use_structured_output = structured; self.is_prefill_chunk = prefill
    p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0"})
    a, b, s, pf = _SReq("a"), _SReq("b"), _SReq("s", structured=True), _SReq("p", prefill=True)
    live = {"a": a, "b": b, "s": s, "p": pf}
    assert p.batch_k(7, [a], live) == 7, "unobserved request pins full length"
    for _ in range(15):
        p.observe("a", 7, 1); p.observe("b", 7, 7)
    assert p.batch_k(7, [a], live) == 2
    assert p.batch_k(7, [b], live) == 7
    assert p.batch_k(7, [a, b], live) == 2, "batch minimum"
    assert p.batch_k(7, [a, s], live) == 7, "structured pins full length"
    assert p.batch_k(7, [a, pf], live) == 2, "prefill chunks are ignored"
    assert p.batch_k(7, [a, None], live) == 2
    assert p.hist.get(2, 0) >= 3

    # batch minimum across two requests
    p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0"})
    a, b = _Req("a"), _Req("b")
    for _ in range(15):
        p.observe("a", 7, 7)
        p.observe("b", 7, 1)
    p.apply([(a, False), (b, False)], {"a": a, "b": b})
    assert len(a.spec_token_ids) == 2 and len(b.spec_token_ids) == 2


def main() -> int:
    for src in (SCHED_SRC, CG_SRC):
        if not src.is_file():
            raise SystemExit(f"missing {src}")
    with tempfile.TemporaryDirectory() as tmp:
        sched = Path(tmp) / "scheduler.py"
        cg = Path(tmp) / "cudagraph_utils.py"
        shutil.copyfile(SCHED_SRC, sched)
        shutil.copyfile(CG_SRC, cg)
        env = os.environ.copy()
        env["GLM53_SCHEDULER_PY"] = str(sched)
        env["GLM53_CUDAGRAPH_UTILS_PY"] = str(cg)
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        st, ct = sched.read_text(), cg.read_text()
        assert st.count("# [glm53-adaptive-k]") >= 6, st.count("# [glm53-adaptive-k]")
        assert "_GLM53_ADAPTIVE_K.observe(" in st and "_GLM53_ADAPTIVE_K.apply(" in st
        assert "_glm53_adaptive_k_query_lens(decode_query_lens" in ct
        compile(st, "scheduler.py", "exec")
        compile(ct, "cudagraph_utils.py", "exec")
        subprocess.check_call([sys.executable, str(PATCH)], env=env)  # idempotent
        assert sched.read_text() == st and cg.read_text() == ct
        # extract the helper class source for policy tests
        start = st.index("class _Glm53AdaptiveK:")
        end = st.index("_GLM53_ADAPTIVE_K = _Glm53AdaptiveK()")
        policy_tests(st[start:end])
        # graph-length helper
        cstart = ct.index("def _glm53_adaptive_k_query_lens(")
        cend = ct.index("@dataclass(frozen=True)\nclass BatchExecutionDescriptor:")
        ns = {}
        exec(ct[cstart:cend], ns)
        fn = ns["_glm53_adaptive_k_query_lens"]
        os.environ["GLM53_ADAPTIVE_K"] = "off"
        assert fn([8], 8) == [8]
        os.environ["GLM53_ADAPTIVE_K"] = "ema"
        assert fn([8], 8) == [3, 5, 8]
    print("adaptive-k patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
