#!/usr/bin/env python3
"""Regression tests for scripts/quality/kl_panel.py admission + condKL math, and for
scripts/quality/compare_arms.py propagating a refused comparison.

CPU-only, no server: panels are synthesized in a tmpdir and compared through the real CLI
entry point, so the exit codes the launchers key off are what gets asserted.

Covers the defects this fixes: empty captures / absent or mismatched panel membership /
differing token sequences / unaligned or invalid logprob records / panels with no scorable
position used to print nothing (or a NaN row) and exit 0; the shared top-K KL used to be
reported without the coverage that makes it readable.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QUALITY = ROOT / "scripts" / "quality"

SPEC = importlib.util.spec_from_file_location("kl_panel", QUALITY / "kl_panel.py")
kp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kp)

L = math.log
# Two captures over the same three tokens: A=(.5,.3,.2), B=(.1,.1,.8)
XA = {0: L(0.5), 1: L(0.3), 2: L(0.2)}
XB = {0: L(0.1), 1: L(0.1), 2: L(0.8)}
# KL(A||B) on the shared support, each side renormalized there (here both already sum to 1)
EXPECT_KL = 0.5 * L(0.5 / 0.1) + 0.3 * L(0.3 / 0.1) + 0.2 * L(0.2 / 0.8)
assert abs(EXPECT_KL - 0.8570437705935049) < 1e-12, EXPECT_KL

# Column indices of a printed row after item + range: npos nKL mask noK nNLL NLL_A NLL_B dNLL condKL covA covB argmax_k dNLL>1
NPOS, NKL, MASK, NOK, NNLL, CONDL, COVA, COVB = 0, 1, 2, 3, 4, 8, 9, 10


def item(ids, pos, kind=None, ts=0, te=0):
    it = {"ids": ids, "pos": pos, "tail_start": ts, "tail_end": te}
    if kind is not None:
        it["kind"] = kind
    return it


def panel(items, k=20, meta=True):
    out = {"items": items}
    if meta:
        out["meta"] = {"k": k, "url": "http://127.0.0.1:8888", "time": "2026-09-15T00:00:00Z"}
    return out


def texts(n, entry=XA):
    """n tokens, position 0 unscored (as vLLM returns it), every later position scoring `entry`."""
    return item(list(range(n)), [None] + [dict(entry)] * (n - 1), kind="text")


def write(tmp: Path, name: str, obj=None, raw=None) -> str:
    p = tmp / name
    p.write_text(raw if raw is not None else json.dumps(obj))
    return str(p)


def run_compare(a, b):
    """Drive the real CLI path; return (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = kp.main(["compare", a, b])
    return rc, out.getvalue(), err.getvalue()


def rows(out):
    """(item, range label) -> row fields; range labels are 'lo-hi' or 'tail lo-hi'."""
    got = {}
    for line in out.splitlines():
        p = line.split()
        if line.startswith("item") or len(p) < 15:
            continue
        got[(p[0], " ".join(p[1:3]) if p[1] == "tail" else p[1])] = p[3:] if p[1] == "tail" else p[2:]
    return got


def check_panel_sources():
    """The capture corpus must be loadable from a clean checkout (it referenced a missing file)."""
    its = kp.items()
    assert its, "the panel corpus is empty"
    for name, it in its.items():
        key = "text" if it["kind"] == "text" else "messages"   # the payload _capture_item will post
        assert it.get(key), (name, key)
    print("panel sources OK (every item loads with the payload its capture path posts)")


def check_valid_comparison(tmp: Path):
    a = write(tmp, "a.json", panel({"code_exl3": texts(8)}))
    b = write(tmp, "b.json", panel({"code_exl3": texts(8, XB)}))
    rc, out, err = run_compare(a, b)
    assert rc == 0, (rc, err)
    f = rows(out)[("code_exl3", "0-2000")]
    assert f[NPOS] == "7" and f[NKL] == "7", f          # 7 tokens after position 0, all scorable
    assert f[CONDL] == f"{EXPECT_KL:.5f}", f            # condKL on the shared support, renormalized per side
    assert f[COVA] == "100.0%" and f[COVB] == "100.0%", f  # both supports fully shared
    print("valid comparison OK (condKL = shared-support renormalized KL, coverage disclosed)")


def check_support_accounting(tmp: Path):
    """Mass on tokens the other side did not return must show up as coverage, not vanish."""
    xa, xb = {0: -0.1, 1: -2.5}, {0: -0.2, 2: -3.0}  # shared support is {0} only
    a = write(tmp, "sa.json", panel({"prose_docs": item([0, 0, 0], [None, dict(xa), dict(xa)], kind="text")}))
    b = write(tmp, "sb.json", panel({"prose_docs": item([0, 0, 0], [None, dict(xb), dict(xb)], kind="text")}))
    rc, out, err = run_compare(a, b)
    assert rc == 0, (rc, err)
    f = rows(out)[("prose_docs", "0-2000")]
    assert f[CONDL] == "0.00000", f                     # one shared token: KL is 0 there...
    assert f[COVA] == f"{math.exp(-0.1):.1%}" and f[COVB] == f"{math.exp(-0.2):.1%}", f  # ...9%/18% of mass omitted
    assert f[NNLL] == "2", f                            # the scored token is in both supports here
    terms = kp._shared_support(xa, xb)
    assert terms is not None and terms[0] == 0.0, terms
    assert abs(terms[1] - math.exp(-0.1)) < 1e-12 and abs(terms[2] - math.exp(-0.2)) < 1e-12, terms
    assert kp._shared_support({1: -1.0}, {2: -1.0}) is None  # disjoint supports: nothing to score
    # NLL needs the scored token in both supports; when it is missing, that is reported, not faked.
    c = write(tmp, "sc.json", panel({"tools_short": item([0, 9, 9], [None, {0: -0.1}, {0: -0.1}], kind="text")}))
    d = write(tmp, "sd.json", panel({"tools_short": item([0, 9, 9], [None, {0: -0.2}, {0: -0.2}], kind="text")}))
    rc, out, err = run_compare(c, d)
    assert rc == 0, (rc, err)
    f = rows(out)[("tools_short", "0-2000")]
    assert f[NKL] == "2" and f[NNLL] == "0" and f[5] == "-" and f[6] == "-", f
    print("support accounting OK (coverage = mass on the shared support; missing NLL reported as '-')")


def check_legit_masking(tmp: Path):
    """Masked regions are legitimate: score the chat tail, do not refuse, do not score the masked span."""
    ids = [0] * 12  # every scored position predicts token 0, which is inside the returned support
    pre, tail = [None] * 5, [dict(XA)] * 5
    pos = pre + tail + [None] * 2
    for tag, kind in (("legacy", None), ("explicit", "chat")):
        a = write(tmp, f"ma-{tag}.json", panel({"tools_short": item(ids, pos, kind=kind, ts=5, te=10)}))
        b = write(tmp, f"mb-{tag}.json", panel({"tools_short": item(ids, pos, kind=kind, ts=5, te=10)}))
        rc, out, err = run_compare(a, b)
        assert rc == 0, (tag, rc, err)
        f = rows(out)[("tools_short", "tail 5-10")]
        assert f[NPOS] == "5" and f[NKL] == "5" and f[MASK] == "0", (tag, f)  # only the tail span is scored
        assert f[NNLL] == "5", (tag, f)
    # A short document leaves high bins empty; that is disclosed, not a refusal. Captures written by the
    # pre-fix script carry no `kind` field and must still compare (tail_start=tail_end=0 => text item).
    legacy_text = item(list(range(8)), [None] + [dict(XA)] * 7)
    a = write(tmp, "ma3.json", panel({"prose_docs": legacy_text}))
    b = write(tmp, "mb3.json", panel({"prose_docs": legacy_text}))
    rc, out, err = run_compare(a, b)
    assert rc == 0, (rc, err)
    f = rows(out)[("prose_docs", "0-2000")]
    assert f[NPOS] == "7" and f[NKL] == "7", f  # the scored bin still yields data; empty bins do not refuse
    print("legit masking OK (chat tail scored, masked spans preserved, empty bins disclosed)")


def check_refusals(tmp: Path):
    """Every noncomparable panel must exit non-zero and name the reason, never print a no-data table."""
    ok = write(tmp, "ok.json", panel({"code_exl3": texts(8)}))
    cases = {
        "empty items": (write(tmp, "r1.json", panel({})), ok, "no items"),
        "absent item in B": (ok, write(tmp, "r2.json", panel({"code_exl3": texts(8), "prose_docs": texts(8)})), "membership differs"),
        "subset capture (--only)": (write(tmp, "r3.json", panel({"code_exl3": texts(8), "tools_long": texts(8)})), ok, "membership differs"),
        "different token counts": (ok, write(tmp, "r4.json", panel({"code_exl3": texts(9)})), "token sequences differ"),
        "same length, different ids": (
            write(tmp, "r5a.json", panel({"code_exl3": item([0, 1, 2, 3], [None] + [dict(XA)] * 3, kind="text")})),
            write(tmp, "r5b.json", panel({"code_exl3": item([0, 1, 9, 3], [None] + [dict(XA)] * 3, kind="text")})),
            "token sequences differ"),
        "unaligned logprob positions": (
            ok, write(tmp, "r6.json", panel({"code_exl3": item(list(range(8)), [None] + [dict(XA)] * 3, kind="text")})), "unaligned"),
        "every position masked": (ok, write(tmp, "r7.json", panel({"code_exl3": item(list(range(8)), [None] * 8, kind="text")})), "none scorable"),
        "disjoint supports only": (ok, write(tmp, "r8.json", panel({"code_exl3": item(list(range(8)), [None] + [{7: -1.0}] * 7, kind="text")})), "none scorable"),
        "NaN logprob": (ok, write(tmp, "r9.json", None,
                                  raw=json.dumps(panel({"code_exl3": texts(8)})).replace(f"{L(0.5)}", "NaN")), "finite value"),
        "logprob > 0": (ok, write(tmp, "r10.json", panel({"code_exl3": item(list(range(8)), [None] + [{0: 0.5, 1: -1.0}] * 7, kind="text")})), "finite value"),
        "string logprob": (ok, write(tmp, "r11.json", panel({"code_exl3": item(list(range(8)), [None] + [{"0": "-1.0"}] * 7, kind="text")})), "finite value"),
        "non-canonical token key": (ok, write(tmp, "r12.json", panel({"code_exl3": item(list(range(8)), [None] + [{"007": -1.0}] * 7, kind="text")})), "canonical"),
        "meta.k widened": (ok, write(tmp, "r13.json", panel({"code_exl3": texts(8)}, k=10)), "top-K width differs"),
        "meta.k absent": (ok, write(tmp, "r14.json", panel({"code_exl3": texts(8)}, meta=False)), "meta.k"),
        "item not an object": (ok, write(tmp, "r15.json", panel({"code_exl3": [1, 2, 3]})), "not an object"),
        "chat item without a tail span": (
            write(tmp, "r16.json", panel({"tools_short": item(list(range(8)), [None] + [dict(XA)] * 7, kind="chat")})), ok, "no scored tail span"),
        "text vs chat kind mismatch": (
            write(tmp, "r17.json", panel({"tools_short": texts(8)})),
            write(tmp, "r18.json", panel({"tools_short": item(list(range(8)), [None] + [dict(XA)] * 7, ts=2, te=8)})), "kind differs"),
        "capture reported failed items": (
            ok, write(tmp, "r19.json", {"meta": {"k": 20, "errors": ["one item failed"]},
                                     "items": {"code_exl3": texts(8)}}), "incomplete data"),
        "capture reported dropped values": (
            ok, write(tmp, "r20.json", {"meta": {"k": 20, "warnings": ["dropped values"]},
                                     "items": {"code_exl3": texts(8)}}), "incomplete data"),
        "different chat scoring spans": (
            write(tmp, "r21.json", panel({"tools": item(list(range(8)), [None] + [dict(XA)] * 7, ts=2, te=8)})),
            write(tmp, "r22.json", panel({"tools": item(list(range(8)), [None] + [dict(XA)] * 7, ts=3, te=8)})), "kind differs"),
        "partial asymmetric mask": (
            ok, write(tmp, "r23.json", panel({"code_exl3": item(list(range(8)), [None, None] + [dict(XA)] * 6, kind="text")})), "masked in only one"),
    }
    for name, (a, b, want) in cases.items():
        rc, out, err = run_compare(a, b)
        assert rc == 2, (name, rc, out, err)
        assert want in err, (name, want, err)
        assert not rows(out), (name, out)  # refused, so no measurement table was printed
    print(f"refusals OK ({len(cases)} noncomparable panels exit 2 with a reason, no table)")


def check_io_errors(tmp: Path):
    ok = write(tmp, "io_ok.json", panel({"code_exl3": texts(8)}))
    rc, _, err = run_compare(str(tmp / "absent.json"), ok)
    assert rc == 3 and "cannot read capture" in err, (rc, err)
    bad = write(tmp, "bad.json", raw="{not json")
    rc, _, err = run_compare(bad, ok)
    assert rc == 3 and "not valid JSON" in err, (rc, err)
    print("io errors OK (unreadable/unparseable captures exit 3, not a silent empty table)")


def check_compare_arms(tmp: Path):
    """compare_arms must not report success when kl_panel refused a comparison."""
    ca = str(QUALITY / "compare_arms.py")
    n = [0]

    def arms_case(layout):
        n[0] += 1
        run_dir = tmp / f"run{n[0]}"
        for arm, content in layout.items():
            (run_dir / arm).mkdir(parents=True)
            if content is not None:
                (run_dir / arm / "kl.json").write_text(json.dumps(content))
        return subprocess.run([sys.executable, ca, str(run_dir)], capture_output=True, text=True)

    valid = panel({"code_exl3": texts(8)})
    r = arms_case({"ref": valid, "arm": panel({"code_exl3": texts(8, XB)})})
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert rows(r.stdout), r.stdout  # the pair was compared: kl_panel's measurement table came through

    r = arms_case({"ref": valid, "bad": panel({})})
    assert r.returncode == 2 and "FAILED" in r.stderr and "bad/kl.json" in r.stderr, (r.returncode, r.stderr)

    r = arms_case({"ref": valid, "short": panel({"code_exl3": texts(9)})})
    assert r.returncode == 2 and "token sequences differ" in r.stderr, (r.returncode, r.stderr)

    r = arms_case({"ref": valid})  # no second capture anywhere: nothing was compared
    assert r.returncode == 2 and "no KL capture pairs" in r.stderr, (r.returncode, r.stderr)

    r = arms_case({"ref": {}, "arm": None})  # reference arm has no usable panel
    assert r.returncode == 2, (r.returncode, r.stdout)

    r = arms_case({"arm": valid})  # --ref defaults to the only arm, which carries the reference itself
    assert r.returncode == 2 and "no KL capture pairs" in r.stderr, (r.returncode, r.stderr)

    r = arms_case({"a_ref": valid, "b_valid": valid, "c_missing": None})
    assert r.returncode == 2 and "c_missing: no KL capture" in r.stderr, (r.returncode, r.stderr)
    print("compare_arms OK (refusals, unusable reference, and zero pairs all exit 2)")


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        check_panel_sources()
        check_valid_comparison(tmp)
        check_support_accounting(tmp)
        check_legit_masking(tmp)
        check_refusals(tmp)
        check_io_errors(tmp)
        check_compare_arms(tmp)
    print("kl_panel/compare_arms admission + condKL regression OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
