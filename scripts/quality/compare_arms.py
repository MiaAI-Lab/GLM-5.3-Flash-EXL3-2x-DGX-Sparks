#!/usr/bin/env python3
"""Summarize every arm under logs/<RUN>/ into one table (tool-calling, long-context recall,
HumanEval/MBPP pass@1) and run the KL panel of each arm against a reference arm.
usage: compare_arms.py logs/quality-RUN [--ref ARM]"""
import argparse, glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    arms = sorted(d for d in os.listdir(a.run) if os.path.isdir(os.path.join(a.run, d)))
    print(f"{'arm':28s} {'env':44s} {'tools':>9s} {'long':>7s} {'leak':>4s} {'badjs':>5s} {'HE':>9s} {'MBPP':>9s} {'trunc':>5s} {'tok/ans':>7s}")
    for arm in arms:
        d = os.path.join(a.run, arm)
        env = ""
        try:
            kv = dict(l.strip().split("=", 1) for l in open(os.path.join(d, "server-env.txt")) if "=" in l)
            ak = load(os.path.join(d, "adaptive_k.json")) or {}
            env = f"fp8={kv.get('GLM53_DENSE_FP8','?')} ablit={kv.get('ABLIT','?')} k={ak.get('mode', kv.get('GLM53_ADAPTIVE_K','?'))}"
        except Exception:  # noqa: BLE001
            pass
        t = load(os.path.join(d, "toolcall.json")); c = load(os.path.join(d, "code.json"))
        ts = t["summary"]["total"] if t else {}
        cs = c["summary"] if c else {}
        he = cs.get("humaneval", {}); mb = cs.get("mbpp", {})
        print(f"{arm:28s} {env:44s} {ts.get('ok','-'):>4}/{ts.get('n','-'):<4} {ts.get('long_ok','-'):>3}/{ts.get('long_n','-'):<3} {ts.get('leaks','-'):>4} {ts.get('bad_json','-'):>5} "
              f"{he.get('pass','-'):>4}/{he.get('n','-'):<4} {mb.get('pass','-'):>4}/{mb.get('n','-'):<4} {he.get('truncated',0)+mb.get('truncated',0):>5} {he.get('mean_completion_tokens','-'):>7}")
    # per-case tool table
    print("\nper-case tool-calling ok/n:")
    cases = {}
    for arm in arms:
        t = load(os.path.join(a.run, arm, "toolcall.json"))
        if t:
            for cid, s in t["summary"]["cases"].items():
                cases.setdefault(cid, {})[arm] = f"{s['ok']}/{s['n']}"
    if cases:
        print(f"{'case':20s}" + "".join(f"{arm[:26]:>28s}" for arm in arms))
        for cid, row in cases.items():
            print(f"{cid:20s}" + "".join(f"{row.get(arm,'-'):>28s}" for arm in arms))
    ref = a.ref or (arms[0] if arms else None)
    if ref:
        refk = os.path.join(a.run, ref, "kl.json")
        for arm in arms:
            for k in sorted(glob.glob(os.path.join(a.run, arm, "kl*.json"))):
                if os.path.abspath(k) == os.path.abspath(refk):
                    continue
                print(f"\n=== KL panel: ref={ref}/kl.json  vs  {arm}/{os.path.basename(k)} ===")
                subprocess.run([sys.executable, os.path.join(HERE, "kl_panel.py"), "compare", refk, k])


if __name__ == "__main__":
    main()
