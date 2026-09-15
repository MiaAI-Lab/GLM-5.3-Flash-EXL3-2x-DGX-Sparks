#!/usr/bin/env python3
"""Long-context logprob panel for the served model (numerics gate, prefill only).

Captures prompt logprobs (top-K per position) for fixed long texts and two tool-calling
transcripts, then compares two captures by context depth. The divergence column is
condKL(A||B): the KL between each capture's distribution *conditioned on the intersection of
the two returned supports* (both sides' returned top-K plus the scored token), renormalized
per side and averaged over scored positions. It is NOT full-vocabulary KL: mass on tokens the
other side did not return, and both sides' unreturned tails, are invisible. compare() therefore
also prints covA/covB, the probability mass each capture spends on that shared support. Read
them together; never quote condKL alone as a distribution-level divergence. Also reports argmax
agreement within the returned support and mean NLL of the actual token. Prefill-only, so it
isolates target numerics (FP8 / ABLIT / kernels) from the decode policy (adaptive-k does not
touch it).

compare() refuses (exit 2) instead of printing an empty table when a panel is not comparable:
no items, absent/mismatched item membership, differing meta.k, differing token sequences,
unaligned or invalid logprob records, or an item/range where no position can be scored.
Exit 3 means a capture file was unreadable or not valid JSON.

usage: kl_panel.py capture OUT.json [--k 20] [--url http://127.0.0.1:8888]
       kl_panel.py compare A.json B.json
"""
import argparse, json, math, os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "GLM-5.3-Flash-EXL3"
BINS = [(0, 2000), (2000, 4000), (4000, 8000), (8000, 16000), (16000, 32000)]
LOGTOL = 1e-6  # a logprob above this is not a probability log (log p <= 0 for p <= 1)
UNDERFLOW = -745.0  # below this exp() is 0.0 in double precision, so the p -> 0 KL term is exactly 0

TOOLS = [
    {"type": "function", "function": {"name": "Read", "description": "Read a file from the filesystem.",
     "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "Bash", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "Edit", "description": "Replace old_string with new_string in file_path. old_string must match the file exactly.",
     "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["file_path", "old_string", "new_string"]}}},
]


class PanelError(Exception):
    """A capture is readable but not comparable: refuse instead of reporting no-data."""


class PanelIOError(PanelError):
    """A capture path could not be read as JSON at all."""


def _fail(msg):
    raise PanelError(msg)


def post(url, path, body, timeout=1800):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def read(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.isfile(path):
        raise PanelError(f"panel source missing: {rel} (looked in {path})")
    with open(path) as fh:
        return fh.read()


def tool_transcript(long: bool):
    """A realistic agentic exchange ending in an Edit call whose old_string is copied from the file."""
    src = read("overlay/exl3.py").splitlines()
    n = 1500 if long else 120
    shown = "\n".join(f"{i+1:>6}\t{l}" for i, l in enumerate(src[:n]))
    target_line = src[n - 40]
    msgs = [
        {"role": "system", "content": "You are a coding agent. Use the tools to inspect and change the repository."},
        {"role": "user", "content": "Read overlay/exl3.py, then rename nothing; just add a comment '# reviewed' at the end of the line I point to next."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": json.dumps({"file_path": "overlay/exl3.py"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": shown},
        {"role": "user", "content": f"Append the comment to line {n-39}."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "Edit", "arguments": json.dumps({"file_path": "overlay/exl3.py", "old_string": target_line, "new_string": target_line + "  # reviewed"})}}]},
    ]
    return msgs


def items():
    # Sources must be tracked files: a missing one used to abort capture with a bare
    # FileNotFoundError, i.e. the panel could not run at all from a clean checkout.
    out = {}
    out["code_exl3"] = {"kind": "text", "text": read("overlay/exl3.py")}
    out["code_start_sh"] = {"kind": "text", "text": read("start.sh")}
    out["prose_docs"] = {"kind": "text", "text": read("docs/DESIGN-apc-per-group-retention.md")}
    out["tools_short"] = {"kind": "chat", "messages": tool_transcript(False)}
    out["tools_long"] = {"kind": "chat", "messages": tool_transcript(True)}
    return out


def _logprob(v):
    """vLLM returns {token_id: {"logprob": .., ..}}; tolerate a bare float, refuse anything else."""
    if isinstance(v, dict):
        v = v.get("logprob")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(float(v)) and float(v) <= LOGTOL else None


def _capture_item(url, it, k, name):
    """Return (record, warnings). Records are exact server output; anomalies are reported, not papered over."""
    warn = []
    if it["kind"] == "text":
        ids = post(url, "/tokenize", {"model": MODEL, "prompt": it["text"]})["tokens"]
        d = post(url, "/v1/completions", {"model": MODEL, "prompt": it["text"], "max_tokens": 1, "temperature": 0, "prompt_logprobs": k})
        pl = d.get("prompt_logprobs") or d["choices"][0].get("prompt_logprobs") or []
        tail_start = tail_end = 0
    else:
        # continue_final_message drops the final assistant turn's tool_calls, so render the
        # transcript with a trailing tool ack + generation prompt and score the span of the
        # final tool-call turn: [tail_start, tail_end).
        msgs = it["messages"] + [{"role": "tool", "tool_call_id": "call_2", "content": "ok"}]
        body = {"model": MODEL, "messages": msgs, "tools": TOOLS}
        ids = post(url, "/tokenize", dict(body, add_generation_prompt=True))["tokens"]
        head = post(url, "/tokenize", dict(body, messages=it["messages"][:-1], add_generation_prompt=True))["tokens"]
        upto = post(url, "/tokenize", dict(body, messages=it["messages"], add_generation_prompt=False))["tokens"]
        tail_start, tail_end = len(head), len(upto)
        if ids[:tail_end] != upto:
            _fail(f"{name}: tokenization of the tool-call turn is not a prefix")
        if not (0 < tail_start < tail_end <= len(ids)):
            warn.append(f"{name}: empty scored tail span {tail_start}-{tail_end} of {len(ids)} tokens")
        d = post(url, "/v1/chat/completions", dict(body, max_tokens=1, temperature=0, prompt_logprobs=k))
        pl = d.get("prompt_logprobs") or []
    if len(pl) != len(ids):
        warn.append(f"{name}: {len(pl)} logprob positions vs {len(ids)} tokens — capture is unaligned and will be refused by compare")
    # compact: per position -> {tok: logprob} (top-k plus the actual token); None where the server returned nothing
    pos, bad = [], 0
    for p in pl:
        if not p:
            pos.append(None)
            continue
        d_ = {}
        for t, v in p.items():
            lp = _logprob(v)
            if lp is None:
                bad += 1
                continue
            d_[str(t)] = lp
        pos.append(d_ or None)
    if bad:
        warn.append(f"{name}: {bad} logprob entries had no numeric value and were dropped")
    if not any(pos):
        warn.append(f"{name}: no scored positions returned (prompt_logprobs empty or all masked)")
    return {"ids": ids, "pos": pos, "tail_start": tail_start, "tail_end": tail_end, "kind": it["kind"]}, warn


def capture(url, out, k, only=None):
    res = {"meta": {"url": url, "k": k, "time": time.strftime("%FT%TZ", time.gmtime()), "errors": [], "warnings": []}, "items": {}}
    for name, it in items().items():
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            rec, warn = _capture_item(url, it, k, name)
        except Exception as exc:  # noqa: BLE001 — keep the other items' data, report the loss
            msg = f"{name}: capture failed: {type(exc).__name__}: {exc}"
            print(f"ERROR {msg}", file=sys.stderr, flush=True)
            res["meta"]["errors"].append(msg)
            continue
        res["items"][name] = rec
        for w in warn:
            print(f"WARN {w}", file=sys.stderr, flush=True)
            res["meta"]["warnings"].append(w)
        nll = [-rec["pos"][i][str(rec["ids"][i])] for i in range(1, min(len(rec["ids"]), len(rec["pos"]))) if rec["pos"][i] and str(rec["ids"][i]) in rec["pos"][i]]
        print(f"{name:14s} tokens={len(rec['ids']):6d} tail={rec['tail_start']}-{rec['tail_end']} mean NLL={sum(nll)/max(1,len(nll)):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if not res["items"]:
        res["meta"]["errors"].append("no selected items were captured")
    with open(out, "w") as fh:
        json.dump(res, fh)
    print("wrote", out)
    if res["meta"]["errors"] or res["meta"]["warnings"]:
        print(f"capture INCOMPLETE: {len(res['items'])} items written, {len(res['meta']['errors'])} failed, "
              f"{len(res['meta']['warnings'])} warnings — this capture will not compare", file=sys.stderr, flush=True)
        return 2
    return 0


def _load_panel(path, label):
    try:
        with open(path) as fh:
            return json.load(fh)
    except OSError as exc:
        raise PanelIOError(f"{label}={path}: cannot read capture: {exc}") from exc
    except ValueError as exc:  # includes json.JSONDecodeError
        raise PanelIOError(f"{label}={path}: not valid JSON: {exc}") from exc


def _check_pos_entry(where, i, entry):
    """Normalize one position to {canonical token str: logprob} or None (server returned nothing)."""
    if entry is None:
        return None
    if not isinstance(entry, dict):
        _fail(f"{where}: position {i} is {type(entry).__name__}, not a token->logprob map")
    if not entry:
        return None
    out = {}
    for t, v in entry.items():
        if not isinstance(t, str) or not t.isascii() or not t.isdigit() or str(int(t)) != t:
            _fail(f"{where}: position {i} has token key {t!r}, not a canonical non-negative int")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) > LOGTOL:
            _fail(f"{where}: position {i} token {t} has logprob {v!r}, not a finite value <= 0")
        out[t] = float(v)
    return out


def _validate_item(name, it, label):
    where = f"{label}:{name}"
    if not isinstance(it, dict):
        _fail(f"{where}: item is {type(it).__name__}, not an object")
    ids, pos = it.get("ids"), it.get("pos")
    if not isinstance(ids, list) or not ids:
        _fail(f"{where}: missing or empty token ids")
    for i, t in enumerate(ids):
        if isinstance(t, bool) or not isinstance(t, int) or t < 0:
            _fail(f"{where}: token id {t!r} at index {i} is not a non-negative int")
    if not isinstance(pos, list) or len(pos) != len(ids):
        _fail(f"{where}: {len(pos) if isinstance(pos, list) else type(pos).__name__} logprob positions vs {len(ids)} tokens — capture is unaligned")
    ts, te = it.get("tail_start", 0), it.get("tail_end", 0)
    for v, nm in ((ts, "tail_start"), (te, "tail_end")):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            _fail(f"{where}: {nm} is {v!r}, not a non-negative int")
    kind = it.get("kind")
    if kind not in (None, "text", "chat"):
        _fail(f"{where}: unknown kind {kind!r}")
    if ts == 0 and te == 0:
        if kind == "chat":
            _fail(f"{where}: chat item carries no scored tail span (tail_start=tail_end=0) — not comparable")
        kind = "text"
    else:
        if kind == "text":
            _fail(f"{where}: text item carries a tail span {ts}-{te}")
        if ts < 1 or te <= ts:
            _fail(f"{where}: chat tail span {ts}-{te} scores nothing")
        if te > len(ids):
            _fail(f"{where}: chat tail span {ts}-{te} exceeds {len(ids)} tokens")
        kind = "chat"
    return {"ids": ids, "pos": [_check_pos_entry(where, i, p) for i, p in enumerate(pos)], "kind": kind, "tail": (ts, te)}


def _validate_panel(obj, label, path):
    if not isinstance(obj, dict):
        _fail(f"{label}={path}: capture is {type(obj).__name__}, not an object")
    its = obj.get("items")
    if not isinstance(its, dict) or not its:
        _fail(f"{label}={path}: capture has no items (empty or missing 'items') — nothing to compare")
    meta = obj.get("meta")
    k = meta.get("k") if isinstance(meta, dict) else None
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        _fail(f"{label}={path}: meta.k is {k!r}, not a positive int — top-K width is unknown")
    if meta.get("errors") or meta.get("warnings"):
        _fail(f"{label}={path}: capture recorded errors or warnings — incomplete data")
    return k, {str(n): _validate_item(str(n), it, label) for n, it in its.items()}


def _logsumexp(vals):
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _argmax(d):
    return max(sorted(d, key=int), key=d.get)


def _shared_support(x, y):
    """condKL(A||B) over the intersection of the two returned supports, plus each side's mass there.

    None when the supports are disjoint (no shared tokens => nothing is comparable at that position).
    covA/covB are sum(exp(logprob)) over the shared tokens, i.e. a probability only under the documented
    assumption that the server returns full-vocabulary log-softmax values (vLLM prompt_logprobs) rather
    than values renormalized over the returned top-K; the meta.k equality check is what keeps the two
    sides' K widths from silently differing. Anything outside the returned supports is invisible here,
    which is why the coverage columns are printed next to the KL.
    """
    keys = sorted(set(x) & set(y), key=int)
    if not keys:
        return None
    lx = [x[t] for t in keys]
    ly = [y[t] for t in keys]
    zx, zy = _logsumexp(lx), _logsumexp(ly)
    kl = 0.0
    for v, w in zip(lx, ly):
        pa = v - zx
        if pa > UNDERFLOW:  # exp(pa) already 0.0 below this and the p->0 KL term is 0
            kl += math.exp(pa) * (pa - (w - zy))
    return kl, math.exp(zx), math.exp(zy)


def _fmt(v, width=7):
    return f"{v:{width}.4f}" if v is not None else f"{'-':>{width}s}"


def _row(name, label, npos, nmask, nnok, kls, cva, cvb, agree, nlla, nllb, big):
    na = sum(nlla) / len(nlla) if nlla else None
    nb = sum(nllb) / len(nllb) if nllb else None
    return (f"{name:14s} {label:>13s} {npos:5d} {len(kls):5d} {nmask:5d} {nnok:5d} {len(nlla):5d} "
            f"{_fmt(na)} {_fmt(nb)} {_fmt(nb - na if na is not None and nb is not None else None)} "
            f"{sum(kls)/len(kls):11.5f} {sum(cva)/len(cva):6.1%} {sum(cvb)/len(cvb):6.1%} "
            f"{agree/len(kls):8.4f} {_fmt(big/len(nlla) if nlla else None)}")


def compare(a_path, b_path):
    A = _load_panel(a_path, "A"); B = _load_panel(b_path, "B")
    ka, ia = _validate_panel(A, "A", a_path); kb, ib = _validate_panel(B, "B", b_path)
    if ka != kb:
        _fail(f"top-K width differs (A meta.k={ka} vs B meta.k={kb}) — supports are not comparable")
    diff = sorted(set(ia) | set(ib)) if set(ia) != set(ib) else []
    if diff:
        _fail("panel membership differs — " + ", ".join(
            f"{n}: A={'present' if n in ia else 'absent'} B={'present' if n in ib else 'absent'}" for n in diff))
    rows, empties, scored = [], [], 0
    for name in sorted(ia):
        a, b = ia[name], ib[name]
        if a["ids"] != b["ids"]:
            first = next((i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), min(len(a["ids"]), len(b["ids"])))
            _fail(f"{name}: token sequences differ (A {len(a['ids'])} tokens, B {len(b['ids'])} tokens; first mismatch at position {first}) — "
                  "captures scored different text and cannot be compared")
        if a["kind"] != b["kind"] or a["tail"] != b["tail"]:
            _fail(f"{name}: item kind differs (A={a['kind']} with tail {a['tail'][0]}-{a['tail'][1]}, "
                  f"B={b['kind']} with tail {b['tail'][0]}-{b['tail'][1]}) — one side scores a span the other does not")
        ranges = ([(a["tail"][0], a["tail"][1], f"tail {a['tail'][0]}-{a['tail'][1]}")] if a["kind"] == "chat"
                  else [(lo, min(hi, len(a["ids"])), f"{lo}-{hi}") for lo, hi in BINS])
        before = len(rows)
        for lo, hi, label in ranges:
            npos = nmask = nnok = 0
            kls, cva, cvb, nlla, nllb, agree, big = [], [], [], [], [], 0, 0
            for i in range(max(lo, 1), hi):
                npos += 1
                x, y = a["pos"][i], b["pos"][i]
                if bool(x) != bool(y):
                    _fail(f"{name}: position {i} is masked in only one capture — none scorable for this pair")
                if not x or not y:
                    nmask += 1
                    continue
                terms = _shared_support(x, y)
                if terms is None:
                    nnok += 1
                    continue
                kl, ca, cb = terms
                kls.append(kl); cva.append(ca); cvb.append(cb)
                agree += _argmax(x) == _argmax(y)
                t = str(a["ids"][i])
                if t in x and t in y:
                    nlla.append(-x[t]); nllb.append(-y[t]); big += abs(x[t] - y[t]) > 1.0
            if npos == 0:
                empties.append(f"{name} {label}")
                continue
            if not kls:
                _fail(f"{name} {label}: {npos} positions, none scorable ({nmask} masked on one side, "
                      f"{nnok} with disjoint supports) — no comparable data")
            scored += len(kls)
            rows.append(_row(name, label, npos, nmask, nnok, kls, cva, cvb, agree, nlla, nllb, big))
        if len(rows) == before:
            _fail(f"{name}: no scorable position in any range — no comparable data")
    print(f"A={a_path}\nB={b_path}   (meta.k={ka})")
    print(f"{'item':14s} {'range':>13s} {'npos':>5s} {'nKL':>5s} {'mask':>5s} {'noK':>5s} {'nNLL':>5s} "
          f"{'NLL_A':>7s} {'NLL_B':>7s} {'dNLL':>7s} {'condKL(A||B)':>11s} {'covA':>6s} {'covB':>6s} {'argmax_k':>8s} {'dNLL>1':>7s}")
    for r in rows:
        print(r)
    print(f"\n{len(ia)} items compared, {scored} scored positions"
          + (f"; ranges with no positions in either capture: {', '.join(empties)}" if empties else ""))
    print("condKL(A||B) is the shared-top-K conditional KL proxy (same term as README): the KL between each")
    print("capture's distribution CONDITIONED on the shared support (intersection of the two returned top-K")
    print("sets). It is not full-vocabulary KL — mass on tokens the other side did not return and both")
    print("unreturned tails are excluded — and it is a screening measurement, not a quality verdict. covA/covB")
    print("are the total probability each capture puts on that shared support, so condKL is only meaningful")
    print("alongside them. mask = both sides returned no logprobs for the position; noK = returns existed but the")
    print("supports were disjoint, so the position is unscorable. argmax_k is agreement within the returned")
    print("support only.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "compare"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--only", default=None, help="comma list of item names to capture")
    a = ap.parse_args(argv)
    try:
        if a.mode == "capture":
            return capture(a.url, a.paths[0], a.k, set(a.only.split(",")) if a.only else None)
        return compare(a.paths[0], a.paths[1])
    except PanelIOError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except PanelError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
