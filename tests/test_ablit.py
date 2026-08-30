#!/usr/bin/env python3
"""ABLIT overlay tests: recipe integrity, orthogonalization math, TP shards,
module walk, and hook gating. Pure torch — drives the shipped ablit_runtime
(exactly the file the image runs), no vLLM import needed.

Runs at image build (Dockerfile self-check) and on the host with torch.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent


def _find_runtime() -> Path:
    for cand in (
        HERE.parent / "overlay" / "ablit_runtime.py",
        HERE / "ablit_runtime.py",
        Path("/opt/glm53/ablit_runtime.py"),
    ):
        if cand.is_file():
            return cand
    raise SystemExit("ablit_runtime.py not found")


def _find_ablit_dir() -> Path:
    for cand in (
        HERE.parent / "ablit",
        HERE / "ablit",
        Path("/opt/glm53/ablit"),
    ):
        if (cand / "LAYER_MAP.json").is_file():
            return cand
    raise SystemExit("ablit/ with LAYER_MAP.json not found")


spec = importlib.util.spec_from_file_location("ablit_runtime", _find_runtime())
ablit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablit)

ABLIT_DIR = _find_ablit_dir()
HIDDEN = 4096
ALPHA = 3.0


def check_recipe_integrity() -> None:
    m = json.loads((ABLIT_DIR / "LAYER_MAP.json").read_text())
    assert m["hidden_size"] == HIDDEN, m["hidden_size"]
    assert m["num_hidden_layers"] == 45, m["num_hidden_layers"]
    assert m["o_proj_dtype"] == "bfloat16"
    assert m["quant_ignore_includes_o_proj"] is True
    layers = m["layers"]
    assert len(layers) == 46, len(layers)  # 45 base + 1 MTP
    pub = m["published"]
    assert pub["method"] == "dealign-oproj-transplant"
    assert (pub["min_layer"], pub["max_layer"]) == (15, 45)
    for entry in layers:
        assert entry["o_proj"].endswith("self_attn.o_proj.weight"), entry
        assert re.fullmatch(r"model-\d+-of-\d+\.safetensors", entry["shard"]), entry
        if entry["layer"] < pub["min_layer"]:
            assert entry["role"] == "safety-anchor", entry
        else:
            assert entry["role"] in ("edit", "mtp-edit"), entry
    assert sum(e["role"] == "edit" for e in layers) == 30
    assert layers[45]["role"] == "mtp-edit"
    print("recipe integrity OK (46 entries, anchors 0-14, edits 15-45)")


def check_direction_files() -> None:
    for fname in ablit.DIRECTION_FILES.values():
        r = ablit.load_direction(ABLIT_DIR / fname)
        assert r.shape == (HIDDEN,), r.shape
        assert r.dtype == torch.float32
        assert 0.9 < float(r.norm()) < 1.1, float(r.norm())
        obj = torch.load(ABLIT_DIR / fname, map_location="cpu", weights_only=True)
        assert obj["directions"].shape in ((HIDDEN,), (1, HIDDEN)), obj["directions"].shape
    dealign = torch.load(
        ABLIT_DIR / ablit.DIRECTION_FILES["dealign"],
        map_location="cpu",
        weights_only=True,
    )
    assert dealign["source"] == "dealign-oproj-svd-L15-35-39-43-45"
    bf = torch.load(
        ABLIT_DIR / ablit.DIRECTION_FILES["bf_oproj"],
        map_location="cpu",
        weights_only=True,
    )
    assert bf["source"] == "blackfrost-oproj-svd"
    assert bf["alpha_ref"] == 3.0
    print("direction files OK (fp32 [4096], unit-norm, sources match)")


def check_parse_layers() -> None:
    assert ablit.parse_layers("15-45") == list(range(15, 46))
    assert ablit.parse_layers("15,17-19") == [15, 17, 18, 19]
    assert ablit.parse_layers("2-44") == list(range(2, 45))
    for bad in ("45-15", "abc", "", "15-,-20"):
        try:
            ablit.parse_layers(bad)
        except ablit.AblitError:
            continue
        raise AssertionError(f"parse_layers accepted {bad!r}")
    print("layer parsing OK")


def _lin(out_f: int, in_f: int, seed: int, dtype: torch.dtype = torch.float32) -> nn.Linear:
    g = torch.Generator().manual_seed(seed)
    m = nn.Linear(in_f, out_f, bias=False, dtype=dtype)
    with torch.no_grad():
        m.weight.copy_(torch.randn(out_f, in_f, generator=g).to(dtype))
    return m


def _orthobasis(r: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """Orthonormal [4096, dim-1] basis of a subspace orthogonal to r."""
    M = torch.randn(HIDDEN, dim)
    M[:, 0] = r / r.norm()
    Q, _ = torch.linalg.qr(M.double())
    Q = Q.float()
    assert (Q[:, 0] - r / r.norm()).abs().max() < 1e-5
    return Q[:, 1:]


def check_math_fp32() -> None:
    torch.manual_seed(0)
    r = torch.randn(HIDDEN)
    r /= r.norm()
    W0 = _lin(HIDDEN, 8192, seed=1)  # fp32: exact math, no roundtrip noise
    pre = W0.weight.detach().clone()  # untouched reference
    rw = r @ pre  # refusal component before the edit

    rep = ablit.apply_to_o_proj(W0, r, ALPHA)
    assert rep["edited"] and rep["shape"] == (HIDDEN, 8192)

    # exact relation: r^T W'(alpha) = (1 - alpha) * r^T W  (alpha=3 -> inverted x2)
    got = r @ W0.weight.detach()
    assert (got - (1.0 - ALPHA) * rw).abs().max() < 1e-4, float(got.abs().max())

    # alpha=1 removes the refusal component exactly
    W1 = _lin(HIDDEN, 8192, seed=1)
    ablit.apply_to_o_proj(W1, r, 1.0)
    assert (r @ W1.weight).abs().max() < 1e-4

    # delta formula: W'(a) - W = -a * outer(r, r^T W) for unit r
    for a in (1.0, ALPHA):
        Wa = _lin(HIDDEN, 8192, seed=1)
        ablit.apply_to_o_proj(Wa, r, a)
        expected = -a * torch.outer(r, rw)
        assert ((Wa.weight.detach() - pre) - expected).abs().max() < 1e-4, a

    # outputs orthogonal to r are preserved exactly (any alpha): (W'^T - W^T) v = 0
    orth = _orthobasis(r)
    preserved = ((W0.weight.detach().T - pre.T) @ orth).abs().max()
    assert preserved < 1e-4, preserved

    # out-dim mismatch is skipped, not edited
    small = _lin(2048, 8192, seed=2)
    rep = ablit.apply_to_o_proj(small, r, ALPHA)
    assert not rep["edited"] and "out_features" in rep["reason"]
    print("orthogonalization math OK (fp32: alpha relation exact, orth preserved)")


def check_bf16_serve_dtype() -> None:
    """Real o_proj is bf16 — verify the fp32-computed edit survives roundtrip."""
    torch.manual_seed(1)
    r = torch.randn(HIDDEN)
    r /= r.norm()
    W = _lin(HIDDEN, 8192, seed=2, dtype=torch.bfloat16)
    pre = (r @ W.weight.float())  # refusal component before the edit
    rep = ablit.apply_to_o_proj(W, r, ALPHA)
    assert rep["edited"]
    # after the edit the component must equal (1 - alpha) * pre, up to bf16 noise
    got = r @ W.weight.float()
    err = (got - (1.0 - ALPHA) * pre).abs().max()
    assert err < 0.05, err
    print(f"bf16 roundtrip OK (post-edit matches (1-alpha)*pre, max err {float(err.detach()):.2e})")


def check_tp_shard_equivalence() -> None:
    torch.manual_seed(3)
    r = torch.randn(HIDDEN)
    r /= r.norm()
    full = _lin(HIDDEN, 8192, seed=4)
    s1 = _lin(HIDDEN, 4096, seed=4)
    s2 = _lin(HIDDEN, 4096, seed=4)
    with torch.no_grad():
        s1.weight.copy_(full.weight[:, :4096])
        s2.weight.copy_(full.weight[:, 4096:])
    ablit.apply_to_o_proj(full, r, ALPHA)
    ablit.apply_to_o_proj(s1, r, ALPHA)
    ablit.apply_to_o_proj(s2, r, ALPHA)
    joined = torch.cat([s1.weight, s2.weight], dim=1)
    err = (joined - full.weight).abs().max()
    assert err < 1e-2, err
    print(f"TP shard equivalence OK (row-space edit, max err {float(err.detach()):.2e})")


class _FakeAttn(nn.Module):
    def __init__(self, out_f: int, in_f: int, seed: int):
        super().__init__()
        self.o_proj = _lin(out_f, in_f, seed)


class _FakeLayer(nn.Module):
    def __init__(self, idx: int):
        super().__init__()
        # in_f alternates like the real map (KDA 8192 / DSA 16384); shrunk to
        # keep the test light — only out_features==4096 matters to the walker
        in_f = 512 if idx % 4 == 3 else 256
        self.self_attn = _FakeAttn(HIDDEN, in_f, seed=idx)


def _fake_model(with_mtp: bool = True):
    model = nn.Module()
    model.layers = nn.ModuleList([_FakeLayer(i) for i in range(45)])
    if with_mtp:
        mtp_layer = nn.Module()
        mtp_layer.mtp_block = nn.Module()
        mtp_layer.mtp_block.self_attn = _FakeAttn(HIDDEN, 256, seed=99)
        model.mtp = nn.Module()
        model.mtp.layers = nn.ModuleDict({"45": mtp_layer})
    return model


def check_module_walk() -> None:
    r = torch.randn(HIDDEN)
    r /= r.norm()

    model = _fake_model()
    rep = ablit.apply_ablit(model, r, ablit.parse_layers("15-45"), ALPHA, True)
    assert rep["edited_layers"] == list(range(15, 45)), rep["edited_layers"]
    assert rep["mtp_edited"] is True
    assert len(rep["skipped"]) == 0

    rep2 = ablit.apply_ablit(_fake_model(), r, ablit.parse_layers("15-45"), ALPHA, False)
    assert rep2["mtp_edited"] is False
    assert rep2["edited_layers"] == list(range(15, 45))

    model3 = nn.Module()
    model3.layers = nn.ModuleList([_FakeLayer(i) for i in range(3)])
    model3.layers[1].self_attn.o_proj = _lin(2048, 8192, seed=5)
    rep3 = ablit.apply_ablit(model3, r, ablit.parse_layers("0-2"), ALPHA, True)
    assert rep3["edited_layers"] == [0, 2]
    assert len(rep3["skipped"]) == 1 and "out_features" in rep3["skipped"][0]["reason"]

    # layer 45 (MTP) does not exist on the target model alone -> empty match
    rep4 = ablit.apply_ablit(
        _fake_model(with_mtp=False), r, ablit.parse_layers("45"), ALPHA, True
    )
    assert rep4["edited_layers"] == []
    print("module walk OK (target 15-44, MTP 45, skips, empty-match)")


def check_maybe_apply_gating() -> None:
    model = _fake_model()
    r_ref = ablit.load_direction(ABLIT_DIR / ablit.DIRECTION_FILES["dealign"])
    before = model.layers[15].self_attn.o_proj.weight.clone()

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["ABLIT_DIR"] = tmp
        # disabled -> untouched
        os.environ.pop("ABLIT", None)
        assert ablit.maybe_apply(model) is None
        assert torch.equal(before, model.layers[15].self_attn.o_proj.weight)

        # enabled, proj method -> edits, custom direction honored via path
        os.environ["ABLIT"] = "1"
        r = torch.randn(HIDDEN)
        r /= r.norm()
        torch.save({"directions": r, "source": "test"}, f"{tmp}/custom.pt")
        os.environ["ABLIT_METHOD"] = "proj"
        os.environ["ABLIT_DIRECTION"] = f"{tmp}/custom.pt"
        os.environ["ABLIT_LAYERS"] = "15-45"
        os.environ["ABLIT_ALPHA"] = "1.0"  # alpha=1 removes the component exactly
        rep = ablit.maybe_apply(model)
        assert rep is not None and rep["edited_layers"] == list(range(15, 45))
        resid = (r @ model.layers[15].self_attn.o_proj.weight.float()).abs().max()
        assert resid < 1e-2, resid

        # enabled but nothing matches -> loud failure (45 = MTP-only, absent here)
        fresh = _fake_model(with_mtp=False)
        os.environ["ABLIT_LAYERS"] = "45"
        try:
            ablit.maybe_apply(fresh)
        except ablit.AblitError as exc:
            assert "no o_proj was edited" in str(exc)
        else:
            raise AssertionError("maybe_apply succeeded on an empty match")
    for k in ("ABLIT", "ABLIT_DIR", "ABLIT_DIRECTION", "ABLIT_LAYERS", "ABLIT_ALPHA",
              "ABLIT_METHOD"):
        os.environ.pop(k, None)
    assert r_ref.shape == (HIDDEN,)
    print("maybe_apply gating OK (off=no-op, on=edits, empty=loud)")


def check_transplant(tmp_root: str) -> None:
    import hashlib

    tdir = Path(tmp_root) / "transplant"
    tdir.mkdir(parents=True, exist_ok=True)
    donors = {}
    entries = {}
    g = torch.Generator().manual_seed(123)
    for L in range(15, 46):
        in_f = 512 if L % 4 == 3 else 256  # matches _FakeLayer / fake-MTP o_proj shapes
        donor = torch.randn(HIDDEN, in_f, generator=g).to(torch.bfloat16)
        raw = donor.contiguous().view(torch.uint8).numpy().tobytes()
        (tdir / f"L{L}.bin").write_bytes(raw)
        entries[str(L)] = {
            "shard": f"model-{L:05d}-of-00120.safetensors",
            "key": f"model.language_model.layers.{L}.self_attn.o_proj.weight",
            "dtype": "BF16", "shape": [HIDDEN, in_f],
            "nbytes": donor.numel() * 2,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    (tdir / "MANIFEST.json").write_text(json.dumps({"donor": "test", "layers": entries}))

    model = _fake_model(with_mtp=True)
    os.environ["ABLIT"] = "1"
    os.environ["ABLIT_DIR"] = str(tmp_root)
    os.environ["ABLIT_METHOD"] = "transplant"
    os.environ["ABLIT_LAYERS"] = "15-45"
    os.environ["ABLIT_INCLUDE_MTP"] = "1"
    rep = ablit.maybe_apply(model)
    assert rep["edited_layers"] == list(range(15, 46)), rep["edited_layers"]  # 45 = MTP block
    assert rep["mtp_edited"] is True
    assert len(rep["deltas"]) == 31
    # weights must be byte-identical to the donor shards (world=1 in tests)
    for L in (15, 44):
        donor = ablit.load_transplant_tensors(str(tmp_root), [L])[L]
        w = model.layers[L].self_attn.o_proj.weight
        assert torch.equal(w.data, donor), L
        assert rep["deltas"][L]["rel_l2"] > 0.1  # donor != stock

    # TP shard slice: two ranks hold stock column-shards; after each rank
    # applies the transplant, the concatenated result must equal the donor.
    ablit._tp_world = lambda: 2
    dinf = lambda L: 512 if L % 4 == 3 else 256
    g2 = torch.Generator().manual_seed(77)
    stock_full = {L: torch.randn(HIDDEN, dinf(L), generator=g2) for L in range(15, 46)}
    donors_loaded = ablit.load_transplant_tensors(str(tmp_root), list(range(15, 46)))
    rank_out = {}
    for rank in (0, 1):
        ablit._tp_rank = (lambda r: lambda: r)(rank)
        m = _fake_model(with_mtp=False)
        with torch.no_grad():
            for L in range(15, 45):  # target decoder layers only (45 = MTP, absent)
                loc = dinf(L) // 2
                # rank-local shard-width module holding this rank's stock slice
                m.layers[L].self_attn.o_proj = _lin(HIDDEN, loc, seed=L,
                                                    dtype=torch.bfloat16)
                m.layers[L].self_attn.o_proj.weight.copy_(
                    stock_full[L][:, rank * loc:(rank + 1) * loc].to(torch.bfloat16))
        ablit.apply_transplant(m, donors_loaded, list(range(15, 46)), True)
        rank_out[rank] = {L: m.layers[L].self_attn.o_proj.weight.detach().clone()
                          for L in range(15, 45)}
    ablit._tp_world = lambda: 1
    ablit._tp_rank = lambda: 0
    for L in (16, 20, 31, 44):
        joined = torch.cat([rank_out[0][L], rank_out[1][L]], dim=1)
        assert torch.equal(joined, donors_loaded[L]), L

    # Live-serve crash: CPU frombuffer donor vs CUDA o_proj at rel_l2.
    if torch.cuda.is_available():
        mcu = _fake_model(with_mtp=False)
        w = mcu.layers[15].self_attn.o_proj.weight
        w.data = w.data.cuda()
        donors_cpu = ablit.load_transplant_tensors(str(tmp_root), [15])
        assert donors_cpu[15].device.type == "cpu"
        ablit.apply_transplant(mcu, donors_cpu, [15], False)
        got = mcu.layers[15].self_attn.o_proj.weight
        assert got.device.type == "cuda"
        assert torch.equal(got.cpu(), donors_cpu[15])
        print("transplant CPU-donor -> CUDA-weight OK")
    else:
        print("transplant CUDA device-mismatch skip (no GPU)")

    # auto fallback when transplant dir empty -> proj path needs a direction
    os.environ["ABLIT_METHOD"] = "auto"
    os.environ["ABLIT_DIR"] = str(tmp_root)
    try:
        os.remove(tdir / "MANIFEST.json")
        for L in range(15, 46):
            (tdir / f"L{L}.bin").unlink()
        ablit.maybe_apply(_fake_model(with_mtp=False))
    except ablit.AblitError:
        pass  # falls to proj path with no direction file -> loud
    else:
        raise AssertionError("auto fallback without direction should fail loud")
    for k in ("ABLIT", "ABLIT_DIR", "ABLIT_METHOD", "ABLIT_LAYERS", "ABLIT_INCLUDE_MTP"):
        os.environ.pop(k, None)
    print("transplant OK (byte-copy, TP shard slice, deltas, auto fallback)")


def main() -> None:
    check_recipe_integrity()
    check_direction_files()
    check_parse_layers()
    check_math_fp32()
    check_bf16_serve_dtype()
    check_tp_shard_equivalence()
    check_module_walk()
    check_maybe_apply_gating()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        check_transplant(tmp)
    print("glm53 ablit overlay verify OK")


if __name__ == "__main__":
    sys.exit(main())
