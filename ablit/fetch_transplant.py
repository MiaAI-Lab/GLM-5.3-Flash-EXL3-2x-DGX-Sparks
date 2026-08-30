#!/usr/bin/env python3
"""Fetch the abliterated o_proj tensors (L15-45) from the Dealign donor repo.

Pulls only the `self_attn.o_proj.weight` byte ranges out of the donor's 120
safetensors shards via HTTP range-requests (~2.7 GiB instead of ~164 GiB),
into ablit/transplant/L{NN}.bin + MANIFEST.json. Resumable: files whose
sha256 matches the manifest are skipped.

Run on the head (needs `python3` + network; no torch):
    ABLIT_DONOR=dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 python3 ablit/fetch_transplant.py
    # token: HF_TOKEN env or ~/.cache/huggingface/token (donor is public; the
    # token only helps with rate limits)

The ABLIT runtime (overlay/ablit_runtime.py) picks these up when
ABLIT_METHOD=auto|transplant and replaces the stock o_proj weights at load —
the exact published "dealign-oproj-transplant" edit (METHOD.md of
drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "ablit" / "transplant"
LAYER_MAP_PATH = REPO_ROOT / "ablit" / "LAYER_MAP.json"

DONOR = os.environ.get("ABLIT_DONOR", "dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4")
BASE = f"https://huggingface.co/{DONOR}/resolve/main"
TOKEN = os.environ.get("HF_TOKEN") or ""
if not TOKEN:
    tok_file = Path.home() / ".cache/huggingface/token"
    if tok_file.is_file():
        TOKEN = tok_file.read_text().strip()

CHUNK = 8 * 1024 * 1024
RETRIES = 4


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def http(url: str, headers: dict | None = None, start: int | None = None, end: int | None = None):
    req = urllib.request.Request(url, headers=dict(headers or {}))
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end if end is not None else ''}")
    return urllib.request.urlopen(req, timeout=120)


def fetch_bytes(url: str, start: int, end: int, headers: dict, what: str) -> bytes:
    """Fetch [start, end] inclusive with retries + resume within the range."""
    got = 0
    total = end - start + 1
    for attempt in range(RETRIES):
        try:
            out = bytearray()
            pos = start + got
            while pos <= end:
                rsp = http(url, headers, pos, end)
                if rsp.status not in (200, 206):
                    raise RuntimeError(f"HTTP {rsp.status}")
                while True:
                    chunk = rsp.read(CHUNK)
                    if not chunk:
                        break
                    out += chunk
                    pos += len(chunk)
                    got = len(out)
                if len(out) == total:
                    return bytes(out)
                rsp.close()
            raise RuntimeError(f"short read {len(out)}/{total}")
        except Exception as exc:  # noqa: BLE001
            wait = 3 * (attempt + 1)
            print(f"  retry {attempt + 1}/{RETRIES} for {what} at {got}/{total} "
                  f"({exc}) — sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"FAILED: {what}")


def safetensors_span(shard_url: str, headers: dict, key: str) -> tuple[int, int, dict]:
    """Resolve (abs_start, abs_end, meta) for one tensor via its header."""
    with http(shard_url, headers, 0, 7) as rsp:
        n = int.from_bytes(rsp.read(8), "little")
    hdr_bytes = fetch_bytes(shard_url, 8, 8 + n - 1, headers, f"header of {shard_url.rsplit('/', 1)[-1]}")
    header = json.loads(hdr_bytes)
    if key not in header:
        raise SystemExit(f"{key} not in {shard_url} header (keys={len(header)})")
    meta = header[key]
    a, b = meta["data_offsets"]
    return 8 + n + a, 8 + n + b - 1, meta


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    lmap = json.loads(LAYER_MAP_PATH.read_text())
    layers = sorted(e["layer"] for e in lmap["layers"]
                    if e["role"] in ("edit", "mtp-edit"))
    print(f"donor : {DONOR}")
    print(f"layers: {layers[0]}..{layers[-1]} ({len(layers)} tensors)")

    # authoritative shard map from the donor's own index
    idx_url = f"{BASE}/model.safetensors.index.json"
    for attempt in range(RETRIES):
        try:
            with http(idx_url, auth()) as rsp:
                idx = json.loads(rsp.read())["weight_map"]
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == RETRIES - 1:
                raise SystemExit(f"cannot fetch index: {exc}")
            time.sleep(3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "MANIFEST.json"
    manifest = {
        "donor": DONOR,
        "donor_sha": None,
        "method": "dealign-oproj-transplant (byte-copy, o_proj L15-45 incl. MTP)",
        "tensors": {},
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("donor") == DONOR:
            manifest = old
            print("resuming: existing manifest found")
    manifest["layers"] = manifest.get("tensors", {})

    for L in layers:
        key = f"model.language_model.layers.{L}.self_attn.o_proj.weight"
        shard = idx.get(key)
        if shard is None:
            raise SystemExit(f"donor index has no {key}")
        lmap_shard = next((e["shard"] for e in lmap["layers"] if e["layer"] == L), None)
        if lmap_shard and lmap_shard != shard:
            print(f"  note: L{L} shard differs from LAYER_MAP ({shard} vs {lmap_shard}) — using donor index")
        out = OUT_DIR / f"L{L}.bin"
        if out.is_file() and L in manifest.get("layers", {}) \
                and manifest["layers"][L].get("sha256") == sha256(out.read_bytes()):
            print(f"L{L}: already fetched ({out.stat().st_size / 1e6:.0f} MB)")
            continue

        shard_url = f"{BASE}/{shard}"
        print(f"L{L}: {key} @ {shard}")
        start, end, meta = safetensors_span(shard_url, auth(), key)
        nbytes = end - start + 1
        print(f"  {meta['dtype']} {meta['shape']} = {nbytes / 1e6:.0f} MB")
        t0 = time.time()
        data = fetch_bytes(shard_url, start, end, auth(), key)
        dt = time.time() - t0
        print(f"  fetched in {dt:.1f}s ({nbytes / 1e6 / max(dt, 1e-6):.0f} MB/s)")
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(out)
        manifest.setdefault("layers", {})[L] = {
            "shard": shard,
            "key": key,
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "nbytes": nbytes,
            "sha256": sha256(data),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    ok = sum(1 for L in layers if L in manifest.get("layers", {}))
    total = sum(m["nbytes"] for m in manifest.get("layers", {}).values())
    print(f"done: {ok}/{len(layers)} tensors, {total / 1e9:.2f} GB -> {OUT_DIR}")


if __name__ == "__main__":
    main()
