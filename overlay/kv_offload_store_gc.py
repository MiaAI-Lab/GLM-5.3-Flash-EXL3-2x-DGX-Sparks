#!/usr/bin/env python3
"""Verify / garbage-collect a GLM53 KV-offload store (stage-1 companion tool).

Walks one per-rank store base directory (``glm53kv_<model>_<ns>_r<rank>``, as
written by ``patch_kv_offload_store_local.py``), header-verifies every chunk
file, cross-references boundary manifests, and reports:

- corrupt/torn files (bad magic, truncated header/payload, CRC mismatch);
- orphan payloads: mamba-group chunks referenced by NO live manifest (a crash
  between group writes and manifest publish leaves exactly these);
- manifest health: per manifest, whether every referenced payload (the four
  mamba chunks at the boundary and the CUMULATIVE g0 chunk chain — plan §4
  C1) exists and matches its recorded size/CRC.

DRY-RUN by default: nothing is deleted or modified unless flags say so.

- ``--sweep``          delete corrupt files and orphan mamba payloads
- ``--keep-boundaries K``  manifest-mediated retention (plan §7): per chain,
                       keep the K most recent boundary manifests (by
                       boundary_token_index), mark the rest superseded
                       (delete the manifest file), then sweep payloads no
                       live manifest references. Full-attention chunks are
                       liveness-protected by ANY live manifest of any chain
                       (cumulative references), so superseding a boundary
                       never strands a shallower restorable boundary.
- ``--verify-crc``     full payload CRC verification (reads every byte)

Chains are identified by their deepest manifest's chunk-hash chain: manifest
A belongs to chain C when A's chunk_hashes is a prefix of C's. Runs on the
host against the mounted store dir or inside the container; stdlib only.

Exit code: 0 clean, 1 findings (corrupt/orphans), 2 usage/IO error.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# The header codec lives in the store overlay (single implementation).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_kv_offload_store_local import read_chunk_header  # noqa: E402


def find_chunk_files(base: Path):
    """Yield (path, hash_hex, group_idx) for every .bin under the base."""
    for sub1 in sorted(base.iterdir()):
        if not sub1.is_dir() or sub1.name == "manifests":
            continue
        for sub2 in sorted(sub1.iterdir()):
            if not sub2.is_dir() or "_g" not in sub2.name:
                continue
            try:
                group_idx = int(sub2.name.rsplit("_g", 1)[1])
            except ValueError:
                continue
            for f in sorted(sub2.glob("*.bin")):
                yield f, f.stem, group_idx


def load_manifests(base: Path):
    import json

    manifests = []
    mdir = base / "manifests"
    if not mdir.is_dir():
        return manifests
    for sub in sorted(mdir.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.glob("*.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    m = json.load(fh)
            except (OSError, ValueError) as exc:
                manifests.append({"_path": f, "_error": str(exc)})
                continue
            m["_path"] = f
            manifests.append(m)
    return manifests


def assign_chains(manifests):
    """Group manifests into chains: A joins C's chain when A.chunk_hashes is a
    prefix of C.chunk_hashes (C the deepest)."""
    valid = [m for m in manifests if "_error" not in m and m.get("chunk_hashes")]
    valid.sort(key=lambda m: len(m["chunk_hashes"]), reverse=True)
    chains: list[list[dict]] = []
    for m in valid:
        placed = False
        for chain in chains:
            deepest = chain[0]["chunk_hashes"]
            mine = m["chunk_hashes"]
            if deepest[: len(mine)] == mine:
                chain.append(m)
                placed = True
                break
        if not placed:
            chains.append([m])
    return chains


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base", help="per-rank store base dir (…_r<rank>)")
    ap.add_argument("--sweep", action="store_true", help="delete corrupt files and orphan mamba payloads")
    ap.add_argument("--keep-boundaries", type=int, default=0, metavar="K", help="keep only the K most recent boundary manifests per chain (0 = keep all)")
    ap.add_argument("--verify-crc", action="store_true", help="full payload CRC verification")
    args = ap.parse_args(argv)

    base = Path(args.base)
    if not base.is_dir():
        print(f"ERROR: not a directory: {base}", file=sys.stderr)
        return 2
    dry = not args.sweep

    corrupt: list[Path] = []
    headers: dict[tuple[str, int], dict] = {}
    total_bytes = 0
    n_files = 0
    for path, hash_hex, group_idx in find_chunk_files(base):
        n_files += 1
        total_bytes += path.stat().st_size
        try:
            h = read_chunk_header(str(path), verify_payload=args.verify_crc)
        except (OSError, ValueError) as exc:
            corrupt.append(path)
            print(f"CORRUPT {path}: {exc}")
            continue
        if h.get("hash") != hash_hex or h.get("group_idx") != group_idx:
            corrupt.append(path)
            print(f"CORRUPT {path}: header identity mismatch (hash/group)")
            continue
        headers[(hash_hex, group_idx)] = h

    manifests = load_manifests(base)
    broken_manifests = [m for m in manifests if "_error" in m]
    for m in broken_manifests:
        print(f"BAD-MANIFEST {m['_path']}: {m['_error']}")

    # Retention: keep K most recent per chain (by boundary_token_index).
    superseded: list[dict] = []
    chains = assign_chains(manifests)
    if args.keep_boundaries > 0:
        for chain in chains:
            chain.sort(key=lambda m: m.get("boundary_token_index", 0), reverse=True)
            for m in chain[args.keep_boundaries :]:
                superseded.append(m)
        for m in superseded:
            print(f"SUPERSEDE {m['_path']} (boundary {m.get('boundary_token_index')})")
            if not dry:
                m["_path"].unlink(missing_ok=True)
        live = [
            m
            for m in manifests
            if "_error" not in m and m not in superseded
        ]
    else:
        live = [m for m in manifests if "_error" not in m]

    # Liveness: any chunk referenced by any live manifest.
    live_keys: set[tuple[str, int]] = set()
    incomplete_manifests = 0
    for m in live:
        ok = True
        chunk_hashes = m.get("chunk_hashes") or []
        for gidx_s, entry in (m.get("cow_groups") or {}).items():
            key = (entry.get("hash"), int(gidx_s))
            live_keys.add(key)
            h = headers.get(key)
            if h is None:
                print(f"MANIFEST-MISSING-PAYLOAD {m['_path']}: g{gidx_s} {entry.get('hash', '?')[:12]}")
                ok = False
            elif (
                h.get("payload_len") != entry.get("payload_len")
                or h.get("payload_crc32") != entry.get("payload_crc32")
            ):
                print(f"MANIFEST-PAYLOAD-MISMATCH {m['_path']}: g{gidx_s}")
                ok = False
        for gidx_s in (m.get("full_groups") or {}):
            for hh in chunk_hashes:
                key = (hh, int(gidx_s))
                live_keys.add(key)
                if key not in headers:
                    print(f"MANIFEST-MISSING-PAYLOAD {m['_path']}: g{gidx_s} {hh[:12]}")
                    ok = False
        if not ok:
            incomplete_manifests += 1

    # Orphans: mamba-group payloads no live manifest references. Full-attn
    # chunks without a manifest are NOT orphans by default (a boundary may
    # legitimately be mid-store); report them only under retention mode.
    mamba_groups = {
        int(g) for m in live for g in (m.get("cow_groups") or {})
    }
    orphans = [
        (key, h)
        for key, h in headers.items()
        if key not in live_keys
        and (key[1] in mamba_groups or (args.keep_boundaries > 0))
    ]
    for (hash_hex, gidx), _h in orphans:
        print(f"ORPHAN {hash_hex} g{gidx}")

    if not dry:
        for path in corrupt:
            path.unlink(missing_ok=True)
            print(f"DELETED {path}")
        for (hash_hex, gidx), _h in orphans:
            p = base / hash_hex[:3] / f"{hash_hex[3:5]}_g{gidx}" / f"{hash_hex}.bin"
            p.unlink(missing_ok=True)
            print(f"DELETED {p}")

    print(
        f"SUMMARY files={n_files} bytes={total_bytes} manifests={len(manifests)} "
        f"chains={len(chains)} corrupt={len(corrupt)} orphans={len(orphans)} "
        f"superseded={len(superseded)} incomplete_manifests={incomplete_manifests} "
        f"broken_manifests={len(broken_manifests)} mode={'DRY-RUN' if dry else 'SWEEP'}"
    )
    return 1 if (corrupt or orphans or broken_manifests or incomplete_manifests) else 0


if __name__ == "__main__":
    sys.exit(main())
