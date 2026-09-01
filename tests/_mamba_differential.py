#!/usr/bin/env python3
"""Stage-3 mamba cross-boot differential harness — fixture side.

SCOPE (Codex OFFLOAD2 finding 14, adopted): this module is a **state-ABI /
codec fixture comparator**, NOT stage-3 evidence. It builds park-at-boundary
fixtures whose payloads are STUBBED KDA state tensors at the receipted ABI
(STAGE0-RECEIPTS R13), and provides the exact bit-comparator stage 3 will run
against REAL captures. The mamba state-index convention is still an OPEN
stage-0 residual ("Strides: derived-contiguous, not observed live; the
state-index convention remains stage-2/3" — STAGE0-RECEIPTS A2); nothing here
proves that an align-mode snapshot restored on another boot is semantically
sufficient. The live instrument is tests/stage3_mamba_differential_protocol.sh
(guarded, not run in stage 2).

The receipted KDA state ABI (deployed config: TP=2, heads 64, head_dim 128,
conv kernel 4, num_spec 7):
  conv_state     (10, 12288)   bfloat16  = 245,760 B   (width = kernel-1+num_spec)
  temporal_state (32, 128, 128) float32  = 2,097,152 B
  real per-layer page = 2,342,912 B  (logged mamba page 2,351,104 is PADDED
  by +8,192 B/layer slot-share padding — padding must never hit disk)
Groups 2/3/4/5 carry 9/9/8/8 KDA layers per rank.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "overlay"))

import patch_kv_offload_store_local as store  # noqa: E402

# The receipted ABI (stage-0 R13). num_spec is part of the ABI: it is baked
# into the conv width, so it rides both the chunk headers and the namespace.
KDA_CONV_SHAPE = (10, 12288)
KDA_CONV_DTYPE = "torch.bfloat16"
KDA_CONV_BYTES = 245_760
KDA_TEMPORAL_SHAPE = (32, 128, 128)
KDA_TEMPORAL_DTYPE = "torch.float32"
KDA_TEMPORAL_BYTES = 2_097_152
KDA_REAL_PAGE = KDA_CONV_BYTES + KDA_TEMPORAL_BYTES  # 2,342,912
KDA_PADDED_PAGE = 2_351_104  # never on disk
NUM_SPEC = 7
MAMBA_GROUP_LAYERS = {2: 9, 3: 9, 4: 8, 5: 8}
STATE_ABI_VERSION = "v0-kda-r13"
TOKENS_PER_CHUNK = 3584


def _tile(seed: str, n: int) -> bytes:
    """Deterministic pseudo-random bytes: sha256 counter-mode tiling."""
    out = bytearray()
    counter = 0
    base = seed.encode()
    while len(out) < n:
        out += hashlib.sha256(base + counter.to_bytes(8, "big")).digest() * 256
        counter += 1
    return bytes(out[:n])


def kda_layer_state(boot: str, group: int, layer: int) -> tuple[bytes, bytes]:
    """One layer's (conv, temporal) stubbed state bytes for a given boot."""
    conv = _tile(f"{boot}:g{group}:l{layer}:conv", KDA_CONV_BYTES)
    temporal = _tile(f"{boot}:g{group}:l{layer}:temporal", KDA_TEMPORAL_BYTES)
    return conv, temporal


def group_payload_and_segments(boot: str, group: int) -> tuple[bytes, list]:
    """The chunk payload (all layers, real bytes only) + its segment table,
    exactly as the stage-1 writer lays them out."""
    n_layers = MAMBA_GROUP_LAYERS[group]
    parts: list[bytes] = []
    segments: list[dict] = []
    offset = 0
    for layer_idx in range(n_layers):
        conv, temporal = kda_layer_state(boot, group, layer_idx)
        layer = f"kda{group}.{layer_idx}"
        for kind, blob, shape, dtype in (
            ("conv_state", conv, KDA_CONV_SHAPE, KDA_CONV_DTYPE),
            ("temporal_state", temporal, KDA_TEMPORAL_SHAPE, KDA_TEMPORAL_DTYPE),
        ):
            strides = []
            acc = 1
            for d in reversed(shape):
                strides.append(acc)
                acc *= int(d)
            strides.reverse()
            segments.append(
                {
                    "layer": layer,
                    "kind": kind,
                    "shape": list(shape),
                    "dtype": dtype,
                    "stride": strides,
                    "stride_provenance": "derived-contiguous",
                    "offset": offset,
                    "length": len(blob),
                }
            )
            parts.append(blob)
            offset += len(blob)
    return b"".join(parts), segments


def boundary_hash(chain: str, k: int) -> str:
    return hashlib.sha256(f"{chain}:boundary:{k}".encode()).hexdigest()


def write_park_fixture(
    root: Path,
    boot: str,
    rank: int = 0,
    boundary_k: int = 1,
    namespace: str = "mamba-diff-ns0",
    chain: str = "chainA",
) -> Path:
    """Write one boot's park-at-boundary-k fixture: the four mamba group
    chunk files + the boundary manifest, in the stage-1 on-disk format
    (format v1 headers, real bytes only, tmp-free direct writes — this is a
    fixture generator, not the durable writer).

    Deliberate fixture scope (review f12 note): only the TERMINAL boundary's
    mamba chunks are materialized even though the manifest chain lists
    earlier boundary hashes — mamba restore needs exactly ONE state block
    per group at the target boundary (plan §3), and this harness compares
    states at that boundary; earlier boundaries would be separate park
    fixtures of their own."""
    base = root / f"glm53kv_test_model_{namespace}_r{rank}"
    bhash = boundary_hash(chain, boundary_k)
    cow_entries: dict[str, dict] = {}
    for group in sorted(MAMBA_GROUP_LAYERS):
        payload, segments = group_payload_and_segments(boot, group)
        header = {
            "namespace_hash": namespace,
            "group_idx": group,
            "spec_kind": "MambaSpec",
            "layers": MAMBA_GROUP_LAYERS[group],
            "block_size_tokens": TOKENS_PER_CHUNK,
            "n_tokens_valid": TOKENS_PER_CHUNK,
            "boundary_token_index": (boundary_k + 1) * TOKENS_PER_CHUNK,
            "tp_rank": rank,
            "tp_world": 2,
            "segment_table": segments,
            "hash": bhash,
            "state_abi_version": STATE_ABI_VERSION,
            "num_speculative_tokens": NUM_SPEC,
        }
        blob = store.encode_chunk(header, payload)
        path = base / bhash[:3] / f"{bhash[3:5]}_g{group}" / f"{bhash}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        h = store.read_chunk_header(str(path))
        cow_entries[str(group)] = {
            "hash": bhash,
            "payload_len": h["payload_len"],
            "payload_crc32": h["payload_crc32"],
        }
    manifest = {
        "format_version": 1,
        "namespace_hash": namespace,
        "boundary_token_index": (boundary_k + 1) * TOKENS_PER_CHUNK,
        "chunk_hashes": [boundary_hash(chain, j) for j in range(boundary_k + 1)],
        "cow_groups": cow_entries,
        "full_groups": {},
        "rank": rank,
        "created_at": 0,
    }
    mpath = base / "manifests" / bhash[:3] / f"{bhash}.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, sort_keys=True))
    return base


def check_state_abi(base: Path, boundary_k: int, chain: str = "chainA") -> list[str]:
    """Field checks against the receipted ABI. Returns a list of violations
    (empty = pass): shapes, dtypes, per-tensor sizes, num_spec in the header,
    real (unpadded) payload sizes — the padding must never be on disk."""
    problems: list[str] = []
    bhash = boundary_hash(chain, boundary_k)
    for group, n_layers in MAMBA_GROUP_LAYERS.items():
        path = base / bhash[:3] / f"{bhash[3:5]}_g{group}" / f"{bhash}.bin"
        try:
            h = store.read_chunk_header(str(path), verify_payload=True)
        except (OSError, ValueError) as exc:
            problems.append(f"g{group}: unreadable chunk: {exc}")
            continue
        if h.get("state_abi_version") != STATE_ABI_VERSION:
            problems.append(f"g{group}: state_abi_version {h.get('state_abi_version')!r}")
        if h.get("num_speculative_tokens") != NUM_SPEC:
            problems.append(f"g{group}: num_spec {h.get('num_speculative_tokens')!r}")
        if h.get("payload_len") != n_layers * KDA_REAL_PAGE:
            problems.append(
                f"g{group}: payload {h.get('payload_len')} != "
                f"{n_layers}x{KDA_REAL_PAGE} (padded bytes on disk?)"
            )
        segs = h.get("segment_table") or []
        if len(segs) != 2 * n_layers:
            problems.append(f"g{group}: {len(segs)} segments != {2 * n_layers}")
            continue
        run = 0
        for i, seg in enumerate(segs):
            layer_idx = i // 2
            if i % 2 == 0:
                want_shape, want_dtype, want_len, want_kind = (
                    list(KDA_CONV_SHAPE), KDA_CONV_DTYPE, KDA_CONV_BYTES,
                    "conv_state",
                )
            else:
                want_shape, want_dtype, want_len, want_kind = (
                    list(KDA_TEMPORAL_SHAPE), KDA_TEMPORAL_DTYPE,
                    KDA_TEMPORAL_BYTES, "temporal_state",
                )
            want_strides = []
            acc = 1
            for d in reversed(want_shape):
                want_strides.append(acc)
                acc *= int(d)
            want_strides.reverse()
            got = (
                seg.get("shape"), seg.get("dtype"), seg.get("length"),
                seg.get("kind"), seg.get("layer"), seg.get("stride"),
                seg.get("stride_provenance"), seg.get("offset"),
            )
            want = (
                want_shape, want_dtype, want_len, want_kind,
                f"kda{group}.{layer_idx}", want_strides,
                "derived-contiguous", run,
            )
            if got != want:
                problems.append(f"g{group} seg{i}: {got} != {want}")
            run += want_len
    return problems


def compare_state_bits(
    base_a: Path, base_b: Path, boundary_k: int, chain: str = "chainA"
) -> dict[tuple[int, str, str], int]:
    """The stage-3 comparator: per-(group, layer, tensor-kind) mismatching
    byte counts between two boots' parked states at the same boundary.
    Walks the VALIDATED segment tables (never raw offsets)."""
    bhash = boundary_hash(chain, boundary_k)
    result: dict[tuple[int, str, str], int] = {}
    for group in sorted(MAMBA_GROUP_LAYERS):
        payloads = []
        tables = []
        for base in (base_a, base_b):
            path = base / bhash[:3] / f"{bhash[3:5]}_g{group}" / f"{bhash}.bin"
            h = store.read_chunk_header(str(path), verify_payload=True)
            raw = path.read_bytes()
            hlen = int.from_bytes(raw[8:12], "little")
            payloads.append(raw[16 + hlen :])
            tables.append(h["segment_table"])
        if tables[0] != tables[1]:
            result[(group, "*", "segment-table")] = -1
            continue
        for seg in tables[0]:
            a = payloads[0][seg["offset"] : seg["offset"] + seg["length"]]
            b = payloads[1][seg["offset"] : seg["offset"] + seg["length"]]
            n = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
            if n:
                result[(group, seg["layer"], seg["kind"])] = n
    return result
