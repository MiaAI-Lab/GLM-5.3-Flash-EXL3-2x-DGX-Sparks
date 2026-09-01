#!/usr/bin/env python3
"""Tests for the stage-3 mamba cross-boot differential harness
(tests/_mamba_differential.py).

WHAT THIS IS (Codex OFFLOAD2 finding 14, adopted): state-ABI/codec fixture
tests + the bit-comparator stage 3 will run against real captures. It is NOT
stage-3 evidence: no live state is captured, the mamba state-index convention
is still an open stage-0 residual, and the live protocol
(tests/stage3_mamba_differential_protocol.sh) is guarded and not run here.

A  Park fixtures: boot-A park-at-boundary-k writes 4 mamba chunk files +
   a boundary manifest in the stage-1 on-disk format, real (unpadded) bytes
   only, num_spec + state-ABI tag in every header.
B  ABI field checks: conv (10,12288) bf16 245,760 B + temporal (32,128,128)
   fp32 2,097,152 B per layer, real page 2,342,912 B, 9/9/8/8 layers,
   padded bytes never on disk; a violated fixture is FLAGGED.
C  Bit comparator: identical boots compare clean; a restored-boot copy of
   boot A compares clean; a single flipped bit is localized to the exact
   (group, layer, tensor); a divergent boot reports every tensor.

Run:  python3 tests/test_mamba_differential_harness.py   (or pytest)
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "overlay"))

import _mamba_differential as md  # noqa: E402
import patch_kv_offload_store_local as store  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


def test_harness() -> None:
    print("Mamba differential harness (ABI/codec fixtures — NOT stage-3 evidence)")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        k = 1
        base_a = md.write_park_fixture(root / "bootA", "bootA", boundary_k=k)
        bhash = md.boundary_hash("chainA", k)

        files = sorted(p.name for p in base_a.rglob("*.bin"))
        check(
            len(files) == 4 and all(n == f"{bhash}.bin" for n in files),
            "A1 boot-A park: one chunk file per mamba group (g2-g5)",
        )
        mpath = base_a / "manifests" / bhash[:3] / f"{bhash}.json"
        check(mpath.is_file(), "A2 boundary manifest published")
        import json

        man = json.loads(mpath.read_text())
        check(
            len(man["chunk_hashes"]) == k + 1
            and set(man["cow_groups"]) == {"2", "3", "4", "5"},
            "A3 manifest: cumulative chain + all four mamba groups",
        )
        g2 = base_a / bhash[:3] / f"{bhash[3:5]}_g2" / f"{bhash}.bin"
        h2 = store.read_chunk_header(str(g2), verify_payload=True)
        check(
            h2["payload_len"] == 9 * md.KDA_REAL_PAGE
            and h2["payload_len"] == 21_086_208,
            f"A4 g2 payload = 9 x 2,342,912 real bytes (got {h2['payload_len']})",
        )
        check(
            h2["payload_len"] != 9 * md.KDA_PADDED_PAGE,
            "A5 slot-share padding never reaches disk",
        )

        # B: ABI checks pass on a good fixture; violations are flagged.
        check(md.check_state_abi(base_a, k) == [], "B1 ABI field checks pass")
        bad_root = root / "bootBad"
        base_bad = md.write_park_fixture(bad_root, "bootBad", boundary_k=k)
        g3 = base_bad / bhash[:3] / f"{bhash[3:5]}_g3" / f"{bhash}.bin"
        payload, segs = md.group_payload_and_segments("bootBad", 3)
        hdr = store.read_chunk_header(str(g3))
        hdr_bad = {
            key: hdr[key]
            for key in hdr
            if key not in ("payload_len", "payload_crc32", "crc_algo", "format_version")
        }
        hdr_bad["num_speculative_tokens"] = 8  # ABI break: conv width changes
        g3.write_bytes(store.encode_chunk(hdr_bad, payload))
        problems = md.check_state_abi(base_bad, k)
        check(
            any("num_spec" in p for p in problems),
            f"B2 num_spec drift is FLAGGED (got {problems})",
        )

        # C: comparator.
        base_a2 = md.write_park_fixture(root / "bootA2", "bootA", boundary_k=k)
        check(
            md.compare_state_bits(base_a, base_a2, k) == {},
            "C1 identical park states compare bit-clean",
        )
        # A 'restored' boot B: byte-copy of boot A's store (what a correct
        # cross-boot restore must reproduce) compares clean.
        restored = root / "bootB-restored" / base_a.name
        shutil.copytree(base_a, restored)
        check(
            md.compare_state_bits(base_a, restored, k) == {},
            "C2 restored-copy boot compares bit-clean",
        )
        # A single flipped bit localizes to the exact (group, layer, tensor).
        g4 = restored / bhash[:3] / f"{bhash[3:5]}_g4" / f"{bhash}.bin"
        raw = bytearray(g4.read_bytes())
        hlen = int.from_bytes(raw[8:12], "little")
        # Flip one bit inside layer kda4.2's temporal state.
        seg = next(
            s
            for s in store.read_chunk_header(str(g4))["segment_table"]
            if s["layer"] == "kda4.2" and s["kind"] == "temporal_state"
        )
        raw[16 + hlen + seg["offset"] + 5] ^= 0x01
        # Re-encode so the CRC stays valid (the comparator verifies payloads;
        # a CRC-invalid file is a CODEC failure, not a state difference).
        hdr4 = store.read_chunk_header(str(g4))
        hdr4_clean = {
            key: hdr4[key]
            for key in hdr4
            if key not in ("payload_len", "payload_crc32", "crc_algo", "format_version")
        }
        g4.write_bytes(store.encode_chunk(hdr4_clean, bytes(raw[16 + hlen :])))
        diff = md.compare_state_bits(base_a, restored, k)
        check(
            diff == {(4, "kda4.2", "temporal_state"): 1},
            f"C3 single bit flip localized to (g4, kda4.2, temporal) (got {diff})",
        )
        # A genuinely different boot reports every tensor.
        base_c = md.write_park_fixture(root / "bootC", "bootC", boundary_k=k)
        diff_c = md.compare_state_bits(base_a, base_c, k)
        check(
            len(diff_c) == 2 * sum(md.MAMBA_GROUP_LAYERS.values()),
            f"C4 divergent boot: every (layer, tensor) differs "
            f"(got {len(diff_c)} of {2 * sum(md.MAMBA_GROUP_LAYERS.values())})",
        )


def main() -> int:
    test_harness()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
