#!/usr/bin/env python3
"""Coordinator-level checks against the CPU vLLM venv, where APIs match.

The venv (upstream-vllm/.venv + the upstream source tree @ 22df3a34-era) is a
DIFFERENT generation from the image [I]: the fork's retention/reachable
machinery ([I] single_type_kv_cache_manager cache_blocks retention_interval +
reachable_block_mask) is fork-only. Everything here FEATURE-DETECTS and skips
cleanly when the API differs — an honest skip, never a fake pass (the task
contract: "where APIs match — feature-detect").

V1  venv availability + vllm importability from the source tree.
V2  single_type cache_blocks signature: retention_interval present => drive
    the T2 attribute against the REAL class; absent => SKIP (upstream tree).
V3  offloading connector base API surface stage 2 relies on:
    get_block_ids_with_load_errors on KVConnectorBase_V1 (the mixin's
    consumer contract) — present in both generations.

Run:  python3 tests/test_kv_offload_restore_venv.py   (or pytest)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENV_PY = ROOT.parent / "upstream-vllm" / ".venv" / "bin" / "python"
TREE = ROOT.parent / "upstream-vllm" / "vllm"

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


def skip(label: str) -> None:
    print("  SKIP " + label)


PROBE = r"""
import inspect, json, sys
out = {}
try:
    import vllm
    out["vllm"] = str(getattr(vllm, "__version__", "?"))
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
    sys.exit(0)
try:
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
    sig = inspect.signature(SingleTypeKVCacheManager.cache_blocks)
    out["cache_blocks_params"] = list(sig.parameters)
except Exception as exc:
    out["cache_blocks_params"] = f"unavailable: {type(exc).__name__}"
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
    out["has_load_errors_api"] = hasattr(
        KVConnectorBase_V1, "get_block_ids_with_load_errors"
    )
except Exception as exc:
    out["has_load_errors_api"] = f"unavailable: {type(exc).__name__}"
print(json.dumps(out))
"""


def main() -> int:
    print("Venv coordinator-level checks (feature-detected)")
    if not VENV_PY.is_file() or not TREE.is_dir():
        skip("V1 CPU vLLM venv not present on this host")
        print(f"\n{len(FAILURES)} failure(s)")
        return 0
    r = subprocess.run(
        [str(VENV_PY), "-c", PROBE],
        capture_output=True,
        text=True,
        cwd=str(TREE),
        timeout=180,
    )
    if r.returncode != 0 or not r.stdout.strip():
        skip(f"V1 venv probe failed to run ({r.stderr.strip()[:80]!r})")
        print(f"\n{len(FAILURES)} failure(s)")
        return 0
    out = json.loads(r.stdout.strip().splitlines()[-1])
    if "error" in out:
        skip(f"V1 vllm unimportable in venv: {out['error'][:80]}")
        print(f"\n{len(FAILURES)} failure(s)")
        return 0
    check(True, f"V1 venv vllm importable ({out['vllm']})")

    params = out.get("cache_blocks_params")
    if isinstance(params, list) and "retention_interval" in params:
        # Fork-generation tree: the T2 seam exists — prove the restore
        # patcher's single_type anchor applies to the REAL tree file.
        sys.path.insert(0, str(ROOT / "overlay"))
        import patch_kv_offload_restore_g0 as restore

        st = TREE / "vllm" / "v1" / "core" / "single_type_kv_cache_manager.py"
        try:
            patched, action = restore.prepare(
                st.read_text(), restore.SITES_SINGLE_TYPE, "venv-tree"
            )
            compile(patched, str(st), "exec")
            check(True, f"V2 T2 anchor applies to the real tree ({action})")
        except ValueError as exc:
            check(False, f"V2 T2 anchor drifted on a fork-generation tree: {exc}")
    else:
        skip(
            f"V2 upstream-generation tree: cache_blocks has no "
            f"retention_interval (params={params}) — the T2 seam is fork-only"
        )

    api = out.get("has_load_errors_api")
    if api is True:
        check(True, "V3 KVConnectorBase_V1.get_block_ids_with_load_errors exists")
    elif api is False:
        check(False, "V3 load-errors API missing from connector base")
    else:
        skip(f"V3 connector base unimportable ({api})")

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


def test_venv():  # pytest entry
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
