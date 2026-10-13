#!/usr/bin/env python3
"""Exercise both restart-test helper paths even when run from a repo checkout.

The Dockerfile flattens the patch and restart test into /opt/glm53 without
patch_adaptive_k.py. A checkout-only run loads the real helper instead and
can miss drift in the fallback used by the image's self-check.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run_restart_test(directory: Path, label: str) -> None:
    result = subprocess.run(
        [sys.executable, str(directory / "test_scheduler_decode_floor_restart.py")],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"{label}:\n{result.stdout}\n{result.stderr}")
    if "scheduler decode-floor restart idempotence OK" not in result.stdout:
        raise AssertionError(f"{label}: missing success output: {result.stdout}")
    print(f"{label}: OK")


def test_restart_layouts() -> None:
    with tempfile.TemporaryDirectory(prefix="glm53-restart-layouts-") as name:
        # Keep the flattened image dir below a fresh root so neither patch
        # resolution nor helper resolution can fall back to the real checkout.
        directory = Path(name) / "glm53"
        directory.mkdir()
        for relative in (
            "overlay/patch_scheduler_decode_floor.py",
            "tests/test_scheduler_decode_floor_restart.py",
        ):
            shutil.copy2(ROOT / relative, directory)
        if (directory / "patch_adaptive_k.py").exists():
            raise AssertionError("image-layout fixture must omit the real helper")
        run_restart_test(directory, "image layout (fallback helper)")

        shutil.copy2(ROOT / "overlay/patch_adaptive_k.py", directory)
        run_restart_test(directory, "flattened layout (real helper)")


if __name__ == "__main__":
    test_restart_layouts()
