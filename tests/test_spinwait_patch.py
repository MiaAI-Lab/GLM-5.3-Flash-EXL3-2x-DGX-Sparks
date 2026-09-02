#!/usr/bin/env python3
"""Focused host checks for the numeric SpinCondition patch."""
from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH_PATH = next(
    path
    for path in (ROOT / "overlay" / "patch_spinwait.py", HERE / "patch_spinwait.py")
    if path.is_file()
)
spec = importlib.util.spec_from_file_location("patch_spinwait", PATCH_PATH)
assert spec and spec.loader
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

FIXTURE = (
    "class SpinCondition:\n"
    "    def __init__(\n"
    "        self,\n"
    "        busy_loop_s: float = 1,\n"
    "    ):\n"
    "        self.busy_loop_s = busy_loop_s\n"
)


def test_parse_contract() -> None:
    assert patch.parse_spinwait_ms(None) is None
    assert patch.parse_spinwait_ms("stock") is None
    for raw, expected in (("1", 1), ("016", 16), ("16", 16), ("1000", 1000)):
        assert patch.parse_spinwait_ms(raw) == expected
    for raw in ("", "0", "1001", "-1", "1.5", "nan", " 16", "16 ", "STOCK"):
        try:
            patch.parse_spinwait_ms(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid value {raw!r}")


def test_seconds_literals() -> None:
    assert patch.seconds_literal(1) == "0.001"
    assert patch.seconds_literal(16) == "0.016"
    assert patch.seconds_literal(100) == "0.1"
    assert patch.seconds_literal(999) == "0.999"
    assert patch.seconds_literal(1000) == "1"


def test_stock_is_exact_noop() -> None:
    out, action = patch.prepare(FIXTURE, None)
    assert out == FIXTURE
    assert action == "stock"
    out, action = patch.prepare(FIXTURE, 1000)
    assert out == FIXTURE
    assert "stock-equivalent" in action


def test_numeric_patch_and_idempotence() -> None:
    out, action = patch.prepare(FIXTURE, 16)
    assert action == "patched"
    assert "busy_loop_s: float = 0.016," in out
    assert patch.STOCK_LINE not in out
    again, action = patch.prepare(out, 16)
    assert again == out
    assert action == "already present"
    compile(out, "fixture.py", "exec")


def test_drift_and_ambiguity_fail_closed() -> None:
    for source in (
        FIXTURE.replace("float = 1", "float = 0.002"),
        FIXTURE + FIXTURE,
        FIXTURE.replace(patch.STOCK_LINE, ""),
    ):
        try:
            patch.prepare(source, 16)
        except ValueError:
            pass
        else:
            raise AssertionError("drifted source was accepted")


def _run(target: Path, value: str | None, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_SPINWAIT_TARGET"] = str(target)
    env.pop(patch.ENV_NAME, None)
    if value is not None:
        env[patch.ENV_NAME] = value
    return subprocess.run(
        [sys.executable, str(PATCH_PATH), *args],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_cli_preflight_apply_mode_and_pyc_cleanup() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        target = root / "shm_broadcast.py"
        target.write_text(FIXTURE)
        target.chmod(0o640)
        cache = root / "__pycache__"
        cache.mkdir()
        pyc = cache / "shm_broadcast.cpython-312.pyc"
        pyc.write_bytes(b"stale")

        preflight = _run(target, "16", "--preflight")
        assert preflight.returncode == 0, preflight.stderr
        assert target.read_text() == FIXTURE
        applied = _run(target, "16")
        assert applied.returncode == 0, applied.stderr
        assert "0.016" in target.read_text()
        assert stat.S_IMODE(target.stat().st_mode) == 0o640
        assert not pyc.exists()
        repeated = _run(target, "16")
        assert repeated.returncode == 0, repeated.stderr
        assert "already present" in repeated.stdout


def test_invalid_cli_never_writes() -> None:
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "shm_broadcast.py"
        target.write_text(FIXTURE)
        for value in ("", "0", "1001", "2ms"):
            result = _run(target, value)
            assert result.returncode != 0, (value, result.stdout, result.stderr)
            assert target.read_text() == FIXTURE


def test_live_source_if_enabled() -> None:
    live = os.environ.get("GLM53_SPINWAIT_LIVE_SOURCE")
    if not live:
        return
    source = Path(live).read_text()
    out, action = patch.prepare(source, 16)
    assert action in ("patched", "already present")
    compile(out, live, "exec")


def test_recipe_wiring() -> None:
    if not all(
        (ROOT / name).is_file() for name in ("start.sh", "Dockerfile", ".env.example")
    ):
        return
    start = (ROOT / "start.sh").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()
    env_example = (ROOT / ".env.example").read_text()
    required_start = (
        'SPINWAIT_PATCH_HOST="${SPINWAIT_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_spinwait.py}"',
        'GLM53_SPINWAIT_MS="${GLM53_SPINWAIT_MS-stock}"',
        "_glm53_validate_spinwait_ms",
        'python3 /opt/glm53/patch_spinwait.py',
        '"GLM53_SPINWAIT_MS=$GLM53_SPINWAIT_MS"',
        '/opt/glm53/patch_spinwait.py:ro',
    )
    for needle in required_start:
        assert needle in start, needle
    assert start.count("python3 /opt/glm53/patch_spinwait.py") == 2
    assert start.count("/opt/glm53/patch_spinwait.py:ro") == 2
    assert "COPY overlay/patch_spinwait.py /opt/glm53/patch_spinwait.py" in dockerfile
    assert "tests/test_spinwait_patch.py" in dockerfile
    assert "GLM53_SPINWAIT_MS=stock" in env_example
    assert "GLM53_SPINWAIT_2MS" not in start + dockerfile + env_example


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"numeric spinwait patch OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
