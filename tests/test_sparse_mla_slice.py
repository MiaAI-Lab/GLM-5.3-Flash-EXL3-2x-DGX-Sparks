#!/usr/bin/env python3
"""overlay/patch_sparse_mla_slice.py: opt-in bounded final sparse-MLA call (TP=4).

CPU-only. The fixture is the stock backend shipped in the published image
(vLLM 0.1.dev20051+g487ecf187), so the hash pins are checked against real
bytes, not a stub. Covers: stock is an exact no-op, 64 produces the pinned
result and compiles, a second application is a no-op, drift and bad values
fail closed without writing, and start-tp4.sh wires the knob on every rank.

Run:  python3 tests/test_sparse_mla_slice.py   (or pytest)
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "overlay" / "patch_sparse_mla_slice.py"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "flashinfer_mla_sparse_sm120-487ecf187.py.txt"

spec = importlib.util.spec_from_file_location("patch_sparse_mla_slice", PATCH_PATH)
assert spec and spec.loader
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

FIXTURE = FIXTURE_PATH.read_text()
ENV = patch.ENV_NAME


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_fixture_is_the_pinned_preimage() -> None:
    assert _sha(FIXTURE) == patch.ORIGINAL_SHA256


def test_parse_contract() -> None:
    for raw in (None, "", "0"):
        assert patch.parse_slice_tokens(raw) is None, raw
    assert patch.parse_slice_tokens("64") == 64
    for bad in ("1", "32", "128", "064", " 64", "64 ", "on", "true"):
        try:
            patch.parse_slice_tokens(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_stock_is_exact_noop() -> None:
    out, action = patch.prepare(FIXTURE, None)
    assert out == FIXTURE and action == "stock"


def test_slice_patch_pinned_and_idempotent() -> None:
    out, action = patch.prepare(FIXTURE, 64)
    assert action == "patched"
    assert _sha(out) == patch.PATCHED_SHA256
    compile(out, "flashinfer_mla_sparse_sm120.py", "exec")
    # the gate the patched backend enforces at call time
    for needle in (
        f'os.getenv(_SLICE_ENV, "0")',
        "_SLICE_TOKENS = 64",
        "_SLICE_REQUIRED_TOPK = 2048",
        "_SLICE_REQUIRED_WORKSPACE_BYTES = 33_685_504",
        "(16, 576)",
        "(64, 656)",
    ):
        assert needle in out, needle
    again, action = patch.prepare(out, 64)
    assert again == out and action == "already present"
    # asking for stock on an already-patched file leaves it alone (inert at env 0)
    same, action = patch.prepare(out, None)
    assert same == out and action == "already present"


def test_drift_fails_closed() -> None:
    for drifted in (
        FIXTURE.replace("bmm2_scale=1.0", "bmm2_scale=1.00", 1),
        FIXTURE + "\n# trailing\n",
        FIXTURE.replace(patch.IMPORT_ANCHOR, "", 1),
    ):
        for tokens in (None, 64):
            try:
                patch.prepare(drifted, tokens)
            except RuntimeError:
                continue
            raise AssertionError("drifted source was accepted")


def _run(target: Path, value: str | None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != ENV}
    if value is not None:
        env[ENV] = value
    return subprocess.run(
        [sys.executable, str(PATCH_PATH), str(target)],
        text=True, capture_output=True, check=False, env=env,
    )


def test_cli_apply_modes_and_pyc_cleanup() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "flashinfer_mla_sparse_sm120.py"
        target.write_text(FIXTURE)
        target.chmod(0o644)
        cache = target.parent / "__pycache__"
        cache.mkdir()
        stale = cache / "flashinfer_mla_sparse_sm120.cpython-312.pyc"
        stale.write_bytes(b"stale")

        r = _run(target, "0")
        assert r.returncode == 0, r.stderr
        assert target.read_text() == FIXTURE and stale.exists()

        r = _run(target, "64")
        assert r.returncode == 0, r.stderr
        assert _sha(target.read_text()) == patch.PATCHED_SHA256
        assert stat.S_IMODE(target.stat().st_mode) == 0o644
        assert not stale.exists()
        assert '"action": "patched"' in r.stdout

        r = _run(target, "64")
        assert r.returncode == 0 and '"action": "already present"' in r.stdout


def test_invalid_cli_never_writes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "flashinfer_mla_sparse_sm120.py"
        target.write_text(FIXTURE)
        for bad in ("32", "128", "yes"):
            r = _run(target, bad)
            assert r.returncode == 2 and ENV in r.stderr, (bad, r.stderr)
            assert target.read_text() == FIXTURE
        drifted = Path(tmp) / "other.py"
        drifted.write_text(FIXTURE + "\n")
        r = _run(drifted, "64")
        assert r.returncode == 2 and "preimage" in r.stderr
        assert drifted.read_text() == FIXTURE + "\n"
        r = _run(Path(tmp) / "missing.py", "64")
        assert r.returncode == 2 and "target missing" in r.stderr


def _guard_source() -> str:
    text = (ROOT / "start-tp4.sh").read_text()
    begin = text.index("# GLM53 numeric config guard (begin)")
    end = text.index("# GLM53 numeric config guard (end)")
    return text[begin:end]


def _validate(value: str | None) -> subprocess.CompletedProcess[str]:
    script = (
        _guard_source()
        + "\nGPU_MEM_UTIL=0.75; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=8; "
        + "MAX_NUM_BATCHED_TOKENS=2048; GLM53_INDEXER_WORKSPACE=stock; "
        + "GLM53_SPINWAIT_MS=stock; SPEC_METHOD=dflash\n"
        + "validate_numeric_config || exit $?\n"
        + 'printf "%s\\n" "${' + ENV + "-unset}\"\n"
    )
    env = {k: v for k, v in os.environ.items() if k != ENV}
    env["LC_ALL"] = "C"
    if value is not None:
        env[ENV] = value
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, check=False, env=env)


def test_launcher_guard_contract() -> None:
    for raw, canonical in ((None, "0"), ("", "0"), ("0", "0"), ("64", "64")):
        r = _validate(raw)
        assert r.returncode == 0, (raw, r.stderr)
        assert r.stdout.strip() == canonical, (raw, r.stdout)
    for bad in ("32", "128", "064", "on", "64 "):
        r = _validate(bad)
        assert r.returncode == 2, (bad, r.returncode, r.stdout)
        assert ENV in r.stderr, bad


def test_recipe_wiring() -> None:
    start = (ROOT / "start-tp4.sh").read_text()
    for needle in (
        'SPARSE_SLICE_PATCH_HOST="${SPARSE_SLICE_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_sparse_mla_slice.py}"',
        f'{ENV}="${{{ENV}-0}}"',
        '[ -f "$SPARSE_SLICE_PATCH_HOST" ] || die "$SPARSE_SLICE_PATCH_HOST missing"',
        f'-e "{ENV}=${ENV}"',
        "-v '/tmp/patch_sparse_mla_slice.py:/opt/glm53/patch_sparse_mla_slice.py:ro'",
        '-v "$SPARSE_SLICE_PATCH_HOST:/opt/glm53/patch_sparse_mla_slice.py:ro"',
    ):
        assert needle in start, needle
    # both inner scripts (head + worker) apply it after the spinwait overlay
    assert start.count("python3 /opt/glm53/patch_sparse_mla_slice.py") == 2
    assert start.count('"${ssh_t}:/tmp/patch_sparse_mla_slice.py"') == 1
    assert start.count('"${WORKER_SSH}:/tmp/patch_sparse_mla_slice.py"') == 1
    # worker serve_env loop forwards the knob
    loop = start[start.index("local serve_env=\"\""):start.index("serve_env+=\" -e VLLM_API_KEY")]
    assert ENV in loop
    # start.sh / start-tp3.sh stay untouched (TP=4-only geometry)
    for other in ("start.sh", "start-tp3.sh"):
        assert ENV not in (ROOT / other).read_text(), other
    env_example = (ROOT / ".env.tp4.example").read_text()
    assert f"{ENV}=0" in env_example


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
