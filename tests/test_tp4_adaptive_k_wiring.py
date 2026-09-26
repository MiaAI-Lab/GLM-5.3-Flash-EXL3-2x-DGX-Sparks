#!/usr/bin/env python3
"""start-tp4.sh forwards adaptive verification length (GLM53_ADAPTIVE_K*) to every rank.

CPU-only static + generator check. The two-node launcher has carried this since
2026-09-08; the four-node launcher never did, so a TP=4 kit setting
GLM53_ADAPTIVE_K=ema got stock k on all ranks with no error. Mirrors the spinwait
overlay's wiring shape (host path, existence check, scp to ranks 1-3, read-only
mount, container-start application, env on head and workers) plus the
capture-size generator start.sh uses.

Run:  python3 tests/test_tp4_adaptive_k_wiring.py   (or pytest)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start-tp4.sh"
KNOBS = (
    "GLM53_ADAPTIVE_K", "GLM53_ADAPTIVE_K_SET", "GLM53_ADAPTIVE_K_ALPHA", "GLM53_ADAPTIVE_K_MARGIN",
    "GLM53_ADAPTIVE_K_MIN_STEPS", "GLM53_ADAPTIVE_K_SATURATE", "GLM53_ADAPTIVE_K_HIST",
)


def test_recipe_wiring() -> None:
    s = START.read_text()
    assert 'ADAPTIVE_K_PATCH_HOST="${ADAPTIVE_K_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_adaptive_k.py}"' in s
    assert '[ -f "$ADAPTIVE_K_PATCH_HOST" ] || die "$ADAPTIVE_K_PATCH_HOST missing"' in s
    assert s.count("python3 /opt/glm53/patch_adaptive_k.py") == 2, "head and worker inner scripts"
    assert s.count('"${ssh_t}:/tmp/patch_adaptive_k.py"') == 1
    assert s.count('"${WORKER_SSH}:/tmp/patch_adaptive_k.py"') == 1
    assert "-v '/tmp/patch_adaptive_k.py:/opt/glm53/patch_adaptive_k.py:ro'" in s
    assert '-v "$ADAPTIVE_K_PATCH_HOST:/opt/glm53/patch_adaptive_k.py:ro"' in s
    loop = s[s.index('local serve_env=""'):s.index('serve_env+=" -e VLLM_API_KEY')]
    for k in KNOBS:
        assert f'{k}="${{{k}:-' in s, f"default for {k}"
        assert f'-e "{k}=${k}"' in s, f"head env for {k}"
        assert k in loop, f"worker env for {k}"
    # defaults match start.sh exactly
    two = (ROOT / "start.sh").read_text()
    for k in KNOBS:
        d4 = re.search(rf'^{k}="\$\{{{k}:-([^}}]*)\}}"', s, re.M)
        d2 = re.search(rf'^{k}="\$\{{{k}:-([^}}]*)\}}"', two, re.M)
        assert d4 and d2 and d4.group(1) == d2.group(1), (k, d4 and d4.group(1), d2 and d2.group(1))
    assert (ROOT / "overlay" / "patch_adaptive_k.py").is_file()


def _generator() -> str:
    s = START.read_text()
    m = re.search(r"capture_sizes=\"\$\(python3 -S -c '(.*?)'", s, re.S)
    assert m, "capture-size generator not found in start-tp4.sh"
    return m.group(1)


def _sizes(mode: str, kset: str, tokens: str, seqs: str) -> list[int]:
    out = subprocess.run([sys.executable, "-S", "-c", _generator(), mode, kset, tokens, seqs],
                         text=True, capture_output=True, check=True).stdout.split()
    return [int(x) for x in out]


def test_capture_sizes_off_is_stock() -> None:
    assert _sizes("off", "2,4,7", "7", "8") == [1, 2, 4, 8, 16, 24, 32]


def test_capture_sizes_ema_covers_every_k_at_every_batch() -> None:
    got = set(_sizes("ema", "2,4,7", "7", "8"))
    for n in range(1, 9):
        for q in (3, 5, 8):
            assert n * q in got, (n, q)
    assert {1, 2, 4, 8, 16, 24, 32} <= got
    assert got == set(_sizes("ON", " 2, 4 ,7 ", "7", "8"))


def test_launcher_syntax() -> None:
    subprocess.run(["bash", "-n", str(START)], check=True)


def test_rank_scripts_syntax() -> None:
    """bash -n on the launcher cannot see inside the quoted heredocs that become
    each rank's /start.sh; check those bodies too (a dropped `fi` there only
    shows up as `/start.sh: syntax error: unexpected end of file` at boot)."""
    import tempfile, os
    s = START.read_text()
    bodies = re.findall(r"<<\s*'([A-Z_]+)'\n(.*?)\n\1\n", s, re.S)
    assert len(bodies) >= 2, "expected head and worker heredocs"
    for tag, body in bodies:
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write(body)
            name = f.name
        try:
            r = subprocess.run(["bash", "-n", name], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
        finally:
            os.unlink(name)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1; print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
