#!/usr/bin/env python3
"""Restart-idempotence regression for overlay/patch_scheduler_decode_floor.py.

The container entrypoint re-runs every overlay installer on each start.
patch_adaptive_k.py shares the decode-floor installer's helper anchor
(`from vllm.compilation.cuda_graph import CUDAGraphStat`), so on a restarted
container the adaptive-k block can sit either before or after the
decode-floor helper. The old verify path unpatched and re-applied the helper
at that fixed anchor, which relocated it and failed the byte-compare with
"v5 helper drifted" — a healthy, correctly patched file was rejected and the
entrypoint exited 1.

These tests build a synthetic scheduler from the installer's own *_OLD
anchors, so they run on any CPU without the vLLM image. They cover:

* apply on a clean tree, then verify (restart) — bytes preserved;
* a subsequent overlay inserting its helper before ours (image order);
* a subsequent overlay inserting its helper after ours (restart order);
* fail-closed rejection of helper-body drift, insertion drift, a missing
  helper, a duplicated helper, and a marker-only file.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_PATCH_CANDIDATES = (
    HERE / "patch_scheduler_decode_floor.py",  # in-image /opt/glm53
    ROOT / "overlay" / "patch_scheduler_decode_floor.py",  # repo checkout
)
PATCH = next((p for p in _PATCH_CANDIDATES if p.is_file()), None)
if PATCH is None:
    raise SystemExit(
        "missing patch_scheduler_decode_floor.py (tried "
        + ", ".join(str(p) for p in _PATCH_CANDIDATES)
        + ")"
    )
ADAPTIVE_K = PATCH.parent / "patch_adaptive_k.py"
CUDA_GRAPH_NEEDLE = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


df = _load("glm53_decode_floor_restart", PATCH)
# The real adaptive-k helper when the overlay tree is beside the test (repo
# checkout); a same-shaped block otherwise (in-image /opt/glm53). Either way
# the foreign block carries the `class _Glm53AdaptiveK:` marker so the
# installer's end-marker search exercises the production path.
if ADAPTIVE_K.is_file():
    FOREIGN_HELPER = _load("glm53_adaptive_k_restart", ADAPTIVE_K).SCHED_HELPER
else:
    FOREIGN_HELPER = (
        "\n\nclass _Glm53AdaptiveK:  # [glm53-adaptive-k]\n"
        "    pass\n\n\n"
        "_GLM53_ADAPTIVE_K = _Glm53AdaptiveK()  # [glm53-adaptive-k]\n\n\n"
    )


def _nest(fragment: str) -> str:
    """Wrap a scheduler fragment in `while` blocks matching its indent.
    `while` (not `if`) so fragments containing `break` stay valid. A trailing
    `pass` closes any block the fragment leaves open (an `if` whose body is
    only a comment, or a comment that dedents out of its block).
    """
    indent = len(fragment) - len(fragment.lstrip())
    out = "".join(f"{' ' * i}while True:\n" for i in range(8, indent, 4))
    out += fragment
    if not out.endswith("\n"):
        out += "\n"
    last = fragment.rstrip("\n").rsplit("\n", 1)[-1]
    last_indent = len(last) - len(last.lstrip())
    if last.lstrip().startswith("#"):
        out += " " * last_indent + "pass\n"
    elif last.rstrip().endswith(":"):
        out += " " * (last_indent + 4) + "pass\n"
    return out


def build_clean_scheduler() -> str:
    """Minimal compilable scheduler containing every v6 anchor exactly once."""
    parts = [
        "# Synthetic scheduler fixture: real anchors, stub bodies.\n",
        df.IMPORT_OLD,
        "from collections import defaultdict\n",
        "\n",
        "class Scheduler:\n",
        "    def schedule(self):\n",
    ]
    parts.extend(_nest(old) for _new, old, _label in df.V6_PAIRS)
    parts.append("        return None\n\n\n")
    parts.append(CUDA_GRAPH_NEEDLE)
    parts.append("from vllm.config import VllmConfig\n")
    text = "".join(parts)
    compile(text, "<fixture>", "exec")
    for _new, old, label in df.V6_PAIRS:
        assert text.count(old) == 1, f"fixture anchor {label} not unique"
    assert text.count(CUDA_GRAPH_NEEDLE) == 1
    return text


def run_patch(target: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "GLM53_SCHEDULER_PY": str(target),
        "GLM53_MIXED_PREFILL_CHUNK": "skip",
    }
    return subprocess.run(
        [sys.executable, str(PATCH)], env=env, capture_output=True, text=True
    )


def insert_at_needle(text: str, block: str) -> str:
    """Replay a subsequent overlay sharing the cuda_graph anchor."""
    assert text.count(CUDA_GRAPH_NEEDLE) == 1
    return text.replace(CUDA_GRAPH_NEEDLE, block + CUDA_GRAPH_NEEDLE, 1)


def apply_and_verify(text: str, tmp: Path, name: str = "scheduler.py") -> str:
    target = tmp / name
    target.write_text(text)
    first = run_patch(target)
    assert first.returncode == 0, first.stderr
    patched = target.read_text()
    second = run_patch(target)
    assert second.returncode == 0, second.stderr
    assert target.read_text() == patched, "verify run must not rewrite the file"
    return patched


def expect_reject(target: Path) -> None:
    before = target.read_text()
    result = run_patch(target)
    assert result.returncode != 0, result.stdout
    assert target.read_text() == before, "rejected run must not modify the file"


def main() -> int:
    clean = build_clean_scheduler()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Apply on a clean tree; a plain restart verifies byte-identical.
        patched = apply_and_verify(clean, tmp)

        # 2. Subsequent overlay inserts its helper AFTER ours — the order a
        #    container restart produces (decode-floor patched first, then
        #    adaptive-k lands at the shared anchor between our helper and the
        #    import). This is the layout that used to fail verification with
        #    "v5 helper drifted".
        after_text = insert_at_needle(patched, FOREIGN_HELPER)
        assert after_text.index("class _Glm53MixedPrefill:") < after_text.index(
            "class _Glm53AdaptiveK:"
        )
        target = tmp / "overlay_after.py"
        target.write_text(after_text)
        result = run_patch(target)
        assert result.returncode == 0, result.stderr
        assert target.read_text() == after_text
        # And the restart after that stays stable too.
        result = run_patch(target)
        assert result.returncode == 0, result.stderr
        assert target.read_text() == after_text

        # 3. Subsequent overlay already sits BEFORE the anchor when our
        #    installer first runs — the order the image produces when
        #    adaptive-k lands first. Apply must succeed and verify must
        #    accept the helper between the foreign block and the import.
        apply_and_verify(insert_at_needle(clean, FOREIGN_HELPER), tmp, "overlay_before.py")

        # 4. Helper-body drift stays fail-closed.
        target = tmp / "drift_helper.py"
        target.write_text(
            patched.replace(
                "return _GLM53_MIXED.cap_for(sched, request, computed)",
                "return _GLM53_MIXED.cap_for(sched, request)",
                1,
            )
        )
        expect_reject(target)

        # 5. Insertion drift stays fail-closed.
        target = tmp / "drift_insertion.py"
        target.write_text(
            patched.replace(
                "_GLM53_MIXED.finish_step(self, scheduler_output)",
                "_GLM53_MIXED.finish_step_renamed(self, scheduler_output)",
                1,
            )
        )
        expect_reject(target)

        # 6. Marker present but helper deleted entirely.
        target = tmp / "no_helper.py"
        helper_start = patched.index("class _Glm53MixedPrefill:")
        helper_end = patched.index(CUDA_GRAPH_NEEDLE)
        target.write_text(patched[:helper_start] + patched[helper_end:])
        expect_reject(target)

        # 7. Duplicated helper is ambiguous, not verified.
        target = tmp / "dup_helper.py"
        dup = patched + "\n" + patched[helper_start:helper_end]
        target.write_text(dup)
        expect_reject(target)

        # 8. Marker alone must not verify.
        target = tmp / "marker_only.py"
        target.write_text(clean + f"\n# {df.MARK_V6} stray\n")
        expect_reject(target)

        # A modified second helper inside the stripped region must not hide
        # behind the one remaining canonical body elsewhere in that region.
        target = tmp / "shadow_helper.py"
        shadow = df._helper_text().replace(
            "return _GLM53_MIXED.cap_for(sched, request, computed)",
            "return None",
            1,
        )
        target.write_text(insert_at_needle(patched, shadow))
        expect_reject(target)

        # The scheduler's owned import must still be present.
        target = tmp / "missing_os.py"
        target.write_text(patched.replace("import os\n", "", 1))
        expect_reject(target)

    print("scheduler decode-floor restart idempotence OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
