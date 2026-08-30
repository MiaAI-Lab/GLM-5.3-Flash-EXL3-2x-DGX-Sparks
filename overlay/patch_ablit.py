#!/usr/bin/env python3
"""Install the ABLIT load-time hook into the glm53-flash vLLM image (idempotent).

Copies ablit_runtime.py next to the glm5next model files and appends a hook
call at the end of Glm5NextModel.load_weights (target layers 0-44) and
Glm5NextMTP.load_weights (checkpoint MTP block, layer 45). The hook no-ops
unless ABLIT=1, so installing it is inert on stock serves.

Runs at image build (Dockerfile) and again at every container start from
/opt/glm53/patch_ablit.py so prebuilt GHCR images get it too.
"""

from __future__ import annotations

import shutil
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
NVIDIA_DIR = SITE / "models/glm5next/nvidia"
MODEL_PY = NVIDIA_DIR / "model.py"
MTP_PY = NVIDIA_DIR / "mtp.py"
RUNTIME_SRC = Path("/opt/glm53/ablit_runtime.py")
RUNTIME_DST = NVIDIA_DIR / "glm53_ablit.py"

MARKER = "ABLIT-HOOK"
HOOK_LINES = (
    "        from vllm.models.glm5next.nvidia.glm53_ablit import maybe_apply "
    "as _glm53_ablit_maybe_apply  # ABLIT-HOOK\n"
    "        _glm53_ablit_maybe_apply(self)  # ABLIT-HOOK\n"
)

MODEL_TAIL = """                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        return loaded_params"""

MTP_TAIL = """                    f"missing from checkpoint."
                )
        return loaded_params"""


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if MARKER in text:
        print(f"{path.name}: {MARKER} already present — skipping")
        return
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected one patch target, found {n}: {old!r}")
    path.write_text(text.replace(old, new))
    print(f"patched {path.name} ({label})")


def main() -> None:
    NVIDIA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RUNTIME_SRC, RUNTIME_DST)
    print(f"installed {RUNTIME_DST}")

    replace_once(
        MODEL_PY,
        MODEL_TAIL,
        MODEL_TAIL.replace(
            "        return loaded_params", HOOK_LINES + "        return loaded_params"
        ),
        "Glm5NextModel.load_weights",
    )

    replace_once(
        MTP_PY,
        MTP_TAIL,
        MTP_TAIL.replace(
            "        return loaded_params", HOOK_LINES + "        return loaded_params"
        ),
        "Glm5NextMTP.load_weights",
    )


if __name__ == "__main__":
    main()
