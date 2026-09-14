#!/usr/bin/env python3
"""Cache the pinned EXL3 expert_map without writing to a read-only property.

pin_exl3_expert_map() moved expert_map to the device once and stored it back on
the layer. With --enable-expert-parallel the owning class exposes expert_map as
a read-only property, so the write raises during the profile run:

    AttributeError: property 'expert_map' of 'RoutedExperts' object has no setter

The intent is only to avoid a CPU->GPU copy inside a CUDA graph, so keep the
pinned tensor in a private attribute instead. Behaviour is unchanged when
expert_map is a plain attribute (TP=2), because the private cache is consulted
first either way.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "EXL3-PIN-EXPERT-MAP-RO"
TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "quantization/exl3.py"
)

OLD = '''    emap = getattr(layer, "expert_map", None)
    if emap is None:
        return None
    if emap.device != device or emap.dtype != torch.long:
        layer.expert_map = emap.to(device=device, dtype=torch.long)
    return layer.expert_map
'''

NEW = '''    # EXL3-PIN-EXPERT-MAP-RO: expert_map is a read-only property under EP.
    cached = getattr(layer, "_exl3_pinned_expert_map", None)
    if cached is not None and cached.device == device and cached.dtype == torch.long:
        return cached
    emap = getattr(layer, "expert_map", None)
    if emap is None:
        return None
    if emap.device != device or emap.dtype != torch.long:
        emap = emap.to(device=device, dtype=torch.long)
        try:
            layer.expert_map = emap
        except AttributeError:
            pass
    object.__setattr__(layer, "_exl3_pinned_expert_map", emap)
    return emap
'''


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit(f"{TARGET}: not in image")
    src = TARGET.read_text()
    if MARKER in src:
        print(f"{TARGET.name}: expert_map pin already patched — no-op")
        return
    if src.count(OLD) != 1:
        raise SystemExit(f"{TARGET}: expected 1 pin_exl3_expert_map body, found {src.count(OLD)}")
    TARGET.write_text(src.replace(OLD, NEW, 1))
    print(f"{TARGET.name}: patched pin_exl3_expert_map for read-only expert_map")


if __name__ == "__main__":
    main()
