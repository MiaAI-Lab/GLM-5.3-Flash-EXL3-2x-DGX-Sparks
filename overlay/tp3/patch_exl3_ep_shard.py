#!/usr/bin/env python3
"""Stop the EXL3 MoE loader from intra-sharding expert weights under EP.

_load_exl3 already maps a global expert id to this rank's local slot via
RoutedExperts._map_global_expert_id_to_local_expert_id — that IS expert
parallelism, each rank owning whole experts. It then *also* narrowed every
expert tensor by the global TP size, which double-partitions. At TP=3 that
surfaces as:

    ValueError: EXL3 TP shard: dim 0 size 2048 is not divisible by tp=3

because w2's dim 0 is moe_intermediate_size=2048 and the EXL3 trellis is
packed 2048-wide (FlyCockpit: do not pad it, use expert parallel).

vLLM already publishes the right number: moe_config.moe_parallel_config.tp_size
is 1 under pure EP and equals the MoE TP degree otherwise. Use it instead of
the global world size. With EP off this is a no-op, so TP=2 is unchanged.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "EXL3-EP-NO-INTRA-SHARD"
TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "quantization/exl3.py"
)

OLD = """        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
"""

NEW = """        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        # EXL3-EP-NO-INTRA-SHARD: under expert parallelism this rank already
        # owns whole experts (see the expert-id mapping above); slicing them
        # again by the global TP size double-partitions and cannot divide a
        # 2048-wide trellis by 3. moe_parallel_config.tp_size is 1 under pure
        # EP and the real MoE TP degree otherwise.
        _mpc = getattr(getattr(owner, "moe_config", None), "moe_parallel_config", None)
        if _mpc is not None:
            _moe_tp = getattr(_mpc, "tp_size", None)
            if _moe_tp:
                tp_size = int(_moe_tp)
                tp_rank = int(getattr(_mpc, "tp_rank", 0) or 0)
                if tp_size <= 1:
                    tp_rank = 0
"""


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit(f"{TARGET}: not in image")
    src = TARGET.read_text()
    if MARKER in src:
        print(f"{TARGET.name}: already patched — no-op")
        return
    if src.count(OLD) != 1:
        raise SystemExit(f"{TARGET}: expected 1 tp_rank/tp_size anchor, found {src.count(OLD)}")
    TARGET.write_text(src.replace(OLD, NEW, 1))
    print(f"{TARGET.name}: patched EXL3 MoE loader to respect expert parallelism")


if __name__ == "__main__":
    main()
