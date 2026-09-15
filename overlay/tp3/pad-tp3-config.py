#!/usr/bin/env python3
"""Idempotent TP pad for GLM-5.3 Flash NVFP4 config.json.

Stock checkpoint: 64 attn heads, moe_intermediate_size 2048. Neither divides
by 3. --hf-overrides is enough for the *target* model; SpeculativeConfig for
native MTP reads this file, so MTP-4 dies with "64 is not divisible by 3"
unless the checkpoint itself is padded.

Do not change vocab_size (154880). VocabParallelEmbedding keeps org_vocab_size
and pads in-module (padding_size = lcm(64, tp)).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _ceil_div(n: int, tp: int) -> int:
    return ((n + tp - 1) // tp) * tp


def pad_config(
    path: Path,
    tp: int,
    heads: int | None = None,
    skip_moe: bool = False,
) -> bool:
    data = json.loads(path.read_text())
    text = data.get("text_config")
    changed = False

    def set_if(obj: dict, key: str, value: int) -> None:
        nonlocal changed
        if obj.get(key) != value:
            obj[key] = value
            changed = True

    if isinstance(text, dict):
        # Always pad from stock 64/2048, not from an already-padded file
        # (96-head SGLang decode pad would otherwise stick).
        orig_path = path.with_suffix(path.suffix + ".orig")
        if orig_path.exists():
            orig_text = json.loads(orig_path.read_text()).get("text_config") or {}
            stock_heads = int(orig_text.get("num_attention_heads", 64))
            stock_moe = int(orig_text.get("moe_intermediate_size", 2048))
        else:
            stock_heads = 64
            stock_moe = 2048
        want_heads = heads if heads is not None else _ceil_div(stock_heads, tp)
        want_moe = stock_moe if skip_moe else _ceil_div(stock_moe, tp)
        for key in (
            "num_attention_heads",
            "num_key_value_heads",
            "linear_num_heads",
        ):
            set_if(text, key, want_heads)
        if not skip_moe:
            set_if(text, "moe_intermediate_size", want_moe)
        la = text.get("linear_attn_config")
        if isinstance(la, dict) and "num_heads" in la:
            set_if(la, "num_heads", want_heads)
        summary = (
            f"heads={want_heads}, moe_i={text.get('moe_intermediate_size')}"
            + (" (unpadded, use EP)" if skip_moe else "")
        )
    else:
        # DFlash2 GQA: always pad from stock 32/8 (ratio 4), not from an
        # already-ceiled config (33/9 would compute ratio 3 and stick).
        # Independent ceil 32→33 / 8→9 gives local 11/3; FlashInfer requires
        # qo % kv == 0. Keep ratio 4: KV 8→9, Q 9*4=36 (local 12/3).
        stock_q, stock_kv = 32, 8
        ratio = stock_q // stock_kv
        want_kv = _ceil_div(stock_kv, tp)
        want_q = _ceil_div(max(_ceil_div(stock_q, tp), want_kv * ratio), tp)
        # --heads on a GQA drafter keeps the 4:1 Q:KV ratio (48 → 12 KV).
        if heads is not None:
            want_q = _ceil_div(heads, tp)
            want_kv = max(want_kv, want_q // ratio)
            want_q = want_kv * ratio
        set_if(data, "num_attention_heads", want_q)
        set_if(data, "num_key_value_heads", want_kv)
        summary = f"heads={want_q}, kv_heads={want_kv}"

    if not changed:
        print(f"{path}: already padded ({summary})")
        return False

    bak = path.with_suffix(path.suffix + ".orig")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"wrote backup {bak}")
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{path}: padded {summary}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--tp", type=int, default=3)
    p.add_argument(
        "--heads",
        type=int,
        default=None,
        help="Force attention head count. Target omit = 64→66 at TP=3.",
    )
    p.add_argument(
        "--skip-moe",
        action="store_true",
        help="Do not pad moe_intermediate_size (EXL3 packed trellis is 2048; use EP).",
    )
    args = p.parse_args()
    if not args.config.is_file():
        raise SystemExit(f"missing {args.config}")
    pad_config(args.config, args.tp, heads=args.heads, skip_moe=args.skip_moe)


if __name__ == "__main__":
    main()
