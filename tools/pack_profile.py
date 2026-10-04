#!/usr/bin/env python3
"""Resolve a staged pack's TP2 H3/DFlash2 profile before lifecycle operations.

No downloads and no weight loads: inspect JSON and safetensors headers only.
The target publisher owns the paired draft identity; there is no default URL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys

PROFILE = "dense-exl3-h3-dflash2-6bpw"
H3 = "kda_in,shared_down,mla_qkv_a,shared_gate_up,kda_o,mla_q_b"
DEFAULT_DRAFT = "incoai/GLM-5.3-Flash-DFlash2"
DEFAULT_REVISION = "dc77ff1c99eeb2df044ee3d4f0094eb033fee410"


def inventory(snapshot: Path) -> tuple[dict, int]:
    """Require every indexed tensor and shard to exist, without reading weights."""
    index = snapshot / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.is_file() else None
    files = set(weight_map.values()) if weight_map else {"model.safetensors"}
    tensors = {}
    for name in sorted(files):
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"invalid shard name: {name!r}")
        path = snapshot / name
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            if not 0 < size <= 100_000_000:
                raise ValueError(f"invalid safetensors header: {path}")
            header = json.loads(f.read(size))
        payload_size = path.stat().st_size - size - 8
        for key, spec in header.items():
            if key == "__metadata__":
                continue
            a, b = spec["data_offsets"]
            if not 0 <= a < b <= payload_size:
                raise ValueError(f"truncated tensor {key} in {path}")
            if weight_map is None or weight_map.get(key) == name:
                tensors[key] = spec
    if weight_map and set(weight_map) != set(tensors):
        raise ValueError(f"missing indexed tensors in {snapshot}")
    if not any(k.endswith(".trellis") for k in tensors):
        raise ValueError(f"no EXL3 assets in {snapshot}")
    return tensors, len(files)


def require_packed(tensors: dict, base: str, bits: int) -> None:
    trellis = tensors.get(base + ".trellis", {})
    shape = trellis.get("shape", [])
    if (bits not in (2, 3, 4, 5, 6) or trellis.get("dtype") != "I16"
            or len(shape) != 3 or shape[-1] != 16 * bits
            or tensors.get(base + ".suh", {}).get("shape") != [shape[0] * 16]
            or tensors.get(base + ".svh", {}).get("shape") != [shape[1] * 16]
            or sum(base + "." + p in tensors for p in ("mcg", "mul1")) != 1):
        raise ValueError(f"missing or incompatible packed tensor: {base}")


def parse_layers(spec: str) -> set[int]:
    """ABLIT_LAYERS syntax ("15-45", "15,17-19"), accepting exactly what
    overlay/ablit_runtime.py parse_layers accepts, so the pre-stop check never
    passes a spec the container then refuses."""
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo_s, sep, hi_s = part.partition("-")
        try:
            lo, hi = int(lo_s), int(hi_s if sep else lo_s)
        except ValueError as exc:
            raise ValueError(f"ABLIT_LAYERS: bad range {part!r}") from exc
        if hi < lo:
            raise ValueError(f"ABLIT_LAYERS: inverted range {part!r}")
        layers.update(range(lo, hi + 1))
    if not layers:
        raise ValueError(f"ABLIT_LAYERS {spec!r} names no layers")
    return layers


def exl3_o_proj_layers(cfg: dict) -> set[int]:
    """Target layers whose o_proj the pack declares EXL3. ABLIT edits a BF16
    o_proj only, so these must stay out of ABLIT_LAYERS."""
    layers = cfg.get("quantization_config", {}).get("non_routed_exl3", {}).get("layers", {})
    out = set()
    for prefix in layers:
        m = re.search(r"(?:^|\.)layers\.(\d+)\.self_attn\.o_proj$", prefix)
        if m:
            out.add(int(m.group(1)))
    return out


def check_ablit(cfg: dict, spec: str) -> None:
    """Refuse ABLIT_LAYERS that reach an EXL3 o_proj. Layer 45 is the
    checkpoint MTP block, outside the 45 target layers; the runtime hook
    edits it only when that block is loaded."""
    text = cfg.get("text_config", cfg)
    targets = set(range(int(text.get("num_hidden_layers", 45))))
    clash = sorted(parse_layers(spec) & targets & exl3_o_proj_layers(cfg))
    if clash:
        raise ValueError(
            f"ABLIT_LAYERS={spec} reaches EXL3 o_proj on layers {clash}; this pack keeps "
            "o_proj BF16 only on the other layers (ABLIT edits BF16 o_proj only)")


def resolve(snapshot: Path, hub: Path, env: dict[str, str], explicit: set[str]) -> dict[str, str]:
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        return {}
    cfg = json.loads(config_path.read_text())
    profile = cfg.get("glm53_profile")
    if profile is None:
        return {}
    if not isinstance(profile, dict) or profile.get("name") != PROFILE:
        raise ValueError("unsupported glm53_profile")
    if env.get("TP", "2") != "2":
        raise ValueError(f"{PROFILE} requires TP=2; TP3/TP4 are not supported")
    text = cfg.get("text_config", cfg)
    if text.get("hidden_size") != 4096 or text.get("num_hidden_layers") != 45:
        raise ValueError("profile requires the GLM-5.3-Flash 45-layer/4096 target")
    quant = cfg.get("quantization_config", {})
    layers = quant.get("non_routed_exl3", {}).get("layers", {})
    if quant.get("quant_method") != "exl3" or not layers:
        raise ValueError("profile requires a non_routed_exl3 target pack")
    for suffix in ("self_attn.in_proj_qkvbfg_a", "mlp.shared_experts.down_proj",
                   "self_attn.fused_qkv_a_proj", "mlp.shared_experts.gate_up_proj",
                   "self_attn.o_proj", "self_attn.q_b_proj"):
        if not any(p.endswith("." + suffix) for p in layers):
            raise ValueError(f"H3 target is missing {suffix}")
    if env.get("ABLIT") == "1":
        check_ablit(cfg, env.get("ABLIT_LAYERS") or "15-45")
    if not (snapshot / "model.safetensors.index.json").is_file():
        raise ValueError("profile target requires model.safetensors.index.json")
    target_tensors, target_shards = inventory(snapshot)
    for prefix, entry in layers.items():
        base = prefix.replace("language_model.model.", "model.language_model.", 1)
        if base == "language_model.lm_head":
            base = "lm_head"
        leaf = base.rsplit(".", 1)[-1]
        shards = {
            "gate_up_proj": ("gate_proj", "up_proj"),
            "in_proj_qkvbfg_a": ("q_proj", "k_proj", "v_proj"),
            "fused_qkv_a_proj": ("q_a_proj", "kv_a_proj_with_mqa"),
        }.get(leaf, (leaf,))
        for shard in shards:
            require_packed(target_tensors, base.rsplit(".", 1)[0] + "." + shard
                           if "." in base else shard, entry["bits"])
    draft = profile["draft"]
    model, revision = draft["model"], draft["revision"]
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", model) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("draft requires a repository id and immutable 40-hex revision")
    draft_dir = hub / ("models--" + model.replace("/", "--")) / "snapshots" / revision
    if not (draft_dir / "model.safetensors").is_file():
        raise ValueError(f"paired draft model.safetensors missing: {draft_dir}")
    raw = (draft_dir / "config.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != draft["config_sha256"]:
        raise ValueError(f"paired draft config hash mismatch: {draft_dir}")
    dc = json.loads(raw)
    dq = dc.get("quantization_config", {})
    dl = dq.get("non_routed_exl3", {}).get("layers", {})
    if (dc.get("architectures") != ["DFlash2DraftModel"]
            or dc.get("hidden_size") != 4096 or dc.get("num_hidden_layers") != 5
            or dq.get("quant_method") != "exl3" or dq.get("scope") != "dflash2_draft"
            or dq.get("bits") != 6 or not dl or any(v.get("bits") != 6 for v in dl.values())):
        raise ValueError("paired draft must be the compatible 5-layer, 6-bpw EXL3 DFlash2 pack")
    tensors, _ = inventory(draft_dir)
    required = {"model.fc"}
    for i in range(5):
        root = f"model.layers.{i}."
        required.update(root + s for s in ("self_attn.qkv_proj", "self_attn.o_proj",
                        "mlp.gate_up_proj", "mlp.down_proj", "attention_conv.kernel_projection",
                        "mlp_conv.kernel_projection"))
        if dl.get(root + "self_attn.qkv_proj", {}).get("bf16_shards") != [1, 2]:
            raise ValueError("draft K/V must remain BF16")
        for part in ("k_proj", "v_proj"):
            key = f"layers.{i}.self_attn.{part}.weight"
            if tensors.get(key, {}).get("dtype") != "BF16":
                raise ValueError(f"draft is missing BF16 {key}")
    if set(dl) != required or any(v["shape"][-1] != 96 for k, v in tensors.items() if k.endswith(".trellis")):
        raise ValueError("draft EXL3 module inventory or 6-bpw trellis geometry mismatch")
    require_packed(tensors, "fc", 6)
    for i in range(5):
        for suffix in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj",
                       "mlp.up_proj", "mlp.down_proj", "attention_conv.kernel_projection",
                       "mlp_conv.kernel_projection"):
            require_packed(tensors, f"layers.{i}.{suffix}", 6)
    ablit = env.get("ABLIT") == "1"
    selected = {
        "GLM53_DENSE_EXL3": "1", "GLM53_DENSE_FP8": "off",
        "GLM53_DENSE_EXL3_PREFILL_BF16": H3, "GLM53_KDA_BF16_LARGE_M": "0",
        "ABLIT": "1" if ablit else "0", "SPEC_METHOD": "dflash", "DFLASH_DRAFT_TP": "2",
        "DFLASH_MODEL": model, "DFLASH_REVISION": revision,
        "DFLASH_CACHE_NAME": "models--" + model.replace("/", "--"),
        "EXPECTED_SHARDS": str(target_shards),
    }
    # Existing .env files retain #281 defaults. Only those known defaults may
    # be replaced automatically; caller exports are always deliberate choices.
    defaults = {"EXPECTED_SHARDS": "83", "GLM53_DENSE_EXL3": "0", "GLM53_DENSE_FP8": "all",
                "GLM53_DENSE_EXL3_PREFILL_BF16": "off", "GLM53_KDA_BF16_LARGE_M": "1",
                "DFLASH_MODEL": DEFAULT_DRAFT, "DFLASH_REVISION": DEFAULT_REVISION,
                "DFLASH_CACHE_NAME": "models--" + DEFAULT_DRAFT.replace("/", "--")}
    for key, wanted in selected.items():
        current = env.get(key)
        if current is None or current == wanted:
            continue
        if key in explicit or current != defaults.get(key):
            raise ValueError(f"{PROFILE} conflicts with {key}={current!r}; requires {wanted!r}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--hub", type=Path)
    parser.add_argument("--explicit", default="")
    parser.add_argument("--check-ablit", metavar="LAYERS",
                        help="only check that ABLIT_LAYERS avoid the pack's EXL3 o_proj")
    args = parser.parse_args()
    try:
        if args.check_ablit is not None:
            check_ablit(json.loads((args.snapshot / "config.json").read_text()), args.check_ablit)
            return 0
        if args.hub is None:
            parser.error("--hub is required")
        values = resolve(args.snapshot, args.hub, dict(os.environ), set(args.explicit.split()))
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        print(f"pack profile refused: {exc}", file=sys.stderr)
        return 2
    for key, value in values.items():
        print(f"{key}\t{value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
