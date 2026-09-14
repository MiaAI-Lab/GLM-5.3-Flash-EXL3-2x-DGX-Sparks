#!/usr/bin/env python3
"""Runtime TP=3 pads for the EXL3 image's Glm5Next model.py.

Pad attention 64→66 (local 22), same as NVFP4 3×. FlashInfer SM120 decode
only instantiates local heads in {8,16,32,64,128}; 22 is not in that table.
Do NOT pad the model to 96 (local 32): one rank becomes all dummy KDA heads
and logits collapse. Kernel-side 22→32 pad lives in the SM120 backend overlay.
Do NOT pad moe_intermediate_size: EXL3 trellis is packed 2048-wide; use EP.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "TP3-HEAD-PAD"
CANDIDATES = (
    Path("/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py"),
    Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm5next/nvidia/model.py"),
)

HEAD_PAD = '''
        # TP3-HEAD-PAD 64→66 (local 22). SM120 decode 22→32 is kernel-side.
        tp_size = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % tp_size != 0:
            padded = ((config.num_attention_heads + tp_size - 1) // tp_size) * tp_size
            config.num_attention_heads = padded
            if getattr(config, "num_key_value_heads", None):
                nkv = config.num_key_value_heads
                if nkv % tp_size != 0:
                    config.num_key_value_heads = padded
            lac = getattr(config, "linear_attn_config", None)
            if isinstance(lac, dict) and lac.get("num_heads"):
                lh = lac.get("num_heads")
                if lh and lh % tp_size != 0:
                    lac = dict(lac)
                    lac["num_heads"] = ((lh + tp_size - 1) // tp_size) * tp_size
                    config.linear_attn_config = lac
            ln = getattr(config, "linear_num_heads", None)
            if ln and ln % tp_size != 0:
                config.linear_num_heads = ((ln + tp_size - 1) // tp_size) * tp_size
        # EXL3: leave moe_intermediate_size=2048; use expert parallel.
'''

ALOG_PAD = '''
        # TP3-HEAD-PAD A_log + NVFP4-style checkpoint pad (64→66, shared I 2048→2112)
        _tp3 = get_tensor_model_parallel_world_size()
        def _tp3_pad_alog(loaded_weight, param=None):
            if (
                hasattr(loaded_weight, "dim")
                and loaded_weight.dim() == 1
                and loaded_weight.numel() == 64
                and _tp3 == 3
            ):
                return torch.nn.functional.pad(loaded_weight, (0, 2))
            if param is None or not hasattr(param, "shape"):
                return loaded_weight
            if tuple(getattr(loaded_weight, "shape", ())) == tuple(param.shape):
                return loaded_weight
            if loaded_weight.dim() != param.dim():
                return loaded_weight
            vocab = getattr(self.config, "vocab_size", None)
            if vocab and loaded_weight.dim() >= 1 and loaded_weight.shape[0] == vocab:
                return loaded_weight
            pad = []
            for ls, ps in zip(reversed(loaded_weight.shape), reversed(param.shape)):
                if ls == ps:
                    extra = 0
                elif ls > ps and ps * _tp3 >= ls:
                    extra = ps * _tp3 - ls
                elif ls < ps:
                    extra = ps - ls
                else:
                    extra = 0
                pad.extend((0, extra))
            if any(pad[1::2]):
                loaded_weight = torch.nn.functional.pad(loaded_weight, tuple(pad))
            return loaded_weight
'''


def _patch(path: Path) -> None:
    src = path.read_text()
    if "from math import gcd" not in src:
        src = src.replace(
            "from collections.abc import Iterable\n",
            "from collections.abc import Iterable\nfrom math import gcd\n",
            1,
        )
    if MARKER not in src:
        needle = "        self.config = config\n"
        if needle not in src:
            raise SystemExit(f"{path}: missing self.config = config anchor")
        src = src.replace(needle, needle + HEAD_PAD, 1)

        load_needle = "    def load_weights(self, weights:"
        idx = src.find(load_needle)
        if idx < 0:
            raise SystemExit(f"{path}: missing load_weights")
        nl = src.find("\n", idx)
        src = src[: nl + 1] + ALOG_PAD + src[nl + 1 :]
        # The 2-arg needle never matches this image: every call has extra
        # args / kwargs. Pad immediately before each loader.
        load_pads = (
            (
                "                param = params_dict[name]\n"
                "                weight_loader = param.weight_loader\n"
                "                weight_loader(param, loaded_weight, shard_id)\n",
                "                param = params_dict[name]\n"
                "                loaded_weight = _tp3_pad_alog(loaded_weight, param)\n"
                "                weight_loader = param.weight_loader\n"
                "                weight_loader(param, loaded_weight, shard_id)\n",
            ),
            (
                "                    param = params_dict[name]\n"
                "                    weight_loader = param.weight_loader\n"
                "                    weight_loader(\n"
                "                        param,\n"
                "                        loaded_weight,\n"
                "                        name,\n",
                "                    param = params_dict[name]\n"
                "                    loaded_weight = _tp3_pad_alog(loaded_weight, param)\n"
                "                    weight_loader = param.weight_loader\n"
                "                    weight_loader(\n"
                "                        param,\n"
                "                        loaded_weight,\n"
                "                        name,\n",
            ),
            (
                "                    param = params_dict[name]\n"
                "                    weight_loader = getattr(\n"
                "                        param, \"weight_loader\", default_weight_loader\n"
                "                    )\n"
                "                    weight_loader(param, loaded_weight, **kwargs)\n",
                "                    param = params_dict[name]\n"
                "                    loaded_weight = _tp3_pad_alog(loaded_weight, param)\n"
                "                    weight_loader = getattr(\n"
                "                        param, \"weight_loader\", default_weight_loader\n"
                "                    )\n"
                "                    weight_loader(param, loaded_weight, **kwargs)\n",
            ),
        )
        for old, new in load_pads:
            if old not in src:
                raise SystemExit(f"{path}: missing weight_loader pad anchor:\n{old}")
            if "loaded_weight = _tp3_pad_alog(loaded_weight, param)" not in new:
                raise SystemExit("pad rewrite lost helper")
            src = src.replace(old, new, 1)
    # Vocab 154880 % 3 != 0. Pass padding_size=lcm(64,tp) like NVFP4 3×.
    # Do not bind-mount a foreign VocabParallelEmbedding.py onto this image.
    old_emb = (
        "            self.embed_tokens = VocabParallelEmbedding(\n"
        "                config.vocab_size,\n"
        "                config.hidden_size,\n"
        "                prefix=f\"{prefix}.embed_tokens\",\n"
        "            )\n"
    )
    new_emb = (
        "            # TP3-VOCAB-PAD\n"
        "            tp = get_tensor_model_parallel_world_size()\n"
        "            vocab_pad = 64 * tp // gcd(64, tp)\n"
        "            self.embed_tokens = VocabParallelEmbedding(\n"
        "                config.vocab_size,\n"
        "                config.hidden_size,\n"
        "                padding_size=vocab_pad,\n"
        "                prefix=f\"{prefix}.embed_tokens\",\n"
        "            )\n"
    )
    if "TP3-VOCAB-PAD" not in src:
        if old_emb not in src:
            raise SystemExit(f"{path}: missing embed_tokens constructor")
        src = src.replace(old_emb, new_emb, 1)
        old_lm = (
            "            self.lm_head = ParallelLMHead(\n"
            "                self.config.vocab_size,\n"
            "                self.config.hidden_size,\n"
            "                quant_config=quant_config,\n"
            "                prefix=maybe_prefix(prefix, \"lm_head\"),\n"
            "            )\n"
        )
        new_lm = (
            "            tp = get_tensor_model_parallel_world_size()\n"
            "            vocab_pad = 64 * tp // gcd(64, tp)\n"
            "            self.lm_head = ParallelLMHead(\n"
            "                self.config.vocab_size,\n"
            "                self.config.hidden_size,\n"
            "                quant_config=quant_config,\n"
            "                padding_size=vocab_pad,\n"
            "                prefix=maybe_prefix(prefix, \"lm_head\"),\n"
            "            )\n"
        )
        if old_lm not in src:
            raise SystemExit(f"{path}: missing lm_head constructor")
        src = src.replace(old_lm, new_lm, 1)
        print(f"{path}: patched vocab padding_size")

    # Shared expert is native BF16. Pad I 2048→2112 and TP-shard it (NVFP4/GLM-5.2).
    # Routed EXL3 trellis stays 2048 + EP. Replicating via disable_tp + later
    # all-reduce 3×s the shared MLP.
    old_shared = (
        "            intermediate_size = config.moe_intermediate_size * config.n_shared_experts\n"
    )
    new_shared = (
        "            # TP3-SHARED-I BF16 pad 2048→2112; routed experts stay 2048+EP\n"
        "            intermediate_size = config.moe_intermediate_size * config.n_shared_experts\n"
        "            _tp_shared = get_tensor_model_parallel_world_size()\n"
        "            if intermediate_size % _tp_shared != 0:\n"
        "                _one = config.moe_intermediate_size\n"
        "                _pad_one = 2112 if _one == 2048 else ((_one + _tp_shared - 1) // _tp_shared) * _tp_shared\n"
        "                intermediate_size = _pad_one * config.n_shared_experts\n"
    )
    if "TP3-SHARED-I" not in src:
        if old_shared not in src:
            raise SystemExit(f"{path}: missing shared-expert intermediate_size")
        src = src.replace(old_shared, new_shared, 1)
        print(f"{path}: patched shared-expert I pad")

    # Shared-expert MLP is native BF16 at moe_intermediate_size=2048.
    # 2048 % 3 != 0, so replicate it (disable_tp) instead of padding EXL3 tiles.
    old_mlp = (
        "            disable_tp=is_sequence_parallel,\n"
        "            prefix=f\"{prefix}.gate_up_proj\","
    )
    new_mlp = (
        "            disable_tp=is_sequence_parallel or (\n"
        "                intermediate_size % get_tensor_model_parallel_world_size() != 0),\n"
        "            prefix=f\"{prefix}.gate_up_proj\","
    )
    if "TP3-SHARED-MLP" not in src and old_mlp in src:
        src = src.replace(old_mlp, "            # TP3-SHARED-MLP\n" + new_mlp, 1)
        old_down = (
            "            disable_tp=is_sequence_parallel,\n"
            "            prefix=f\"{prefix}.down_proj\","
        )
        new_down = (
            "            disable_tp=is_sequence_parallel or (\n"
            "                intermediate_size % get_tensor_model_parallel_world_size() != 0),\n"
            "            prefix=f\"{prefix}.down_proj\","
        )
        src = src.replace(old_down, new_down, 1)
        print(f"{path}: patched shared-expert disable_tp")

    path.write_text(src)
    print(f"{path}: patched TP=3 head / A_log pads")


def main() -> None:
    found = [p for p in CANDIDATES if p.is_file()]
    if not found:
        raise SystemExit("glm5next model.py not in image")
    for p in found:
        _patch(p)


if __name__ == "__main__":
    main()
