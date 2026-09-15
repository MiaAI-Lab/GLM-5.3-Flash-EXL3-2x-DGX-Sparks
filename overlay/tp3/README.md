# TP=3 overlays

Vendored verbatim from [FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks](https://github.com/FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks)
(MIT), used only by `start-tp3.sh`. Nothing here is on the TP=2 path.

TP=3 is not a flags change. Almost nothing in GLM-5.3-Flash divides by three:

| what | value | handled by |
|---|---|---|
| attention / KV heads | 64 | `patch_tp3_glm.py` pads to 66 (local 22) |
| vocab | 154880 | `patch_tp3_glm.py` passes `padding_size=lcm(64,tp)=192` -> 154944 |
| shared-expert intermediate | 2048 | padded to 2112, or `disable_tp` when it still does not divide |
| routed `moe_intermediate_size` | 2048 | NOT padded — EXL3 trellis is packed 2048-wide; use `--enable-expert-parallel` |
| A_log (1-D, 64) | 64 | padded to 66 at load time |
| FlashInfer SM120 decode | local 22 | `flashinfer_mla_sparse_sm120.py` pads Q 22->32 for decode-sized calls and slices back |

Two traps recorded upstream, both load-bearing:
- Do **not** pad heads to 96 (local 32): one rank becomes all dummy KDA heads and logits collapse.
- Do **not** pad `moe_intermediate_size`; expert parallel is the supported route.

`patch_tp3_glm.py` rewrites the image's `glm5next/nvidia/model.py` in place at
container start; the four files under `vllm/` are bind-mounted over their
counterparts. Verified to apply cleanly to `glm53-flash-sm121:e3-20260907`
(vLLM v0.1.dev20051+g487ecf187) on 2026-09-14.
