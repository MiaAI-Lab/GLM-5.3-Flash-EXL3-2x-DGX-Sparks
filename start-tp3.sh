#!/usr/bin/env bash
# ============================================================================
# start-tp3.sh — 3× DGX Spark TP=3 launcher for GLM-5.3-Flash EXL3
# ============================================================================
#
# Optional sibling of start.sh. Does not change the supported 2× TP=2 path,
# and start.sh never reads anything this script writes. Booted on this kit
# 2026-09-14 (see README "3x Spark (TP=3)").
#
#   ./start-tp3.sh
#
# Layout (mp executor, not Ray):
#   rank 0  HEAD_IP     (default 10.0.0.1)  — vLLM API on :8888
#   rank 1  WORKER_IP   (default 10.0.0.2)  — --headless
#   rank 2  WORKER2_IP  (default 10.0.0.3)  — --headless
#   --tensor-parallel-size 3  --nnodes 3
#
# TP=3 needs three shape fixes that TP=2 and TP=4 do not. GLM-5.3-Flash has
# 64 attention heads, 64 KV heads and moe_intermediate_size=2048 — none of
# which divides by 3:
#   --hf-overrides             pads the head counts to 66 (3 × 22).
#                              Knob: TP3_HEAD_OVERRIDE (empty = no override).
#   --enable-expert-parallel   gives each rank whole experts (288 / 3 = 96)
#                              instead of slicing the expert intermediate.
#                              Knob: ENABLE_EXPERT_PARALLEL.
#   --mm-encoder-tp-mode data  runs the vision tower data-parallel, same
#                              divisibility reason. Knob: MM_ENCODER_TP_MODE.
# The DFlash2 drafter has 32 heads and 8 KV heads, so it cannot shard by 3
# either: DFLASH_DRAFT_TP defaults to 1 (rank 0 only) here, not 3.
#
# Technique from FlyCockpit's 3× recipe by way of jakejharris/jspark3 and
# outstandly/glm53-flash-3x-dgx-spark. Those run a different launcher
# (jspark3's fleetctl.py); only the flags above are borrowed. The
# orchestration, overlays, E3 kernels, adaptive-k and FP8 paths below are
# this repo's, unchanged from start-tp4.sh.
#
# Fabric: three Sparks need a RoCE path among all ranks (QSFP triangle on
# both CX7 ports, or a switch). Defaults set NCCL_CROSS_NIC=1. Pin per-rank
# WORKER2_CX7_IF / WORKER2_CX7_IB / WORKER2_GID in .env.tp3.
# Wire the triangle as a *directed ring* — each node Port0 to the next node
# Port1 — or one NIC pair can never connect (NCCL pairs NIC index to NIC
# index per channel). A mirrored ring does not work.
# Dual-port example: RANK NCCL_IB_HCA="rocep1s0f0,rocep1s0f1".
#
# Same image/weights as start.sh. Container names are glm53-exl3-tp3-* so a
# TP=2 serve is not accidentally reused. Stop with ./stop.sh (detects TP=2
# and/or TP=3) or ./start-tp3.sh stop. ./start.sh stop does not know rank 2.
#
# Usage:
#   ./start-tp3.sh                 start TP=3
#   ./start-tp3.sh download        head HF cache only
#   ./start-tp3.sh stop|restart|status
#   ./start-tp3.sh share           re-export the head HF cache, remount ranks
#   ./start-tp3.sh logs            follow head
#   ./start-tp3.sh logs 1|2        follow that worker rank
#
# Extra knobs live in .env.tp3 (copied from .env.tp3.example). start.sh
# never reads that file. Shared tokens/IPs can stay in .env.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    [ -f "$SCRIPT_DIR/.env.example" ] || {
        echo "ERROR: missing .env.example" >&2
        exit 1
    }
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    printf '\033[1;36m[glm53-exl3-tp3]\033[0m wrote .env from .env.example\n'
fi
if [ ! -f "$SCRIPT_DIR/.env.tp3" ]; then
    [ -f "$SCRIPT_DIR/.env.tp3.example" ] || {
        echo "ERROR: missing .env.tp3.example" >&2
        exit 1
    }
    cp "$SCRIPT_DIR/.env.tp3.example" "$SCRIPT_DIR/.env.tp3"
fi
# Caller exports (MTP_TOKENS=2 ./start.sh restart) must win over .env.
_cli_mtp="${MTP_TOKENS-}"
_cli_spec="${SPEC_METHOD-}"
_cli_eager="${ENFORCE_EAGER-}"
_cli_fused="${EXL3_FUSED_MOE-}"
_cli_row_tile="${EXL3_MOE_ROW_TILE-}"
_cli_temp_rows="${EXL3_TEMP_ROWS_FUSED-}"
_cli_fat_sorted="${EXL3_FAT_SORTED-}"
_cli_fat_batched="${EXL3_FAT_BATCHED-}"
_cli_fat_kernel="${EXL3_FAT_KERNEL-}"
_cli_fat_grouped="${EXL3_FAT_GROUPED-}"
_cli_mnbt="${MAX_NUM_BATCHED_TOKENS-}"
_cli_long_prefill_set="${LONG_PREFILL_TOKEN_THRESHOLD+1}"
_cli_long_prefill="${LONG_PREFILL_TOKEN_THRESHOLD-}"
_cli_image="${IMAGE-}"
_cli_util="${GPU_MEM_UTIL-}"
_cli_lm="${LANGUAGE_MODEL_ONLY-}"
_cli_max_num_seqs="${MAX_NUM_SEQS-}"
_cli_ablit="${ABLIT-}"
_cli_ablit_method="${ABLIT_METHOD-}"
_cli_ablit_direction="${ABLIT_DIRECTION-}"
_cli_ablit_layers="${ABLIT_LAYERS-}"
_cli_ablit_alpha="${ABLIT_ALPHA-}"
_cli_ablit_mtp="${ABLIT_INCLUDE_MTP-}"
# Setness-aware: an explicitly empty caller value is an operator error and
# must reach validate_numeric_config, not be swallowed by a .env value.
_cli_indexer_workspace_set="${GLM53_INDEXER_WORKSPACE+1}"
_cli_indexer_workspace="${GLM53_INDEXER_WORKSPACE-}"
_cli_spinwait_ms_set="${GLM53_SPINWAIT_MS+1}"
_cli_spinwait_ms="${GLM53_SPINWAIT_MS-}"
_cli_apc_swa_set="${GLM53_APC_RETENTION_INTERVAL_SWA+1}"
_cli_apc_swa="${GLM53_APC_RETENTION_INTERVAL_SWA-}"
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
# TP=3 overlay wins over the 2× knobs in .env.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env.tp3"
set +a
[ -n "${_cli_mtp}" ] && MTP_TOKENS="$_cli_mtp"
[ -n "${_cli_spec}" ] && SPEC_METHOD="$_cli_spec"
[ -n "${_cli_eager}" ] && ENFORCE_EAGER="$_cli_eager"
[ -n "${_cli_fused}" ] && EXL3_FUSED_MOE="$_cli_fused"
[ -n "${_cli_row_tile}" ] && EXL3_MOE_ROW_TILE="$_cli_row_tile"
[ -n "${_cli_temp_rows}" ] && EXL3_TEMP_ROWS_FUSED="$_cli_temp_rows"
[ -n "${_cli_fat_sorted}" ] && EXL3_FAT_SORTED="$_cli_fat_sorted"
[ -n "${_cli_fat_batched}" ] && EXL3_FAT_BATCHED="$_cli_fat_batched"
[ -n "${_cli_fat_kernel}" ] && EXL3_FAT_KERNEL="$_cli_fat_kernel"
[ -n "${_cli_fat_grouped}" ] && EXL3_FAT_GROUPED="$_cli_fat_grouped"
[ -n "${_cli_mnbt}" ] && MAX_NUM_BATCHED_TOKENS="$_cli_mnbt"
[ -n "${_cli_long_prefill_set}" ] && LONG_PREFILL_TOKEN_THRESHOLD="$_cli_long_prefill"
[ -n "${_cli_image}" ] && IMAGE="$_cli_image"
[ -n "${_cli_util}" ] && GPU_MEM_UTIL="$_cli_util"
[ -n "${_cli_lm}" ] && LANGUAGE_MODEL_ONLY="$_cli_lm"
[ -n "${_cli_max_num_seqs}" ] && MAX_NUM_SEQS="$_cli_max_num_seqs"
[ -n "${_cli_ablit}" ] && ABLIT="$_cli_ablit"
[ -n "${_cli_ablit_method}" ] && ABLIT_METHOD="$_cli_ablit_method"
[ -n "${_cli_ablit_direction}" ] && ABLIT_DIRECTION="$_cli_ablit_direction"
[ -n "${_cli_ablit_layers}" ] && ABLIT_LAYERS="$_cli_ablit_layers"
[ -n "${_cli_ablit_alpha}" ] && ABLIT_ALPHA="$_cli_ablit_alpha"
[ -n "${_cli_ablit_mtp}" ] && ABLIT_INCLUDE_MTP="$_cli_ablit_mtp"
[ -n "${_cli_indexer_workspace_set}" ] && GLM53_INDEXER_WORKSPACE="$_cli_indexer_workspace"
[ -n "${_cli_spinwait_ms_set}" ] && GLM53_SPINWAIT_MS="$_cli_spinwait_ms"
[ -n "${_cli_apc_swa_set}" ] && GLM53_APC_RETENTION_INTERVAL_SWA="$_cli_apc_swa"

# ----------------------------- configuration -------------------------------
MODEL="${MODEL:-Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw}"
# If the durable mirror is empty/moved, download.sh falls back to this id.
MODEL_FALLBACK="${MODEL_FALLBACK:-brandonmusic/GLM-5.3-Flash-tr3-4bpw}"
MODEL_CACHE_NAME="${MODEL_CACHE_NAME:-models--${MODEL//\//--}}"
MODEL_FALLBACK_CACHE_NAME="${MODEL_FALLBACK_CACHE_NAME:-models--${MODEL_FALLBACK//\//--}}"
# Hub commit on the Mia-AiLab mirror (the 5ab363a8-byte-identical upload).
MODEL_REVISION="${MODEL_REVISION:-25a44fdbf16862a46b7cc9921142c6c81350af2f}"
IMAGE="${IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.3-Flash-EXL3}"
GHCR_USER="${GHCR_USER:-MiaAI-Lab}"

HEAD_IP="${HEAD_IP:-10.0.0.1}"
WORKER_IP="${WORKER_IP:-10.0.0.2}"
# Same OS user on both Sparks unless .env sets WORKER_USER (mixed-account kits).
WORKER_USER="${WORKER_USER:-$USER}"
if [ "$WORKER_USER" = "$USER" ]; then
    WORKER_HOME="${WORKER_HOME:-$HOME}"
else
    WORKER_HOME="${WORKER_HOME:-/home/${WORKER_USER}}"
fi
WORKER_SSH="${WORKER_SSH:-${WORKER_USER}@${WORKER_IP}}"

# Rank 2 (experimental TP=3). Same OS user as WORKER_USER unless set.
WORKER2_IP="${WORKER2_IP:-10.0.0.3}"
WORKER2_USER="${WORKER2_USER:-$WORKER_USER}"
if [ "$WORKER2_USER" = "$USER" ]; then
    WORKER2_HOME="${WORKER2_HOME:-$HOME}"
else
    WORKER2_HOME="${WORKER2_HOME:-/home/${WORKER2_USER}}"
fi
WORKER2_SSH="${WORKER2_SSH:-${WORKER2_USER}@${WORKER2_IP}}"

HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f1np1}"
# Control plane (gloo + NCCL bootstrap). TP=2 can put this on the CX7 pin
# because head<->worker share one cable. A 3-node ring cannot: every pair has
# its own /24, so no single fabric interface reaches both peers, and gloo takes
# exactly ONE interface name (a comma list makes it fall back to the default
# route). Point these at an interface that reaches ALL ranks — the management
# LAN — and leave NCCL_IB_HCA on the CX7 HCAs so data still moves over RoCE.
# Empty = use the CX7 pins, i.e. the old behaviour.
SOCKET_IFNAME="${SOCKET_IFNAME:-}"
HEAD_SOCKET_IFNAME="${HEAD_SOCKET_IFNAME:-$SOCKET_IFNAME}"
WORKER_SOCKET_IFNAME="${WORKER_SOCKET_IFNAME:-$SOCKET_IFNAME}"
WORKER2_SOCKET_IFNAME="${WORKER2_SOCKET_IFNAME:-$SOCKET_IFNAME}"
# The address each rank advertises for the process group. Must live on the
# socket interface above. Empty = the rank's 10.0.0.x address.
HEAD_HOST_IP="${HEAD_HOST_IP:-}"
WORKER_HOST_IP="${WORKER_HOST_IP:-}"
WORKER2_HOST_IP="${WORKER2_HOST_IP:-}"
WORKER_CX7_IF="${WORKER_CX7_IF:-enp1s0f0np0}"
HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"
WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f0}"
WORKER2_CX7_IF="${WORKER2_CX7_IF:-$WORKER_CX7_IF}"
WORKER2_CX7_IB="${WORKER2_CX7_IB:-$WORKER_CX7_IB}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# The RoCEv2 GID index is per-NIC: the usable entry is the one whose GID matches
# that node's own fabric IP. Most pairs share a good index; some do not (this kit
# needs head=4, worker=3). Unset, both inherit NCCL_IB_GID_INDEX -> unchanged.
HEAD_GID="${HEAD_GID:-$NCCL_IB_GID_INDEX}"
WORKER_GID="${WORKER_GID:-$NCCL_IB_GID_INDEX}"
WORKER2_GID="${WORKER2_GID:-$NCCL_IB_GID_INDEX}"
# vLLM subtracts a CUDA-graph memory ESTIMATE from the KV pool. On this kit the
# estimate is 2.43 GiB while the captured graphs actually consume -0.19 GiB, so
# ~2.6 GiB of KV is reserved and never used. 0 keeps CUDA graphs ON and drops only
# the deduction. 1 = upstream default.
CG_ESTIMATE="${CG_ESTIMATE:-1}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
# A 3-node ring cannot index-match its NICs: every cable joins dev0 on one side
# to dev1 on the other, and a 3-cycle with two ports per node has no consistent
# labelling. NCCL pairs NIC index to NIC index by default, so it tries to reach
# a peer device on a subnet it has no path to and the QP transition times out
# (ibv_modify_qp ... 110 Connection timed out -> "NCCL error: unhandled system
# error"). Subnet-aware routing matches by subnet instead. 0 = stock.
NCCL_IB_SUBNET_AWARE_ROUTING="${NCCL_IB_SUBNET_AWARE_ROUTING:-1}"
# The rest of the ring settings, copied from ~/NewModels/DS4.1 which already
# runs three of these Sparks over this same cabling. P2P/SHM off keeps NCCL on
# IB instead of guessing a local transport; the buffer/proto/channel caps are
# what keep pinned host memory sane on a GB10 (512 connections x 9 MiB
# otherwise). Do not "tune" these without re-measuring there first.
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-1048576}"
NCCL_LL128_BUFFSIZE="${NCCL_LL128_BUFFSIZE:-262144}"
NCCL_PROTO="${NCCL_PROTO:-^LL128}"
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
NCCL_HOST_DIR="${NCCL_HOST_DIR:-$HOME/nccl-2.30.7}"
WORKER_NCCL_HOST_DIR="${WORKER_NCCL_HOST_DIR:-$WORKER_HOME/nccl-2.30.7}"
WORKER2_NCCL_HOST_DIR="${WORKER2_NCCL_HOST_DIR:-$WORKER2_HOME/nccl-2.30.7}"
NCCL_SO_NAME="${NCCL_SO_NAME:-libnccl.so.2.30.7}"
# glm53-flash already ships nvidia-nccl. LD_PRELOAD of the host 2.30.7 SO
# makes DeepEP assert duplicate NCCL (/nccl/... vs nvidia/nccl/lib/...).
# Set USE_HOST_NCCL=1 only if image NCCL cannot talk CX7.
USE_HOST_NCCL="${USE_HOST_NCCL:-0}"

TP="${TP:-3}"
NNODES="${NNODES:-3}"
PORT="${PORT:-8888}"
MASTER_PORT="${MASTER_PORT:-29521}"

MTP_TOKENS="${MTP_TOKENS:-2}"
# dflash (default, incoai/GLM-5.3-Flash-DFlash2, k=7) | mtp | none
SPEC_METHOD="${SPEC_METHOD:-dflash}"
DFLASH_MODEL="${DFLASH_MODEL:-incoai/GLM-5.3-Flash-DFlash2}"
DFLASH_CACHE_NAME="${DFLASH_CACHE_NAME:-models--${DFLASH_MODEL//\//--}}"
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
# Same pin as start.sh. Without it resolve_dflash_dir falls back to refs/main,
# which on this kit points at the older 2026-08-28 snapshot — a different
# drafter than the supported TP=2 path, silently. Empty = follow refs/main.
DFLASH_REVISION="${DFLASH_REVISION-dc77ff1c99eeb2df044ee3d4f0094eb033fee410}"
# 2 = shard the ~2.3 GiB DFlash2 drafter across TP (C4 keep, 2026-08-30:
# idle 8k 938 / 16k 972 / 100k 997; decode structured 65.1 / prose 27.1).
# 1 = rank 0 only (no CX7 on every draft step). Empty = inherit target TP.
# Do not pin attention_backend: SM121 already prefers FLASH_ATTN for
# non-causal dense SWA. TRITON_ATTN was an SM120 mask-fix this image lacks.
DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.87}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# 8192 chunk × long history oversubscribes GB10 persistent_topk smem (300k crash).
# E2 one-shot 2026-09-01: 7168 keep (100k ~1148 / 300k ~1107); 2048/3548 similar or slower.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-7168}"
# Cap a long chunked prefill so it cannot monopolize MNBT (issue #110).
# Explicit empty disables the flag. Must be <= MAX_NUM_BATCHED_TOKENS.
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD-3584}"
CHAT_TEMPLATE_HOST="${CHAT_TEMPLATE_HOST:-$SCRIPT_DIR/files/chat_template.jinja}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/opt/glm53/chat_template.jinja}"
VIDEO_PATCH_HOST="${VIDEO_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_glm_video_placeholders.py}"
STOP_PATCH_HOST="${STOP_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_suppress_stops_in_reasoning.py}"
SCHED_PATCH_HOST="${SCHED_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_scheduler_decode_floor.py}"
DRAFTER_PATCH_HOST="${DRAFTER_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_glm5_drafter_group.py}"
APC_PATCH_HOST="${APC_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_hybrid_prefix_hit.py}"
XGRAMMAR_PATCH_HOST="${XGRAMMAR_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_xgrammar_termination.py}"
KPOOL_TAIL_PATCH_HOST="${KPOOL_TAIL_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_kpool_tail_slotmap.py}"
SPINWAIT_PATCH_HOST="${SPINWAIT_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_spinwait.py}"
# Decode stack, same overlays start.sh uses at TP=2. Without these the TP=3
# path runs stock k=7 and BF16 dense projections: adaptive-k is worth +13-21 %
# on prose and dense FP8 about -11 ms/step on this kit, so leaving them out
# spends the third node's gain paying for missing optimisations.
# patch_dense_fp8.py installs this into site-packages, so it must be mounted
# even when GLM53_DENSE_FP8 is off (the patch refreshes the module either way).
EXL3_OVERLAY_HOST="${EXL3_OVERLAY_HOST:-$SCRIPT_DIR/overlay/exl3.py}"
ADAPTIVE_K_PATCH_HOST="${ADAPTIVE_K_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_adaptive_k.py}"
DENSE_FP8_PATCH_HOST="${DENSE_FP8_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_dense_fp8.py}"
# Same defaults as start.sh. Docker -e VAR= (empty) hides the Python fallbacks
# in overlay/patch_adaptive_k.py — EngineCore then dies on float('').
GLM53_ADAPTIVE_K="${GLM53_ADAPTIVE_K:-off}"
GLM53_ADAPTIVE_K_SET="${GLM53_ADAPTIVE_K_SET:-2,4,7}"
GLM53_ADAPTIVE_K_ALPHA="${GLM53_ADAPTIVE_K_ALPHA:-0.25}"
GLM53_ADAPTIVE_K_MARGIN="${GLM53_ADAPTIVE_K_MARGIN:-1.0}"
GLM53_ADAPTIVE_K_MIN_STEPS="${GLM53_ADAPTIVE_K_MIN_STEPS:-4}"
GLM53_ADAPTIVE_K_SATURATE="${GLM53_ADAPTIVE_K_SATURATE:-max}"
GLM53_ADAPTIVE_K_HIST="${GLM53_ADAPTIVE_K_HIST:-200}"
GLM53_DENSE_FP8="${GLM53_DENSE_FP8:-dense,kda}"
# TP=3 shape overlays (FlyCockpit, MIT — see overlay/tp3/README.md). Nothing in
# here is on the TP=2 path. Empty disables them, which will not boot at TP=3.
TP3_OVERLAY_HOST="${TP3_OVERLAY_HOST:-$SCRIPT_DIR/overlay/tp3}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
QUANTIZATION="${QUANTIZATION:-exl3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
# JSON default cannot sit in ${LIMIT_MM:-{...}} — } ends the expansion.
if [ -z "${LIMIT_MM:-}" ]; then
    LIMIT_MM='{"image":48,"video":1}'
fi
# Vision cost caps, same reasoning as start.sh: SKIP_MM_PROFILING reserves
# nothing for the tower, so LIMIT_MM has to be a ceiling this kit can encode.
# ${VAR-default} not ${VAR:-default}: an explicitly empty value means stock vLLM.
MM_IMAGE_TOKENS="${MM_IMAGE_TOKENS-2048}"
VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES-}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB-1}"

# --- TP=3 shape fixes (see header) -----------------------------------------
# 64 attention/KV heads do not divide by 3; pad to 66 = 3 x 22. Empty = leave
# the checkpoint's head counts alone (for a model that already divides by 3).
TP3_HEAD_OVERRIDE="${TP3_HEAD_OVERRIDE-66}"
# moe_intermediate_size=2048 does not divide by 3 either, so the MoE has to be
# expert-parallel (288 routed experts / 3 = 96 per rank) rather than sliced.
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
# Vision tower is data-parallel across ranks for the same reason. Empty = stock.
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE-data}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
# Graph-safe fused apply (device-side expert grouping). MTP k=2 decode is
# 1..4 seqs × 3 tokens (must include 3). DFlash2 k=7 is 1..4 seqs × 8 tokens
# (must include 8, 16, 24, 32).
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
if [ "${ENFORCE_EAGER}" != "1" ]; then
    case " ${EXTRA_ARGS:-} " in
        *" --cudagraph-capture-sizes "*|*" cudagraph-capture-sizes "*) ;;
        *)
            if [ "$SPEC_METHOD" = "dflash" ]; then
                EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--cudagraph-capture-sizes 1 2 4 8 16 24 32"
            else
                EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--cudagraph-capture-sizes 1 2 3 4 6 8 12"
            fi
            ;;
    esac
fi
# 1 = fused exl3_moe (decode). 0 restores the unique-expert LinearEXL3 loop.
EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"
# 1 = GPU row tiles for fat experts (prefill). 0 = LinearEXL3 fallback.
# Tile (P2a) and TEMP_ROWS=1024 (P2b) both lost at MNBT=1024 — leave 128.
EXL3_MOE_ROW_TILE="${EXL3_MOE_ROW_TILE:-0}"
# E3 grouped fat-expert kernels (default ON, same as start.sh). Must reach
# every rank: overlay/exl3.py treats a missing EXL3_FAT_GROUPED as off, so a
# host .env of 1 is a no-op unless docker -e forwards it.
EXL3_FAT_GROUPED="${EXL3_FAT_GROUPED:-1}"
# Fused exl3_moe temp rows/expert; experts above it are "fat". E3 wants 32
# (>= MAX_NUM_SEQS x (DFLASH_TOKENS+1)); E2 wants 256. Explicit value wins.
if [ "${EXL3_FAT_GROUPED}" != "0" ]; then
    EXL3_TEMP_ROWS_FUSED="${EXL3_TEMP_ROWS_FUSED:-32}"
else
    EXL3_TEMP_ROWS_FUSED="${EXL3_TEMP_ROWS_FUSED:-256}"
fi
# Sorted routing tier; higher tiers imply it even when this is 0.
EXL3_FAT_SORTED="${EXL3_FAT_SORTED:-0}"
# E1 batched tier: persistent scratch + combined gate/up; implies SORTED=1.
EXL3_FAT_BATCHED="${EXL3_FAT_BATCHED:-0}"
# E2 direct trellis kernel (default on). Implies BATCHED=1 and SORTED=1.
# Needs the patched extension — start.sh rebuilds when the recipe stamp drifts.
# Set all three flags to 0 for the legacy fat-expert path.
EXL3_FAT_KERNEL="${EXL3_FAT_KERNEL:-1}"

# --- abliteration (ablit/) --------------------------------------------------
# Load-time o_proj orthogonalization (overlay/ablit_runtime.py). Published
# recipe: layers 15-45 edited with the dealign direction, 0-14 stay stock
# safety anchors, MTP block included. 0 = stock weights. Applied identically
# on both TP ranks; the DFlash2 drafter is never touched.
ABLIT="${ABLIT:-0}"
ABLIT_METHOD="${ABLIT_METHOD:-auto}"           # auto | transplant | proj
ABLIT_DIRECTION="${ABLIT_DIRECTION:-dealign}"  # dealign | bf_oproj | /path/dir.pt
ABLIT_LAYERS="${ABLIT_LAYERS:-15-45}"          # inclusive; 45 = checkpoint MTP block
ABLIT_ALPHA="${ABLIT_ALPHA:-3.0}"              # 1.0 = plain projection, >1 over-projects
ABLIT_INCLUDE_MTP="${ABLIT_INCLUDE_MTP:-1}"

READY_TIMEOUT="${READY_TIMEOUT:-3600}"
# 1 = suppress client stop strings until </think> (DSpark #42 class).
GLM53_SUPPRESS_STOPS_IN_REASONING="${GLM53_SUPPRESS_STOPS_IN_REASONING:-1}"
# Mixed-step prefill policy when a peer is already decoding (issue #6).
# 0 = off (TP=3 default; fair is unmeasured on this topology);
# skip = do not mix; N>0 = cap tokens; fair = time-share mixing (opt-in).
# Fair knobs are forwarded on every rank even when CHUNK is not fair.
GLM53_MIXED_PREFILL_CHUNK="${GLM53_MIXED_PREFILL_CHUNK:-0}"
GLM53_FAIR_PREFILL_CHUNK="${GLM53_FAIR_PREFILL_CHUNK:-256}"
GLM53_FAIR_PREFILL_SHARE="${GLM53_FAIR_PREFILL_SHARE:-0.20}"
GLM53_FAIR_PREFILL_MAX_INTERVAL_MS="${GLM53_FAIR_PREFILL_MAX_INTERVAL_MS:-2000}"
GLM53_FAIR_PREFILL_MAX_STEP_MS="${GLM53_FAIR_PREFILL_MAX_STEP_MS:-1000}"
GLM53_FAIR_PREFILL_MAX_CHUNKS="${GLM53_FAIR_PREFILL_MAX_CHUNKS:-1}"
# Sparse-indexer prefill gather workspace (overlay/patch_indexer_workspace.py).
# stock = max_model_len * 40 entries (5036.40 MB locked at 1M, measured);
# rightsize = the legal per-step maximum, ~+26% KV. Default applies only
# when UNSET: an explicitly empty value is an operator error and
# validate_numeric_config rejects it rather than guessing a serving mode.
GLM53_INDEXER_WORKSPACE="${GLM53_INDEXER_WORKSPACE-stock}"
# SpinCondition reader busy-loop window. "stock" preserves vLLM's 1 s default;
# 1..1000 selects milliseconds. The frozen TP=2 sweep selected 16 ms.
GLM53_SPINWAIT_MS="${GLM53_SPINWAIT_MS-stock}"
# EngineCore stock timeout is 300s; mid-serve Triton/TileLang JIT on TP=2 can
# exceed that without being a true hang. NCCL watchdog is still 600s.
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
# 1 = after /health, burn DFlash2 BLOCK / sampler / kpool shapes. Nonfatal.
GLM53_BOOT_SHAPE_WARMUP="${GLM53_BOOT_SHAPE_WARMUP:-1}"
GLM53_WARMUP_REQ_TIMEOUT="${GLM53_WARMUP_REQ_TIMEOUT:-240}"

# OpenAI-compatible API bearer token. Read the native VLLM_API_KEY env var
# (vLLM falls back to it when --api-key is absent on the CLI), so the key
# never lands in argv / `non-default args` startup log. Empty = no auth.
# Same single-key semantics as the DeepSeek V4 Flash DSpark deployment.
VLLM_API_KEY="${VLLM_API_KEY:-}"

CONTAINER_HEAD="${CONTAINER_HEAD:-glm53-exl3-tp3-head}"
CONTAINER_WORKER="${CONTAINER_WORKER:-glm53-exl3-tp3-w1}"
CONTAINER_WORKER2="${CONTAINER_WORKER2:-glm53-exl3-tp3-w2}"

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_PATH="$HF_CACHE_DIR/hub/$MODEL_CACHE_NAME"
FALLBACK_MODEL_PATH="$HF_CACHE_DIR/hub/$MODEL_FALLBACK_CACHE_NAME"
DFLASH_PATH="$HF_CACHE_DIR/hub/$DFLASH_CACHE_NAME"
WORKER_CACHE_DIR="$WORKER_HOME/.cache/huggingface"
WORKER2_CACHE_DIR="${WORKER2_CACHE_DIR:-$WORKER2_HOME/.cache/huggingface}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/vllm-glm53-flash}"
WORKER_VLLM_CACHE="${WORKER_VLLM_CACHE:-$WORKER_HOME/.cache/vllm-glm53-flash}"
WORKER2_VLLM_CACHE="${WORKER2_VLLM_CACHE:-$WORKER2_HOME/.cache/vllm-glm53-flash}"
# Overlay FS ~/.triton and ~/.tilelang die on container recreate (TP=2 JIT
# stall → 600s NCCL watchdog). Persist next to the vLLM cache.
TRITON_HOST_CACHE="${TRITON_HOST_CACHE:-$CACHE_ROOT/triton}"
TILELANG_HOST_CACHE="${TILELANG_HOST_CACHE:-$CACHE_ROOT/tilelang}"
WORKER_TRITON_CACHE="${WORKER_TRITON_CACHE:-$WORKER_VLLM_CACHE/triton}"
WORKER_TILELANG_CACHE="${WORKER_TILELANG_CACHE:-$WORKER_VLLM_CACHE/tilelang}"
WORKER2_TRITON_CACHE="${WORKER2_TRITON_CACHE:-$WORKER2_VLLM_CACHE/triton}"
WORKER2_TILELANG_CACHE="${WORKER2_TILELANG_CACHE:-$WORKER2_VLLM_CACHE/tilelang}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.triton/cache}"
TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/root/.tilelang/cache}"

LOGDIR="$SCRIPT_DIR/logs"
HEAD_SCRIPT="$SCRIPT_DIR/.glm53-exl3-tp3-head.inner.sh"
WORKER_SCRIPT="$SCRIPT_DIR/.glm53-exl3-tp3-worker.inner.sh"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-120}"

# ------------------------------- helpers -----------------------------------
log()  { printf '\033[1;36m[glm53-exl3-tp3]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm53-exl3-tp3]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[glm53-exl3-tp3]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

# Worker weight distribution. 0 (default) = rsync a full copy to every rank.
# 1 = ranks 1-2 mount the head's HF cache over NFSv4 on ConnectX, so neither
# keeps its own ~164 GiB copy. Opt in from .env.tp3 (or .env).
NFS_SHARE="${NFS_SHARE:-0}"
NFS_RANKS="1 2"
# shellcheck source=files/nfs-share.sh
if [ -f "$SCRIPT_DIR/files/nfs-share.sh" ]; then
    source "$SCRIPT_DIR/files/nfs-share.sh"
elif [ "$NFS_SHARE" = "1" ]; then
    warn "files/nfs-share.sh missing — falling back to rsync copies (NFS_SHARE=0)"
    NFS_SHARE=0
fi

# What rank r bind-mounts at /root/.cache/huggingface: its own directory when
# rsyncing, or the read-only NFS docker volume when it reads the head's cache.
# Safe read-only: the container runs HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
# and the writable Triton/TileLang/vLLM caches are separate node-local mounts.
_tp3_hf_mount() {
    if [ "${NFS_SHARE:-0}" = "1" ]; then
        nfs_hf_mount_spec
    else
        printf '%s:/root/.cache/huggingface' "$(_tp3_rank_hf "$1")"
    fi
}

# GLM53 numeric config guard (begin)
_glm53_canonical_positive_int() {
    local name="$1" value="$2" maximum="$3" canonical
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$name must be a positive base-10 integer (got: $value)" >&2
        return 2
    fi
    canonical="$value"
    while [ "${canonical#0}" != "$canonical" ]; do canonical="${canonical#0}"; done
    [ -n "$canonical" ] || canonical=0
    if [ "$canonical" = 0 ] \
       || [ "${#canonical}" -gt "${#maximum}" ] \
       || [ "$canonical" -gt "$maximum" ]; then
        echo "$name must be between 1 and $maximum (got: $value)" >&2
        return 2
    fi
    printf -v "$name" '%s' "$canonical"
    # $name is a validated integer configuration variable.
    # shellcheck disable=SC2163
    export "$name"
}

# Enum knobs are exactly one of a fixed set. Not "non-empty means on": a
# typo'd knob must not silently pick a serving mode. GLM53_INDEXER_WORKSPACE
# sizes the sparse-indexer prefill workspace, and the patched
# get_max_prefill_buffer_size itself raises on anything but stock/rightsize
# (overlay/patch_indexer_workspace.py, _glm53_workspace_mode), so catching it
# here turns a container boot failure into a launcher error. The match is
# literal on both sides -- the "-stock" default applies only to an UNSET var,
# so "", " rightsize " and "RIGHTSIZE" all fail here and would fail there.
_glm53_validate_enum() {
    local name="$1" value="$2" allowed
    shift 2
    for allowed in "$@"; do
        [ "$value" = "$allowed" ] && return 0
    done
    echo "$name must be one of: $* (got: $value)" >&2
    return 2
}

_glm53_validate_spinwait_ms() {
    if [ "$GLM53_SPINWAIT_MS" = "stock" ]; then
        export GLM53_SPINWAIT_MS
        return 0
    fi
    _glm53_canonical_positive_int \
        GLM53_SPINWAIT_MS "$GLM53_SPINWAIT_MS" 1000
}

_glm53_validate_mixed_prefill() {
    if [ -n "${GLM53_MIXED_PREFILL_CHUNK+x}" ]; then
        case "$GLM53_MIXED_PREFILL_CHUNK" in
            skip|-1|0|off|no|fair) ;;
            *)
                _glm53_canonical_positive_int GLM53_MIXED_PREFILL_CHUNK \
                    "$GLM53_MIXED_PREFILL_CHUNK" "$MAX_NUM_BATCHED_TOKENS" || return
                ;;
        esac
        export GLM53_MIXED_PREFILL_CHUNK
    fi
    if [ -n "${GLM53_FAIR_PREFILL_CHUNK:-}" ]; then
        _glm53_canonical_positive_int GLM53_FAIR_PREFILL_CHUNK \
            "$GLM53_FAIR_PREFILL_CHUNK" "$MAX_NUM_BATCHED_TOKENS" || return
    fi
    if [ -n "${GLM53_FAIR_PREFILL_MAX_INTERVAL_MS:-}" ]; then
        _glm53_canonical_positive_int GLM53_FAIR_PREFILL_MAX_INTERVAL_MS \
            "$GLM53_FAIR_PREFILL_MAX_INTERVAL_MS" 600000 || return
    fi
    if [ -n "${GLM53_FAIR_PREFILL_MAX_STEP_MS:-}" ]; then
        _glm53_canonical_positive_int GLM53_FAIR_PREFILL_MAX_STEP_MS \
            "$GLM53_FAIR_PREFILL_MAX_STEP_MS" 600000 || return
    fi
    if [ -n "${GLM53_FAIR_PREFILL_MAX_CHUNKS:-}" ]; then
        _glm53_canonical_positive_int GLM53_FAIR_PREFILL_MAX_CHUNKS \
            "$GLM53_FAIR_PREFILL_MAX_CHUNKS" 16 || return
    fi
    if [ -n "${GLM53_FAIR_PREFILL_SHARE:-}" ]; then
        if ! [[ "$GLM53_FAIR_PREFILL_SHARE" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0+)?)$ ]] \
           || ! awk -v u="$GLM53_FAIR_PREFILL_SHARE" 'BEGIN { exit !(u >= 0 && u <= 1) }'; then
            echo "GLM53_FAIR_PREFILL_SHARE must be between 0 and 1 (got: $GLM53_FAIR_PREFILL_SHARE)" >&2
            return 2
        fi
        export GLM53_FAIR_PREFILL_SHARE
    fi
}

validate_numeric_config() {
    if ! [[ "$GPU_MEM_UTIL" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0+)?)$ ]] \
       || ! awk -v u="$GPU_MEM_UTIL" 'BEGIN { exit !(u > 0 && u <= 1) }'; then
        echo "GPU_MEM_UTIL must be greater than 0 and at most 1 (got: $GPU_MEM_UTIL)" >&2
        return 2
    fi
    _glm53_canonical_positive_int MAX_MODEL_LEN "$MAX_MODEL_LEN" 1000000 || return
    _glm53_canonical_positive_int MAX_NUM_SEQS "$MAX_NUM_SEQS" 4096 || return
    _glm53_canonical_positive_int MAX_NUM_BATCHED_TOKENS "$MAX_NUM_BATCHED_TOKENS" 8388608 || return
    if [ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ]; then
        _glm53_canonical_positive_int LONG_PREFILL_TOKEN_THRESHOLD \
            "$LONG_PREFILL_TOKEN_THRESHOLD" "$MAX_NUM_BATCHED_TOKENS" || return
    fi
    _glm53_validate_enum GLM53_INDEXER_WORKSPACE "${GLM53_INDEXER_WORKSPACE-stock}" \
        stock rightsize || return
    _glm53_validate_spinwait_ms || return
    _glm53_validate_mixed_prefill || return
    if [ -n "${GLM53_APC_RETENTION_INTERVAL:-}" ]; then
        echo "GLM53_APC_RETENTION_INTERVAL is supported only by start.sh (TP=2); unset it for start-tp3.sh" >&2
        return 2
    fi
    if [ -n "${GLM53_APC_RETENTION_INTERVAL_SWA:-}" ]; then
        echo "GLM53_APC_RETENTION_INTERVAL_SWA is supported only by start.sh (TP=2); unset it for start-tp3.sh" >&2
        return 2
    fi
}
# GLM53 numeric config guard (end)

banner() {
    local label="${1:-start-tp3.sh}"
    printf '\n'
    printf '  \033[1;36m┌────────────────────────────────────────────┐\033[0m\n'
    printf '  \033[1;36m│\033[0m  \033[1mGLM-5.3 Flash EXL3\033[0m  \033[2m·  %-11s\033[0m        \033[1;36m│\033[0m\n' "$label"
    printf '  \033[1;36m└────────────────────────────────────────────┘\033[0m\n'
    printf '\n'
}

# rank 1..3
_tp3_ssh_target() {
    case "$1" in
        1) printf '%s' "$WORKER_SSH" ;;
        2) printf '%s' "$WORKER2_SSH" ;;
        *) die "internal: bad worker rank $1" ;;
    esac
}
_tp3_rank_ip() {
    case "$1" in
        1) printf '%s' "$WORKER_IP" ;;
        2) printf '%s' "$WORKER2_IP" ;;
    esac
}
_tp3_rank_home() {
    case "$1" in
        1) printf '%s' "$WORKER_HOME" ;;
        2) printf '%s' "$WORKER2_HOME" ;;
    esac
}
_tp3_rank_hf() {
    case "$1" in
        1) printf '%s' "$WORKER_CACHE_DIR" ;;
        2) printf '%s' "$WORKER2_CACHE_DIR" ;;
    esac
}
_tp3_rank_vllm() {
    case "$1" in
        1) printf '%s' "$WORKER_VLLM_CACHE" ;;
        2) printf '%s' "$WORKER2_VLLM_CACHE" ;;
    esac
}
_tp3_rank_triton() {
    case "$1" in
        1) printf '%s' "$WORKER_TRITON_CACHE" ;;
        2) printf '%s' "$WORKER2_TRITON_CACHE" ;;
    esac
}
_tp3_rank_tilelang() {
    case "$1" in
        1) printf '%s' "$WORKER_TILELANG_CACHE" ;;
        2) printf '%s' "$WORKER2_TILELANG_CACHE" ;;
    esac
}
_tp3_rank_cx7_if() {
    case "$1" in
        1) printf '%s' "$WORKER_CX7_IF" ;;
        2) printf '%s' "$WORKER2_CX7_IF" ;;
    esac
}
# Socket interface for gloo/NCCL bootstrap; falls back to the CX7 pin.
_tp3_rank_socket_if() {
    local v=""
    case "$1" in
        1) v="$WORKER_SOCKET_IFNAME" ;;
        2) v="$WORKER2_SOCKET_IFNAME" ;;
    esac
    [ -n "$v" ] || v="$(_tp3_rank_cx7_if "$1")"
    printf '%s' "$v"
}
# Address the rank advertises; falls back to its 10.0.0.x address.
_tp3_rank_host_ip() {
    local v=""
    case "$1" in
        1) v="$WORKER_HOST_IP" ;;
        2) v="$WORKER2_HOST_IP" ;;
    esac
    [ -n "$v" ] || v="$(_tp3_rank_ip "$1")"
    printf '%s' "$v"
}
_tp3_rank_cx7_ib() {
    case "$1" in
        1) printf '%s' "$WORKER_CX7_IB" ;;
        2) printf '%s' "$WORKER2_CX7_IB" ;;
    esac
}
_tp3_rank_gid() {
    case "$1" in
        1) printf '%s' "$WORKER_GID" ;;
        2) printf '%s' "$WORKER2_GID" ;;
    esac
}
_tp3_rank_container() {
    case "$1" in
        1) printf '%s' "$CONTAINER_WORKER" ;;
        2) printf '%s' "$CONTAINER_WORKER2" ;;
    esac
}
_tp3_rank_nccl_dir() {
    case "$1" in
        1) printf '%s' "$WORKER_NCCL_HOST_DIR" ;;
        2) printf '%s' "$WORKER2_NCCL_HOST_DIR" ;;
    esac
}
_tp3_first_ib() { printf '%s' "${1%%,*}"; }

worker_ssh() { ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH" "$@"; }
worker_ssh_n() {
    local r="$1"; shift
    ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$(_tp3_ssh_target "$r")" "$@"
}

# Print the whole header block: everything between the shebang and
# `set -euo pipefail`, minus the ==== rulers. Beats a magic line number, which
# silently truncated ./start.sh status/logs/share out of --help.
usage() {
    sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" \
        | sed -e '/^set -euo pipefail/d' -e '/^# =\{10,\}$/d' -e 's/^# \{0,1\}//'
}

count_shards() {
    find "$1/snapshots" -name '*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]' || true
}

ensure_refs_main() {
    local ref="$MODEL_PATH/refs/main" snap
    [ -f "$ref" ] && [ -n "$(<"$ref")" ] && return 0
    snap="$(ls -1t "$MODEL_PATH/snapshots" 2>/dev/null | head -n 1 || true)"
    [ -n "$snap" ] || die "no snapshots under $MODEL_PATH — re-run download"
    mkdir -p "$MODEL_PATH/refs"
    printf '%s' "$snap" >"$ref"
    log "wrote refs/main -> $snap (hf download left it empty)"
}

resolve_model_dir() {
    local ref="$MODEL_PATH/refs/main" hash dir
    ensure_refs_main
    hash="$(<"$ref")"
    dir="$MODEL_PATH/snapshots/$hash"
    [ -f "$dir/config.json" ] || die "config.json missing in $dir — re-run with REFRESH_WEIGHTS=1"
    printf '/root/.cache/huggingface/hub/%s/snapshots/%s' "$MODEL_CACHE_NAME" "$hash"
}

ensure_dflash_refs_main() {
    local ref="$DFLASH_PATH/refs/main" snap
    [ -f "$ref" ] && [ -n "$(<"$ref")" ] && return 0
    snap="$(ls -1t "$DFLASH_PATH/snapshots" 2>/dev/null | head -n 1 || true)"
    [ -n "$snap" ] || die "no snapshots under $DFLASH_PATH — re-run download"
    mkdir -p "$DFLASH_PATH/refs"
    printf '%s' "$snap" >"$ref"
    log "wrote DFlash2 refs/main -> $snap"
}

# Even with draft_tensor_parallel_size=1 the drafter process still reads the
# world TP (3) — vllm/model_executor/models/qwen3_dflash.py asserts
# total_num_heads % tp_size == 0 and stock GQA is 32/8. FlyCockpit pads the
# draft config to 36/9 (local 12/3; the 4:1 Q:KV ratio is kept because
# FlashInfer requires qo % kv == 0). The shared snapshot must NOT be edited —
# start.sh at TP=2 loads the same files — so build a TP=3-only copy beside it.
# It lives under $HF_CACHE_DIR so the ranks already see it over NFS, and
# model.safetensors is hardlinked (same fs, zero extra bytes).
prepare_tp3_draft() {
    local src dst rev
    src="$1"
    rev="$(basename "$src")"
    dst="$HF_CACHE_DIR/glm53-tp3-draft/$rev"
    mkdir -p "$dst"
    # Hardlink the RESOLVED blob, not the snapshot symlink: that symlink is
    # relative (../../blobs/...) and would resolve outside this directory,
    # which the NFS-mounted ranks then cannot follow (FileNotFoundError).
    local blob
    blob="$(readlink -f "$src/model.safetensors")"
    [ -f "$blob" ] || die "draft model.safetensors missing: $src/model.safetensors"
    if [ ! -f "$dst/model.safetensors" ] || [ "$dst/model.safetensors" -ot "$blob" ]; then
        rm -f "$dst/model.safetensors"
        ln -f "$blob" "$dst/model.safetensors" 2>/dev/null \
            || cp -a "$blob" "$dst/model.safetensors"
    fi
    cp -f "$(readlink -f "$src/config.json")" "$dst/config.json"
    rm -f "$dst/config.json.orig"
    python3 "$TP3_OVERLAY_HOST/pad-tp3-config.py" "$dst/config.json" --tp "$TP" >&2 \
        || die "could not pad the TP=3 draft config at $dst"
    printf '/root/.cache/huggingface/glm53-tp3-draft/%s' "$rev"
}

resolve_dflash_dir() {
    local ref="$DFLASH_PATH/refs/main" hash dir
    if [ -n "${DFLASH_REVISION:-}" ]; then
        hash="$DFLASH_REVISION"
    else
        ensure_dflash_refs_main
        hash="$(<"$ref")"
    fi
    dir="$DFLASH_PATH/snapshots/$hash"
    [ -f "$dir/config.json" ] || die "DFlash2 config.json missing in $dir"
    printf '/root/.cache/huggingface/hub/%s/snapshots/%s' "$DFLASH_CACHE_NAME" "$hash"
}

check_port_free() {
    local port="$1" envname="$2"
    command -v ss >/dev/null 2>&1 || return 0
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
        if docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
            die "port ${port} is held by ${CONTAINER_HEAD} — use './start-tp3.sh restart' or './start-tp3.sh stop' first"
        fi
        die "port ${port} is already in use — stop it or rerun with ${envname}=<free-port>"
    fi
}

trap 'warn "interrupted — containers keep running ('"'"'./start-tp3.sh logs'"'"' to watch, '"'"'./start-tp3.sh stop'"'"' to stop)"; exit 130' INT

# ------------------------------ preflight ----------------------------------
preflight() {
    command -v docker  >/dev/null 2>&1 || die "docker not found on head"
    command -v curl    >/dev/null 2>&1 || die "curl not found on head"
    command -v rsync   >/dev/null 2>&1 || die "rsync not found on head"
    docker info >/dev/null 2>&1 || die "cannot talk to docker daemon on head"

    ip -4 addr show 2>/dev/null | grep -q "inet ${HEAD_IP}/" \
        || die "HEAD_IP=${HEAD_IP} is not assigned on this host — set it in .env"

    [ "$TP" = "3" ] || warn "TP=${TP} — this script is meant for TP=3"
    [ "$NNODES" = "3" ] || warn "NNODES=${NNODES} — this script is meant for nnodes=3"
    local r ssh_t
    for r in 1 2; do
        ssh_t="$(_tp3_ssh_target "$r")"
        log "checking worker rank ${r} ${ssh_t} ..."
        worker_ssh_n "$r" true 2>/dev/null \
            || die "cannot ssh (key-based) to ${ssh_t} — set up passwordless ssh first"
        worker_ssh_n "$r" "docker info >/dev/null 2>&1" \
            || die "rank ${r} (${ssh_t}) cannot talk to its docker daemon"
        worker_ssh_n "$r" "nvidia-smi -L 2>/dev/null | grep -q GB10" \
            || warn "no GB10 GPU visible on rank ${r} (${ssh_t})"
    done

    # Each rank's GID index must name a populated entry on ITS OWN CX7 device.
    # An empty (all-zero) entry passes every earlier check and then kills that
    # rank ~60 s in with ibv_modify_qp errno 61 "No data available". The index is
    # per-NIC, so validate head and worker separately: some pairs share one good
    # index, others need different ones (HEAD_GID / WORKER_GID).
    local gid_head gid_worker gid_path ib r ssh_t gid_ok=1
    ib="$(_tp3_first_ib "$HEAD_CX7_IB")"
    gid_path="/sys/class/infiniband/${ib}/ports/1/gids/${HEAD_GID}"
    gid_head=$(cat "$gid_path" 2>/dev/null | tr -d ':0' || true)
    [ -n "$gid_head" ] || { warn "head GID index ${HEAD_GID} is EMPTY on ${ib}"; gid_ok=0; }
    for r in 1 2; do
        ib="$(_tp3_first_ib "$(_tp3_rank_cx7_ib "$r")")"
        gid_path="/sys/class/infiniband/${ib}/ports/1/gids/$(_tp3_rank_gid "$r")"
        gid_worker=$(worker_ssh_n "$r" "cat '$gid_path' 2>/dev/null" | tr -d ':0' || true)
        if [ -z "$gid_worker" ]; then
            warn "rank ${r} GID index $(_tp3_rank_gid "$r") is EMPTY on ${ib}"
            gid_ok=0
        fi
    done
    if [ "$gid_ok" != "1" ]; then
        if [ -z "$gid_head" ]; then
            warn "head GID index ${HEAD_GID} is EMPTY on $(_tp3_first_ib "$HEAD_CX7_IB")"
        fi
        warn "GID tables — pick each node's ::ffff:<ip> entry whose type is RoCE v2;"
        warn "the two indices need not match, and a v1 entry at the same index will not work:"
        for i in 0 1 2 3 4 5 6 7; do
            printf '    head   gid%s: %-40s %s\n' "$i" \
                "$(cat "/sys/class/infiniband/${HEAD_CX7_IB}/ports/1/gids/$i" 2>/dev/null)" \
                "$(cat "/sys/class/infiniband/${HEAD_CX7_IB}/ports/1/gid_attrs/types/$i" 2>/dev/null)" >&2
        done
        worker_ssh "for i in 0 1 2 3 4 5 6 7; do printf '    worker gid%s: %-40s %s\n' \"\$i\" \"\$(cat /sys/class/infiniband/${WORKER_CX7_IB}/ports/1/gids/\$i 2>/dev/null)\" \"\$(cat /sys/class/infiniband/${WORKER_CX7_IB}/ports/1/gid_attrs/types/\$i 2>/dev/null)\"; done" >&2 || true
        die "set NCCL_IB_GID_INDEX (same index both ranks) or HEAD_GID/WORKER_GID (per rank) in .env to populated indices"
    fi

    [ "$TP" = "3" ] || warn "TP=${TP} — expected TP=3 for start-tp3.sh"
    [ "$NNODES" = "3" ] || warn "NNODES=${NNODES} — expected 3"

    local others cname
    for r in 1 2; do
        cname="$(_tp3_rank_container "$r")"
        others=$(worker_ssh_n "$r" "docker ps --format '  {{.Names}}  ({{.Image}})'" 2>/dev/null | grep -v "^  ${cname}" || true)
        if [ -n "$others" ]; then
            warn "other containers are running on rank ${r} ($(_tp3_ssh_target "$r")):"
            echo "$others" >&2
            warn "this model needs most of each GB10 — stop GPU containers on that Spark first"
        fi
    done

    check_port_free "$PORT" PORT
    check_port_free "$MASTER_PORT" MASTER_PORT

    [ -f "$STOP_PATCH_HOST" ] || die "$STOP_PATCH_HOST missing"
    [ -f "$SCHED_PATCH_HOST" ] || die "$SCHED_PATCH_HOST missing"
    [ -f "$DRAFTER_PATCH_HOST" ] || die "$DRAFTER_PATCH_HOST missing"
    [ -f "$APC_PATCH_HOST" ] || die "$APC_PATCH_HOST missing"
    [ -f "$XGRAMMAR_PATCH_HOST" ] || die "$XGRAMMAR_PATCH_HOST missing"
    [ -f "$KPOOL_TAIL_PATCH_HOST" ] || die "$KPOOL_TAIL_PATCH_HOST missing"
    [ -f "$SPINWAIT_PATCH_HOST" ] || die "$SPINWAIT_PATCH_HOST missing"
    [ -f "$SCRIPT_DIR/overlay/patch_ablit.py" ] || die "$SCRIPT_DIR/overlay/patch_ablit.py missing"
    [ -f "$SCRIPT_DIR/overlay/ablit_runtime.py" ] || die "$SCRIPT_DIR/overlay/ablit_runtime.py missing"
    [ -f "$SCRIPT_DIR/ablit/LAYER_MAP.json" ] || die "$SCRIPT_DIR/ablit/LAYER_MAP.json missing"
    if [ "$ABLIT" = "1" ]; then
        log "ablit: ON (method=${ABLIT_METHOD} direction=${ABLIT_DIRECTION} layers=${ABLIT_LAYERS} alpha=${ABLIT_ALPHA} mtp=${ABLIT_INCLUDE_MTP})"
    fi

    local need_kb=$((180 * 1024 * 1024)) avail
    mkdir -p "$HF_CACHE_DIR"
    avail=$(df -Pk "$HF_CACHE_DIR" 2>/dev/null | awk 'NR==2{print $4}' || true)
    [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on head for a ~164 GiB model"
    if [ "${NFS_SHARE:-0}" = "1" ]; then
        log "NFS_SHARE=1 — ranks read the head HF cache, no per-rank copy to size for"
    else
        for r in 1 2; do
            avail=$(worker_ssh_n "$r" "df -Pk '$(_tp3_rank_home "$r")' 2>/dev/null" | awk 'NR==2{print $4}' || true)
            [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on rank ${r} for a ~164 GiB model"
        done
    fi

    # The worker HF cache must be writable by the SSH user before the ~164 GiB
    # sync starts. A root-owned ~/.cache/huggingface (prior sudo/docker
    # prepare on the worker) otherwise fails mid-sync with a bare mkdir
    # permission error. mkdir -p is idempotent and is what sync does anyway.
    if [ "${NFS_SHARE:-0}" != "1" ]; then
        for r in 1 2; do
            if ! worker_ssh_n "$r" "mkdir -p '$(_tp3_rank_hf "$r")/hub' && test -w '$(_tp3_rank_hf "$r")/hub'"; then
                die "rank ${r} cannot write $(_tp3_rank_hf "$r")/hub — fix ownership, e.g. ssh $(_tp3_ssh_target "$r") \"sudo chown -R \$USER: '$(_tp3_rank_hf "$r")'\""
            fi
        done
    fi

}

# ------------------------------ image --------------------------------------
image_from_registry() {
    case "$IMAGE" in
        */*) return 0 ;;
        *) return 1 ;;
    esac
}

login_ghcr_if_token() {
    [ -n "${GHCR_TOKEN:-}" ] || return 0
    log "docker login ghcr.io as ${GHCR_USER} (GHCR_TOKEN)"
    echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin >/dev/null
}

login_ghcr_if_token_worker() {
    [ -n "${GHCR_TOKEN:-}" ] || return 0
    local r
    for r in 1 2; do
        log "docker login ghcr.io on rank ${r} as ${GHCR_USER} (GHCR_TOKEN)"
        echo "$GHCR_TOKEN" | worker_ssh_n "$r" "docker login ghcr.io -u '$GHCR_USER' --password-stdin" >/dev/null
    done
}

# Identity for "does the worker already have the head's image?". No single
# field survives every path: overlay2 and containerd disagree on .Id (config
# digest vs index digest, issue #8), and docker save | docker load drops
# RepoDigests, so a shipped image never matched the GHCR tag it came from and
# we re-shipped the whole image on every run. RootFS.Layers (diff IDs) is
# identical on both sides in both cases — fold it into a short digest (the
# full layer list does not belong in a log line) and keep RepoDigest/.Id only
# as fallbacks for the rare inspect that reports no layers.
_IMAGE_KEY_FMT='{{if .RootFS.Layers}}layers {{join .RootFS.Layers ","}}{{else if .RepoDigests}}other {{index .RepoDigests 0}}{{else}}other {{.Id}}{{end}}'

parse_image_key() {
    local raw
    raw="$(tr -d '\r' | sed -n 's/^GLM53KEY //p' | tail -n 1)"
    case "$raw" in
        "layers "*) printf 'layers:%s' "$(printf '%s' "${raw#layers }" | sha256sum | cut -c1-16)" ;;
        "other "*)  printf '%s' "${raw#other }" ;;
    esac
}

local_image_key() {
    docker image inspect -f "GLM53KEY ${_IMAGE_KEY_FMT}" "$IMAGE" 2>/dev/null | parse_image_key
}

worker_image_key() {
    worker_ssh_n "${1:-1}" "docker image inspect -f 'GLM53KEY ${_IMAGE_KEY_FMT}' '$IMAGE' 2>/dev/null" | parse_image_key
}

images_match() {
    [ -n "${1:-}" ] && [ -n "${2:-}" ] && [ "$1" = "$2" ]
}

image_platform() {
    if [ -n "${IMAGE_PLATFORM:-}" ]; then
        printf '%s' "$IMAGE_PLATFORM"
        return
    fi
    local p
    p="$(docker image inspect -f '{{.Os}}/{{.Architecture}}' "$IMAGE" 2>/dev/null || true)"
    printf '%s' "${p:-linux/arm64}"
}

# Hash of Dockerfile + overlay/tests/files/ablit inputs that docker COPY.
# Compared to LABEL glm53.recipe.stamp so a git pull rebuilds once.
overlay_recipe_hash() {
    {
        printf '%s\n' "$SCRIPT_DIR/Dockerfile"
        find "$SCRIPT_DIR/overlay" "$SCRIPT_DIR/files" "$SCRIPT_DIR/tests" \
            "$SCRIPT_DIR/ablit" \
            -type f \
            ! -path '*/__pycache__/*' \
            ! -path '*/.pytest_cache/*' \
            ! -path '*/ablit/transplant/*' \
            ! -path '*/files/nfs-server/*' \
            ! -path '*/files/nfs-share.sh' \
            ! -name '*.pyc' \
            2>/dev/null
    } | LC_ALL=C sort | xargs -d '\n' -r sha256sum | sha256sum | awk '{print $1}'
}

image_recipe_stamp() {
    local stamp
    stamp="$(docker image inspect -f '{{ index .Config.Labels "glm53.recipe.stamp" }}' "$IMAGE" 2>/dev/null || true)"
    case "$stamp" in
        ""|"<no value>"|"<nil>") printf '' ;;
        *) printf '%s' "$stamp" ;;
    esac
}

build_image() {
    local stamp
    stamp="$(overlay_recipe_hash)"
    log "building ${IMAGE} from Dockerfile stamp=${stamp:0:12} (log: $LOGDIR/build-sm121.log) ..."
    docker build --build-arg "GLM53_RECIPE_STAMP=$stamp" -t "$IMAGE" "$SCRIPT_DIR" \
        >"$LOGDIR/build-sm121.log" 2>&1 \
        || { tail -n 40 "$LOGDIR/build-sm121.log" >&2; die "docker build of $IMAGE failed"; }
}

pull_image() {
    login_ghcr_if_token
    log "pulling ${IMAGE} ..."
    docker pull "$IMAGE" && return 0
    die "docker pull ${IMAGE} failed.
  :exl3 is a public GHCR package — check network / disk.
  If you still get 401/403: echo YOUR_PAT | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin
  Overlay rebuild: BUILD=1 ./start.sh. Recipe-stamp drift also rebuilds; SKIP_BUILD=1 keeps GHCR."
}

pull_image_on_worker() {
    local r="${1:-1}"
    login_ghcr_if_token_worker
    log "pulling ${IMAGE} on rank ${r} ..."
    worker_ssh_n "$r" "docker pull '$IMAGE'"
}

# ~21 GiB over ssh. Without a counter in the pipe this logs one line and then
# looks hung for minutes, which is indistinguishable from a stalled launch.
# dd bs=4M status=progress writes bytes/rate to stderr; no extra dependency
# (pv is not installed on this kit).
ship_image_to_worker() {
    local r="${1:-1}" platform size_gib rc=0
    platform="$(image_platform)"
    size_gib="$(docker image inspect "$IMAGE" --format '{{.Size}}' 2>/dev/null \
        | awk '$1 ~ /^[0-9]+$/ {printf "%.1f", $1/1073741824}')"
    log "shipping ${IMAGE} (${platform}, ~${size_gib:-?} GiB) to rank ${r} via docker save | ssh docker load ..."
    log "  (progress below is bytes sent; the rank is silent until docker load finishes)"
    if dd --help 2>/dev/null | grep -q 'status='; then
        docker save --platform "$platform" "$IMAGE" \
            | dd bs=4M status=progress \
            | worker_ssh_n "$r" docker load || rc=$?
    else
        docker save --platform "$platform" "$IMAGE" | worker_ssh_n "$r" docker load || rc=$?
    fi
    [ "$rc" = "0" ] && return 0
    warn "docker save --platform ${platform} failed — retrying without --platform"
    docker save "$IMAGE" | worker_ssh_n "$r" docker load
}

ensure_image() {
    mkdir -p "$LOGDIR"
    local head_ok=0 worker_ok=0 head_key="" worker_key=""
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        head_ok=1
        head_key="$(local_image_key || true)"
    fi
    worker_ok=1
    for r in 1 2; do
        if worker_ssh_n "$r" "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
            worker_key="$(worker_image_key "$r" || true)"
            if images_match "$head_key" "$worker_key"; then
                :
            else
                worker_ok=0
                log "rank ${r} image differs (head=${head_key:-none} worker=${worker_key:-none}) — will refresh"
            fi
        else
            worker_ok=0
        fi
    done
    local skip_pull="${SKIP_PULL:-0}"
    [ "${PULL:-0}" = "1" ] && skip_pull=0
    local wanted_stamp have_stamp have_short
    wanted_stamp="$(overlay_recipe_hash)"
    have_stamp=""
    [ "$head_ok" = "1" ] && have_stamp="$(image_recipe_stamp)"
    have_short="${have_stamp:0:12}"
    if [ "${BUILD:-0}" != "1" ] && [ "${SKIP_BUILD:-0}" != "1" ]; then
        if [ "$head_ok" = "0" ] || [ "$have_stamp" != "$wanted_stamp" ]; then
            log "image recipe ${have_short:-none} != repo ${wanted_stamp:0:12} — rebuilding (SKIP_BUILD=1 keeps GHCR)"
            BUILD=1
        fi
    elif [ "${SKIP_BUILD:-0}" = "1" ] && [ "$have_stamp" != "$wanted_stamp" ]; then
        warn "SKIP_BUILD=1 — not rebuilding; stamp ${have_short:-none} != repo ${wanted_stamp:0:12}"
    fi
    if [ "${BUILD:-0}" = "1" ]; then
        build_image
        head_key="$(local_image_key || true)"
        head_ok=1
        worker_ok=0
    elif image_from_registry && [ "$skip_pull" != "1" ]; then
        local before_key="$head_key"
        pull_image
        head_key="$(local_image_key || true)"
        head_ok=1
        if [ "$head_key" != "$before_key" ]; then
            log "pulled ${IMAGE} (${before_key:-missing} -> ${head_key})"
        else
            log "${IMAGE} already current"
        fi
        if images_match "$worker_key" "$head_key"; then
            worker_ok=1
        else
            worker_ok=0
        fi
    elif [ "$head_ok" = "0" ]; then
        if image_from_registry && [ "$skip_pull" = "1" ]; then
            die "SKIP_PULL=1 but ${IMAGE} is not on the head"
        fi
        build_image
        head_key="$(local_image_key || true)"
        head_ok=1
        worker_ok=0
    fi
    if [ "${SKIP_SHIP:-0}" = "1" ]; then
        [ "$worker_ok" = "1" ] || warn "SKIP_SHIP=1 — not copying ${IMAGE} to workers"
    elif [ "$worker_ok" = "0" ]; then
        for r in 1 2; do
            local wok=0
            worker_key="$(worker_image_key "$r" || true)"
            if images_match "$head_key" "$worker_key"; then
                continue
            fi
            if image_from_registry && [ "$skip_pull" != "1" ] && [ "${BUILD:-0}" != "1" ]; then
                if pull_image_on_worker "$r"; then
                    worker_key="$(worker_image_key "$r" || true)"
                    if images_match "$head_key" "$worker_key"; then
                        wok=1
                        log "rank ${r} pulled ${IMAGE} — matches head"
                    else
                        warn "rank ${r} pull left a different image — shipping"
                    fi
                else
                    warn "rank ${r} docker pull failed — shipping over SSH"
                fi
            fi
            if [ "$wok" = "0" ]; then
                ship_image_to_worker "$r"
                worker_key="$(worker_image_key "$r" || true)"
                if images_match "$head_key" "$worker_key"; then
                    :
                elif worker_ssh_n "$r" "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
                    warn "rank ${r} has ${IMAGE} after ship but keys still differ — continuing"
                else
                    die "rank ${r} still missing ${IMAGE} after ship"
                fi
            fi
        done
    fi
    if [ "${SKIP_OVERLAY_VERIFY:-0}" != "1" ]; then
        log "GPU EXL3 self-check on ${IMAGE} (log: $LOGDIR/overlay-verify.log) ..."
        docker run --rm --gpus all \
            -e EXL3_SELFCHECK_GPU=1 \
            --entrypoint python3 "$IMAGE" /opt/glm53/test_exl3_overlay.py \
            >"$LOGDIR/overlay-verify.log" 2>&1 \
            || { tail -n 80 "$LOGDIR/overlay-verify.log" >&2; die "EXL3 overlay GPU self-check failed"; }
        log "overlay verify OK"
    fi
    log "image ready on all three nodes"
}

# ---------------------------- weight download ------------------------------
# Use an already-complete local tree (primary or upstream fallback). If the
# durable Mia-AiLab mirror is still filling / 404s, keep serving from the
# brandonmusic cache folder without a second 164 GiB pull.
adopt_complete_weights() {
    local have
    have="$(count_shards "$MODEL_PATH")"
    if [ "${have:-0}" -ge "$EXPECTED_SHARDS" ]; then
        ensure_refs_main
        log "weights already present: $MODEL_PATH ($have shards)"
        return 0
    fi
    have="$(count_shards "$FALLBACK_MODEL_PATH")"
    if [ "${have:-0}" -ge "$EXPECTED_SHARDS" ]; then
        log "primary cache incomplete — using fallback ${MODEL_FALLBACK} at $FALLBACK_MODEL_PATH ($have shards)"
        MODEL_PATH="$FALLBACK_MODEL_PATH"
        MODEL_CACHE_NAME="$MODEL_FALLBACK_CACHE_NAME"
        ensure_refs_main
        return 0
    fi
    return 1
}

# Resolve the HF CLI even when it lives outside PATH (venv installs), with a
# python huggingface_hub fallback when no binary exists (issue #22, item 1).
# Sets the global HF_BIN_CMD array. HF_BIN (may contain arguments) wins when
# its first word resolves. Returns 1 when nothing usable is found.
resolve_hf_bin() {
    HF_BIN_CMD=()
    if [ -n "${HF_BIN:-}" ]; then
        read -ra HF_BIN_CMD <<< "$HF_BIN"
        if command -v "${HF_BIN_CMD[0]}" >/dev/null 2>&1; then return 0; fi
        HF_BIN_CMD=()
    fi
    local cand
    for cand in hf huggingface-cli "$HOME/.local/bin/hf" "$HOME/.hf-cli/venv/bin/hf" /opt/hf-cli/venv/bin/hf; do
        if command -v "$cand" >/dev/null 2>&1; then HF_BIN_CMD=("$cand"); return 0; fi
    done
    if command -v python3 >/dev/null 2>&1 && python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
        HF_BIN_CMD=(python3 -m huggingface_hub.commands.huggingface_cli)
        return 0
    fi
    return 1
}

hf_download_repo() {
    local repo="$1"
    shift
    local -a args=("$repo")
    if [ -n "${MODEL_REVISION:-}" ] && [ "$repo" = "$MODEL" ]; then
        args+=(--revision "$MODEL_REVISION")
    fi
    args+=("$@")
    HF_HOME="$HF_CACHE_DIR" "${HF_BIN_CMD[@]}" download "${args[@]}"
}

download_weights() {
    [ "${SKIP_DOWNLOAD:-0}" = "1" ] && { log "SKIP_DOWNLOAD=1 — skipping download check"; return; }
    if [ "${REFRESH_WEIGHTS:-0}" != "1" ] && adopt_complete_weights; then
        return
    fi

    resolve_hf_bin || die "no 'hf' / 'huggingface-cli' on PATH and no python huggingface_hub — pip install --user -U 'huggingface_hub[cli]' (or set HF_BIN=/path/to/hf)"

    mkdir -p "$HF_CACHE_DIR"
    local -a hf_excl=()
    local pat
    IFS=',' read -ra _excl_pats <<< "${HF_DOWNLOAD_EXCLUDE:-runtime-results/**,src/**,runtime/src/**,scripts/**,docs/**,results/**,.materialization/**,runtime/scripts/**}"
    for pat in "${_excl_pats[@]}"; do
        [ -n "$pat" ] && hf_excl+=(--exclude "$pat")
    done

    log "downloading ${MODEL} (~164 GiB / ${EXPECTED_SHARDS} shards) into ${HF_CACHE_DIR} ..."
    hf_download_repo "$MODEL" "${hf_excl[@]}" || warn "download of ${MODEL} failed — will try ${MODEL_FALLBACK}"
    if adopt_complete_weights; then
        return
    fi

    if [ "$MODEL_FALLBACK" != "$MODEL" ]; then
        log "falling back to ${MODEL_FALLBACK} ..."
        hf_download_repo "$MODEL_FALLBACK" "${hf_excl[@]}" \
            || die "download of ${MODEL} and ${MODEL_FALLBACK} both failed"
    fi
    adopt_complete_weights \
        || die "download finished with $(count_shards "$MODEL_PATH") / $EXPECTED_SHARDS shards"
}

download_dflash() {
    [ "$SPEC_METHOD" = "dflash" ] || return 0
    [ "${SKIP_DOWNLOAD:-0}" = "1" ] && { log "SKIP_DOWNLOAD=1 — skipping DFlash2 download check"; return; }
    local have
    have="$(find "$DFLASH_PATH/snapshots" -name 'model.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]' || true)"
    if [ "${have:-0}" -ge 1 ] && [ "${REFRESH_WEIGHTS:-0}" != "1" ]; then
        log "DFlash2 already present: $DFLASH_PATH"
        ensure_dflash_refs_main
        return
    fi
    resolve_hf_bin || die "no 'hf' / 'huggingface-cli' on PATH and no python huggingface_hub — pip install --user -U 'huggingface_hub[cli]' (or set HF_BIN=/path/to/hf)"
    mkdir -p "$HF_CACHE_DIR"
    log "downloading ${DFLASH_MODEL} (~2.3 GiB) into ${HF_CACHE_DIR} ..."
    HF_HOME="$HF_CACHE_DIR" "${HF_BIN_CMD[@]}" download "$DFLASH_MODEL"
    ensure_dflash_refs_main
    have="$(find "$DFLASH_PATH/snapshots" -name 'model.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]' || true)"
    [ "${have:-0}" -ge 1 ] || die "DFlash2 download finished without model.safetensors"
    log "DFlash2 download complete"
}

# Head-only Hub fetch. No docker, no SSH, no worker rsync.
download_only() {
    local have
    resolve_hf_bin || die "no 'hf' / 'huggingface-cli' on PATH and no python huggingface_hub — pip install --user -U 'huggingface_hub[cli]' (or set HF_BIN=/path/to/hf)"
    mkdir -p "$HF_CACHE_DIR"
    local need_kb=$((180 * 1024 * 1024)) avail
    avail=$(df -Pk "$HF_CACHE_DIR" 2>/dev/null | awk 'NR==2{print $4}' || true)
    [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on this disk for a ~164 GiB model"

    # Explicit download: do not honor SKIP_DOWNLOAD from .env.
    SKIP_DOWNLOAD=0
    download_weights
    download_dflash

    have="$(count_shards "$MODEL_PATH")"
    log "======================================================================"
    log "head HF cache : ${HF_CACHE_DIR}"
    log "  target      : ${MODEL}  (${have} / ${EXPECTED_SHARDS} shards)"
    log "  snapshot    : ${MODEL_PATH}"
    if [ "$SPEC_METHOD" = "dflash" ]; then
        log "  DFlash2     : ${DFLASH_MODEL}"
        log "  draft cache : ${DFLASH_PATH}"
    else
        log "  DFlash2     : skipped (SPEC_METHOD=${SPEC_METHOD})"
    fi
    log "workers were not touched. ./start-tp3.sh will rsync on launch unless SKIP_SYNC=1."
    log "======================================================================"
}

# ------------------------------ weight sync --------------------------------
# Keyed on the snapshot commit (refs/main, with the same repair fallback as
# ensure_refs_main), not on MODEL_REVISION: the marker lives inside each
# synced repo folder, so a MODEL / revision switch re-syncs automatically.
# Without it, every ./start.sh pays a full size+mtime re-verification walk
# over ~164 GiB / 120 shards on both ends for zero bytes of difference
# (issue #22, item 2). FORCE_SYNC=1 bypasses the marker; deleting the
# marker file on the worker has the same effect.
sync_repo_marker_rev() {
    local src="$1"
    local rev
    rev="$(cat "$src/refs/main" 2>/dev/null || true)"
    [ -n "$rev" ] || rev="$(ls -1t "$src/snapshots" 2>/dev/null | head -n 1 || true)"
    [ -n "$rev" ] || rev="unknown"
    printf '%s' "$rev"
}

sync_repo_to_one_worker() {
    local r="$1" src="$2" cache_name="$3" label="$4"
    local marker rev hf ssh_t
    hf="$(_tp3_rank_hf "$r")"
    ssh_t="$(_tp3_ssh_target "$r")"
    marker="${hf}/hub/${cache_name}/.glm53-exl3-synced"
    rev="$(sync_repo_marker_rev "$src")"
    if [ "${FORCE_SYNC:-0}" != "1" ] \
       && [ "$(worker_ssh_n "$r" "cat '$marker' 2>/dev/null" || true)" = "$rev" ]; then
        log "rank ${r} ${cache_name} already at ${rev} — rsync skipped"
        return 0
    fi
    log "syncing ${label} to rank ${r} (${ssh_t}) ..."
    worker_ssh_n "$r" "mkdir -p '${hf}/hub/${cache_name}'"
    rsync -a --partial --info=progress2 \
        "$src/" "${ssh_t}:${hf}/hub/${cache_name}/"
    worker_ssh_n "$r" "printf '%s' '$rev' > '$marker'"
}

sync_weights() {
    [ "${SKIP_SYNC:-0}" = "1" ] && { log "SKIP_SYNC=1 — not syncing to workers"; return; }
    [ -d "$MODEL_PATH" ] || die "weights missing at $MODEL_PATH — run without SKIP_DOWNLOAD first"
    if [ "${NFS_SHARE:-0}" = "1" ]; then
        if [ "$SPEC_METHOD" = "dflash" ] && [ ! -d "$DFLASH_PATH" ]; then
            die "DFlash2 weights missing at $DFLASH_PATH"
        fi
        nfs_share_weights
        return
    fi
    local r
    for r in 1 2; do
        sync_repo_to_one_worker "$r" "$MODEL_PATH" "$MODEL_CACHE_NAME" "weights"
        if [ "$SPEC_METHOD" = "dflash" ]; then
            [ -d "$DFLASH_PATH" ] || die "DFlash2 weights missing at $DFLASH_PATH"
            sync_repo_to_one_worker "$r" "$DFLASH_PATH" "$DFLASH_CACHE_NAME" "DFlash2 draft"
        fi
    done
    log "all worker weights in sync"
}

# ------------------------ inner container scripts --------------------------
write_inner_scripts() {
    cat > "$HEAD_SCRIPT" <<'EOF'
#!/bin/bash
set -euo pipefail
say() { echo "[glm53-exl3-head] $*"; }

ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --nnodes "${NNODES}"
    --node-rank 0
    --master-addr "${HEAD_IP}"
    --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
)
[ "${ENFORCE_EAGER:-1}" = "1" ] && ARGS+=(--enforce-eager)
# TP=3 shape fixes. 64 attention/KV heads and moe_intermediate_size=2048 do
# not divide by 3; pad the head counts and give each rank whole experts.
if [ -n "${TP3_HEAD_OVERRIDE:-}" ]; then
    ARGS+=(--hf-overrides "$(python3 -S -c 'import json,os
n=int(os.environ["TP3_HEAD_OVERRIDE"])
h={"num_attention_heads":n,"num_key_value_heads":n,"linear_num_heads":n}
print(json.dumps({**h,"text_config":dict(h)},separators=(",",":")))')")
fi
[ "${ENABLE_EXPERT_PARALLEL:-1}" = "1" ] && ARGS+=(--enable-expert-parallel)
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ] && ARGS+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
if [ "${SPEC_METHOD:-mtp}" = "dflash" ]; then
    ARGS+=(--speculative-config "$(python3 -S -c 'import json,os
spec={"method":"dflash","model":os.environ["DFLASH_MODEL_DIR"],"num_speculative_tokens":int(os.environ.get("DFLASH_TOKENS","7")),"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}
tp=os.environ.get("DFLASH_DRAFT_TP","").strip()
if tp:
    spec["draft_tensor_parallel_size"]=int(tp)
print(json.dumps(spec,separators=(",",":")))')")
elif [ "${SPEC_METHOD:-mtp}" = "none" ]; then
    :
elif [ "${MTP_TOKENS:-0}" != "0" ]; then
    ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi
if [ -n "${CHAT_TEMPLATE:-}" ] && [ -f "${CHAT_TEMPLATE}" ]; then
    ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    ARGS+=(--language-model-only)
    say "language-model-only: no vision tower"
else
    [ -n "${LIMIT_MM:-}" ] && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
    [ -n "${MM_IMAGE_TOKENS:-}" ] && ARGS+=(--mm-processor-kwargs "{\"max_image_tokens\":${MM_IMAGE_TOKENS}}")
    [ -n "${VIDEO_NUM_FRAMES:-}" ] && ARGS+=(--media-io-kwargs "{\"video\":{\"num_frames\":${VIDEO_NUM_FRAMES}}}")
    [ -n "${MM_PROCESSOR_CACHE_GB:-}" ] && ARGS+=(--mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB}")
    [ -n "${MM_ENCODER_TP_MODE:-}" ] && ARGS+=(--mm-encoder-tp-mode "${MM_ENCODER_TP_MODE}")
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
    say "vision on: limit-mm=${LIMIT_MM:-} image-tokens=${MM_IMAGE_TOKENS:-8000} video-frames=${VIDEO_NUM_FRAMES:-32} mm-cache-gb=${MM_PROCESSOR_CACHE_GB:-4} mm-encoder-tp=${MM_ENCODER_TP_MODE:-weights} skip-mm-profiling=${SKIP_MM_PROFILING:-1} chat-template=${CHAT_TEMPLATE:-}"
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
if [ -f /opt/glm53/patch_glm_video_placeholders.py ]; then
    python3 /opt/glm53/patch_glm_video_placeholders.py
fi
if [ -f /opt/glm53/patch_suppress_stops_in_reasoning.py ]; then
    python3 /opt/glm53/patch_suppress_stops_in_reasoning.py
fi
if [ -f /opt/glm53/patch_scheduler_decode_floor.py ]; then
    python3 /opt/glm53/patch_scheduler_decode_floor.py
fi
if [ -f /opt/glm53/patch_glm5_drafter_group.py ]; then
    python3 /opt/glm53/patch_glm5_drafter_group.py
fi
if [ -f /opt/glm53/patch_hybrid_prefix_hit.py ]; then
    python3 /opt/glm53/patch_hybrid_prefix_hit.py
fi
if [ -f /opt/glm53/patch_xgrammar_termination.py ]; then
    python3 /opt/glm53/patch_xgrammar_termination.py
fi
if [ -f /opt/glm53/patch_kpool_tail_slotmap.py ]; then
    python3 /opt/glm53/patch_kpool_tail_slotmap.py
fi
if [ -f /opt/glm53/patch_tp3_glm.py ]; then
    # Rewrites glm5next/nvidia/model.py: head 64->66, vocab padding_size
    # lcm(64,tp), shared-expert I pad / disable_tp, A_log load pad.
    python3 /opt/glm53/patch_tp3_glm.py
fi
if [ -f /opt/glm53/patch_adaptive_k.py ]; then
    python3 /opt/glm53/patch_adaptive_k.py
fi
if [ -f /opt/glm53/patch_dense_fp8.py ]; then
    python3 /opt/glm53/patch_dense_fp8.py
fi
# AFTER patch_dense_fp8: it reinstalls exl3.py from /opt/glm53, which would
# otherwise wipe both EP fixes below.
if [ -f /opt/glm53/patch_exl3_ep_shard.py ]; then
    # EXL3 MoE loader: whole experts under EP, no intra-expert TP slicing.
    python3 /opt/glm53/patch_exl3_ep_shard.py
fi
if [ -f /opt/glm53/patch_exl3_expert_map.py ]; then
    # expert_map is a read-only property under EP; cache the pinned copy.
    python3 /opt/glm53/patch_exl3_expert_map.py
fi
if [ -f /opt/glm53/patch_spinwait.py ]; then
    python3 /opt/glm53/patch_spinwait.py
fi
if [ -f /opt/glm53/patch_indexer_workspace.py ]; then
    python3 /opt/glm53/patch_indexer_workspace.py
fi
if [ -f /opt/glm53/patch_ablit.py ]; then
    python3 /opt/glm53/patch_ablit.py
fi
if [ "${ABLIT:-0}" = "1" ]; then
    say "ablit: o_proj orthogonalization ON (method=${ABLIT_METHOD:-auto} direction=${ABLIT_DIRECTION:-dealign} layers=${ABLIT_LAYERS:-15-45} alpha=${ABLIT_ALPHA:-3.0})"
else
    say "ablit: off — stock o_proj weights"
fi
say "launching: vllm serve ${MODEL_DIR} ${ARGS[*]}"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
EOF

    cat > "$WORKER_SCRIPT" <<'EOF'
#!/bin/bash
set -euo pipefail
say() { echo "[glm53-exl3-worker] $*"; }

ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --nnodes "${NNODES}"
    --node-rank "${NODE_RANK}"
    --master-addr "${HEAD_IP}"
    --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
    --headless
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
)
[ "${ENFORCE_EAGER:-1}" = "1" ] && ARGS+=(--enforce-eager)
# TP=3 shape fixes. 64 attention/KV heads and moe_intermediate_size=2048 do
# not divide by 3; pad the head counts and give each rank whole experts.
if [ -n "${TP3_HEAD_OVERRIDE:-}" ]; then
    ARGS+=(--hf-overrides "$(python3 -S -c 'import json,os
n=int(os.environ["TP3_HEAD_OVERRIDE"])
h={"num_attention_heads":n,"num_key_value_heads":n,"linear_num_heads":n}
print(json.dumps({**h,"text_config":dict(h)},separators=(",",":")))')")
fi
[ "${ENABLE_EXPERT_PARALLEL:-1}" = "1" ] && ARGS+=(--enable-expert-parallel)
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ] && ARGS+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
if [ "${SPEC_METHOD:-mtp}" = "dflash" ]; then
    ARGS+=(--speculative-config "$(python3 -S -c 'import json,os
spec={"method":"dflash","model":os.environ["DFLASH_MODEL_DIR"],"num_speculative_tokens":int(os.environ.get("DFLASH_TOKENS","7")),"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}
tp=os.environ.get("DFLASH_DRAFT_TP","").strip()
if tp:
    spec["draft_tensor_parallel_size"]=int(tp)
print(json.dumps(spec,separators=(",",":")))')")
elif [ "${SPEC_METHOD:-mtp}" = "none" ]; then
    :
elif [ "${MTP_TOKENS:-0}" != "0" ]; then
    ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi
if [ -n "${CHAT_TEMPLATE:-}" ] && [ -f "${CHAT_TEMPLATE}" ]; then
    ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    ARGS+=(--language-model-only)
else
    [ -n "${LIMIT_MM:-}" ] && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
    [ -n "${MM_IMAGE_TOKENS:-}" ] && ARGS+=(--mm-processor-kwargs "{\"max_image_tokens\":${MM_IMAGE_TOKENS}}")
    [ -n "${VIDEO_NUM_FRAMES:-}" ] && ARGS+=(--media-io-kwargs "{\"video\":{\"num_frames\":${VIDEO_NUM_FRAMES}}}")
    [ -n "${MM_PROCESSOR_CACHE_GB:-}" ] && ARGS+=(--mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB}")
    [ -n "${MM_ENCODER_TP_MODE:-}" ] && ARGS+=(--mm-encoder-tp-mode "${MM_ENCODER_TP_MODE}")
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
if [ -f /opt/glm53/patch_glm_video_placeholders.py ]; then
    python3 /opt/glm53/patch_glm_video_placeholders.py
fi
if [ -f /opt/glm53/patch_suppress_stops_in_reasoning.py ]; then
    python3 /opt/glm53/patch_suppress_stops_in_reasoning.py
fi
if [ -f /opt/glm53/patch_scheduler_decode_floor.py ]; then
    python3 /opt/glm53/patch_scheduler_decode_floor.py
fi
if [ -f /opt/glm53/patch_glm5_drafter_group.py ]; then
    python3 /opt/glm53/patch_glm5_drafter_group.py
fi
if [ -f /opt/glm53/patch_hybrid_prefix_hit.py ]; then
    python3 /opt/glm53/patch_hybrid_prefix_hit.py
fi
if [ -f /opt/glm53/patch_xgrammar_termination.py ]; then
    python3 /opt/glm53/patch_xgrammar_termination.py
fi
if [ -f /opt/glm53/patch_kpool_tail_slotmap.py ]; then
    python3 /opt/glm53/patch_kpool_tail_slotmap.py
fi
if [ -f /opt/glm53/patch_tp3_glm.py ]; then
    # Rewrites glm5next/nvidia/model.py: head 64->66, vocab padding_size
    # lcm(64,tp), shared-expert I pad / disable_tp, A_log load pad.
    python3 /opt/glm53/patch_tp3_glm.py
fi
if [ -f /opt/glm53/patch_adaptive_k.py ]; then
    python3 /opt/glm53/patch_adaptive_k.py
fi
if [ -f /opt/glm53/patch_dense_fp8.py ]; then
    python3 /opt/glm53/patch_dense_fp8.py
fi
# AFTER patch_dense_fp8: it reinstalls exl3.py from /opt/glm53, which would
# otherwise wipe both EP fixes below.
if [ -f /opt/glm53/patch_exl3_ep_shard.py ]; then
    # EXL3 MoE loader: whole experts under EP, no intra-expert TP slicing.
    python3 /opt/glm53/patch_exl3_ep_shard.py
fi
if [ -f /opt/glm53/patch_exl3_expert_map.py ]; then
    # expert_map is a read-only property under EP; cache the pinned copy.
    python3 /opt/glm53/patch_exl3_expert_map.py
fi
if [ -f /opt/glm53/patch_spinwait.py ]; then
    python3 /opt/glm53/patch_spinwait.py
fi
if [ -f /opt/glm53/patch_indexer_workspace.py ]; then
    python3 /opt/glm53/patch_indexer_workspace.py
fi
if [ -f /opt/glm53/patch_ablit.py ]; then
    python3 /opt/glm53/patch_ablit.py
fi
if [ "${ABLIT:-0}" = "1" ]; then
    say "ablit: o_proj orthogonalization ON (method=${ABLIT_METHOD:-auto} direction=${ABLIT_DIRECTION:-dealign} layers=${ABLIT_LAYERS:-15-45} alpha=${ABLIT_ALPHA:-3.0})"
else
    say "ablit: off — stock o_proj weights"
fi
say "joining TP${TP} at ${HEAD_IP}:${MASTER_PORT} as rank ${NODE_RANK}"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
EOF
    chmod +x "$HEAD_SCRIPT" "$WORKER_SCRIPT"
}

# ------------------------------- launch ------------------------------------
_tp3_scp_runtime() {
    local r="$1" ssh_t cname
    ssh_t="$(_tp3_ssh_target "$r")"
    cname="$(_tp3_rank_container "$r")"
    scp -q -o BatchMode=yes "$WORKER_SCRIPT" "${ssh_t}:/tmp/${cname}.sh"
    scp -q -o BatchMode=yes "$CHAT_TEMPLATE_HOST" "${ssh_t}:/tmp/glm53-chat_template.jinja"
    scp -q -o BatchMode=yes "$VIDEO_PATCH_HOST" "${ssh_t}:/tmp/patch_glm_video_placeholders.py"
    scp -q -o BatchMode=yes "$STOP_PATCH_HOST" "${ssh_t}:/tmp/patch_suppress_stops_in_reasoning.py"
    scp -q -o BatchMode=yes "$SCHED_PATCH_HOST" "${ssh_t}:/tmp/patch_scheduler_decode_floor.py"
    scp -q -o BatchMode=yes "$DRAFTER_PATCH_HOST" "${ssh_t}:/tmp/patch_glm5_drafter_group.py"
    scp -q -o BatchMode=yes "$APC_PATCH_HOST" "${ssh_t}:/tmp/patch_hybrid_prefix_hit.py"
    scp -q -o BatchMode=yes "$XGRAMMAR_PATCH_HOST" "${ssh_t}:/tmp/patch_xgrammar_termination.py"
    scp -q -o BatchMode=yes "$KPOOL_TAIL_PATCH_HOST" "${ssh_t}:/tmp/patch_kpool_tail_slotmap.py"
    scp -q -o BatchMode=yes "$SPINWAIT_PATCH_HOST" "${ssh_t}:/tmp/patch_spinwait.py"
    scp -q -o BatchMode=yes "$EXL3_OVERLAY_HOST" "${ssh_t}:/tmp/glm53-exl3.py"
    scp -q -o BatchMode=yes "$ADAPTIVE_K_PATCH_HOST" "${ssh_t}:/tmp/patch_adaptive_k.py"
    scp -q -o BatchMode=yes "$DENSE_FP8_PATCH_HOST" "${ssh_t}:/tmp/patch_dense_fp8.py"
    if [ -d "$TP3_OVERLAY_HOST" ]; then
        ssh -o BatchMode=yes "$ssh_t" "rm -rf /tmp/glm53-tp3"
        scp -q -r -o BatchMode=yes "$TP3_OVERLAY_HOST" "${ssh_t}:/tmp/glm53-tp3"
    fi
    worker_ssh_n "$r" "rm -rf /tmp/glm53-ablit"
    scp -q -r -o BatchMode=yes "$SCRIPT_DIR/ablit" "${ssh_t}:/tmp/glm53-ablit"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/ablit_runtime.py" "${ssh_t}:/tmp/glm53-ablit_runtime.py"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/patch_ablit.py" "${ssh_t}:/tmp/patch_ablit.py"
}

launch_cluster() {
    local r
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || true
    for r in 1 2; do
        worker_ssh_n "$r" "docker rm -f '$(_tp3_rank_container "$r")'" >/dev/null 2>&1 || true
    done

    mkdir -p "$CACHE_ROOT" "$TRITON_HOST_CACHE" "$TILELANG_HOST_CACHE"
    [ -f "$CHAT_TEMPLATE_HOST" ] || die "missing chat template: $CHAT_TEMPLATE_HOST"
    for r in 1 2; do
        worker_ssh_n "$r" "mkdir -p '$(_tp3_rank_vllm "$r")' '$(_tp3_rank_triton "$r")' '$(_tp3_rank_tilelang "$r")'"
        _tp3_scp_runtime "$r"
    done
    # skip the old single-worker scp block
    true
    : <<'TP3_SKIP_OLD_SCP'
    [ -f "$CHAT_TEMPLATE_HOST" ] || die "missing chat template: $CHAT_TEMPLATE_HOST"
    scp -q -o BatchMode=yes "$CHAT_TEMPLATE_HOST" "${WORKER_SSH}:/tmp/glm53-chat_template.jinja"
    [ -f "$VIDEO_PATCH_HOST" ] || die "missing $VIDEO_PATCH_HOST"
    scp -q -o BatchMode=yes "$VIDEO_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_glm_video_placeholders.py"
    [ -f "$STOP_PATCH_HOST" ] || die "missing $STOP_PATCH_HOST"
    scp -q -o BatchMode=yes "$STOP_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_suppress_stops_in_reasoning.py"
    [ -f "$SCHED_PATCH_HOST" ] || die "missing $SCHED_PATCH_HOST"
    scp -q -o BatchMode=yes "$SCHED_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_scheduler_decode_floor.py"
    [ -f "$DRAFTER_PATCH_HOST" ] || die "missing $DRAFTER_PATCH_HOST"
    scp -q -o BatchMode=yes "$DRAFTER_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_glm5_drafter_group.py"
    [ -f "$APC_PATCH_HOST" ] || die "missing $APC_PATCH_HOST"
    scp -q -o BatchMode=yes "$APC_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_hybrid_prefix_hit.py"
    [ -f "$XGRAMMAR_PATCH_HOST" ] || die "missing $XGRAMMAR_PATCH_HOST"
    scp -q -o BatchMode=yes "$XGRAMMAR_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_xgrammar_termination.py"
    [ -f "$KPOOL_TAIL_PATCH_HOST" ] || die "missing $KPOOL_TAIL_PATCH_HOST"
    scp -q -o BatchMode=yes "$KPOOL_TAIL_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_kpool_tail_slotmap.py"
    [ -f "$SPINWAIT_PATCH_HOST" ] || die "missing $SPINWAIT_PATCH_HOST"
    scp -q -o BatchMode=yes "$SPINWAIT_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_spinwait.py"
    if [ -d "$TP3_OVERLAY_HOST" ]; then
        worker_ssh "rm -rf /tmp/glm53-tp3"
        scp -q -r -o BatchMode=yes "$TP3_OVERLAY_HOST" "${WORKER_SSH}:/tmp/glm53-tp3"
    fi

    worker_ssh "rm -rf /tmp/glm53-ablit"
    scp -q -r -o BatchMode=yes "$SCRIPT_DIR/ablit" "${WORKER_SSH}:/tmp/glm53-ablit"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/ablit_runtime.py" "${WORKER_SSH}:/tmp/glm53-ablit_runtime.py"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/patch_ablit.py" "${WORKER_SSH}:/tmp/patch_ablit.py"
TP3_SKIP_OLD_SCP

    local -a nccl_common=(
        -e NCCL_IB_DISABLE=0
        -e NCCL_IB_ROCE_VERSION_NUM=2
        -e NCCL_NET=IB
        -e NCCL_NET_PLUGIN=none
        -e NCCL_NVLS_ENABLE=0
        -e NCCL_CUMEM_ENABLE=0
        -e NCCL_IB_MERGE_NICS=0
        -e "NCCL_CROSS_NIC=$NCCL_CROSS_NIC"
        -e "NCCL_IB_SUBNET_AWARE_ROUTING=$NCCL_IB_SUBNET_AWARE_ROUTING"
        -e "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
        -e "NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
        -e "NCCL_BUFFSIZE=$NCCL_BUFFSIZE"
        -e "NCCL_LL128_BUFFSIZE=$NCCL_LL128_BUFFSIZE"
        -e "NCCL_PROTO=$NCCL_PROTO"
        -e "NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS"
        -e NCCL_IGNORE_CPU_AFFINITY=1
        -e "NCCL_DEBUG=$NCCL_DEBUG"
        -e HF_HUB_OFFLINE=1
        -e TRANSFORMERS_OFFLINE=1
        -e HF_HOME=/root/.cache/huggingface
        -e VLLM_CACHE_ROOT=/root/.cache/vllm
        -e "GLM53_SUPPRESS_STOPS_IN_REASONING=$GLM53_SUPPRESS_STOPS_IN_REASONING"
        -e "GLM53_MIXED_PREFILL_CHUNK=$GLM53_MIXED_PREFILL_CHUNK"
        -e "GLM53_FAIR_PREFILL_CHUNK=$GLM53_FAIR_PREFILL_CHUNK"
        -e "GLM53_FAIR_PREFILL_SHARE=$GLM53_FAIR_PREFILL_SHARE"
        -e "GLM53_FAIR_PREFILL_MAX_INTERVAL_MS=$GLM53_FAIR_PREFILL_MAX_INTERVAL_MS"
        -e "GLM53_FAIR_PREFILL_MAX_STEP_MS=$GLM53_FAIR_PREFILL_MAX_STEP_MS"
        -e "GLM53_FAIR_PREFILL_MAX_CHUNKS=$GLM53_FAIR_PREFILL_MAX_CHUNKS"
        -e "GLM53_INDEXER_WORKSPACE=$GLM53_INDEXER_WORKSPACE"
        -e "GLM53_SPINWAIT_MS=$GLM53_SPINWAIT_MS"
        -e "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
        -e "TILELANG_CACHE_DIR=$TILELANG_CACHE_DIR"
        -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
        -e "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
        -e "FLASHINFER_CUDA_ARCH_LIST=$FLASHINFER_CUDA_ARCH_LIST"
        -e FLASHINFER_DISABLE_VERSION_CHECK=1
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        -e "VLLM_ENGINE_READY_TIMEOUT_S=$READY_TIMEOUT"
        # py-cpuinfo JSON-parses empty output on Grace/aarch64; the usage
        # thread then dumps JSONDecodeError. Stats are off on this private kit.
        -e VLLM_NO_USAGE_STATS=1
        -e DO_NOT_TRACK=1
        -e PYTHONFAULTHANDLER=1
        -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=$CG_ESTIMATE"
    )
    local worker_nccl="" e
    for e in "${nccl_common[@]}"; do
        [ "$e" = "-e" ] && continue
        worker_nccl+=" -e $e"
    done

    local -a head_preload=()
    if [ "$USE_HOST_NCCL" = "1" ]; then
        if [ -f "$NCCL_HOST_DIR/$NCCL_SO_NAME" ]; then
            head_preload=(-v "$NCCL_HOST_DIR:/nccl:ro" -e "LD_PRELOAD=/nccl/$NCCL_SO_NAME")
            log "head: LD_PRELOAD $NCCL_SO_NAME"
        else
            warn "head: $NCCL_HOST_DIR/$NCCL_SO_NAME missing — using image NCCL"
        fi
    fi

    local serve_env=""
    local v
    for v in SERVED_MODEL_NAME PORT TP NNODES HEAD_IP MASTER_PORT QUANTIZATION \
             MAX_MODEL_LEN GPU_MEM_UTIL MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
             LONG_PREFILL_TOKEN_THRESHOLD \
             KV_CACHE_DTYPE MTP_TOKENS SPEC_METHOD DFLASH_TOKENS DFLASH_MODEL_DIR \
             DFLASH_DRAFT_TP DFLASH_REVISION \
             SOCKET_IFNAME HEAD_SOCKET_IFNAME WORKER_SOCKET_IFNAME WORKER2_SOCKET_IFNAME \
             HEAD_HOST_IP WORKER_HOST_IP WORKER2_HOST_IP NCCL_IB_SUBNET_AWARE_ROUTING \
             NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_BUFFSIZE NCCL_LL128_BUFFSIZE NCCL_PROTO NCCL_MAX_NCHANNELS \
             LANGUAGE_MODEL_ONLY SKIP_MM_PROFILING \
             MM_IMAGE_TOKENS VIDEO_NUM_FRAMES MM_PROCESSOR_CACHE_GB MM_ENCODER_TP_MODE \
             TP3_HEAD_OVERRIDE ENABLE_EXPERT_PARALLEL \
             LIMIT_MM CHAT_TEMPLATE ENFORCE_EAGER EXL3_FUSED_MOE EXL3_MOE_ROW_TILE EXL3_TEMP_ROWS_FUSED EXL3_FAT_SORTED EXL3_FAT_BATCHED EXL3_FAT_KERNEL EXL3_FAT_GROUPED MODEL_DIR EXTRA_ARGS \
             ABLIT ABLIT_METHOD ABLIT_DIRECTION ABLIT_LAYERS ABLIT_ALPHA ABLIT_INCLUDE_MTP \
             GLM53_ADAPTIVE_K GLM53_ADAPTIVE_K_SET GLM53_ADAPTIVE_K_ALPHA GLM53_ADAPTIVE_K_MARGIN \
             GLM53_ADAPTIVE_K_MIN_STEPS GLM53_ADAPTIVE_K_SATURATE GLM53_ADAPTIVE_K_HIST GLM53_DENSE_FP8; do
        serve_env+=" -e $v='${!v:-}'"
    done
    # VLLM_API_KEY is read by the head (rank 0) API server for bearer auth; the
    # worker runs --headless so it only needs the var for argv-parity, and
    # start.sh below passes it explicitly on the head. Keep it out of the
    # generic loop so the key never shows in process listings of either node
    # beyond the container env (same as the DeepSeek deployment).
    serve_env+=" -e VLLM_API_KEY='${VLLM_API_KEY:-}'"

    local worker_preload="" nccl_dir cname
    for r in 1 2; do
        worker_preload=""
        if [ "$USE_HOST_NCCL" = "1" ]; then
            nccl_dir="$(_tp3_rank_nccl_dir "$r")"
            if worker_ssh_n "$r" "test -f '$nccl_dir/$NCCL_SO_NAME'"; then
                worker_preload="-v '$nccl_dir:/nccl:ro' -e LD_PRELOAD='/nccl/$NCCL_SO_NAME'"
                log "rank ${r}: LD_PRELOAD $NCCL_SO_NAME"
            else
                warn "rank ${r}: $nccl_dir/$NCCL_SO_NAME missing — using image NCCL"
            fi
        fi
        cname="$(_tp3_rank_container "$r")"
        log "starting rank ${r} on $(_tp3_ssh_target "$r") (NCCL if=$(_tp3_rank_cx7_if "$r") hca=$(_tp3_rank_cx7_ib "$r")) ..."
        worker_ssh_n "$r" "docker run -d --name '$cname' \
            --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
            --device /dev/infiniband --cap-add IPC_LOCK \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v '$(_tp3_hf_mount "$r")' \
            -v '$(_tp3_rank_vllm "$r"):/root/.cache/vllm' \
            -v '$(_tp3_rank_triton "$r"):/root/.triton/cache' \
            -v '$(_tp3_rank_tilelang "$r"):/root/.tilelang/cache' \
            -v '/tmp/${cname}.sh:/start.sh:ro' \
            -v '/tmp/glm53-chat_template.jinja:${CHAT_TEMPLATE}:ro' \
            -v '/tmp/patch_glm_video_placeholders.py:/opt/glm53/patch_glm_video_placeholders.py:ro' \
            -v '/tmp/patch_suppress_stops_in_reasoning.py:/opt/glm53/patch_suppress_stops_in_reasoning.py:ro' \
            -v '/tmp/patch_scheduler_decode_floor.py:/opt/glm53/patch_scheduler_decode_floor.py:ro' \
            -v '/tmp/patch_glm5_drafter_group.py:/opt/glm53/patch_glm5_drafter_group.py:ro' \
            -v '/tmp/patch_hybrid_prefix_hit.py:/opt/glm53/patch_hybrid_prefix_hit.py:ro' \
            -v '/tmp/patch_xgrammar_termination.py:/opt/glm53/patch_xgrammar_termination.py:ro' \
            -v '/tmp/patch_kpool_tail_slotmap.py:/opt/glm53/patch_kpool_tail_slotmap.py:ro' \
            -v '/tmp/patch_spinwait.py:/opt/glm53/patch_spinwait.py:ro' \
            -v '/tmp/glm53-exl3.py:/opt/glm53/exl3.py:ro' \
            -v '/tmp/patch_adaptive_k.py:/opt/glm53/patch_adaptive_k.py:ro' \
            -v '/tmp/patch_dense_fp8.py:/opt/glm53/patch_dense_fp8.py:ro' \
            -v '/tmp/glm53-tp3/patch_tp3_glm.py:/opt/glm53/patch_tp3_glm.py:ro' \
            -v '/tmp/glm53-tp3/patch_exl3_ep_shard.py:/opt/glm53/patch_exl3_ep_shard.py:ro' \
            -v '/tmp/glm53-tp3/patch_exl3_expert_map.py:/opt/glm53/patch_exl3_expert_map.py:ro' \
            -v '/tmp/glm53-tp3/vllm/model_executor/parameter.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/parameter.py:ro' \
            -v '/tmp/glm53-tp3/vllm/model_executor/model_loader/weight_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py:ro' \
            -v '/tmp/glm53-tp3/vllm/model_executor/layers/vocab_parallel_embedding.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/vocab_parallel_embedding.py:ro' \
            -v '/tmp/glm53-tp3/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:ro' \
            -v '/tmp/glm53-ablit:/opt/glm53/ablit:ro' \
            -v '/tmp/glm53-ablit_runtime.py:/opt/glm53/ablit_runtime.py:ro' \
            -v '/tmp/patch_ablit.py:/opt/glm53/patch_ablit.py:ro' \
            ${worker_preload} \
            ${worker_nccl} \
            -e NCCL_SOCKET_IFNAME='$(_tp3_rank_socket_if "$r")' \
            -e GLOO_SOCKET_IFNAME='$(_tp3_rank_socket_if "$r")' \
            -e NCCL_IB_HCA='$(_tp3_rank_cx7_ib "$r")' \
            -e NCCL_IB_GID_INDEX='$(_tp3_rank_gid "$r")' \
            -e VLLM_HOST_IP='$(_tp3_rank_host_ip "$r")' \
            -e NODE_RANK='$r' \
            ${serve_env} \
            --entrypoint bash '$IMAGE' /start.sh" >/dev/null
    done

    log "starting head (vLLM API :${PORT}; NCCL if=${HEAD_CX7_IF} hca=${HEAD_CX7_IB}) ..."
    docker run -d --name "$CONTAINER_HEAD" \
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
        -v "$CACHE_ROOT:/root/.cache/vllm" \
        -v "$TRITON_HOST_CACHE:/root/.triton/cache" \
        -v "$TILELANG_HOST_CACHE:/root/.tilelang/cache" \
        -v "$HEAD_SCRIPT:/start.sh:ro" \
        -v "$CHAT_TEMPLATE_HOST:$CHAT_TEMPLATE:ro" \
        -v "$VIDEO_PATCH_HOST:/opt/glm53/patch_glm_video_placeholders.py:ro" \
        -v "$STOP_PATCH_HOST:/opt/glm53/patch_suppress_stops_in_reasoning.py:ro" \
        -v "$SCHED_PATCH_HOST:/opt/glm53/patch_scheduler_decode_floor.py:ro" \
        -v "$DRAFTER_PATCH_HOST:/opt/glm53/patch_glm5_drafter_group.py:ro" \
        -v "$APC_PATCH_HOST:/opt/glm53/patch_hybrid_prefix_hit.py:ro" \
        -v "$XGRAMMAR_PATCH_HOST:/opt/glm53/patch_xgrammar_termination.py:ro" \
        -v "$KPOOL_TAIL_PATCH_HOST:/opt/glm53/patch_kpool_tail_slotmap.py:ro" \
        -v "$SPINWAIT_PATCH_HOST:/opt/glm53/patch_spinwait.py:ro" \
        -v "$EXL3_OVERLAY_HOST:/opt/glm53/exl3.py:ro" \
        -v "$ADAPTIVE_K_PATCH_HOST:/opt/glm53/patch_adaptive_k.py:ro" \
        -v "$DENSE_FP8_PATCH_HOST:/opt/glm53/patch_dense_fp8.py:ro" \
        -v "$TP3_OVERLAY_HOST/patch_tp3_glm.py:/opt/glm53/patch_tp3_glm.py:ro" \
        -v "$TP3_OVERLAY_HOST/patch_exl3_ep_shard.py:/opt/glm53/patch_exl3_ep_shard.py:ro" \
        -v "$TP3_OVERLAY_HOST/patch_exl3_expert_map.py:/opt/glm53/patch_exl3_expert_map.py:ro" \
        -v "$TP3_OVERLAY_HOST/vllm/model_executor/parameter.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/parameter.py:ro" \
        -v "$TP3_OVERLAY_HOST/vllm/model_executor/model_loader/weight_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py:ro" \
        -v "$TP3_OVERLAY_HOST/vllm/model_executor/layers/vocab_parallel_embedding.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/vocab_parallel_embedding.py:ro" \
        -v "$TP3_OVERLAY_HOST/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:ro" \
        -v "$SCRIPT_DIR/ablit:/opt/glm53/ablit:ro" \
        -v "$SCRIPT_DIR/overlay/ablit_runtime.py:/opt/glm53/ablit_runtime.py:ro" \
        -v "$SCRIPT_DIR/overlay/patch_ablit.py:/opt/glm53/patch_ablit.py:ro" \
        "${head_preload[@]}" \
        "${nccl_common[@]}" \
        -e NCCL_SOCKET_IFNAME="${HEAD_SOCKET_IFNAME:-$HEAD_CX7_IF}" \
        -e GLOO_SOCKET_IFNAME="${HEAD_SOCKET_IFNAME:-$HEAD_CX7_IF}" \
        -e NCCL_IB_HCA="$HEAD_CX7_IB" \
        -e NCCL_IB_GID_INDEX="$HEAD_GID" \
        -e VLLM_HOST_IP="${HEAD_HOST_IP:-$HEAD_IP}" \
        -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
        -e PORT="$PORT" -e TP="$TP" -e NNODES="$NNODES" \
        -e HEAD_IP="$HEAD_IP" -e MASTER_PORT="$MASTER_PORT" \
        -e QUANTIZATION="$QUANTIZATION" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
        -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
        -e LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-}" \
        -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" -e MTP_TOKENS="$MTP_TOKENS" \
        -e SPEC_METHOD="$SPEC_METHOD" \
        -e DFLASH_TOKENS="${DFLASH_TOKENS:-7}" \
        -e DFLASH_MODEL_DIR="${DFLASH_MODEL_DIR:-}" \
        -e DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP:-}" \
        -e LANGUAGE_MODEL_ONLY="$LANGUAGE_MODEL_ONLY" \
        -e SKIP_MM_PROFILING="$SKIP_MM_PROFILING" \
        -e LIMIT_MM="$LIMIT_MM" \
        -e GLM53_ADAPTIVE_K="$GLM53_ADAPTIVE_K" \
        -e GLM53_ADAPTIVE_K_SET="$GLM53_ADAPTIVE_K_SET" \
        -e GLM53_ADAPTIVE_K_ALPHA="$GLM53_ADAPTIVE_K_ALPHA" \
        -e GLM53_ADAPTIVE_K_MARGIN="$GLM53_ADAPTIVE_K_MARGIN" \
        -e GLM53_ADAPTIVE_K_MIN_STEPS="$GLM53_ADAPTIVE_K_MIN_STEPS" \
        -e GLM53_ADAPTIVE_K_SATURATE="$GLM53_ADAPTIVE_K_SATURATE" \
        -e GLM53_ADAPTIVE_K_HIST="$GLM53_ADAPTIVE_K_HIST" \
        -e GLM53_DENSE_FP8="$GLM53_DENSE_FP8" \
        -e MM_IMAGE_TOKENS="${MM_IMAGE_TOKENS:-}" \
        -e VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:-}" \
        -e MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-}" \
        -e MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-}" \
        -e TP3_HEAD_OVERRIDE="${TP3_HEAD_OVERRIDE:-}" \
        -e ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-}" \
        -e CHAT_TEMPLATE="$CHAT_TEMPLATE" \
        -e ENFORCE_EAGER="$ENFORCE_EAGER" \
        -e EXL3_FUSED_MOE="$EXL3_FUSED_MOE" \
        -e EXL3_MOE_ROW_TILE="$EXL3_MOE_ROW_TILE" \
        -e EXL3_TEMP_ROWS_FUSED="$EXL3_TEMP_ROWS_FUSED" \
        -e EXL3_FAT_SORTED="$EXL3_FAT_SORTED" \
        -e EXL3_FAT_BATCHED="$EXL3_FAT_BATCHED" \
        -e EXL3_FAT_KERNEL="$EXL3_FAT_KERNEL" \
        -e EXL3_FAT_GROUPED="$EXL3_FAT_GROUPED" \
        -e ABLIT="$ABLIT" \
        -e ABLIT_METHOD="$ABLIT_METHOD" \
        -e ABLIT_DIRECTION="$ABLIT_DIRECTION" \
        -e ABLIT_LAYERS="$ABLIT_LAYERS" \
        -e ABLIT_ALPHA="$ABLIT_ALPHA" \
        -e ABLIT_INCLUDE_MTP="$ABLIT_INCLUDE_MTP" \
        -e MODEL_DIR="$MODEL_DIR" \
        -e VLLM_API_KEY="$VLLM_API_KEY" \
        -e EXTRA_ARGS="${EXTRA_ARGS:-}" \
        --entrypoint bash "$IMAGE" /start.sh >/dev/null

}

# ---------------------------- health wait ----------------------------------
wait_for_health() {
    local url="http://127.0.0.1:${PORT}/health"
    log "waiting for ${url} (weight load + warmup on a 320B MoE is slow; timeout ${READY_TIMEOUT}s) ..."
    log "streaming head logs live — Ctrl-C detaches, the server keeps running"

    local logpid=""
    _stop_logtail() {
        [ -n "$logpid" ] && kill "$logpid" 2>/dev/null || true
        wait "$logpid" 2>/dev/null || true
        logpid=""
    }
    trap '_stop_logtail; warn "interrupted — containers keep running ('"'"'./start-tp3.sh logs'"'"' / '"'"'./start-tp3.sh stop'"'"')"; exit 130' INT
    docker logs -f --tail 0 "$CONTAINER_HEAD" 2>&1 &
    logpid=$!

    local elapsed=0 healthy=0 exited=0 dead_side="" worker_fail=0
    while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
        if curl -fsS -m 5 "$url" >/dev/null 2>&1; then healthy=1; break; fi
        if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
            log "head container exited during startup"
            exited=1; dead_side="head"; break
        fi
        # A dead worker rank can never make the head healthy — fail fast with
        # the log dump instead of polling for the full READY_TIMEOUT (issue
        # #22, item 4). Transient ssh/docker hiccups are tolerated; only
        # three consecutive non-running answers (~30 s) count as a dead
        # worker.
        local all_up=1 wr
        for wr in 1 2; do
            if worker_ssh_n "$wr" "docker inspect -f '{{.State.Running}}' '$(_tp3_rank_container "$wr")' 2>/dev/null" | grep -q true; then
                :
            else
                all_up=0
                dead_side="rank${wr}"
            fi
        done
        if [ "$all_up" = "1" ]; then
            worker_fail=0
        else
            worker_fail=$((worker_fail + 1))
            if [ "$worker_fail" -ge 3 ]; then
                log "a worker container is not running (${dead_side}, 3 consecutive checks)"
                exited=1; break
            fi
        fi
        sleep 10; elapsed=$((elapsed + 10))
    done

    _stop_logtail
    trap 'warn "interrupted — containers keep running ('"'"'./start-tp3.sh logs'"'"' / '"'"'./start-tp3.sh stop'"'"')"; exit 130' INT

    if [ "$healthy" = "1" ]; then
        log "health check passed after ${elapsed}s — server is up"
    elif [ "$exited" = "1" ]; then
        warn "${dead_side:-head} container exited/stopped after ${elapsed}s"
    else
        warn "timed out after ${elapsed}s without becoming healthy"
    fi
    [ "$healthy" = "1" ]
}

post_ready_warmup() {
    if [ "${GLM53_BOOT_SHAPE_WARMUP:-1}" = "0" ]; then
        log "boot shape warmup skipped (GLM53_BOOT_SHAPE_WARMUP=0)"
        return 0
    fi
    [ -f "$SCRIPT_DIR/scripts/boot-shape-warmup.sh" ] \
        || { warn "boot-shape-warmup.sh missing — skipping"; return 0; }
    log "post-ready DFlash2/sampler warmup (nonfatal; timeout ${GLM53_WARMUP_REQ_TIMEOUT}s/req) ..."
    GLM53_WARMUP_MAX_CONCURRENCY="$MAX_NUM_SEQS" \
    GLM53_WARMUP_REQ_TIMEOUT="$GLM53_WARMUP_REQ_TIMEOUT" \
    GLM53_WARMUP_DFLASH_K="${DFLASH_TOKENS:-7}" \
    GLM53_WARMUP_TRITON_CACHE_DIR="$TRITON_HOST_CACHE" \
    GLM53_WARMUP_BEARER="${VLLM_API_KEY:-}" \
        bash "$SCRIPT_DIR/scripts/boot-shape-warmup.sh" \
            "http://127.0.0.1:${PORT}" "$SERVED_MODEL_NAME" \
        || warn "boot shape warmup incomplete — uncovered shapes may JIT mid-serve on TP=3"
}

collect_failure_logs() {
    mkdir -p "$LOGDIR"
    docker logs "$CONTAINER_HEAD" >"$LOGDIR/head.log" 2>&1 || true
    local r
    for r in 1 2; do
        worker_ssh_n "$r" "docker logs '$(_tp3_rank_container "$r")' 2>&1" >"$LOGDIR/worker${r}.log" 2>&1 || true
    done
}

on_ready() {
    log "======================================================================"
    log "GLM-5.3-Flash EXL3 is UP (TP=${TP}, nnodes=${NNODES})"
    log "  endpoints  : http://127.0.0.1:${PORT}/v1   (LAN: ${HEAD_IP}:${PORT})"
    log "  model name : ${SERVED_MODEL_NAME}"
    log "  weights    : ${MODEL}  quant=${QUANTIZATION}  kv=${KV_CACHE_DTYPE}"
    local vision=on
    [ "${LANGUAGE_MODEL_ONLY}" = "1" ] && vision=off
    local spec="MTP k=${MTP_TOKENS}"
    [ "$SPEC_METHOD" = "dflash" ] && spec="DFlash2 k=${DFLASH_TOKENS} (${DFLASH_MODEL})"
    [ "$SPEC_METHOD" = "none" ] && spec=off
    local ablit="off (stock weights)"
    [ "$ABLIT" = "1" ] && ablit="ON method=${ABLIT_METHOD} direction=${ABLIT_DIRECTION} layers=${ABLIT_LAYERS} alpha=${ABLIT_ALPHA}"
    log "  features   : tools=glm47+auto, reasoning=glm45, spec=${spec}, vision=${vision}, ablit=${ablit}"
    local auth_line="none (VLLM_API_KEY empty)"
    if [ -n "${VLLM_API_KEY:-}" ]; then
        auth_line="bearer token set (VLLM_API_KEY) — send Authorization: Bearer <key> on /v1 requests"
    fi
    log "  auth       : ${auth_line}"
    log "  quick test :"
    log "    curl -s http://127.0.0.1:${PORT}/v1/chat/completions \\"
    if [ -n "${VLLM_API_KEY:-}" ]; then
        log "      -H 'Authorization: Bearer <KEY>' \\"
    fi
    log "      -H 'Content-Type: application/json' \\"
    log "      -d '{\"model\": \"${SERVED_MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"hello!\"}]}'"
    log "  manage     : ./start-tp3.sh status | ./start-tp3.sh logs | ./start-tp3.sh logs 1 | ./start-tp3.sh stop"
    log "======================================================================"
    if [ "${TAIL:-0}" = "1" ]; then
        log "tailing head logs — Ctrl-C just detaches, the server keeps running"
        trap '' INT
        docker logs -f --tail 20 "$CONTAINER_HEAD" || true
        trap 'warn "interrupted — containers keep running"; exit 130' INT
    fi
}

# ------------------------------- start -------------------------------------
start() {
    preflight
    ensure_image
    download_weights
    download_dflash
    sync_weights
    write_inner_scripts

    MODEL_DIR="$(resolve_model_dir)"
    DFLASH_MODEL_DIR=""
    if [ "$SPEC_METHOD" = "dflash" ]; then
        DFLASH_MODEL_DIR="$(resolve_dflash_dir)"
        if [ -f "$TP3_OVERLAY_HOST/pad-tp3-config.py" ]; then
            DFLASH_MODEL_DIR="$(prepare_tp3_draft "$DFLASH_PATH/snapshots/${DFLASH_REVISION:-$(cat "$DFLASH_PATH/refs/main")}")"
            log "TP=3 draft copy (GQA padded): ${DFLASH_MODEL_DIR}"
        fi
        log "DFlash2 load path (in-container): ${DFLASH_MODEL_DIR}"
    fi
    log "model load path (in-container): ${MODEL_DIR}"
    log "config: image=${IMAGE} tp=${TP} nnodes=${NNODES} quant=${QUANTIZATION} spec=${SPEC_METHOD} mtp=${MTP_TOKENS} dflash_k=${DFLASH_TOKENS} max-len=${MAX_MODEL_LEN} gpu-util=${GPU_MEM_UTIL} kv=${KV_CACHE_DTYPE} lm-only=${LANGUAGE_MODEL_ONLY} port=${PORT} adaptive-k=${GLM53_ADAPTIVE_K} set=${GLM53_ADAPTIVE_K_SET} alpha=${GLM53_ADAPTIVE_K_ALPHA} dense_fp8=${GLM53_DENSE_FP8} fat_grouped=${EXL3_FAT_GROUPED} temp_rows=${EXL3_TEMP_ROWS_FUSED}"

    launch_cluster
    if wait_for_health; then
        post_ready_warmup
        on_ready
        return
    fi
    collect_failure_logs
    echo "---- last 60 lines of head log ($LOGDIR/head.log) ----"
    tail -n 60 "$LOGDIR/head.log" || true
    echo "---- last 40 lines of worker1 log ($LOGDIR/worker1.log) ----"
    tail -n 40 "$LOGDIR/worker1.log" || true
    echo "---- last 40 lines of worker2 log ($LOGDIR/worker2.log) ----"
    tail -n 40 "$LOGDIR/worker2.log" || true
    die "server did not become healthy — full logs in $LOGDIR/"
}

# ------------------------------- stop --------------------------------------
stop() {
    local r
    log "stopping head container ..."
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || log "  (no head container was running)"
    for r in 1 2; do
        log "stopping rank ${r} on $(_tp3_ssh_target "$r") ..."
        worker_ssh_n "$r" "docker rm -f '$(_tp3_rank_container "$r")'" >/dev/null 2>&1 \
            || log "  (no rank ${r} container was running)"
    done
    if [ "${NFS_SHARE:-0}" = "1" ]; then
        log "removing the rank NFS volumes (the exporter stays up) ..."
        nfs_unmount_workers
    fi
    log "stopped."
}

# ------------------------------ status -------------------------------------
status() {
    log "head (${CONTAINER_HEAD} on $(hostname)):"
    docker ps -a --filter "name=${CONTAINER_HEAD}" --format '  {{.Names}}  {{.Status}}' || true
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        log "  API: healthy — http://127.0.0.1:${PORT}/v1"
    else
        log "  API: not responding"
    fi
    local r
    for r in 1 2; do
        log "rank ${r} ($(_tp3_rank_container "$r") on $(_tp3_ssh_target "$r")):"
        worker_ssh_n "$r" "docker ps -a --filter name=$(_tp3_rank_container "$r") --format '  {{.Names}}  {{.Status}}'" 2>/dev/null \
            || log "  (rank ${r} unreachable)"
    done
}

# ------------------------------- logs --------------------------------------
logs() {
    case "${1:-head}" in
        1|2|3|worker|worker1)
            local r="${1}"
            if [ "$r" = "worker" ] || [ "$r" = "worker1" ]; then r=1; fi
            log "following rank ${r} logs on $(_tp3_ssh_target "$r") ..."
            trap '' INT
            worker_ssh_n "$r" "docker logs -f --tail 100 '$(_tp3_rank_container "$r")'" || true
            trap 'warn "interrupted"; exit 130' INT
            ;;
        head|*)
            log "following head logs (driver + API server) ..."
            trap '' INT
            docker logs -f --tail 100 "$CONTAINER_HEAD" || true
            trap 'warn "interrupted"; exit 130' INT
            ;;
    esac
}

# ------------------------------- main --------------------------------------
main() {
    local cmd="${1:-start}"
    case "$cmd" in
        start|restart) validate_numeric_config ;;
    esac
    case "$cmd" in
        stop)     banner stop.sh ;;
        download) banner download.sh ;;
        *)        banner start-tp3.sh ;;
    esac
    case "$cmd" in
        start)    shift || true; start ;;
        download) download_only ;;
        stop)     stop ;;
        restart)  stop; start ;;
        status)   status ;;
        logs)     shift || true; logs "$@" ;;
        share)    [ "${NFS_SHARE:-0}" = "1" ] || die "NFS_SHARE=0 in .env.tp3 — nothing to share"
                  nfs_share_weights ;;
        -h|--help|help) usage ;;
        *) usage; exit 1 ;;
    esac
}

main "$@"
