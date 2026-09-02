#!/usr/bin/env bash
# ============================================================================
# start-tp4.sh — EXPERIMENTAL 4× DGX Spark TP=4 launcher for GLM-5.3-Flash EXL3
# ============================================================================
#
# Optional sibling of start.sh. Does not change the supported 2× TP=2 path.
# Untested here (no 4-Spark kit). A user with four GB10s can try:
#
#   ./start-tp4.sh
#
# Layout (mp executor, not Ray):
#   rank 0  HEAD_IP     (default 10.0.0.1)  — vLLM API on :8888
#   rank 1  WORKER_IP   (default 10.0.0.2)  — --headless
#   rank 2  WORKER2_IP  (default 10.0.0.3)  — --headless
#   rank 3  WORKER3_IP  (default 10.0.0.4)  — --headless
#   --tensor-parallel-size 4  --nnodes 4
#
# Fabric: four Sparks need a RoCE path among all ranks (QSFP ring using both
# CX7 ports, or a switch). Defaults set NCCL_CROSS_NIC=1. Pin per-rank
# WORKER2_CX7_IF / WORKER2_CX7_IB / WORKER2_GID (same for 3) in .env.
# Dual-port example: RANK NCCL_IB_HCA="rocep1s0f0,rocep1s0f1".
#
# Same image/weights as start.sh. Container names are glm53-exl3-tp4-* so a
# TP=2 serve is not accidentally reused. Stop with ./start-tp4.sh stop
# (./start.sh stop does not know about ranks 2/3).
#
# Usage:
#   ./start-tp4.sh                 start TP=4
#   ./start-tp4.sh download        head HF cache only
#   ./start-tp4.sh stop|restart|status
#   ./start-tp4.sh logs            follow head
#   ./start-tp4.sh logs 1|2|3      follow that worker rank
#
# Extra knobs live in .env.tp4 (copied from .env.tp4.example). start.sh
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
    printf '\033[1;36m[glm53-exl3-tp4]\033[0m wrote .env from .env.example\n'
fi
if [ ! -f "$SCRIPT_DIR/.env.tp4" ]; then
    [ -f "$SCRIPT_DIR/.env.tp4.example" ] || {
        echo "ERROR: missing .env.tp4.example" >&2
        exit 1
    }
    cp "$SCRIPT_DIR/.env.tp4.example" "$SCRIPT_DIR/.env.tp4"
    printf '\033[1;36m[glm53-exl3-tp4]\033[0m wrote .env.tp4 from .env.tp4.example — edit WORKER2_IP / WORKER3_IP / CX7 pins\n'
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
_cli_mnbt="${MAX_NUM_BATCHED_TOKENS-}"
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
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
# TP=4 overlay wins over the 2× knobs in .env.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env.tp4"
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
[ -n "${_cli_mnbt}" ] && MAX_NUM_BATCHED_TOKENS="$_cli_mnbt"
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

# Ranks 2 and 3 (experimental TP=4). Same OS user as WORKER_USER unless set.
WORKER2_IP="${WORKER2_IP:-10.0.0.3}"
WORKER3_IP="${WORKER3_IP:-10.0.0.4}"
WORKER2_USER="${WORKER2_USER:-$WORKER_USER}"
WORKER3_USER="${WORKER3_USER:-$WORKER_USER}"
if [ "$WORKER2_USER" = "$USER" ]; then
    WORKER2_HOME="${WORKER2_HOME:-$HOME}"
else
    WORKER2_HOME="${WORKER2_HOME:-/home/${WORKER2_USER}}"
fi
if [ "$WORKER3_USER" = "$USER" ]; then
    WORKER3_HOME="${WORKER3_HOME:-$HOME}"
else
    WORKER3_HOME="${WORKER3_HOME:-/home/${WORKER3_USER}}"
fi
WORKER2_SSH="${WORKER2_SSH:-${WORKER2_USER}@${WORKER2_IP}}"
WORKER3_SSH="${WORKER3_SSH:-${WORKER3_USER}@${WORKER3_IP}}"

HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f1np1}"
WORKER_CX7_IF="${WORKER_CX7_IF:-enp1s0f0np0}"
HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"
WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f0}"
WORKER2_CX7_IF="${WORKER2_CX7_IF:-$WORKER_CX7_IF}"
WORKER2_CX7_IB="${WORKER2_CX7_IB:-$WORKER_CX7_IB}"
WORKER3_CX7_IF="${WORKER3_CX7_IF:-$WORKER_CX7_IF}"
WORKER3_CX7_IB="${WORKER3_CX7_IB:-$WORKER_CX7_IB}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# The RoCEv2 GID index is per-NIC: the usable entry is the one whose GID matches
# that node's own fabric IP. Most pairs share a good index; some do not (this kit
# needs head=4, worker=3). Unset, both inherit NCCL_IB_GID_INDEX -> unchanged.
HEAD_GID="${HEAD_GID:-$NCCL_IB_GID_INDEX}"
WORKER_GID="${WORKER_GID:-$NCCL_IB_GID_INDEX}"
WORKER2_GID="${WORKER2_GID:-$NCCL_IB_GID_INDEX}"
WORKER3_GID="${WORKER3_GID:-$NCCL_IB_GID_INDEX}"
# vLLM subtracts a CUDA-graph memory ESTIMATE from the KV pool. On this kit the
# estimate is 2.43 GiB while the captured graphs actually consume -0.19 GiB, so
# ~2.6 GiB of KV is reserved and never used. 0 keeps CUDA graphs ON and drops only
# the deduction. 1 = upstream default.
CG_ESTIMATE="${CG_ESTIMATE:-1}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
NCCL_HOST_DIR="${NCCL_HOST_DIR:-$HOME/nccl-2.30.7}"
WORKER_NCCL_HOST_DIR="${WORKER_NCCL_HOST_DIR:-$WORKER_HOME/nccl-2.30.7}"
WORKER2_NCCL_HOST_DIR="${WORKER2_NCCL_HOST_DIR:-$WORKER2_HOME/nccl-2.30.7}"
WORKER3_NCCL_HOST_DIR="${WORKER3_NCCL_HOST_DIR:-$WORKER3_HOME/nccl-2.30.7}"
NCCL_SO_NAME="${NCCL_SO_NAME:-libnccl.so.2.30.7}"
# glm53-flash already ships nvidia-nccl. LD_PRELOAD of the host 2.30.7 SO
# makes DeepEP assert duplicate NCCL (/nccl/... vs nvidia/nccl/lib/...).
# Set USE_HOST_NCCL=1 only if image NCCL cannot talk CX7.
USE_HOST_NCCL="${USE_HOST_NCCL:-0}"

TP="${TP:-4}"
NNODES="${NNODES:-4}"
PORT="${PORT:-8888}"
MASTER_PORT="${MASTER_PORT:-29521}"

MTP_TOKENS="${MTP_TOKENS:-2}"
# dflash (default, incoai/GLM-5.3-Flash-DFlash2, k=7) | mtp | none
SPEC_METHOD="${SPEC_METHOD:-dflash}"
DFLASH_MODEL="${DFLASH_MODEL:-incoai/GLM-5.3-Flash-DFlash2}"
DFLASH_CACHE_NAME="${DFLASH_CACHE_NAME:-models--${DFLASH_MODEL//\//--}}"
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
# 2 = shard the ~2.3 GiB DFlash2 drafter across TP (C4 keep, 2026-08-30:
# idle 8k 938 / 16k 972 / 100k 997; decode structured 65.1 / prose 27.1).
# 1 = rank 0 only (no CX7 on every draft step). Empty = inherit target TP.
# Do not pin attention_backend: SM121 already prefers FLASH_ATTN for
# non-causal dense SWA. TRITON_ATTN was an SM120 mask-fix this image lacks.
DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.87}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# 8192 chunk × long history oversubscribes GB10 persistent_topk smem (300k crash).
# E2 one-shot 2026-09-01: 7168 keep (100k ~1148 / 300k ~1107); 2048/3548 similar or slower.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-7168}"
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
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
QUANTIZATION="${QUANTIZATION:-exl3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
# JSON default cannot sit in ${LIMIT_MM:-{...}} — } ends the expansion.
if [ -z "${LIMIT_MM:-}" ]; then
    LIMIT_MM='{"image":4,"video":1}'
fi
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
# Fused exl3_moe temp rows/expert. 1024 was slower than 128+fallback (P2b).
EXL3_TEMP_ROWS_FUSED="${EXL3_TEMP_ROWS_FUSED:-128}"
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
# skip = do not mix; N>0 = cap tokens; 0 = off.
GLM53_MIXED_PREFILL_CHUNK="${GLM53_MIXED_PREFILL_CHUNK:-skip}"
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

CONTAINER_HEAD="${CONTAINER_HEAD:-glm53-exl3-tp4-head}"
CONTAINER_WORKER="${CONTAINER_WORKER:-glm53-exl3-tp4-w1}"
CONTAINER_WORKER2="${CONTAINER_WORKER2:-glm53-exl3-tp4-w2}"
CONTAINER_WORKER3="${CONTAINER_WORKER3:-glm53-exl3-tp4-w3}"

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_PATH="$HF_CACHE_DIR/hub/$MODEL_CACHE_NAME"
FALLBACK_MODEL_PATH="$HF_CACHE_DIR/hub/$MODEL_FALLBACK_CACHE_NAME"
DFLASH_PATH="$HF_CACHE_DIR/hub/$DFLASH_CACHE_NAME"
WORKER_CACHE_DIR="$WORKER_HOME/.cache/huggingface"
WORKER2_CACHE_DIR="${WORKER2_CACHE_DIR:-$WORKER2_HOME/.cache/huggingface}"
WORKER3_CACHE_DIR="${WORKER3_CACHE_DIR:-$WORKER3_HOME/.cache/huggingface}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/vllm-glm53-flash}"
WORKER_VLLM_CACHE="${WORKER_VLLM_CACHE:-$WORKER_HOME/.cache/vllm-glm53-flash}"
WORKER2_VLLM_CACHE="${WORKER2_VLLM_CACHE:-$WORKER2_HOME/.cache/vllm-glm53-flash}"
WORKER3_VLLM_CACHE="${WORKER3_VLLM_CACHE:-$WORKER3_HOME/.cache/vllm-glm53-flash}"
# Overlay FS ~/.triton and ~/.tilelang die on container recreate (TP=2 JIT
# stall → 600s NCCL watchdog). Persist next to the vLLM cache.
TRITON_HOST_CACHE="${TRITON_HOST_CACHE:-$CACHE_ROOT/triton}"
TILELANG_HOST_CACHE="${TILELANG_HOST_CACHE:-$CACHE_ROOT/tilelang}"
WORKER_TRITON_CACHE="${WORKER_TRITON_CACHE:-$WORKER_VLLM_CACHE/triton}"
WORKER_TILELANG_CACHE="${WORKER_TILELANG_CACHE:-$WORKER_VLLM_CACHE/tilelang}"
WORKER2_TRITON_CACHE="${WORKER2_TRITON_CACHE:-$WORKER2_VLLM_CACHE/triton}"
WORKER2_TILELANG_CACHE="${WORKER2_TILELANG_CACHE:-$WORKER2_VLLM_CACHE/tilelang}"
WORKER3_TRITON_CACHE="${WORKER3_TRITON_CACHE:-$WORKER3_VLLM_CACHE/triton}"
WORKER3_TILELANG_CACHE="${WORKER3_TILELANG_CACHE:-$WORKER3_VLLM_CACHE/tilelang}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.triton/cache}"
TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/root/.tilelang/cache}"

LOGDIR="$SCRIPT_DIR/logs"
HEAD_SCRIPT="$SCRIPT_DIR/.glm53-exl3-tp4-head.inner.sh"
WORKER_SCRIPT="$SCRIPT_DIR/.glm53-exl3-tp4-worker.inner.sh"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-120}"

# ------------------------------- helpers -----------------------------------
log()  { printf '\033[1;36m[glm53-exl3-tp4]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm53-exl3-tp4]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[glm53-exl3-tp4]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

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

validate_numeric_config() {
    if ! [[ "$GPU_MEM_UTIL" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0+)?)$ ]] \
       || ! awk -v u="$GPU_MEM_UTIL" 'BEGIN { exit !(u > 0 && u <= 1) }'; then
        echo "GPU_MEM_UTIL must be greater than 0 and at most 1 (got: $GPU_MEM_UTIL)" >&2
        return 2
    fi
    _glm53_canonical_positive_int MAX_MODEL_LEN "$MAX_MODEL_LEN" 1000000 || return
    _glm53_canonical_positive_int MAX_NUM_SEQS "$MAX_NUM_SEQS" 4096 || return
    _glm53_canonical_positive_int MAX_NUM_BATCHED_TOKENS "$MAX_NUM_BATCHED_TOKENS" 8388608 || return
    _glm53_validate_enum GLM53_INDEXER_WORKSPACE "${GLM53_INDEXER_WORKSPACE-stock}" \
        stock rightsize || return
    _glm53_validate_spinwait_ms || return
}
# GLM53 numeric config guard (end)

banner() {
    local label="${1:-start-tp4.sh}"
    printf '\n'
    printf '  \033[1;36m┌────────────────────────────────────────────┐\033[0m\n'
    printf '  \033[1;36m│\033[0m  \033[1mGLM-5.3 Flash EXL3\033[0m  \033[2m·  %-11s\033[0m        \033[1;36m│\033[0m\n' "$label"
    printf '  \033[1;36m└────────────────────────────────────────────┘\033[0m\n'
    printf '\n'
}

# rank 1..3
_tp4_ssh_target() {
    case "$1" in
        1) printf '%s' "$WORKER_SSH" ;;
        2) printf '%s' "$WORKER2_SSH" ;;
        3) printf '%s' "$WORKER3_SSH" ;;
        *) die "internal: bad worker rank $1" ;;
    esac
}
_tp4_rank_ip() {
    case "$1" in
        1) printf '%s' "$WORKER_IP" ;;
        2) printf '%s' "$WORKER2_IP" ;;
        3) printf '%s' "$WORKER3_IP" ;;
    esac
}
_tp4_rank_home() {
    case "$1" in
        1) printf '%s' "$WORKER_HOME" ;;
        2) printf '%s' "$WORKER2_HOME" ;;
        3) printf '%s' "$WORKER3_HOME" ;;
    esac
}
_tp4_rank_hf() {
    case "$1" in
        1) printf '%s' "$WORKER_CACHE_DIR" ;;
        2) printf '%s' "$WORKER2_CACHE_DIR" ;;
        3) printf '%s' "$WORKER3_CACHE_DIR" ;;
    esac
}
_tp4_rank_vllm() {
    case "$1" in
        1) printf '%s' "$WORKER_VLLM_CACHE" ;;
        2) printf '%s' "$WORKER2_VLLM_CACHE" ;;
        3) printf '%s' "$WORKER3_VLLM_CACHE" ;;
    esac
}
_tp4_rank_triton() {
    case "$1" in
        1) printf '%s' "$WORKER_TRITON_CACHE" ;;
        2) printf '%s' "$WORKER2_TRITON_CACHE" ;;
        3) printf '%s' "$WORKER3_TRITON_CACHE" ;;
    esac
}
_tp4_rank_tilelang() {
    case "$1" in
        1) printf '%s' "$WORKER_TILELANG_CACHE" ;;
        2) printf '%s' "$WORKER2_TILELANG_CACHE" ;;
        3) printf '%s' "$WORKER3_TILELANG_CACHE" ;;
    esac
}
_tp4_rank_cx7_if() {
    case "$1" in
        1) printf '%s' "$WORKER_CX7_IF" ;;
        2) printf '%s' "$WORKER2_CX7_IF" ;;
        3) printf '%s' "$WORKER3_CX7_IF" ;;
    esac
}
_tp4_rank_cx7_ib() {
    case "$1" in
        1) printf '%s' "$WORKER_CX7_IB" ;;
        2) printf '%s' "$WORKER2_CX7_IB" ;;
        3) printf '%s' "$WORKER3_CX7_IB" ;;
    esac
}
_tp4_rank_gid() {
    case "$1" in
        1) printf '%s' "$WORKER_GID" ;;
        2) printf '%s' "$WORKER2_GID" ;;
        3) printf '%s' "$WORKER3_GID" ;;
    esac
}
_tp4_rank_container() {
    case "$1" in
        1) printf '%s' "$CONTAINER_WORKER" ;;
        2) printf '%s' "$CONTAINER_WORKER2" ;;
        3) printf '%s' "$CONTAINER_WORKER3" ;;
    esac
}
_tp4_rank_nccl_dir() {
    case "$1" in
        1) printf '%s' "$WORKER_NCCL_HOST_DIR" ;;
        2) printf '%s' "$WORKER2_NCCL_HOST_DIR" ;;
        3) printf '%s' "$WORKER3_NCCL_HOST_DIR" ;;
    esac
}
_tp4_first_ib() { printf '%s' "${1%%,*}"; }

worker_ssh() { ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH" "$@"; }
worker_ssh_n() {
    local r="$1"; shift
    ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$(_tp4_ssh_target "$r")" "$@"
}

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

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

resolve_dflash_dir() {
    local ref="$DFLASH_PATH/refs/main" hash dir
    ensure_dflash_refs_main
    hash="$(<"$ref")"
    dir="$DFLASH_PATH/snapshots/$hash"
    [ -f "$dir/config.json" ] || die "DFlash2 config.json missing in $dir"
    printf '/root/.cache/huggingface/hub/%s/snapshots/%s' "$DFLASH_CACHE_NAME" "$hash"
}

check_port_free() {
    local port="$1" envname="$2"
    command -v ss >/dev/null 2>&1 || return 0
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
        if docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
            die "port ${port} is held by ${CONTAINER_HEAD} — use './start-tp4.sh restart' or './start-tp4.sh stop' first"
        fi
        die "port ${port} is already in use — stop it or rerun with ${envname}=<free-port>"
    fi
}

trap 'warn "interrupted — containers keep running ('"'"'./start-tp4.sh logs'"'"' to watch, '"'"'./start-tp4.sh stop'"'"' to stop)"; exit 130' INT

# ------------------------------ preflight ----------------------------------
preflight() {
    command -v docker  >/dev/null 2>&1 || die "docker not found on head"
    command -v curl    >/dev/null 2>&1 || die "curl not found on head"
    command -v rsync   >/dev/null 2>&1 || die "rsync not found on head"
    docker info >/dev/null 2>&1 || die "cannot talk to docker daemon on head"

    ip -4 addr show 2>/dev/null | grep -q "inet ${HEAD_IP}/" \
        || die "HEAD_IP=${HEAD_IP} is not assigned on this host — set it in .env"

    [ "$TP" = "4" ] || warn "TP=${TP} — this script is meant for TP=4"
    [ "$NNODES" = "4" ] || warn "NNODES=${NNODES} — this script is meant for nnodes=4"
    local r ssh_t
    for r in 1 2 3; do
        ssh_t="$(_tp4_ssh_target "$r")"
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
    ib="$(_tp4_first_ib "$HEAD_CX7_IB")"
    gid_path="/sys/class/infiniband/${ib}/ports/1/gids/${HEAD_GID}"
    gid_head=$(cat "$gid_path" 2>/dev/null | tr -d ':0' || true)
    [ -n "$gid_head" ] || { warn "head GID index ${HEAD_GID} is EMPTY on ${ib}"; gid_ok=0; }
    for r in 1 2 3; do
        ib="$(_tp4_first_ib "$(_tp4_rank_cx7_ib "$r")")"
        gid_path="/sys/class/infiniband/${ib}/ports/1/gids/$(_tp4_rank_gid "$r")"
        gid_worker=$(worker_ssh_n "$r" "cat '$gid_path' 2>/dev/null" | tr -d ':0' || true)
        if [ -z "$gid_worker" ]; then
            warn "rank ${r} GID index $(_tp4_rank_gid "$r") is EMPTY on ${ib}"
            gid_ok=0
        fi
    done
    if [ "$gid_ok" != "1" ]; then
        if [ -z "$gid_head" ]; then
            warn "head GID index ${HEAD_GID} is EMPTY on $(_tp4_first_ib "$HEAD_CX7_IB")"
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

    [ "$TP" = "4" ] || warn "TP=${TP} — expected TP=4 for start-tp4.sh"
    [ "$NNODES" = "4" ] || warn "NNODES=${NNODES} — expected 4"

    local others cname
    for r in 1 2 3; do
        cname="$(_tp4_rank_container "$r")"
        others=$(worker_ssh_n "$r" "docker ps --format '  {{.Names}}  ({{.Image}})'" 2>/dev/null | grep -v "^  ${cname}" || true)
        if [ -n "$others" ]; then
            warn "other containers are running on rank ${r} ($(_tp4_ssh_target "$r")):"
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
    for r in 1 2 3; do
        avail=$(worker_ssh_n "$r" "df -Pk '$(_tp4_rank_home "$r")' 2>/dev/null" | awk 'NR==2{print $4}' || true)
        [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on rank ${r} for a ~164 GiB model"
    done

    # The worker HF cache must be writable by the SSH user before the ~164 GiB
    # sync starts. A root-owned ~/.cache/huggingface (prior sudo/docker
    # prepare on the worker) otherwise fails mid-sync with a bare mkdir
    # permission error. mkdir -p is idempotent and is what sync does anyway.
    for r in 1 2 3; do
        if ! worker_ssh_n "$r" "mkdir -p '$(_tp4_rank_hf "$r")/hub' && test -w '$(_tp4_rank_hf "$r")/hub'"; then
            die "rank ${r} cannot write $(_tp4_rank_hf "$r")/hub — fix ownership, e.g. ssh $(_tp4_ssh_target "$r") \"sudo chown -R \$USER: '$(_tp4_rank_hf "$r")'\""
        fi
    done

    log "preflight OK (head=$(hostname) ${HEAD_IP}, workers=${WORKER_SSH} ${WORKER2_SSH} ${WORKER3_SSH})"
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
    for r in 1 2 3; do
        log "docker login ghcr.io on rank ${r} as ${GHCR_USER} (GHCR_TOKEN)"
        echo "$GHCR_TOKEN" | worker_ssh_n "$r" "docker login ghcr.io -u '$GHCR_USER' --password-stdin" >/dev/null
    done
}

# RepoDigest is stable across overlay2 vs containerd. Those snapshotters
# disagree on .Id (config digest vs index digest), so start.sh used to
# ship even after the worker had already pulled the same GHCR tag (issue #8).
# Local builds have no RepoDigest — use RootFS layer diffs, then .Id.
_IMAGE_KEY_FMT='{{if .RepoDigests}}{{index .RepoDigests 0}}{{else if .RootFS.Layers}}{{join .RootFS.Layers ","}}{{else}}{{.Id}}{{end}}'

parse_image_key() {
    tr -d '\r' | sed -n 's/^GLM53KEY //p' | tail -n 1
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
            ! -path '*/ablit/transplant/*' \
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

ship_image_to_worker() {
    local r="${1:-1}" platform
    platform="$(image_platform)"
    log "shipping ${IMAGE} (${platform}) to rank ${r} via docker save | ssh docker load ..."
    if docker save --platform "$platform" "$IMAGE" | worker_ssh_n "$r" docker load; then
        return 0
    fi
    warn "docker save --platform ${platform} failed — retrying without --platform"
    docker save "$IMAGE" | worker_ssh_n "$r" docker load
}

ensure_image() {
    mkdir -p "$LOGDIR"
    local head_ok=0 worker_ok=0 head_key="" worker_key=""
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        head_ok=1
        head_key="$(local_image_key)"
    fi
    worker_ok=1
    for r in 1 2 3; do
        if worker_ssh_n "$r" "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
            worker_key="$(worker_image_key "$r")"
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
    local wanted_stamp have_stamp
    wanted_stamp="$(overlay_recipe_hash)"
    have_stamp=""
    [ "$head_ok" = "1" ] && have_stamp="$(image_recipe_stamp)"
    if [ "${BUILD:-0}" != "1" ] && [ "${SKIP_BUILD:-0}" != "1" ]; then
        if [ "$head_ok" = "0" ] || [ "$have_stamp" != "$wanted_stamp" ]; then
            log "image recipe ${have_stamp:-none} != repo ${wanted_stamp:0:12} — rebuilding (SKIP_BUILD=1 keeps GHCR)"
            BUILD=1
        fi
    elif [ "${SKIP_BUILD:-0}" = "1" ] && [ "$have_stamp" != "$wanted_stamp" ]; then
        warn "SKIP_BUILD=1 — not rebuilding; stamp ${have_stamp:-none} != repo ${wanted_stamp:0:12}"
    fi
    if [ "${BUILD:-0}" = "1" ]; then
        build_image
        head_key="$(local_image_key)"
        head_ok=1
        worker_ok=0
    elif image_from_registry && [ "$skip_pull" != "1" ]; then
        local before_key="$head_key"
        pull_image
        head_key="$(local_image_key)"
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
        head_key="$(local_image_key)"
        head_ok=1
        worker_ok=0
    fi
    if [ "${SKIP_SHIP:-0}" = "1" ]; then
        [ "$worker_ok" = "1" ] || warn "SKIP_SHIP=1 — not copying ${IMAGE} to workers"
    elif [ "$worker_ok" = "0" ]; then
        for r in 1 2 3; do
            local wok=0
            worker_key="$(worker_image_key "$r")"
            if images_match "$head_key" "$worker_key"; then
                continue
            fi
            if image_from_registry && [ "$skip_pull" != "1" ] && [ "${BUILD:-0}" != "1" ]; then
                if pull_image_on_worker "$r"; then
                    worker_key="$(worker_image_key "$r")"
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
                worker_key="$(worker_image_key "$r")"
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
    log "image ready on all four nodes"
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
    log "workers were not touched. ./start-tp4.sh will rsync on launch unless SKIP_SYNC=1."
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
    hf="$(_tp4_rank_hf "$r")"
    ssh_t="$(_tp4_ssh_target "$r")"
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
    local r
    for r in 1 2 3; do
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
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
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
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
    say "vision on: limit-mm=${LIMIT_MM:-} skip-mm-profiling=${SKIP_MM_PROFILING:-1} chat-template=${CHAT_TEMPLATE:-}"
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
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
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
_tp4_scp_runtime() {
    local r="$1" ssh_t cname
    ssh_t="$(_tp4_ssh_target "$r")"
    cname="$(_tp4_rank_container "$r")"
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
    worker_ssh_n "$r" "rm -rf /tmp/glm53-ablit"
    scp -q -r -o BatchMode=yes "$SCRIPT_DIR/ablit" "${ssh_t}:/tmp/glm53-ablit"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/ablit_runtime.py" "${ssh_t}:/tmp/glm53-ablit_runtime.py"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/patch_ablit.py" "${ssh_t}:/tmp/patch_ablit.py"
}

launch_cluster() {
    local r
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || true
    for r in 1 2 3; do
        worker_ssh_n "$r" "docker rm -f '$(_tp4_rank_container "$r")'" >/dev/null 2>&1 || true
    done

    mkdir -p "$CACHE_ROOT" "$TRITON_HOST_CACHE" "$TILELANG_HOST_CACHE"
    [ -f "$CHAT_TEMPLATE_HOST" ] || die "missing chat template: $CHAT_TEMPLATE_HOST"
    for r in 1 2 3; do
        worker_ssh_n "$r" "mkdir -p '$(_tp4_rank_vllm "$r")' '$(_tp4_rank_triton "$r")' '$(_tp4_rank_tilelang "$r")'"
        _tp4_scp_runtime "$r"
    done
    # skip the old single-worker scp block
    true
    : <<'TP4_SKIP_OLD_SCP'
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

    worker_ssh "rm -rf /tmp/glm53-ablit"
    scp -q -r -o BatchMode=yes "$SCRIPT_DIR/ablit" "${WORKER_SSH}:/tmp/glm53-ablit"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/ablit_runtime.py" "${WORKER_SSH}:/tmp/glm53-ablit_runtime.py"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/patch_ablit.py" "${WORKER_SSH}:/tmp/patch_ablit.py"
TP4_SKIP_OLD_SCP

    local -a nccl_common=(
        -e NCCL_IB_DISABLE=0
        -e NCCL_IB_ROCE_VERSION_NUM=2
        -e NCCL_NET=IB
        -e NCCL_NET_PLUGIN=none
        -e NCCL_NVLS_ENABLE=0
        -e NCCL_CUMEM_ENABLE=0
        -e NCCL_IB_MERGE_NICS=0
        -e "NCCL_CROSS_NIC=$NCCL_CROSS_NIC"
        -e NCCL_IGNORE_CPU_AFFINITY=1
        -e "NCCL_DEBUG=$NCCL_DEBUG"
        -e HF_HUB_OFFLINE=1
        -e TRANSFORMERS_OFFLINE=1
        -e HF_HOME=/root/.cache/huggingface
        -e VLLM_CACHE_ROOT=/root/.cache/vllm
        -e "GLM53_SUPPRESS_STOPS_IN_REASONING=$GLM53_SUPPRESS_STOPS_IN_REASONING"
        -e "GLM53_MIXED_PREFILL_CHUNK=$GLM53_MIXED_PREFILL_CHUNK"
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
             KV_CACHE_DTYPE MTP_TOKENS SPEC_METHOD DFLASH_TOKENS DFLASH_MODEL_DIR \
             DFLASH_DRAFT_TP \
             LANGUAGE_MODEL_ONLY SKIP_MM_PROFILING \
             LIMIT_MM CHAT_TEMPLATE ENFORCE_EAGER EXL3_FUSED_MOE EXL3_MOE_ROW_TILE EXL3_TEMP_ROWS_FUSED EXL3_FAT_SORTED EXL3_FAT_BATCHED EXL3_FAT_KERNEL MODEL_DIR EXTRA_ARGS \
             ABLIT ABLIT_METHOD ABLIT_DIRECTION ABLIT_LAYERS ABLIT_ALPHA ABLIT_INCLUDE_MTP; do
        serve_env+=" -e $v='${!v:-}'"
    done
    # VLLM_API_KEY is read by the head (rank 0) API server for bearer auth; the
    # worker runs --headless so it only needs the var for argv-parity, and
    # start.sh below passes it explicitly on the head. Keep it out of the
    # generic loop so the key never shows in process listings of either node
    # beyond the container env (same as the DeepSeek deployment).
    serve_env+=" -e VLLM_API_KEY='${VLLM_API_KEY:-}'"

    local worker_preload="" nccl_dir cname
    for r in 1 2 3; do
        worker_preload=""
        if [ "$USE_HOST_NCCL" = "1" ]; then
            nccl_dir="$(_tp4_rank_nccl_dir "$r")"
            if worker_ssh_n "$r" "test -f '$nccl_dir/$NCCL_SO_NAME'"; then
                worker_preload="-v '$nccl_dir:/nccl:ro' -e LD_PRELOAD='/nccl/$NCCL_SO_NAME'"
                log "rank ${r}: LD_PRELOAD $NCCL_SO_NAME"
            else
                warn "rank ${r}: $nccl_dir/$NCCL_SO_NAME missing — using image NCCL"
            fi
        fi
        cname="$(_tp4_rank_container "$r")"
        log "starting rank ${r} on $(_tp4_ssh_target "$r") (NCCL if=$(_tp4_rank_cx7_if "$r") hca=$(_tp4_rank_cx7_ib "$r")) ..."
        worker_ssh_n "$r" "docker run -d --name '$cname' \
            --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
            --device /dev/infiniband --cap-add IPC_LOCK \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v '$(_tp4_rank_hf "$r"):/root/.cache/huggingface' \
            -v '$(_tp4_rank_vllm "$r"):/root/.cache/vllm' \
            -v '$(_tp4_rank_triton "$r"):/root/.triton/cache' \
            -v '$(_tp4_rank_tilelang "$r"):/root/.tilelang/cache' \
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
            -v '/tmp/glm53-ablit:/opt/glm53/ablit:ro' \
            -v '/tmp/glm53-ablit_runtime.py:/opt/glm53/ablit_runtime.py:ro' \
            -v '/tmp/patch_ablit.py:/opt/glm53/patch_ablit.py:ro' \
            ${worker_preload} \
            ${worker_nccl} \
            -e NCCL_SOCKET_IFNAME='$(_tp4_rank_cx7_if "$r")' \
            -e GLOO_SOCKET_IFNAME='$(_tp4_rank_cx7_if "$r")' \
            -e NCCL_IB_HCA='$(_tp4_rank_cx7_ib "$r")' \
            -e NCCL_IB_GID_INDEX='$(_tp4_rank_gid "$r")' \
            -e VLLM_HOST_IP='$(_tp4_rank_ip "$r")' \
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
        -v "$SCRIPT_DIR/ablit:/opt/glm53/ablit:ro" \
        -v "$SCRIPT_DIR/overlay/ablit_runtime.py:/opt/glm53/ablit_runtime.py:ro" \
        -v "$SCRIPT_DIR/overlay/patch_ablit.py:/opt/glm53/patch_ablit.py:ro" \
        "${head_preload[@]}" \
        "${nccl_common[@]}" \
        -e NCCL_SOCKET_IFNAME="$HEAD_CX7_IF" \
        -e GLOO_SOCKET_IFNAME="$HEAD_CX7_IF" \
        -e NCCL_IB_HCA="$HEAD_CX7_IB" \
        -e NCCL_IB_GID_INDEX="$HEAD_GID" \
        -e VLLM_HOST_IP="$HEAD_IP" \
        -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
        -e PORT="$PORT" -e TP="$TP" -e NNODES="$NNODES" \
        -e HEAD_IP="$HEAD_IP" -e MASTER_PORT="$MASTER_PORT" \
        -e QUANTIZATION="$QUANTIZATION" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
        -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
        -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" -e MTP_TOKENS="$MTP_TOKENS" \
        -e SPEC_METHOD="$SPEC_METHOD" \
        -e DFLASH_TOKENS="${DFLASH_TOKENS:-7}" \
        -e DFLASH_MODEL_DIR="${DFLASH_MODEL_DIR:-}" \
        -e DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP:-}" \
        -e LANGUAGE_MODEL_ONLY="$LANGUAGE_MODEL_ONLY" \
        -e SKIP_MM_PROFILING="$SKIP_MM_PROFILING" \
        -e LIMIT_MM="$LIMIT_MM" \
        -e CHAT_TEMPLATE="$CHAT_TEMPLATE" \
        -e ENFORCE_EAGER="$ENFORCE_EAGER" \
        -e EXL3_FUSED_MOE="$EXL3_FUSED_MOE" \
        -e EXL3_MOE_ROW_TILE="$EXL3_MOE_ROW_TILE" \
        -e EXL3_TEMP_ROWS_FUSED="$EXL3_TEMP_ROWS_FUSED" \
        -e EXL3_FAT_SORTED="$EXL3_FAT_SORTED" \
        -e EXL3_FAT_BATCHED="$EXL3_FAT_BATCHED" \
        -e EXL3_FAT_KERNEL="$EXL3_FAT_KERNEL" \
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

    log "containers up — head=${CONTAINER_HEAD}, workers=${CONTAINER_WORKER} ${CONTAINER_WORKER2} ${CONTAINER_WORKER3}"
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
    trap '_stop_logtail; warn "interrupted — containers keep running ('"'"'./start-tp4.sh logs'"'"' / '"'"'./start-tp4.sh stop'"'"')"; exit 130' INT
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
        for wr in 1 2 3; do
            if worker_ssh_n "$wr" "docker inspect -f '{{.State.Running}}' '$(_tp4_rank_container "$wr")' 2>/dev/null" | grep -q true; then
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
    trap 'warn "interrupted — containers keep running ('"'"'./start-tp4.sh logs'"'"' / '"'"'./start-tp4.sh stop'"'"')"; exit 130' INT

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
        || warn "boot shape warmup incomplete — uncovered shapes may JIT mid-serve on TP=4"
}

collect_failure_logs() {
    mkdir -p "$LOGDIR"
    docker logs "$CONTAINER_HEAD" >"$LOGDIR/head.log" 2>&1 || true
    local r
    for r in 1 2 3; do
        worker_ssh_n "$r" "docker logs '$(_tp4_rank_container "$r")' 2>&1" >"$LOGDIR/worker${r}.log" 2>&1 || true
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
    log "  manage     : ./start-tp4.sh status | ./start-tp4.sh logs | ./start-tp4.sh logs 1 | ./start-tp4.sh stop"
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
        log "DFlash2 load path (in-container): ${DFLASH_MODEL_DIR}"
    fi
    log "model load path (in-container): ${MODEL_DIR}"
    log "config: image=${IMAGE} tp=${TP} nnodes=${NNODES} quant=${QUANTIZATION} spec=${SPEC_METHOD} mtp=${MTP_TOKENS} dflash_k=${DFLASH_TOKENS} max-len=${MAX_MODEL_LEN} gpu-util=${GPU_MEM_UTIL} kv=${KV_CACHE_DTYPE} lm-only=${LANGUAGE_MODEL_ONLY} port=${PORT}"

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
    echo "---- last 40 lines of worker3 log ($LOGDIR/worker3.log) ----"
    tail -n 40 "$LOGDIR/worker3.log" || true
    die "server did not become healthy — full logs in $LOGDIR/"
}

# ------------------------------- stop --------------------------------------
stop() {
    local r
    log "stopping head container ..."
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || log "  (no head container was running)"
    for r in 1 2 3; do
        log "stopping rank ${r} on $(_tp4_ssh_target "$r") ..."
        worker_ssh_n "$r" "docker rm -f '$(_tp4_rank_container "$r")'" >/dev/null 2>&1 \
            || log "  (no rank ${r} container was running)"
    done
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
    for r in 1 2 3; do
        log "rank ${r} ($(_tp4_rank_container "$r") on $(_tp4_ssh_target "$r")):"
        worker_ssh_n "$r" "docker ps -a --filter name=$(_tp4_rank_container "$r") --format '  {{.Names}}  {{.Status}}'" 2>/dev/null \
            || log "  (rank ${r} unreachable)"
    done
}

# ------------------------------- logs --------------------------------------
logs() {
    case "${1:-head}" in
        1|2|3|worker|worker1)
            local r="${1}"
            if [ "$r" = "worker" ] || [ "$r" = "worker1" ]; then r=1; fi
            log "following rank ${r} logs on $(_tp4_ssh_target "$r") ..."
            trap '' INT
            worker_ssh_n "$r" "docker logs -f --tail 100 '$(_tp4_rank_container "$r")'" || true
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
        *)        banner start-tp4.sh ;;
    esac
    case "$cmd" in
        start)    shift || true; start ;;
        download) download_only ;;
        stop)     stop ;;
        restart)  stop; start ;;
        status)   status ;;
        logs)     shift || true; logs "$@" ;;
        -h|--help|help) usage ;;
        *) usage; exit 1 ;;
    esac
}

main "$@"
