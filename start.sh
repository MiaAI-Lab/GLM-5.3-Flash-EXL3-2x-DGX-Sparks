#!/usr/bin/env bash
# ============================================================================
# start.sh — Spark runtime for GLM-5.3-Flash EXL3 (SM121 / GB10)
# ============================================================================
#
# We serve Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw (mirror of
# brandonmusic/GLM-5.3-Flash-tr3-4bpw @ 5ab363a8) on this 2× DGX Spark (GB10 /
# SM121) kit: vLLM TP=2 over CX7, OpenAI API on :8888, NoPE-MLA overlay image.
# DFlash2-7 is the default speculator. Target KV stays packed fp8_ds_mla;
# the SM120 B12X recipe (EP2/DCP2 + nvfp4_ds_mla) is a different image/arch.
#
#   head   : this machine (HEAD_IP, default 10.0.0.1) — vLLM rank 0 + API
#   worker : WORKER_USER@WORKER_IP (default: $USER@10.0.0.2) — vLLM rank 1, --headless
#   layout : --tensor-parallel-size 2, --nnodes 2, mp executor (not Ray)
#
# EXL3, not NVFP4. Do not pass --moe-backend marlin.
#
# What we do:
#   1. preflight  — docker/ssh/disk on both nodes
#   2. image      — docker pull IMAGE from GHCR (public :exl3 tag). If the
#                   worker is missing that digest, try docker pull there,
#                   then fall back to docker save --platform | ssh docker
#                   load (issue #8). SKIP_PULL=1 keeps a local copy.
#                   BUILD=1 rebuilds from this repo instead. Local-only
#                   tags (no slash) skip pull. SKIP_SHIP=1 never copies.
#   3. download   — EXL3/TR3 (+ DFlash2) into the local HF cache if missing
#   4. sync       — rsync that cache to the worker (each rank loads local disk)
#   5. launch     — worker --headless, then head + `vllm serve` (both
#                   --network host --ipc=host)
#   6. wait       — poll /health up to READY_TIMEOUT, then a nonfatal
#                   DFlash2/sampler shape warmup (GLM53_BOOT_SHAPE_WARMUP)
#
# Usage:
#   ./start.sh                    start (download/sync/launch) — default
#   ./start.sh download           EXL3 (+ DFlash2) into the head HF cache only
#                                 (no worker). Same as ./download.sh
#   ./start.sh stop               stop both nodes
#   ./start.sh restart            stop + start
#   ./start.sh status             containers + API health
#   ./start.sh logs               follow head logs
#   ./start.sh logs worker        follow worker container logs
#
# Node IPs live in .env (copied from .env.example on first run).
# Handy overrides: SKIP_DOWNLOAD=1 SKIP_SYNC=1 SKIP_PULL=1 SKIP_SHIP=1 PULL=1 BUILD=1 TAIL=1 HF_TOKEN=...
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    [ -f "$SCRIPT_DIR/.env.example" ] || {
        echo "ERROR: missing .env.example" >&2
        exit 1
    }
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    printf '\033[1;36m[glm53-exl3]\033[0m wrote .env from .env.example — edit HEAD_IP / WORKER_IP if needed\n'
fi
# Caller exports (MTP_TOKENS=2 ./start.sh restart) must win over .env.
_cli_mtp="${MTP_TOKENS-}"
_cli_spec="${SPEC_METHOD-}"
_cli_eager="${ENFORCE_EAGER-}"
_cli_fused="${EXL3_FUSED_MOE-}"
_cli_row_tile="${EXL3_MOE_ROW_TILE-}"
_cli_temp_rows="${EXL3_TEMP_ROWS_FUSED-}"
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
_cli_chunk="${GLM53_MIXED_PREFILL_CHUNK-}"
_cli_warm="${GLM53_MIXED_PREFILL_WARM_TOKENS-}"
_cli_wait="${GLM53_MIXED_PREFILL_MAX_WAIT_MS-}"
_cli_late="${GLM53_MIXED_PREFILL_LATE_CAP-}"
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a
[ -n "${_cli_mtp}" ] && MTP_TOKENS="$_cli_mtp"
[ -n "${_cli_chunk}" ] && GLM53_MIXED_PREFILL_CHUNK="$_cli_chunk"
[ -n "${_cli_warm}" ] && GLM53_MIXED_PREFILL_WARM_TOKENS="$_cli_warm"
[ -n "${_cli_wait}" ] && GLM53_MIXED_PREFILL_MAX_WAIT_MS="$_cli_wait"
[ -n "${_cli_late}" ] && GLM53_MIXED_PREFILL_LATE_CAP="$_cli_late"
[ -n "${_cli_spec}" ] && SPEC_METHOD="$_cli_spec"
[ -n "${_cli_eager}" ] && ENFORCE_EAGER="$_cli_eager"
[ -n "${_cli_fused}" ] && EXL3_FUSED_MOE="$_cli_fused"
[ -n "${_cli_row_tile}" ] && EXL3_MOE_ROW_TILE="$_cli_row_tile"
[ -n "${_cli_temp_rows}" ] && EXL3_TEMP_ROWS_FUSED="$_cli_temp_rows"
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

HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f1np1}"
WORKER_CX7_IF="${WORKER_CX7_IF:-enp1s0f0np0}"
HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"
WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f0}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# The RoCEv2 GID index is per-NIC: the usable entry is the one whose GID matches
# that node's own fabric IP. Most pairs share a good index; some do not (this kit
# needs head=4, worker=3). Unset, both inherit NCCL_IB_GID_INDEX -> unchanged.
HEAD_GID="${HEAD_GID:-$NCCL_IB_GID_INDEX}"
WORKER_GID="${WORKER_GID:-$NCCL_IB_GID_INDEX}"
# vLLM subtracts a CUDA-graph memory ESTIMATE from the KV pool. On this kit the
# estimate is 2.43 GiB while the captured graphs actually consume -0.19 GiB, so
# ~2.6 GiB of KV is reserved and never used. 0 keeps CUDA graphs ON and drops only
# the deduction. 1 = upstream default.
CG_ESTIMATE="${CG_ESTIMATE:-1}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
NCCL_HOST_DIR="${NCCL_HOST_DIR:-$HOME/nccl-2.30.7}"
WORKER_NCCL_HOST_DIR="${WORKER_NCCL_HOST_DIR:-$WORKER_HOME/nccl-2.30.7}"
NCCL_SO_NAME="${NCCL_SO_NAME:-libnccl.so.2.30.7}"
# glm53-flash already ships nvidia-nccl. LD_PRELOAD of the host 2.30.7 SO
# makes DeepEP assert duplicate NCCL (/nccl/... vs nvidia/nccl/lib/...).
# Set USE_HOST_NCCL=1 only if image NCCL cannot talk CX7.
USE_HOST_NCCL="${USE_HOST_NCCL:-0}"

TP="${TP:-2}"
NNODES="${NNODES:-2}"
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
DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.87}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# 8192 chunk × long history oversubscribes GB10 persistent_topk smem (300k crash).
# P1 ladder 2026-08-29: 2048 keep; 3584/4096 revert (fat LinearEXL3 tax).
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
CHAT_TEMPLATE_HOST="${CHAT_TEMPLATE_HOST:-$SCRIPT_DIR/files/chat_template.jinja}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/opt/glm53/chat_template.jinja}"
VIDEO_PATCH_HOST="${VIDEO_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_glm_video_placeholders.py}"
STOP_PATCH_HOST="${STOP_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_suppress_stops_in_reasoning.py}"
SCHED_PATCH_HOST="${SCHED_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_scheduler_decode_floor.py}"
DRAFTER_PATCH_HOST="${DRAFTER_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_glm5_drafter_group.py}"
APC_PATCH_HOST="${APC_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_hybrid_prefix_hit.py}"
XGRAMMAR_PATCH_HOST="${XGRAMMAR_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_xgrammar_termination.py}"
KPOOL_TAIL_PATCH_HOST="${KPOOL_TAIL_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_kpool_tail_slotmap.py}"
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
# Gate v2: cached follow-ups (uncached remainder <= WARM_TOKENS) bypass `skip`; held
# cold prefills proceed after MAX_WAIT_MS under LATE_CAP tokens/step (0 = wait forever).
GLM53_MIXED_PREFILL_WARM_TOKENS="${GLM53_MIXED_PREFILL_WARM_TOKENS:-3584}"
GLM53_MIXED_PREFILL_MAX_WAIT_MS="${GLM53_MIXED_PREFILL_MAX_WAIT_MS:-1500}"
GLM53_MIXED_PREFILL_LATE_CAP="${GLM53_MIXED_PREFILL_LATE_CAP:-512}"
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

CONTAINER_HEAD="${CONTAINER_HEAD:-glm53-exl3-head}"
CONTAINER_WORKER="${CONTAINER_WORKER:-glm53-exl3-worker}"

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_PATH="$HF_CACHE_DIR/hub/$MODEL_CACHE_NAME"
FALLBACK_MODEL_PATH="$HF_CACHE_DIR/hub/$MODEL_FALLBACK_CACHE_NAME"
DFLASH_PATH="$HF_CACHE_DIR/hub/$DFLASH_CACHE_NAME"
WORKER_CACHE_DIR="$WORKER_HOME/.cache/huggingface"
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/vllm-glm53-flash}"
WORKER_VLLM_CACHE="${WORKER_VLLM_CACHE:-$WORKER_HOME/.cache/vllm-glm53-flash}"
# Overlay FS ~/.triton and ~/.tilelang die on container recreate (TP=2 JIT
# stall → 600s NCCL watchdog). Persist next to the vLLM cache.
TRITON_HOST_CACHE="${TRITON_HOST_CACHE:-$CACHE_ROOT/triton}"
TILELANG_HOST_CACHE="${TILELANG_HOST_CACHE:-$CACHE_ROOT/tilelang}"
WORKER_TRITON_CACHE="${WORKER_TRITON_CACHE:-$WORKER_VLLM_CACHE/triton}"
WORKER_TILELANG_CACHE="${WORKER_TILELANG_CACHE:-$WORKER_VLLM_CACHE/tilelang}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.triton/cache}"
TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/root/.tilelang/cache}"

LOGDIR="$SCRIPT_DIR/logs"
HEAD_SCRIPT="$SCRIPT_DIR/.glm53-exl3-head.inner.sh"
WORKER_SCRIPT="$SCRIPT_DIR/.glm53-exl3-worker.inner.sh"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-120}"

# ------------------------------- helpers -----------------------------------
log()  { printf '\033[1;36m[glm53-exl3]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm53-exl3]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[glm53-exl3]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

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
    # $name is one of three fixed names below.
    # shellcheck disable=SC2163
    export "$name"
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
    # mixed-prefill gate v2 knobs (self-defaulting so the guard block runs standalone; 0 = feature off for the first two)
    GLM53_MIXED_PREFILL_WARM_TOKENS="${GLM53_MIXED_PREFILL_WARM_TOKENS:-3584}"
    GLM53_MIXED_PREFILL_MAX_WAIT_MS="${GLM53_MIXED_PREFILL_MAX_WAIT_MS:-1500}"
    GLM53_MIXED_PREFILL_LATE_CAP="${GLM53_MIXED_PREFILL_LATE_CAP:-512}"
    local _k _v
    local _max
    for _k in GLM53_MIXED_PREFILL_WARM_TOKENS GLM53_MIXED_PREFILL_MAX_WAIT_MS; do
        _v="${!_k}"
        [ "$_k" = GLM53_MIXED_PREFILL_WARM_TOKENS ] && _max=1000000 || _max=600000
        if ! [[ "$_v" =~ ^[0-9]+$ ]] || [ "${#_v}" -gt 7 ] || [ "$((10#$_v))" -gt "$_max" ]; then
            echo "$_k must be an integer between 0 and $_max (got: $_v)" >&2; return 2
        fi
    done
    _glm53_canonical_positive_int GLM53_MIXED_PREFILL_LATE_CAP "$GLM53_MIXED_PREFILL_LATE_CAP" 8192 || return
    if [ "$GLM53_MIXED_PREFILL_LATE_CAP" -lt 64 ]; then
        echo "GLM53_MIXED_PREFILL_LATE_CAP must be between 64 and 8192 (got: $GLM53_MIXED_PREFILL_LATE_CAP)" >&2; return 2
    fi
}
# GLM53 numeric config guard (end)

banner() {
    local label="${1:-start.sh}"
    printf '\n'
    printf '  \033[1;36m┌────────────────────────────────────────────┐\033[0m\n'
    printf '  \033[1;36m│\033[0m  \033[1mGLM-5.3 Flash EXL3\033[0m  \033[2m·  %-11s\033[0m        \033[1;36m│\033[0m\n' "$label"
    printf '  \033[1;36m└────────────────────────────────────────────┘\033[0m\n'
    printf '\n'
}

worker_ssh() { ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH" "$@"; }

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
            die "port ${port} is held by ${CONTAINER_HEAD} — use './start.sh restart' or './start.sh stop' first"
        fi
        die "port ${port} is already in use — stop it or rerun with ${envname}=<free-port>"
    fi
}

trap 'warn "interrupted — containers keep running ('"'"'./start.sh logs'"'"' to watch, '"'"'./start.sh stop'"'"' to stop)"; exit 130' INT

# ------------------------------ preflight ----------------------------------
preflight() {
    command -v docker  >/dev/null 2>&1 || die "docker not found on head"
    command -v curl    >/dev/null 2>&1 || die "curl not found on head"
    command -v rsync   >/dev/null 2>&1 || die "rsync not found on head"
    docker info >/dev/null 2>&1 || die "cannot talk to docker daemon on head"

    ip -4 addr show 2>/dev/null | grep -q "inet ${HEAD_IP}/" \
        || die "HEAD_IP=${HEAD_IP} is not assigned on this host — set it in .env"

    log "checking worker ${WORKER_SSH} ..."
    worker_ssh true 2>/dev/null \
        || die "cannot ssh (key-based) to ${WORKER_SSH} — set up passwordless ssh first"
    worker_ssh "docker info >/dev/null 2>&1" \
        || die "worker cannot talk to its docker daemon (docker group?)"
    worker_ssh "nvidia-smi -L 2>/dev/null | grep -q GB10" \
        || warn "no GB10 GPU visible on worker"

    # Each rank's GID index must name a populated entry on ITS OWN CX7 device.
    # An empty (all-zero) entry passes every earlier check and then kills that
    # rank ~60 s in with ibv_modify_qp errno 61 "No data available". The index is
    # per-NIC, so validate head and worker separately: some pairs share one good
    # index, others need different ones (HEAD_GID / WORKER_GID).
    local gid_head gid_worker gid_path
    gid_path="/sys/class/infiniband/${HEAD_CX7_IB}/ports/1/gids/${HEAD_GID}"
    gid_head=$(cat "$gid_path" 2>/dev/null | tr -d ':0' || true)
    gid_path="/sys/class/infiniband/${WORKER_CX7_IB}/ports/1/gids/${WORKER_GID}"
    gid_worker=$(worker_ssh "cat '$gid_path' 2>/dev/null" | tr -d ':0' || true)
    if [ -z "$gid_head" ] || [ -z "$gid_worker" ]; then
        if [ -z "$gid_head" ]; then
            warn "head GID index ${HEAD_GID} is EMPTY on ${HEAD_CX7_IB}"
        fi
        if [ -z "$gid_worker" ]; then
            warn "worker GID index ${WORKER_GID} is EMPTY on ${WORKER_CX7_IB}"
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

    [ "$TP" = "2" ] || warn "TP=${TP} on a 2×1-GPU cluster — expected TP=2"
    [ "$NNODES" = "2" ] || warn "NNODES=${NNODES} — expected 2"

    local others
    others=$(worker_ssh "docker ps --format '  {{.Names}}  ({{.Image}})'" 2>/dev/null | grep -v "^  ${CONTAINER_WORKER}" || true)
    if [ -n "$others" ]; then
        warn "other containers are running on the worker:"
        echo "$others" >&2
        warn "this model needs most of each GB10 — stop GPU containers on the worker first"
    fi

    check_port_free "$PORT" PORT
    check_port_free "$MASTER_PORT" MASTER_PORT

    [ -f "$STOP_PATCH_HOST" ] || die "$STOP_PATCH_HOST missing"
    [ -f "$SCHED_PATCH_HOST" ] || die "$SCHED_PATCH_HOST missing"
    [ -f "$DRAFTER_PATCH_HOST" ] || die "$DRAFTER_PATCH_HOST missing"
    [ -f "$APC_PATCH_HOST" ] || die "$APC_PATCH_HOST missing"
    [ -f "$XGRAMMAR_PATCH_HOST" ] || die "$XGRAMMAR_PATCH_HOST missing"
    [ -f "$KPOOL_TAIL_PATCH_HOST" ] || die "$KPOOL_TAIL_PATCH_HOST missing"
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
    avail=$(worker_ssh "df -Pk '$WORKER_HOME' 2>/dev/null" | awk 'NR==2{print $4}' || true)
    [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on worker for a ~164 GiB model"

    # The worker HF cache must be writable by the SSH user before the ~164 GiB
    # sync starts. A root-owned ~/.cache/huggingface (prior sudo/docker
    # prepare on the worker) otherwise fails mid-sync with a bare mkdir
    # permission error. mkdir -p is idempotent and is what sync does anyway.
    if ! worker_ssh "mkdir -p '$WORKER_CACHE_DIR/hub' && test -w '$WORKER_CACHE_DIR/hub'"; then
        die "worker cannot write $WORKER_CACHE_DIR/hub as $( [ -n "${WORKER_USER:-}" ] && echo "$WORKER_USER" || echo "$USER" ) — fix ownership on the worker, e.g.: ssh $WORKER_SSH \"sudo chown -R ${WORKER_USER:-\$USER}: '$WORKER_CACHE_DIR'\""
    fi

    log "preflight OK (head=$(hostname) ${HEAD_IP}, worker=${WORKER_SSH})"
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
    log "docker login ghcr.io on worker as ${GHCR_USER} (GHCR_TOKEN)"
    echo "$GHCR_TOKEN" | worker_ssh "docker login ghcr.io -u '$GHCR_USER' --password-stdin" >/dev/null
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
    worker_ssh "docker image inspect -f 'GLM53KEY ${_IMAGE_KEY_FMT}' '$IMAGE' 2>/dev/null" | parse_image_key
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

build_image() {
    log "building ${IMAGE} from Dockerfile (log: $LOGDIR/build-sm121.log) ..."
    docker build -t "$IMAGE" "$SCRIPT_DIR" \
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
  Or set GHCR_TOKEN + GHCR_USER in .env. Overlay rebuild: BUILD=1 ./start.sh"
}

pull_image_on_worker() {
    login_ghcr_if_token_worker
    log "pulling ${IMAGE} on worker ..."
    worker_ssh "docker pull '$IMAGE'"
}

ship_image_to_worker() {
    local platform
    platform="$(image_platform)"
    log "shipping ${IMAGE} (${platform}) to worker via docker save | ssh docker load ..."
    # A multi-arch OCI index references blobs docker save does not pack
    # (only the native platform is local). docker load then dies with:
    #   open /var/lib/docker/tmp/docker-import-*/blobs/sha256/<id>: no such file
    # (issue #8). --platform emits a complete single-manifest tar.
    if docker save --platform "$platform" "$IMAGE" | worker_ssh docker load; then
        return 0
    fi
    warn "docker save --platform ${platform} failed — retrying without --platform"
    docker save "$IMAGE" | worker_ssh docker load
}

ensure_image() {
    mkdir -p "$LOGDIR"
    local head_ok=0 worker_ok=0 head_key="" worker_key=""
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        head_ok=1
        head_key="$(local_image_key)"
    fi
    if worker_ssh "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
        worker_key="$(worker_image_key)"
        if images_match "$head_key" "$worker_key"; then
            worker_ok=1
        else
            worker_ok=0
            log "worker image differs (head=${head_key:-none} worker=${worker_key:-none}) — will refresh worker"
        fi
    fi
    local skip_pull="${SKIP_PULL:-0}"
    [ "${PULL:-0}" = "1" ] && skip_pull=0
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
        [ "$worker_ok" = "1" ] || warn "SKIP_SHIP=1 — not copying ${IMAGE} to the worker"
    elif [ "$worker_ok" = "0" ]; then
        if image_from_registry && [ "$skip_pull" != "1" ] && [ "${BUILD:-0}" != "1" ]; then
            if pull_image_on_worker; then
                worker_key="$(worker_image_key)"
                if images_match "$head_key" "$worker_key"; then
                    worker_ok=1
                    log "worker pulled ${IMAGE} — matches head"
                else
                    warn "worker pull left a different image (head=${head_key:-none} worker=${worker_key:-none}) — shipping"
                fi
            else
                warn "worker docker pull failed — shipping over SSH (worker does not need GHCR)"
            fi
        fi
        if [ "$worker_ok" = "0" ]; then
            ship_image_to_worker
            worker_key="$(worker_image_key)"
            if images_match "$head_key" "$worker_key"; then
                worker_ok=1
            elif worker_ssh "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
                warn "worker has ${IMAGE} after ship but keys still differ (head=${head_key:-none} worker=${worker_key:-none}) — continuing"
                worker_ok=1
            else
                die "worker still missing ${IMAGE} after ship"
            fi
        fi
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
    log "image ready on both nodes"
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
    log "worker was not touched. ./start.sh will rsync on launch unless SKIP_SYNC=1."
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

sync_repo_to_worker() {
    local src="$1" cache_name="$2" label="$3"
    local marker rev
    marker="${WORKER_CACHE_DIR}/hub/${cache_name}/.glm53-exl3-synced"
    rev="$(sync_repo_marker_rev "$src")"
    if [ "${FORCE_SYNC:-0}" != "1" ] \
       && [ "$(worker_ssh "cat '$marker' 2>/dev/null" || true)" = "$rev" ]; then
        log "worker ${cache_name} already at ${rev} — rsync skipped (FORCE_SYNC=1 to force)"
        return 0
    fi
    log "syncing ${label} to worker (first run moves ~164 GiB over the p2p link) ..."
    worker_ssh "mkdir -p '${WORKER_CACHE_DIR}/hub/${cache_name}'"
    rsync -a --partial --info=progress2 \
        "$src/" "${WORKER_SSH}:${WORKER_CACHE_DIR}/hub/${cache_name}/"
    worker_ssh "printf '%s' '$rev' > '$marker'"
}

sync_weights() {
    [ "${SKIP_SYNC:-0}" = "1" ] && { log "SKIP_SYNC=1 — not syncing to worker"; return; }
    [ -d "$MODEL_PATH" ] || die "weights missing at $MODEL_PATH — run without SKIP_DOWNLOAD first"
    sync_repo_to_worker "$MODEL_PATH" "$MODEL_CACHE_NAME" "weights"
    if [ "$SPEC_METHOD" = "dflash" ]; then
        [ -d "$DFLASH_PATH" ] || die "DFlash2 weights missing at $DFLASH_PATH"
        sync_repo_to_worker "$DFLASH_PATH" "$DFLASH_CACHE_NAME" "DFlash2 draft"
    fi
    log "worker weights in sync"
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
    --node-rank 1
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
if [ -f /opt/glm53/patch_ablit.py ]; then
    python3 /opt/glm53/patch_ablit.py
fi
if [ "${ABLIT:-0}" = "1" ]; then
    say "ablit: o_proj orthogonalization ON (method=${ABLIT_METHOD:-auto} direction=${ABLIT_DIRECTION:-dealign} layers=${ABLIT_LAYERS:-15-45} alpha=${ABLIT_ALPHA:-3.0})"
else
    say "ablit: off — stock o_proj weights"
fi
say "joining TP2 at ${HEAD_IP}:${MASTER_PORT} as rank 1"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
EOF
    chmod +x "$HEAD_SCRIPT" "$WORKER_SCRIPT"
}

# ------------------------------- launch ------------------------------------
launch_cluster() {
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || true
    worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 || true

    mkdir -p "$CACHE_ROOT" "$TRITON_HOST_CACHE" "$TILELANG_HOST_CACHE"
    worker_ssh "mkdir -p '$WORKER_VLLM_CACHE' '$WORKER_TRITON_CACHE' '$WORKER_TILELANG_CACHE'"
    scp -q -o BatchMode=yes "$WORKER_SCRIPT" "${WORKER_SSH}:/tmp/${CONTAINER_WORKER}.sh"
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

    worker_ssh "rm -rf /tmp/glm53-ablit"
    scp -q -r -o BatchMode=yes "$SCRIPT_DIR/ablit" "${WORKER_SSH}:/tmp/glm53-ablit"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/ablit_runtime.py" "${WORKER_SSH}:/tmp/glm53-ablit_runtime.py"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/patch_ablit.py" "${WORKER_SSH}:/tmp/patch_ablit.py"

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
        -e "GLM53_MIXED_PREFILL_WARM_TOKENS=$GLM53_MIXED_PREFILL_WARM_TOKENS"
        -e "GLM53_MIXED_PREFILL_MAX_WAIT_MS=$GLM53_MIXED_PREFILL_MAX_WAIT_MS"
        -e "GLM53_MIXED_PREFILL_LATE_CAP=$GLM53_MIXED_PREFILL_LATE_CAP"
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

    local -a head_preload=() worker_preload=""
    if [ "$USE_HOST_NCCL" = "1" ]; then
        if [ -f "$NCCL_HOST_DIR/$NCCL_SO_NAME" ]; then
            head_preload=(-v "$NCCL_HOST_DIR:/nccl:ro" -e "LD_PRELOAD=/nccl/$NCCL_SO_NAME")
            log "head: LD_PRELOAD $NCCL_SO_NAME"
        else
            warn "head: $NCCL_HOST_DIR/$NCCL_SO_NAME missing — using image NCCL"
        fi
        if worker_ssh "test -f '$WORKER_NCCL_HOST_DIR/$NCCL_SO_NAME'"; then
            worker_preload="-v '$WORKER_NCCL_HOST_DIR:/nccl:ro' -e LD_PRELOAD='/nccl/$NCCL_SO_NAME'"
            log "worker: LD_PRELOAD $NCCL_SO_NAME"
        else
            warn "worker: $WORKER_NCCL_HOST_DIR/$NCCL_SO_NAME missing — using image NCCL"
        fi
    fi

    local serve_env=""
    local v
    for v in SERVED_MODEL_NAME PORT TP NNODES HEAD_IP MASTER_PORT QUANTIZATION \
             MAX_MODEL_LEN GPU_MEM_UTIL MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
             KV_CACHE_DTYPE MTP_TOKENS SPEC_METHOD DFLASH_TOKENS DFLASH_MODEL_DIR \
             DFLASH_DRAFT_TP \
             LANGUAGE_MODEL_ONLY SKIP_MM_PROFILING \
             LIMIT_MM CHAT_TEMPLATE ENFORCE_EAGER EXL3_FUSED_MOE EXL3_MOE_ROW_TILE EXL3_TEMP_ROWS_FUSED MODEL_DIR EXTRA_ARGS \
             ABLIT ABLIT_METHOD ABLIT_DIRECTION ABLIT_LAYERS ABLIT_ALPHA ABLIT_INCLUDE_MTP; do
        serve_env+=" -e $v='${!v:-}'"
    done
    # VLLM_API_KEY is read by the head (rank 0) API server for bearer auth; the
    # worker runs --headless so it only needs the var for argv-parity, and
    # start.sh below passes it explicitly on the head. Keep it out of the
    # generic loop so the key never shows in process listings of either node
    # beyond the container env (same as the DeepSeek deployment).
    serve_env+=" -e VLLM_API_KEY='${VLLM_API_KEY:-}'"

    log "starting worker on ${WORKER_SSH} (NCCL if=${WORKER_CX7_IF} hca=${WORKER_CX7_IB}) ..."
    worker_ssh "docker run -d --name '$CONTAINER_WORKER' \
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v '$WORKER_CACHE_DIR:/root/.cache/huggingface' \
        -v '$WORKER_VLLM_CACHE:/root/.cache/vllm' \
        -v '$WORKER_TRITON_CACHE:/root/.triton/cache' \
        -v '$WORKER_TILELANG_CACHE:/root/.tilelang/cache' \
        -v '/tmp/${CONTAINER_WORKER}.sh:/start.sh:ro' \
        -v '/tmp/glm53-chat_template.jinja:${CHAT_TEMPLATE}:ro' \
        -v '/tmp/patch_glm_video_placeholders.py:/opt/glm53/patch_glm_video_placeholders.py:ro' \
        -v '/tmp/patch_suppress_stops_in_reasoning.py:/opt/glm53/patch_suppress_stops_in_reasoning.py:ro' \
        -v '/tmp/patch_scheduler_decode_floor.py:/opt/glm53/patch_scheduler_decode_floor.py:ro' \
        -v '/tmp/patch_glm5_drafter_group.py:/opt/glm53/patch_glm5_drafter_group.py:ro' \
        -v '/tmp/patch_hybrid_prefix_hit.py:/opt/glm53/patch_hybrid_prefix_hit.py:ro' \
        -v '/tmp/patch_xgrammar_termination.py:/opt/glm53/patch_xgrammar_termination.py:ro' \
        -v '/tmp/patch_kpool_tail_slotmap.py:/opt/glm53/patch_kpool_tail_slotmap.py:ro' \
        -v '/tmp/glm53-ablit:/opt/glm53/ablit:ro' \
        -v '/tmp/glm53-ablit_runtime.py:/opt/glm53/ablit_runtime.py:ro' \
        -v '/tmp/patch_ablit.py:/opt/glm53/patch_ablit.py:ro' \
        ${worker_preload} \
        ${worker_nccl} \
        -e NCCL_SOCKET_IFNAME='$WORKER_CX7_IF' \
        -e GLOO_SOCKET_IFNAME='$WORKER_CX7_IF' \
        -e NCCL_IB_HCA='$WORKER_CX7_IB' \
        -e NCCL_IB_GID_INDEX='$WORKER_GID' \
        -e VLLM_HOST_IP='$WORKER_IP' \
        ${serve_env} \
        --entrypoint bash '$IMAGE' /start.sh" >/dev/null

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

    log "containers up — head=${CONTAINER_HEAD}, worker=${CONTAINER_WORKER}"
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
    trap '_stop_logtail; warn "interrupted — containers keep running ('"'"'./start.sh logs'"'"' / '"'"'./start.sh stop'"'"')"; exit 130' INT
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
        if worker_ssh "docker inspect -f '{{.State.Running}}' '$CONTAINER_WORKER' 2>/dev/null" | grep -q true; then
            worker_fail=0
        else
            worker_fail=$((worker_fail + 1))
            if [ "$worker_fail" -ge 3 ]; then
                log "worker container '$CONTAINER_WORKER' not running on ${WORKER_SSH} (3 consecutive checks)"
                exited=1; dead_side="worker"; break
            fi
        fi
        sleep 10; elapsed=$((elapsed + 10))
    done

    _stop_logtail
    trap 'warn "interrupted — containers keep running ('"'"'./start.sh logs'"'"' / '"'"'./start.sh stop'"'"')"; exit 130' INT

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
        || warn "boot shape warmup incomplete — uncovered shapes may JIT mid-serve on TP=2"
}

collect_failure_logs() {
    mkdir -p "$LOGDIR"
    docker logs "$CONTAINER_HEAD" >"$LOGDIR/head.log" 2>&1 || true
    worker_ssh "docker logs '$CONTAINER_WORKER' 2>&1" >"$LOGDIR/worker.log" 2>&1 || true
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
    log "  manage     : ./start.sh status | ./start.sh logs | ./start.sh logs worker | ./start.sh stop"
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
    echo "---- last 40 lines of worker log ($LOGDIR/worker.log) ----"
    tail -n 40 "$LOGDIR/worker.log" || true
    die "server did not become healthy — full logs in $LOGDIR/"
}

# ------------------------------- stop --------------------------------------
stop() {
    log "stopping head container ..."
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || log "  (no head container was running)"
    log "stopping worker container on ${WORKER_SSH} ..."
    worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 \
        || log "  (no worker container was running)"
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
    log "worker (${CONTAINER_WORKER} on ${WORKER_SSH}):"
    worker_ssh "docker ps -a --filter name=${CONTAINER_WORKER} --format '  {{.Names}}  {{.Status}}'" 2>/dev/null \
        || log "  (worker unreachable)"
}

# ------------------------------- logs --------------------------------------
logs() {
    case "${1:-head}" in
        worker)
            log "following worker container logs on ${WORKER_SSH} ..."
            trap '' INT
            worker_ssh "docker logs -f --tail 100 '$CONTAINER_WORKER'" || true
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
        *)        banner start.sh ;;
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
