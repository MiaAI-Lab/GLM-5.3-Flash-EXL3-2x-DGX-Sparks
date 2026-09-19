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
#   2. image      — docker pull IMAGE from GHCR (public :exl3-instanttensor
#                   tag). If the
#                   worker is missing that digest, try docker pull there,
#                   then fall back to docker save --platform | ssh docker
#                   load (issue #8). SKIP_PULL=1 keeps a local copy.
#                   BUILD=1 rebuilds from this repo. A git pull that changes
#                   Dockerfile/overlay also rebuilds once (recipe stamp);
#                   SKIP_BUILD=1 keeps GHCR. Local-only tags (no slash) skip
#                   pull. SKIP_SHIP=1 never copies.
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
#   ./start.sh share              NFS_SHARE=1 only: re-export the head HF
#                                 cache and remount it on the worker
#
# Lifecycle commands on this checkout are serialized by a flock on
# logs/cluster.lock: start/restart refuse immediately when another lifecycle
# command owns it, and stop waits up to 30s and then exits 1 WITHOUT stopping
# anything — retry once the running command exits. The lock is per checkout
# and covers TP=2 only: another clone, manual docker commands and the
# start-tp3.sh / start-tp4.sh stacks are not serialized by it.
#
# Node IPs live in .env (copied from .env.example on first run).
# Handy overrides: SKIP_DOWNLOAD=1 SKIP_SYNC=1 SKIP_PULL=1 SKIP_SHIP=1 SKIP_BUILD=1 PULL=1 BUILD=1 TAIL=1 HF_TOKEN=...
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
# Caller exports, including explicit empties, must win over .env.
# Snapshot exports rather than parsing .env: it is sourced as shell code.
_caller_overrides=()
while IFS= read -r _k; do
    _flags="$(declare -p "$_k")"
    _flags="${_flags#declare -}"; _flags="${_flags%% *}"
    case "$_flags" in *r*) continue ;; esac
    if [ -n "${!_k+x}" ]; then _caller_overrides+=("$_k=${!_k}"); fi
done < <(compgen -e)
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a
# Stock o_proj unless the caller exported ABLIT. Only this flag is cleared;
# every other .env knob stays, including GLM53_APC_RETENTION_INTERVAL_SWA
# from #207. Use ABLIT=1 ./start.sh to opt in.
ABLIT=0
# Each entry is NAME=value; quoting preserves whitespace and empty values.
# shellcheck disable=SC2163
for _kv in ${_caller_overrides[@]+"${_caller_overrides[@]}"}; do export "$_kv"; done
unset _k _kv _flags _caller_overrides

# ----------------------------- configuration -------------------------------
MODEL="${MODEL:-Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw}"
# If the durable mirror is empty/moved, download.sh falls back to this id.
MODEL_FALLBACK="${MODEL_FALLBACK:-brandonmusic/GLM-5.3-Flash-tr3-4bpw}"
MODEL_CACHE_NAME="${MODEL_CACHE_NAME:-models--${MODEL//\//--}}"
MODEL_FALLBACK_CACHE_NAME="${MODEL_FALLBACK_CACHE_NAME:-models--${MODEL_FALLBACK//\//--}}"
# Hub commit on the Mia-AiLab mirror (the 5ab363a8-byte-identical upload).
MODEL_REVISION="${MODEL_REVISION:-25a44fdbf16862a46b7cc9921142c6c81350af2f}"
# Optional pinned-checkpoint preset (start-abliterated.sh). It pins repo,
# fallback, revision and inventory, and that exact snapshot is then required on
# every path: refs/main is not consulted for it, and a complete but different
# cached revision does not satisfy the checks.
MODEL_SNAPSHOT=""
MODEL_FALLBACK_SNAPSHOT=""
MODEL_PINNED_SHARDS=""
GLM53_MODEL_PRESET="${GLM53_MODEL_PRESET:-}"
case "$GLM53_MODEL_PRESET" in
    "") ;;
    abliterated)
        MODEL="bullerwins/GLM-5.3-Flash-exl3-4bpw-ablit"
        MODEL_FALLBACK="$MODEL"
        MODEL_REVISION="14858211ed81d7fa773f8a0db02f38f36d230252"
        MODEL_SNAPSHOT="$MODEL_REVISION"
        MODEL_FALLBACK_SNAPSHOT="$MODEL_SNAPSHOT"
        MODEL_CACHE_NAME="models--bullerwins--GLM-5.3-Flash-exl3-4bpw-ablit"
        MODEL_FALLBACK_CACHE_NAME="$MODEL_CACHE_NAME"
        # Published manifest at the pin: 120 safetensors shards (175642157752
        # bytes) plus config.json / model.safetensors.index.json / ABLIT_META.json.
        MODEL_PINNED_SHARDS=120
        ;;
    *)
        printf 'FATAL: unknown GLM53_MODEL_PRESET: %s\n' "$GLM53_MODEL_PRESET" >&2
        exit 2
        ;;
esac
IMAGE="${IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
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
# Empty = let NCCL pick the channel count. A positive integer pins both
# NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS on both ranks (Entrpi 2× Spark: 8).
NCCL_NCHANNELS="${NCCL_NCHANNELS:-}"
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
# Receipt-matched DFlash2 checkpoint used by the 2026-08-30 TP=2 results.
# A mutable Hub main has already changed weights, so fresh and warm installs
# must resolve the same snapshot unless the operator deliberately overrides it.
DFLASH_REVISION="${DFLASH_REVISION-dc77ff1c99eeb2df044ee3d4f0094eb033fee410}"
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
# 2 = shard the ~2.3 GiB DFlash2 drafter across TP (C4 keep, 2026-08-30:
# idle 8k 938 / 16k 972 / 100k 997; decode structured 65.1 / prose 27.1).
# 1 = rank 0 only (no CX7 on every draft step). Empty = inherit target TP.
# Do not pin attention_backend: SM121 already prefers FLASH_ATTN for
# non-causal dense SWA. TRITON_ATTN was an SM120 mask-fix this image lacks.
DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP-2}"
# 900k with the E3 grouped tier (default since 2026-09-07). One request needs ~7.4 GiB
# + 7.1 GiB per 1M tokens of KV at MNBT 7168; E3 keeps a 560 MiB fat-row scratch that
# vLLM charges to the KV budget, so 1M no longer fits at util <= 0.87 on this kit.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-850000}"
# 0.85 leaves ~2.4 GiB more host headroom than 0.87 (long prefills need it; a 256k
# prefill at 0.87 with zero MemAvailable crashed a head on 2026-09-06).
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# 8192 chunk × long history oversubscribes GB10 persistent_topk smem (300k crash).
# E2 one-shot 2026-09-01: 7168 keep (100k ~1148 / 300k ~1107); 2048/3548 similar or slower.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-7168}"
# Decode hygiene (issue #43): without a server default, omitted chat limits
# may consume the entire remaining context budget and grow KV enough to
# preempt other sessions. A bounded omitted-request default limits that risk.
# This is not a cap: overlay/patch_default_max_new_tokens.py changes only the serving
# layer's omitted-request fallback. Explicit client limits override this
# default; independently configured server/platform and context caps remain.
# Admission in this vLLM is chunk-based (allocate_slots per
# chunk), so this does NOT gate admission; long-context concurrency is
# governed by the effective KV pool (see issue #43 measurements).
# Unset-only expansion: explicit empty preserves stock model/server limits,
# and a caller export — including empty — beats .env.
# Two-node start.sh only; start-tp4.sh is unchanged.
DEFAULT_MAX_NEW_TOKENS="${DEFAULT_MAX_NEW_TOKENS-65536}"
# Empty preserves the stock scheduler; opt in after measuring contention.
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-}"
CHAT_TEMPLATE_HOST="${CHAT_TEMPLATE_HOST:-$SCRIPT_DIR/files/chat_template.jinja}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/opt/glm53/chat_template.jinja}"
VIDEO_PATCH_HOST="${VIDEO_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_glm_video_placeholders.py}"
STOP_PATCH_HOST="${STOP_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_suppress_stops_in_reasoning.py}"
SCHED_PATCH_HOST="${SCHED_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_scheduler_decode_floor.py}"
DRAFTER_PATCH_HOST="${DRAFTER_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_glm5_drafter_group.py}"
APC_PATCH_HOST="${APC_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_hybrid_prefix_hit.py}"
PERGROUP_PATCH_HOST="${PERGROUP_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_apc_per_group_retention.py}"
NOSTORE_PATCH_HOST="${NOSTORE_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_apc_no_store.py}"
KVCAP_PATCH_HOST="${KVCAP_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_kv_capacity_log.py}"
TOOLCHOICE_PATCH_HOST="${TOOLCHOICE_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_tool_choice_none.py}"
XGRAMMAR_PATCH_HOST="${XGRAMMAR_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_xgrammar_termination.py}"
CACHE_RESET_PATCH_HOST="${CACHE_RESET_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_cache_reset.py}"
COLD_LOAD_PATCH_HOST="${COLD_LOAD_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_cold_load_uma.py}"
KPOOL_TAIL_PATCH_HOST="${KPOOL_TAIL_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_kpool_tail_slotmap.py}"
SPINWAIT_PATCH_HOST="${SPINWAIT_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_spinwait.py}"
ADAPTIVE_K_PATCH_HOST="${ADAPTIVE_K_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_adaptive_k.py}"
DENSE_FP8_PATCH_HOST="${DENSE_FP8_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_dense_fp8.py}"
DEFAULT_TOKENS_PATCH_HOST="${DEFAULT_TOKENS_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_default_max_new_tokens.py}"
EXL3_OVERLAY_HOST="${EXL3_OVERLAY_HOST:-$SCRIPT_DIR/overlay/exl3.py}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# Direct-I/O safetensors on the published InstantTensor image. Unset follows
# IMAGE (*instanttensor* → on). Explicit empty (LOAD_FORMAT=) is vLLM auto.
# PREFIX_MATCH_UNIT empty = vLLM default hash grain.
if [ -z "${LOAD_FORMAT+x}" ]; then
    case "$IMAGE" in
        *instanttensor*) LOAD_FORMAT=instanttensor ;;
        *) LOAD_FORMAT= ;;
    esac
fi
PREFIX_MATCH_UNIT="${PREFIX_MATCH_UNIT:-}"
QUANTIZATION="${QUANTIZATION:-exl3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
# JSON default cannot sit in ${LIMIT_MM:-{...}} — } ends the expansion.
if [ -z "${LIMIT_MM:-}" ]; then
    LIMIT_MM='{"image":48,"video":1}'
fi
# Vision cost caps. 2026-09-14: a chat client split an 11.9 MB video into ~33
# frames and posted them as images. The checkpoint's processor_config.json
# allows max_image_tokens=8000, so each frame cost ~7.2k tokens and the prompt
# reached 236k tokens of vision encode. SKIP_MM_PROFILING reserves nothing for
# the tower, so the host OOM-killer took VLLM::Worker_TP and the engine died.
# ${VAR-default} not ${VAR:-default}: an explicitly empty value means stock vLLM.
#   MM_IMAGE_TOKENS        per-image token budget. Must stay <=
#                          MAX_NUM_BATCHED_TOKENS, which is also vLLM's encoder
#                          cache size — the checkpoint's 8000 exceeds our 7168.
#                          A 1080p frame is 2691 tokens uncapped, 2040 at 2048.
#   VIDEO_NUM_FRAMES       frames sampled from a real video_url item. Empty =
#                          vLLM's VideoMediaIO default (32). Untested here: the
#                          09-14 crash came in over the image path.
#   MM_PROCESSOR_CACHE_GB  host RAM held for processed media. vLLM defaults to
#                          4 GiB; on UMA that is 4 GiB the model cannot have.
MM_IMAGE_TOKENS="${MM_IMAGE_TOKENS-2048}"
VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES-}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB-1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
# Graph-safe fused apply (device-side expert grouping). MTP k=2 decode is
# 1..4 seqs × 3 tokens (must include 3). DFlash2 k=7 is 1..4 seqs × 8 tokens
# (must include 8, 16, 24, 32).
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
# Called only after start/restart configuration validation, before any stop.
configure_capture_sizes() {
    local capture_sizes
    if [ "${ENFORCE_EAGER}" != "1" ]; then
        # Accept both CLI spellings and shell whitespace without duplicating an override.
        if [[ ! " ${EXTRA_ARGS:-} " =~ [[:space:]](--)?cudagraph-capture-sizes([[:space:]]|=) ]]; then
            if [ "$SPEC_METHOD" = "dflash" ]; then
                # Match the runtime's mode normalization and boot-time query lengths.
                # Keep stock captures and add every enabled length at each batch size.
                capture_sizes="$(python3 -S -c '
import sys
mode, raw, tokens, seqs = sys.argv[1:]
sizes = {1, 2, 4, 8, 16, 24, 32}
if mode.strip().lower() in ("ema", "on", "1"):
    decode_query_len = int(tokens) + 1
    ks = {int(x) for x in raw.split(",") if x.strip()}
    lens = {k + 1 for k in ks if 0 < k + 1 <= decode_query_len}
    lens.add(decode_query_len)
    sizes.update(n * q for n in range(1, int(seqs) + 1) for q in lens)
print(" ".join(map(str, sorted(sizes))))
' "${GLM53_ADAPTIVE_K:-off}" "${GLM53_ADAPTIVE_K_SET:-2,4,7}" "$DFLASH_TOKENS" "$MAX_NUM_SEQS")"
                EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--cudagraph-capture-sizes $capture_sizes"
            else
                EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--cudagraph-capture-sizes 1 2 3 4 6 8 12"
            fi
        fi
    fi
}
# 1 = fused exl3_moe (decode). 0 restores the unique-expert LinearEXL3 loop.
EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"
# 1 = GPU row tiles for fat experts (prefill). 0 = LinearEXL3 fallback.
# Tile (P2a) and TEMP_ROWS=1024 (P2b) both lost at MNBT=1024 — leave 128.
EXL3_MOE_ROW_TILE="${EXL3_MOE_ROW_TILE:-0}"
# E3 grouped fat-expert kernels (default ON since 2026-09-07: +37-45% cold prefill):
# one gather + gate/up + down launch per layer for every fat expert from device-side
# tables, no host sync. Needs the exl3_fat_moe kernels in the image (fails closed at
# load otherwise; start.sh rebuilds when the recipe stamp drifts). 0 = the E2 kernel path.
EXL3_FAT_GROUPED="${EXL3_FAT_GROUPED:-1}"
# Fused exl3_moe temp rows/expert; experts above it are "fat". E3 wants 32 (>= MAX_NUM_SEQS
# x (DFLASH_TOKENS+1) so decode stays one graph-safe launch); E2 wants 256 (its per-expert
# loop is host-bound). 1024 was slower than 128+fallback (P2b). Explicit value always wins.
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
# safety anchors, MTP block included. 0 leaves checkpoint weights unchanged.
# Applied identically on both TP ranks; the DFlash2 drafter is never touched.
ABLIT="${ABLIT:-0}"
ABLIT_METHOD="${ABLIT_METHOD:-auto}"           # auto | transplant | proj
ABLIT_DIRECTION="${ABLIT_DIRECTION:-dealign}"  # dealign | bf_oproj | /path/dir.pt
ABLIT_LAYERS="${ABLIT_LAYERS:-15-45}"          # inclusive; 45 = checkpoint MTP block
ABLIT_ALPHA="${ABLIT_ALPHA:-3.0}"              # 1.0 = plain projection, >1 over-projects
ABLIT_INCLUDE_MTP="${ABLIT_INCLUDE_MTP:-1}"
# The preset's checkpoint already carries the o_proj transplant: applying the
# runtime edit again would produce a different model, so the preset forces
# ABLIT=0 over any caller value.
[ "$GLM53_MODEL_PRESET" = "abliterated" ] && ABLIT=0

READY_TIMEOUT="${READY_TIMEOUT:-3600}"
# 1 = suppress client stop strings until </think> (DSpark #42 class).
GLM53_SUPPRESS_STOPS_IN_REASONING="${GLM53_SUPPRESS_STOPS_IN_REASONING:-1}"
# Mixed-step prefill policy when a peer is already decoding (issue #6).
# fair = time-share mixing (default since 2026-09-15, overlay v5);
# skip = do not mix (starves waiting prefills until the decode ends);
# N>0 = cap mixed prefill tokens; 0 / off = no isolation.
# Fair knobs are forwarded on every rank even when CHUNK is not fair.
# fair v5: fixed+per-token step-cost fit, largest chunk that fits MAX_STEP_MS,
# prompt step-bounded newcomer probe, bounded contention credit, decode first.
GLM53_MIXED_PREFILL_CHUNK="${GLM53_MIXED_PREFILL_CHUNK:-fair}"
GLM53_FAIR_PREFILL_CHUNK="${GLM53_FAIR_PREFILL_CHUNK:-256}"
GLM53_FAIR_PREFILL_SHARE="${GLM53_FAIR_PREFILL_SHARE:-0.30}"
GLM53_FAIR_PREFILL_MAX_INTERVAL_MS="${GLM53_FAIR_PREFILL_MAX_INTERVAL_MS:-2000}"
GLM53_FAIR_PREFILL_MAX_STEP_MS="${GLM53_FAIR_PREFILL_MAX_STEP_MS:-2000}"
GLM53_FAIR_PREFILL_MAX_CHUNKS="${GLM53_FAIR_PREFILL_MAX_CHUNKS:-1}"
# Space-separated NAME=VALUE list of extra env for both container ranks (diagnostics, e.g. VLLM_DEBUG_WORKSPACE=1).
GLM53_EXTRA_ENV="${GLM53_EXTRA_ENV:-}"
# 1 = at boot, after vLLM's "GPU KV cache size: N tokens" line (which is
# max_concurrency x max_model_len, not a pool size), log one line per KV-cache
# group (spec, block_size, blocks per max_model_len request) and a summary with
# the usable block ids, the ids one aligned cached segment costs across groups
# and the resulting cached-conversation capacity (overlay
# patch_kv_capacity_log.py). 0 = do not log (one line saying so). Log-only:
# no serving behaviour changes either way. Default applies only when UNSET: an
# explicitly empty value is an operator error and validate_numeric_config
# rejects it.
GLM53_KV_CAPACITY_LOG="${GLM53_KV_CAPACITY_LOG-1}"
# Adaptive verification length (overlay/patch_adaptive_k.py). off = stock k=7 every step.
GLM53_ADAPTIVE_K="${GLM53_ADAPTIVE_K:-off}"
GLM53_ADAPTIVE_K_SET="${GLM53_ADAPTIVE_K_SET:-2,4,7}"
GLM53_ADAPTIVE_K_ALPHA="${GLM53_ADAPTIVE_K_ALPHA:-0.25}"
GLM53_ADAPTIVE_K_MARGIN="${GLM53_ADAPTIVE_K_MARGIN:-1.0}"
GLM53_ADAPTIVE_K_MIN_STEPS="${GLM53_ADAPTIVE_K_MIN_STEPS:-4}"
GLM53_ADAPTIVE_K_SATURATE="${GLM53_ADAPTIVE_K_SATURATE:-max}"
GLM53_ADAPTIVE_K_HIST="${GLM53_ADAPTIVE_K_HIST:-200}"
# Thin/small-M EXL3 routed-expert decode path (overlay/patch_exl3_decode_pipeline.py).
# 1 = opt-in SM121 K4/N256 kernels (frag-1/shared-8, optional gate/up transform
# reuse); 0 = stock exl3_moe kernels. Requires an image built from a tree that
# includes the decode-pipeline patch, otherwise model load fails closed.
GLM53_EXL3_MOE_FAST="${GLM53_EXL3_MOE_FAST-0}"
# Dense projections FP8 weight-only via Marlin (overlay/patch_dense_fp8.py). off = BF16 as shipped.
# PROVISIONAL (changes target numerics; needs a KLD panel). Groups: shared,dense,kda,mla.
GLM53_DENSE_FP8="${GLM53_DENSE_FP8:-off}"
# Cooperative MoE tile geometry (0 both-narrow, 1 both-wide, 2 A-wide/B-narrow).
# Empty uses the adapter default (1). Must be identical on both ranks and set
# before native prepare / CUDA-graph capture; it is not a live graph switch.
GLM53_COOP_GEOMETRY="${GLM53_COOP_GEOMETRY:-}"
# Empty leaves the template's omitted-effort fallback unchanged.
GLM53_DEFAULT_REASONING_EFFORT="${GLM53_DEFAULT_REASONING_EFFORT-}"
# 1 = honour the per-request GPU prefix-cache no-store flag
# (vllm_xargs {"skip_writing_prefix_cache": 1}; overlay patch_apc_no_store.py);
# 0 = ignore it (logged once). Requests never opt in by default, so 1 changes
# nothing until a client asks. Default applies only when UNSET: an explicitly
# empty value is an operator error and validate_numeric_config rejects it.
GLM53_APC_NO_STORE="${GLM53_APC_NO_STORE-1}"
# Sparse-indexer prefill gather workspace (overlay/patch_indexer_workspace.py).
# stock = max_model_len * 40 entries (5036.40 MB locked at 1M, measured);
# rightsize = the legal per-step maximum, ~+26% KV (default since 2026-09-07:
# the E3 recipe needs that KV back). Default applies only when UNSET: an
# explicitly empty value is an operator error and validate_numeric_config
# rejects it rather than guessing a serving mode.
GLM53_INDEXER_WORKSPACE="${GLM53_INDEXER_WORKSPACE-rightsize}"
# SpinCondition reader busy-loop window. "stock" preserves vLLM's 1 s default;
# 1..1000 selects milliseconds. The frozen TP=2 sweep selected 16 ms.
GLM53_SPINWAIT_MS="${GLM53_SPINWAIT_MS-stock}"
# EngineCore stock timeout is 300s; mid-serve Triton/TileLang JIT on TP=2 can
# exceed that without being a true hang. NCCL watchdog is still 600s.
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
# --- host memory hygiene before launch (docs/uvm-livelock-gb10.md) ----------
# On UMA the page cache is "used" device memory to the CUDA driver: after a
# 164 GiB rsync or a previous serve, MemFree is ~2 GiB and the InstantTensor
# loader either dies (buffer > budget) or shrinks io_depth to double digits.
# 1 = drop clean page cache + cycle swap on BOTH nodes (needs passwordless
# sudo; skipped with a warning otherwise), then wait until CUDA free memory
# clears GPU_MEM_UTIL x total (the driver returns a torn-down context a few
# GiB behind MemFree). Set 0 to leave the host alone.
GLM53_HOST_MEM_HYGIENE="${GLM53_HOST_MEM_HYGIENE:-1}"
# 1 = after /health, burn DFlash2 BLOCK / sampler / kpool shapes. Nonfatal.
GLM53_BOOT_SHAPE_WARMUP="${GLM53_BOOT_SHAPE_WARMUP:-1}"
GLM53_WARMUP_REQ_TIMEOUT="${GLM53_WARMUP_REQ_TIMEOUT:-240}"
# 1 = mount ONLY the cache-reset dev routes (/reset_prefix_cache, /reset_mm_cache,
# /reset_encoder_cache — issue #31) on the head API server, so cold bench runs
# can reset the prefix cache without a restart. Opt-in (default 0) on purpose:
# the caveat is auth, not stability — root-mounted routes sit outside the bearer
# guard (GUARDED_PREFIX), so a shared kit must ask for this explicitly.
# This flag does not enable other dev routes or override independent
# VLLM_SERVER_DEV_MODE, which retains precedence. Restart applies the flag.
GLM53_EXPOSE_CACHE_RESET="${GLM53_EXPOSE_CACHE_RESET:-0}"

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
# TP2 lifecycle lock (helpers below). start/restart take it without waiting;
# stop waits CLUSTER_LOCK_WAIT seconds and then refuses with exit 1.
CLUSTER_LOCK="$LOGDIR/cluster.lock"
CLUSTER_LOCK_PID="$LOGDIR/cluster.lock.pid"
CLUSTER_LOCK_WAIT=30
HEAD_SCRIPT="$SCRIPT_DIR/.glm53-exl3-head.inner.sh"
WORKER_SCRIPT="$SCRIPT_DIR/.glm53-exl3-worker.inner.sh"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-120}"
# Under a preset the pinned inventory wins: a caller EXPECTED_SHARDS must not
# lower the gate for one exact checkpoint.
[ -n "$MODEL_PINNED_SHARDS" ] && EXPECTED_SHARDS="$MODEL_PINNED_SHARDS"

# ------------------------------- helpers -----------------------------------
log()  { printf '\033[1;36m[glm53-exl3]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm53-exl3]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[glm53-exl3]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

# Worker weight distribution. 0 (default) = rsync a full copy of the ~164 GiB
# checkpoint to the worker, which is what this script has always done. 1 = the
# worker mounts the head's HF cache read-only over NFSv4 on ConnectX and keeps
# no copy (files/nfs-share.sh; same pattern as ~/NewModels/DS4.1). Opt in from
# .env — nothing below changes while it is 0.
NFS_SHARE="${NFS_SHARE:-0}"
NFS_RANKS="1"
# shellcheck source=files/nfs-share.sh
if [ -f "$SCRIPT_DIR/files/nfs-share.sh" ]; then
    source "$SCRIPT_DIR/files/nfs-share.sh"
elif [ "$NFS_SHARE" = "1" ]; then
    warn "files/nfs-share.sh missing — falling back to an rsync copy (NFS_SHARE=0)"
    NFS_SHARE=0
fi

# What the worker bind-mounts at /root/.cache/huggingface. Read-only over NFS is
# safe: the container runs HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 and its
# writable Triton/TileLang/vLLM caches are separate node-local mounts.
_hf_mount() {
    if [ "${NFS_SHARE:-0}" = "1" ]; then
        nfs_hf_mount_spec
    else
        printf '%s:/root/.cache/huggingface' "$WORKER_CACHE_DIR"
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

# Prefix-cache retention intervals are token counts on the scheduler-block
# grid. "" (unset = inherit the global policy) and 0 pass as-is; anything
# else must be a positive multiple of GLM53_APC_BLOCK_TOKENS no larger than
# GLM53_APC_RETENTION_MAX -- the same rule overlay/patch_apc_per_group_retention.py
# re-checks at coordinator init against the live scheduler_block_size. main()
# runs this guard before `restart` stops anything, so a typo is a launcher
# error with the healthy pair still serving, not a boot failure after the old
# containers are already gone. The canonical value (leading zeros stripped) is
# what both ranks receive.
GLM53_APC_BLOCK_TOKENS=3584
GLM53_APC_RETENTION_MAX=1000000
_glm53_validate_retention_interval() {
    local name="$1" value="$2" canonical
    [ -n "$value" ] || return 0
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$name must be empty, 0, or a positive multiple of $GLM53_APC_BLOCK_TOKENS <= $GLM53_APC_RETENTION_MAX (got: $value)" >&2
        return 2
    fi
    canonical="$value"
    while [ "${canonical#0}" != "$canonical" ]; do canonical="${canonical#0}"; done
    [ -n "$canonical" ] || canonical=0
    if [ "$canonical" != 0 ] \
       && { [ "${#canonical}" -gt "${#GLM53_APC_RETENTION_MAX}" ] \
            || [ "$canonical" -gt "$GLM53_APC_RETENTION_MAX" ] \
            || [ $((canonical % GLM53_APC_BLOCK_TOKENS)) -ne 0 ]; }; then
        echo "$name must be empty, 0, or a positive multiple of $GLM53_APC_BLOCK_TOKENS <= $GLM53_APC_RETENTION_MAX (got: $value)" >&2
        return 2
    fi
    printf -v "$name" '%s' "$canonical"
    # shellcheck disable=SC2163
    export "$name"
}

# Validate no-store and KV-capacity-log switches before restart stops anything;
# both overlays reject values other than exactly 0 or 1 at runtime.
_glm53_validate_bool_flag() {
    local name="$1" value="$2"
    if [ "$value" != 0 ] && [ "$value" != 1 ]; then
        echo "$name must be exactly 0 or 1 (got: $value)" >&2
        return 2
    fi
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

# skip / -1 / 0 / off / no / fair / positive integer <= MNBT.
# Companion fair knobs are validated when set so a typo cannot reach one rank.
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
    # Empty preserves stock limits; a set value must canonicalize
    # before restart stops the healthy pair.
    if [ -n "${DEFAULT_MAX_NEW_TOKENS:-}" ]; then
        _glm53_canonical_positive_int DEFAULT_MAX_NEW_TOKENS \
            "$DEFAULT_MAX_NEW_TOKENS" 1000000 || return
    fi
    _glm53_validate_enum GLM53_INDEXER_WORKSPACE "${GLM53_INDEXER_WORKSPACE-rightsize}" \
        stock rightsize || return
    _glm53_validate_bool_flag GLM53_EXL3_MOE_FAST "${GLM53_EXL3_MOE_FAST-0}" || return
    _glm53_validate_spinwait_ms || return
    _glm53_validate_bool_flag GLM53_APC_NO_STORE "${GLM53_APC_NO_STORE-1}" || return
    _glm53_validate_bool_flag GLM53_HOST_MEM_HYGIENE "$GLM53_HOST_MEM_HYGIENE" || return
    _glm53_validate_bool_flag GLM53_KV_CAPACITY_LOG "${GLM53_KV_CAPACITY_LOG-1}" || return
    # The template treats medium as max, so do not advertise it as a level.
    if [ -n "${GLM53_DEFAULT_REASONING_EFFORT-}" ]; then
        _glm53_validate_enum GLM53_DEFAULT_REASONING_EFFORT \
            "$GLM53_DEFAULT_REASONING_EFFORT" low high max || return
    fi
    _glm53_validate_mixed_prefill || return
    _glm53_validate_retention_interval GLM53_APC_RETENTION_INTERVAL "${GLM53_APC_RETENTION_INTERVAL-}" || return
    _glm53_validate_retention_interval GLM53_APC_RETENTION_INTERVAL_SWA "${GLM53_APC_RETENTION_INTERVAL_SWA-}" || return
    if [ -n "${GLM53_APC_RETENTION_INTERVAL_SWA:-}" ] && [ "$SPEC_METHOD" != "dflash" ]; then
        echo "GLM53_APC_RETENTION_INTERVAL_SWA requires SPEC_METHOD=dflash (got: $SPEC_METHOD)" >&2
        return 2
    fi
    if [ -n "${GLM53_COOP_GEOMETRY:-}" ]; then
        _glm53_validate_enum GLM53_COOP_GEOMETRY "$GLM53_COOP_GEOMETRY" 0 1 2 || return
    fi
}
# GLM53 numeric config guard (end)

# GLM53 overlay artifact guard (begin)
# Every file both rank containers mount. main() runs validate_overlay_artifacts
# on start|restart BEFORE `restart` stops anything: a missing, empty,
# mis-pointed, truncated or syntactically broken input is a launcher error
# with the healthy pair left serving, not a container that dies at boot after
# the old one is gone. Python overlays: path|identity string|last line -
# the identity string (MARK / target path / hook marker) is distinct per file
# so a *_PATCH_HOST pointed at a different overlay is caught, and the last
# non-blank line must match exactly so a copy truncated anywhere before EOF
# (even one that still parses) is caught too. This guards against operator
# error (wrong path, stale checkout, truncated copy); it is not a
# tamper-proof manifest. Needs python3 on the head (DGX OS ships it).
# preflight() re-checks existence later; this is the fail-closed early gate.
# The chat-template parse below needs jinja2 on the host. The caller's
# `python3` can be a venv/brew interpreter without it, so probe the caller
# first, then common system interpreters. An explicit override is the sole
# candidate (one executable name/path, no arguments); even empty is an error.
_glm53_template_python() {
    local candidate
    local -a candidates=(python3 python3.12 python3.11 /usr/bin/python3)
    if [ "${GLM53_VALIDATE_PYTHON+x}" = x ]; then
        candidates=("$GLM53_VALIDATE_PYTHON")
    fi
    for candidate in "${candidates[@]}"; do
        [ -n "$candidate" ] || continue
        command -v "$candidate" >/dev/null 2>&1 || continue
        if "$candidate" -c 'import jinja2' >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

validate_overlay_artifacts() {
    # Sentinels that contain quotes live in single-quoted locals.
    local main_guard='    sys.exit(main())'
    local video_end='    print("glm53: overlay install ok aligned=True", file=sys.stderr)'
    local ablit_marker='MARKER = "ABLIT-HOOK"'
    local -a artifacts=(
        "$EXL3_OVERLAY_HOST|class Exl3Config(QuantizationConfig):|        )"
        "$VIDEO_PATCH_HOST|vllm/model_executor/layers/|$video_end"
        "$STOP_PATCH_HOST|[suppress-stops-in-reasoning]|    raise SystemExit(main(sys.argv))"
        "$SCHED_PATCH_HOST|[glm53-decode-floor]|$main_guard"
        "$DRAFTER_PATCH_HOST|vllm/v1/core/kv_cache_utils.py|$main_guard"
        "$APC_PATCH_HOST|[glm53-hybrid-apc]|$main_guard"
        "$PERGROUP_PATCH_HOST|glm53-apc-per-group-contract:explicit-v1|$main_guard"
        "$NOSTORE_PATCH_HOST|[glm53-apc-no-store]|$main_guard"
        "$KVCAP_PATCH_HOST|[glm53-kv-capacity-log]|$main_guard"
        "$TOOLCHOICE_PATCH_HOST|[glm53-tool-choice-none]|$main_guard"
        "$XGRAMMAR_PATCH_HOST|vllm/v1/structured_output/|$main_guard"
        "$KPOOL_TAIL_PATCH_HOST|[glm53-kpool-tail-slotmap]|$main_guard"
        "$SPINWAIT_PATCH_HOST|device_communicators/shm_broadcast.py|$main_guard"
        "$ADAPTIVE_K_PATCH_HOST|[glm53-adaptive-k]|$main_guard"
        "$DENSE_FP8_PATCH_HOST|[glm53-dense-fp8]|$main_guard"
        "$DEFAULT_TOKENS_PATCH_HOST|[glm53-default-max-new-tokens]|    raise SystemExit(main(sys.argv))"
        "$CACHE_RESET_PATCH_HOST|# [glm53-cache-reset]|$main_guard"
        "$COLD_LOAD_PATCH_HOST|[glm53-cold-load-uma:v1]|    sys.exit(main())"
        "$SCRIPT_DIR/overlay/patch_ablit.py|$ablit_marker|    main()"
        "$SCRIPT_DIR/overlay/ablit_runtime.py|o_proj abliteration (ABLIT)|    return report"
    )
    local entry path rest tag tail last stock_last
    if [ "${#artifacts[@]}" -eq 0 ]; then
        echo "overlay artifact list is empty - refusing to launch" >&2
        return 2
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 is required on the head to verify overlay artifacts before launch" >&2
        return 2
    fi
    for entry in "${artifacts[@]}"; do
        path="${entry%%|*}"
        rest="${entry#*|}"
        tag="${rest%%|*}"
        tail="${rest#*|}"
        if [ ! -f "$path" ] || [ ! -r "$path" ] || [ ! -s "$path" ]; then
            echo "overlay artifact missing, unreadable or empty: $path" >&2
            return 2
        fi
        if ! grep -qF -- "$tag" "$path"; then
            echo "overlay artifact $path does not carry its identity string '$tag' (wrong file?)" >&2
            return 2
        fi
        # `|| true`: under pipefail a whitespace-only file makes grep exit 1,
        # which must surface as the rc=2 diagnostic below, not a bare exit 1.
        last="$(grep -v '^[[:space:]]*$' "$path" | tail -n 1 || true)"
        if [ "$last" != "$tail" ]; then
            # Generated cooperative overlay appends an install footer after stock
            # Exl3Config. Still require the stock closer in the body so a truncated
            # copy cannot hide behind the footer.
            if [ "$path" = "$EXL3_OVERLAY_HOST" ] && [[ "$last" == _coop_setup\[\"install\"\]* ]]; then
                stock_last="$(python3 -c 'import sys
p=[ln.rstrip("\n") for ln in open(sys.argv[1], encoding="utf-8")]
while p and (not p[-1].strip() or p[-1].startswith("_coop_") or p[-1].startswith("import runpy as _coop") or p[-1].startswith("import sys as _coop") or p[-1].startswith("# Explicit fixed cooperative")):
    p.pop()
print(p[-1] if p else "")
' "$path")"
                if [ "$stock_last" != "$tail" ]; then
                    echo "overlay artifact $path cooperative footer is present but stock closer is '$stock_last' (truncated copy?)" >&2
                    return 2
                fi
            else
                echo "overlay artifact $path does not end with '$tail' (truncated copy? last line: '$last')" >&2
                return 2
            fi
        fi
        # Parse-only: proves the file is importable Python without executing it
        # or leaving __pycache__ litter in the checkout.
        if ! python3 -c 'import ast, sys; ast.parse(open(sys.argv[1], encoding="utf-8").read(), sys.argv[1])' "$path" 2>/dev/null; then
            echo "overlay artifact does not parse as Python: $path" >&2
            return 2
        fi
    done
    # Non-Python inputs both ranks mount: the chat template and the ablit
    # layer map (the .pt payloads are ABLIT's own concern at hook time).
    if [ ! -f "$CHAT_TEMPLATE_HOST" ] || [ ! -r "$CHAT_TEMPLATE_HOST" ] || [ ! -s "$CHAT_TEMPLATE_HOST" ]; then
        echo "chat template missing, unreadable, empty or not a regular file: $CHAT_TEMPLATE_HOST" >&2
        return 2
    fi
    local template_python
    if ! template_python="$(_glm53_template_python)"; then
        if [ "${GLM53_VALIDATE_PYTHON+x}" = x ]; then
            echo "GLM53_VALIDATE_PYTHON must name an executable Python with jinja2 (no fallback): ${GLM53_VALIDATE_PYTHON}" >&2
        else
            echo "no host python3 with jinja2 found (chat-template validation; set GLM53_VALIDATE_PYTHON)" >&2
        fi
        return 2
    fi
    if ! "$template_python" -c 'from jinja2 import Environment; import sys; Environment(extensions=["jinja2.ext.loopcontrols"]).parse(open(sys.argv[1], encoding="utf-8").read())' "$CHAT_TEMPLATE_HOST" 2>/dev/null; then
        echo "chat template is invalid: $CHAT_TEMPLATE_HOST" >&2
        return 2
    fi
    if ! python3 -c 'import json, sys; json.load(open(sys.argv[1], encoding="utf-8"))' "$SCRIPT_DIR/ablit/LAYER_MAP.json" 2>/dev/null; then
        echo "ablit layer map missing or not JSON: $SCRIPT_DIR/ablit/LAYER_MAP.json" >&2
        return 2
    fi
}
# GLM53 overlay artifact guard (end)

# Serialize the TP2 lifecycle commands (start / restart / stop) so a second
# launcher cannot docker-rm the first's containers mid wait_for_health (false
# "head container exited" + empty logs). The flock on $CLUSTER_LOCK is the
# authoritative owner; $CLUSTER_LOCK_PID is advisory diagnostics only — never
# proof of ownership, and no PID is ever signalled.
with_cluster_lock() {
    mkdir -p "$LOGDIR"
    exec 9>"$CLUSTER_LOCK"
    if ! flock -n 9; then
        local holder
        holder="$(tr -d '[:space:]' <"$CLUSTER_LOCK_PID" 2>/dev/null || true)"
        die "another start.sh/restart is already running${holder:+ (pid $holder)} — retry after it exits"
    fi
    echo $$ >"$CLUSTER_LOCK_PID" 2>/dev/null || true
}

# stop waits up to CLUSTER_LOCK_WAIT for the same lock and then refuses with
# exit 1: no container is stopped, no PID metadata is written, and the lock
# file is left alone. Retry once the start/restart holding it exits.
with_cluster_lock_for_stop() {
    mkdir -p "$LOGDIR"
    exec 9>"$CLUSTER_LOCK"
    if ! flock -w "$CLUSTER_LOCK_WAIT" 9; then
        local holder
        holder="$(tr -d '[:space:]' <"$CLUSTER_LOCK_PID" 2>/dev/null || true)"
        die "cluster lock still held after ${CLUSTER_LOCK_WAIT}s${holder:+ (pid $holder)} — nothing was stopped; retry once that start.sh/restart exits"
    fi
    echo $$ >"$CLUSTER_LOCK_PID" 2>/dev/null || true
}

banner() {
    local label="${1:-start.sh}"
    printf '\n'
    printf '  \033[1;36m┌────────────────────────────────────────────┐\033[0m\n'
    printf '  \033[1;36m│\033[0m  \033[1mGLM-5.3 Flash EXL3\033[0m  \033[2m·  %-11s\033[0m        \033[1;36m│\033[0m\n' "$label"
    printf '  \033[1;36m└────────────────────────────────────────────┘\033[0m\n'
    printf '\n'
}

worker_ssh() { ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH" "$@"; }

# Print the whole header block: everything between the shebang and
# `set -euo pipefail`, minus the ==== rulers. Beats a magic line number, which
# silently truncated ./start.sh status/logs/share out of --help.
usage() {
    sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" \
        | sed -e '/^set -euo pipefail/d' -e '/^# =\{10,\}$/d' -e 's/^# \{0,1\}//'
}

count_shards() {
    local repo_path="$1" snapshot="${2:-}" ref
    if [ -n "$snapshot" ]; then
        ref="$snapshot"
    else
        ref="$(cat "$repo_path/refs/main" 2>/dev/null || true)"
        [ -n "$ref" ] || ref="$(ls -1t "$repo_path/snapshots" 2>/dev/null | head -n 1 || true)"
    fi
    if [ -z "$ref" ]; then
        printf '0'
        return
    fi
    # -L + -type f: a shard entry counts only when the link resolves to a real
    # regular file, so dangling links and directories are not counted.
    find -L "$repo_path/snapshots/$ref" -maxdepth 1 -type f -name '*.safetensors' 2>/dev/null \
        | wc -l | tr -d '[:space:]' || true
}

# Completeness of the selected snapshot: the shard count (count_shards follows
# shard links to their blobs) plus the two loader-required sidecars. A preset
# passes its pinned revision, so an unrelated complete revision cannot satisfy
# it; the required count is EXPECTED_SHARDS, which the preset pins to its own
# inventory.
model_tree_complete() {
    local repo_path="$1" snapshot="${2:-}" have
    have="$(count_shards "$repo_path" "$snapshot")"
    [ "${have:-0}" -ge "$EXPECTED_SHARDS" ] || return 1
    [ -z "$snapshot" ] && return 0
    [ -f "$repo_path/snapshots/$snapshot/config.json" ] \
        && [ -f "$repo_path/snapshots/$snapshot/model.safetensors.index.json" ]
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

# Fail-closed gate for every pinned path: a preset resolves, syncs and serves
# exactly its own snapshot or the launch stops.
require_model_snapshot() {
    [ -n "$MODEL_SNAPSHOT" ] || return 0
    model_tree_complete "$MODEL_PATH" "$MODEL_SNAPSHOT" \
        || die "pinned model snapshot is incomplete: $MODEL_PATH/snapshots/$MODEL_SNAPSHOT"
}

resolve_model_dir() {
    local ref="$MODEL_PATH/refs/main" hash dir
    if [ -n "$MODEL_SNAPSHOT" ]; then
        require_model_snapshot
        hash="$MODEL_SNAPSHOT"
    else
        ensure_refs_main
        hash="$(<"$ref")"
    fi
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
    if [ -n "${DFLASH_REVISION:-}" ]; then
        hash="$DFLASH_REVISION"
    else
        ensure_dflash_refs_main
        hash="$(<"$ref")"
    fi
    dir="$DFLASH_PATH/snapshots/$hash"
    [ -f "$dir/config.json" ] || die "DFlash2 config.json missing in $dir"
    [ -f "$dir/model.safetensors" ] || die "DFlash2 model.safetensors missing in $dir"
    printf '/root/.cache/huggingface/hub/%s/snapshots/%s' "$DFLASH_CACHE_NAME" "$hash"
}

check_port_free() {
    local port="$1" envname="$2"
    command -v ss >/dev/null 2>&1 || return 0
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
        # Do not pipe docker inspect into grep -q: pipefail can turn grep's
        # early close into a false negative when docker gets SIGPIPE.
        if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null || true)" = "true" ]; then
            die "port ${port} is held by ${CONTAINER_HEAD} — use './start.sh restart' or './start.sh stop' first"
        fi
        die "port ${port} is already in use — stop it or rerun with ${envname}=<free-port>"
    fi
}

# GLM53 preflight memory guard (begin)
read_meminfo_kib() {
    local source_file="${1:-/proc/meminfo}"
    awk '
      /^MemTotal:/ { total=$2 }
      /^MemAvailable:/ { available=$2 }
      END {
        if (!total || !available) exit 1
        print total, available
      }
    ' "$source_file"
}

preflight_memory() {
    local label="$1" total_kib="$2" available_kib="$3" util="$4"
    local headroom_kib="${GLM53_PREFLIGHT_MEMORY_HEADROOM_KIB:-2097152}"
    local total_gib available_gib requested_gib headroom_gib

    if ! [[ "$total_kib" =~ ^[0-9]+$ && "$available_kib" =~ ^[0-9]+$ && "$headroom_kib" =~ ^[0-9]+$ ]]; then
        echo "PREFLIGHT FAIL [$label]: invalid memory reading" >&2
        return 2
    fi
    if ! [[ "$util" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0+)?)$ ]] \
       || ! awk -v u="$util" 'BEGIN { exit !(u > 0 && u <= 1) }'; then
        echo "PREFLIGHT FAIL [$label]: GPU_MEM_UTIL must be greater than 0 and at most 1: $util" >&2
        return 2
    fi

    total_gib=$(awk -v k="$total_kib" 'BEGIN { printf "%.2f", k/1048576 }')
    available_gib=$(awk -v k="$available_kib" 'BEGIN { printf "%.2f", k/1048576 }')
    requested_gib=$(awk -v t="$total_kib" -v u="$util" 'BEGIN { printf "%.2f", (t*u)/1048576 }')
    headroom_gib=$(awk -v k="$headroom_kib" 'BEGIN { printf "%.2f", k/1048576 }')

    if ! awk -v a="$available_kib" -v t="$total_kib" -v u="$util" -v h="$headroom_kib" \
        'BEGIN { exit !(a >= (t*u)+h) }'; then
        echo "PREFLIGHT FAIL [$label]: MemAvailable=${available_gib} GiB of ${total_gib} GiB; GPU_MEM_UTIL=${util} requests ${requested_gib} GiB plus ${headroom_gib} GiB headroom" >&2
        return 1
    fi
    echo "PREFLIGHT OK [$label]: MemAvailable=${available_gib}/${total_gib} GiB; request=${requested_gib} GiB plus ${headroom_gib} GiB headroom"
}
# GLM53 preflight memory guard (end)

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

    # Each rank's GID index must be populated on EVERY selected CX7 device.
    # HEAD_CX7_IB / WORKER_CX7_IB are literal names or comma-separated lists;
    # pass the original values unchanged to NCCL below.
    local gid_head=ok gid_worker=ok gid_path hca i
    local -a head_hcas worker_hcas
    IFS=, read -r -a head_hcas <<< "$HEAD_CX7_IB"
    IFS=, read -r -a worker_hcas <<< "$WORKER_CX7_IB"
    for hca in "${head_hcas[@]}"; do
        gid_path="/sys/class/infiniband/${hca}/ports/1/gids/${HEAD_GID}"
        if [ -z "$(cat "$gid_path" 2>/dev/null | tr -d ':0' || true)" ]; then
            gid_head=""
            warn "head GID index ${HEAD_GID} is EMPTY on ${hca}"
        fi
    done
    for hca in "${worker_hcas[@]}"; do
        gid_path="/sys/class/infiniband/${hca}/ports/1/gids/${WORKER_GID}"
        if [ -z "$(worker_ssh "cat '$gid_path' 2>/dev/null" | tr -d ':0' || true)" ]; then
            gid_worker=""
            warn "worker GID index ${WORKER_GID} is EMPTY on ${hca}"
        fi
    done
    if [ -z "$gid_head" ] || [ -z "$gid_worker" ]; then
        warn "GID tables — pick each node's ::ffff:<ip> entry whose type is RoCE v2;"
        warn "the two indices need not match, and a v1 entry at the same index will not work:"
        for hca in "${head_hcas[@]}"; do
            for i in 0 1 2 3 4 5 6 7; do
                printf '    head   %s gid%s: %-40s %s\n' "$hca" "$i" \
                    "$(cat "/sys/class/infiniband/${hca}/ports/1/gids/$i" 2>/dev/null)" \
                    "$(cat "/sys/class/infiniband/${hca}/ports/1/gid_attrs/types/$i" 2>/dev/null)" >&2
            done
        done
        for hca in "${worker_hcas[@]}"; do
            worker_ssh "for i in 0 1 2 3 4 5 6 7; do printf '    worker %s gid%s: %-40s %s\n' '${hca}' \"\$i\" \"\$(cat /sys/class/infiniband/${hca}/ports/1/gids/\$i 2>/dev/null)\" \"\$(cat /sys/class/infiniband/${hca}/ports/1/gid_attrs/types/\$i 2>/dev/null)\"; done" >&2 || true
        done
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

    local head_mem worker_mem head_total head_available worker_total worker_available
    head_mem="$(read_meminfo_kib /proc/meminfo)" \
        || die "cannot read MemTotal/MemAvailable on head"
    worker_mem="$(worker_ssh "cat /proc/meminfo" | read_meminfo_kib /dev/stdin)" \
        || die "cannot read MemTotal/MemAvailable on worker"
    read -r head_total head_available <<< "$head_mem"
    read -r worker_total worker_available <<< "$worker_mem"
    preflight_memory head "$head_total" "$head_available" "$GPU_MEM_UTIL" || return
    preflight_memory worker "$worker_total" "$worker_available" "$GPU_MEM_UTIL" || return

    [ -f "$STOP_PATCH_HOST" ] || die "$STOP_PATCH_HOST missing"
    [ -f "$SCHED_PATCH_HOST" ] || die "$SCHED_PATCH_HOST missing"
    [ -f "$DRAFTER_PATCH_HOST" ] || die "$DRAFTER_PATCH_HOST missing"
    [ -f "$APC_PATCH_HOST" ] || die "$APC_PATCH_HOST missing"
    [ -f "$PERGROUP_PATCH_HOST" ] || die "$PERGROUP_PATCH_HOST missing"
    [ -f "$NOSTORE_PATCH_HOST" ] || die "$NOSTORE_PATCH_HOST missing"
    [ -f "$KVCAP_PATCH_HOST" ] || die "$KVCAP_PATCH_HOST missing"
    [ -f "$TOOLCHOICE_PATCH_HOST" ] || die "$TOOLCHOICE_PATCH_HOST missing"
    [ -f "$XGRAMMAR_PATCH_HOST" ] || die "$XGRAMMAR_PATCH_HOST missing"
    [ -f "$CACHE_RESET_PATCH_HOST" ] || die "$CACHE_RESET_PATCH_HOST missing"
    [ -f "$COLD_LOAD_PATCH_HOST" ] || die "$COLD_LOAD_PATCH_HOST missing"
    [ -f "$KPOOL_TAIL_PATCH_HOST" ] || die "$KPOOL_TAIL_PATCH_HOST missing"
    [ -f "$SPINWAIT_PATCH_HOST" ] || die "$SPINWAIT_PATCH_HOST missing"
    [ -f "$ADAPTIVE_K_PATCH_HOST" ] || die "$ADAPTIVE_K_PATCH_HOST missing"
    [ -f "$DENSE_FP8_PATCH_HOST" ] || die "$DENSE_FP8_PATCH_HOST missing"
    [ -f "$DEFAULT_TOKENS_PATCH_HOST" ] || die "$DEFAULT_TOKENS_PATCH_HOST missing"
    [ -f "$EXL3_OVERLAY_HOST" ] || die "$EXL3_OVERLAY_HOST missing"
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
        log "NFS_SHARE=1 — worker reads the head HF cache, no local copy to size for"
    else
        avail=$(worker_ssh "df -Pk '$WORKER_HOME' 2>/dev/null" | awk 'NR==2{print $4}' || true)
        [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on worker for a ~164 GiB model"

        # The worker HF cache must be writable by the SSH user before the ~164 GiB
        # sync starts. A root-owned ~/.cache/huggingface (prior sudo/docker
        # prepare on the worker) otherwise fails mid-sync with a bare mkdir
        # permission error. mkdir -p is idempotent and is what sync does anyway.
        if ! worker_ssh "mkdir -p '$WORKER_CACHE_DIR/hub' && test -w '$WORKER_CACHE_DIR/hub'"; then
            die "worker cannot write $WORKER_CACHE_DIR/hub as $( [ -n "${WORKER_USER:-}" ] && echo "$WORKER_USER" || echo "$USER" ) — fix ownership on the worker, e.g.: ssh $WORKER_SSH \"sudo chown -R ${WORKER_USER:-\$USER}: '$WORKER_CACHE_DIR'\""
        fi
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
  :exl3-instanttensor is a public GHCR package — check network / disk.
  If you still get 401/403: echo YOUR_PAT | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin
  Overlay rebuild: BUILD=1 ./start.sh. Recipe-stamp drift also rebuilds; SKIP_BUILD=1 keeps GHCR."
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
        head_key="$(local_image_key || true)"
    fi
    if worker_ssh "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
        worker_key="$(worker_image_key || true)"
        if images_match "$head_key" "$worker_key"; then
            worker_ok=1
        else
            worker_ok=0
            log "worker image differs (head=${head_key:-none} worker=${worker_key:-none}) — will refresh worker"
        fi
    fi
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
        [ "$worker_ok" = "1" ] || warn "SKIP_SHIP=1 — not copying ${IMAGE} to the worker"
    elif [ "$worker_ok" = "0" ]; then
        if image_from_registry && [ "$skip_pull" != "1" ] && [ "${BUILD:-0}" != "1" ]; then
            if pull_image_on_worker; then
                worker_key="$(worker_image_key || true)"
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
            worker_key="$(worker_image_key || true)"
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
    have="$(count_shards "$MODEL_PATH" "$MODEL_SNAPSHOT")"
    if model_tree_complete "$MODEL_PATH" "$MODEL_SNAPSHOT"; then
        [ -n "$MODEL_SNAPSHOT" ] || ensure_refs_main
        log "weights already present: $MODEL_PATH ($have shards)"
        return 0
    fi
    have="$(count_shards "$FALLBACK_MODEL_PATH" "$MODEL_FALLBACK_SNAPSHOT")"
    if model_tree_complete "$FALLBACK_MODEL_PATH" "$MODEL_FALLBACK_SNAPSHOT"; then
        log "primary cache incomplete — using fallback ${MODEL_FALLBACK} at $FALLBACK_MODEL_PATH ($have shards)"
        MODEL_PATH="$FALLBACK_MODEL_PATH"
        MODEL_CACHE_NAME="$MODEL_FALLBACK_CACHE_NAME"
        MODEL_SNAPSHOT="$MODEL_FALLBACK_SNAPSHOT"
        [ -n "$MODEL_SNAPSHOT" ] || ensure_refs_main
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
        || die "download finished with $(count_shards "$MODEL_PATH" "$MODEL_SNAPSHOT") / $EXPECTED_SHARDS shards${MODEL_SNAPSHOT:+ in the selected snapshot}"
}

download_dflash() {
    [ "$SPEC_METHOD" = "dflash" ] || return 0
    [ "${SKIP_DOWNLOAD:-0}" = "1" ] && { log "SKIP_DOWNLOAD=1 — skipping DFlash2 download check"; return; }
    local have=0 selected=""
    if [ -n "${DFLASH_REVISION:-}" ]; then
        selected="$DFLASH_PATH/snapshots/$DFLASH_REVISION"
    elif [ -s "$DFLASH_PATH/refs/main" ]; then
        selected="$DFLASH_PATH/snapshots/$(<"$DFLASH_PATH/refs/main")"
    fi
    [ -n "$selected" ] && [ -f "$selected/model.safetensors" ] && have=1
    if [ "${have:-0}" -ge 1 ] && [ "${REFRESH_WEIGHTS:-0}" != "1" ]; then
        log "DFlash2 already present: $DFLASH_PATH"
        ensure_dflash_refs_main
        return
    fi
    resolve_hf_bin || die "no 'hf' / 'huggingface-cli' on PATH and no python huggingface_hub — pip install --user -U 'huggingface_hub[cli]' (or set HF_BIN=/path/to/hf)"
    mkdir -p "$HF_CACHE_DIR"
    log "downloading ${DFLASH_MODEL} (~2.3 GiB) into ${HF_CACHE_DIR} ..."
    local -a dflash_args=("$DFLASH_MODEL")
    [ -n "${DFLASH_REVISION:-}" ] && dflash_args+=(--revision "$DFLASH_REVISION")
    HF_HOME="$HF_CACHE_DIR" "${HF_BIN_CMD[@]}" download "${dflash_args[@]}"
    resolve_dflash_dir >/dev/null
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

    have="$(count_shards "$MODEL_PATH" "$MODEL_SNAPSHOT")"
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
# Keyed on the selected snapshot commit (refs/main, with the same repair
# fallback as ensure_refs_main, or a preset's pinned revision), not on
# MODEL_REVISION: the marker lives inside each synced repo folder, so a MODEL /
# revision switch re-syncs automatically.
# Without it, every ./start.sh pays a full size+mtime re-verification walk
# over ~164 GiB / 120 shards on both ends for zero bytes of difference
# (issue #22, item 2). FORCE_SYNC=1 bypasses the marker; deleting the
# marker file on the worker has the same effect.
sync_repo_marker_rev() {
    local src="$1" preferred="${2:-}"
    local rev
    if [ -n "$preferred" ] && [ -d "$src/snapshots/$preferred" ]; then
        rev="$preferred"
    else
        rev="$(cat "$src/refs/main" 2>/dev/null || true)"
    fi
    [ -n "$rev" ] || rev="$(ls -1t "$src/snapshots" 2>/dev/null | head -n 1 || true)"
    [ -n "$rev" ] || rev="unknown"
    printf '%s' "$rev"
}

sync_repo_to_worker() {
    local src="$1" cache_name="$2" label="$3" preferred="${4:-}"
    local marker rev
    marker="${WORKER_CACHE_DIR}/hub/${cache_name}/.glm53-exl3-synced"
    rev="$(sync_repo_marker_rev "$src" "$preferred")"
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

# Worker-side twin of require_model_snapshot for the rsync/SKIP_SYNC paths.
verify_worker_model_snapshot() {
    [ -n "$MODEL_SNAPSHOT" ] || return 0
    # NFS_SHARE=1: the rank reads the head's tree over NFS (nfs_share_weights
    # proves it can) — there is no worker copy to count, and the export is the
    # tree require_model_snapshot already validated on the head.
    [ "${NFS_SHARE:-0}" = "1" ] && return 0
    local dir="${WORKER_CACHE_DIR}/hub/${MODEL_CACHE_NAME}/snapshots/${MODEL_SNAPSHOT}"
    worker_ssh "test -f '$dir/config.json' \
        && test -f '$dir/model.safetensors.index.json' \
        && [ \"\$(find -L '$dir' -maxdepth 1 -type f -name '*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]')\" -ge '$EXPECTED_SHARDS' ]" \
        || die "pinned model snapshot is incomplete on worker: $dir"
}

sync_weights() {
    require_model_snapshot
    if [ "${SKIP_SYNC:-0}" = "1" ]; then
        verify_worker_model_snapshot
        log "SKIP_SYNC=1 — not syncing to worker"
        return
    fi
    [ -d "$MODEL_PATH" ] || die "weights missing at $MODEL_PATH — run without SKIP_DOWNLOAD first"
    if [ "${NFS_SHARE:-0}" = "1" ]; then
        if [ "$SPEC_METHOD" = "dflash" ] && [ ! -d "$DFLASH_PATH" ]; then
            die "DFlash2 weights missing at $DFLASH_PATH"
        fi
        nfs_share_weights
        return
    fi
    sync_repo_to_worker "$MODEL_PATH" "$MODEL_CACHE_NAME" "weights" "$MODEL_SNAPSHOT"
    if [ "$SPEC_METHOD" = "dflash" ]; then
        [ -d "$DFLASH_PATH" ] || die "DFlash2 weights missing at $DFLASH_PATH"
        sync_repo_to_worker "$DFLASH_PATH" "$DFLASH_CACHE_NAME" "DFlash2 draft" "$DFLASH_REVISION"
    fi
    verify_worker_model_snapshot
    log "worker weights in sync"
}

# ------------------------ inner container scripts --------------------------
# Both ranks apply the same checked overlays. Hybrid replay precedes retention
# because they share the coordinator helper insertion point; patch_apc_no_store
# follows retention (sampling_params / request / block_pool only, no shared
# anchors with the coordinator overlays). The KV-capacity log edits only
# kv_cache_utils.py and follows patch_glm5_drafter_group.py, the other overlay
# editing that file.
GLM53_OVERLAY_ORDER=(
    patch_cold_load_uma.py
    patch_glm_video_placeholders.py
    patch_suppress_stops_in_reasoning.py
    patch_scheduler_decode_floor.py
    patch_glm5_drafter_group.py
    patch_hybrid_prefix_hit.py
    patch_apc_per_group_retention.py
    patch_apc_no_store.py
    patch_kv_capacity_log.py
    patch_tool_choice_none.py
    patch_xgrammar_termination.py
    patch_kpool_tail_slotmap.py
    patch_spinwait.py
    patch_adaptive_k.py
    patch_dense_fp8.py
    patch_default_max_new_tokens.py
    patch_indexer_workspace.py
    patch_cache_reset.py
    patch_ablit.py
)

# Emits the in-container apply block for GLM53_OVERLAY_ORDER (same bytes for
# both ranks).
emit_overlay_block() {
    local p
    for p in "${GLM53_OVERLAY_ORDER[@]}"; do
        printf 'if [ -f /opt/glm53/%s ]; then\n    python3 /opt/glm53/%s\nfi\n' "$p" "$p"
    done
}

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
[ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ] && ARGS+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
[ -n "${LOAD_FORMAT:-}" ] && ARGS+=(--load-format "${LOAD_FORMAT}")
[ -n "${PREFIX_MATCH_UNIT:-}" ] && ARGS+=(--prefix-match-unit "${PREFIX_MATCH_UNIT}")
if [ -n "${GLM53_DEFAULT_REASONING_EFFORT:-}" ]; then
    ARGS+=(--default-chat-template-kwargs "{\"reasoning_effort\":\"${GLM53_DEFAULT_REASONING_EFFORT}\"}")
fi
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
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
    say "vision on: limit-mm=${LIMIT_MM:-} image-tokens=${MM_IMAGE_TOKENS:-8000} video-frames=${VIDEO_NUM_FRAMES:-32} mm-cache-gb=${MM_PROCESSOR_CACHE_GB:-4} skip-mm-profiling=${SKIP_MM_PROFILING:-1} chat-template=${CHAT_TEMPLATE:-}"
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
EOF
    emit_overlay_block >> "$HEAD_SCRIPT"
    cat >> "$HEAD_SCRIPT" <<'EOF'
if [ "${ABLIT:-0}" = "1" ]; then
    say "ablit: o_proj orthogonalization ON (method=${ABLIT_METHOD:-auto} direction=${ABLIT_DIRECTION:-dealign} layers=${ABLIT_LAYERS:-15-45} alpha=${ABLIT_ALPHA:-3.0})"
else
    say "runtime ablit: off; checkpoint o_proj unchanged"
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
[ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ] && ARGS+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
[ -n "${LOAD_FORMAT:-}" ] && ARGS+=(--load-format "${LOAD_FORMAT}")
[ -n "${PREFIX_MATCH_UNIT:-}" ] && ARGS+=(--prefix-match-unit "${PREFIX_MATCH_UNIT}")
if [ -n "${GLM53_DEFAULT_REASONING_EFFORT:-}" ]; then
    ARGS+=(--default-chat-template-kwargs "{\"reasoning_effort\":\"${GLM53_DEFAULT_REASONING_EFFORT}\"}")
fi
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
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
EOF
    emit_overlay_block >> "$WORKER_SCRIPT"
    cat >> "$WORKER_SCRIPT" <<'EOF'
if [ "${ABLIT:-0}" = "1" ]; then
    say "ablit: o_proj orthogonalization ON (method=${ABLIT_METHOD:-auto} direction=${ABLIT_DIRECTION:-dealign} layers=${ABLIT_LAYERS:-15-45} alpha=${ABLIT_ALPHA:-3.0})"
else
    say "runtime ablit: off; checkpoint o_proj unchanged"
fi
say "joining TP2 at ${HEAD_IP}:${MASTER_PORT} as rank 1"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
EOF
    chmod +x "$HEAD_SCRIPT" "$WORKER_SCRIPT"
}

_glm53_coop_overlay_selected() {
    local last
    [ -f "$EXL3_OVERLAY_HOST" ] || return 1
    last="$(grep -v '^[[:space:]]*$' "$EXL3_OVERLAY_HOST" | tail -n 1 || true)"
    [[ "$last" == _coop_setup\[\"install\"\]* ]]
}

_glm53_coop_src_dir() {
    local dir
    dir="$(dirname -- "$EXL3_OVERLAY_HOST")"
    if [ -f "$dir/runtime.py" ] && [ -f "$dir/cooperative_moe.so" ]; then
        printf '%s\n' "$dir"
        return 0
    fi
    dir="$CACHE_ROOT/cooperative_moe"
    if [ -f "$dir/runtime.py" ] && [ -f "$dir/cooperative_moe.so" ]; then
        printf '%s\n' "$dir"
        return 0
    fi
    return 1
}

# Generated overlay run_path's /root/.cache/vllm/cooperative_moe/runtime.py.
# Copy adapter + .so into the worker host cache that is bind-mounted there.
_glm53_stage_coop_runtime_worker() {
    local src dest
    _glm53_coop_overlay_selected || return 0
    src="$(_glm53_coop_src_dir)" || die "cooperative overlay $EXL3_OVERLAY_HOST needs runtime.py and cooperative_moe.so beside it or in $CACHE_ROOT/cooperative_moe"
    mkdir -p "$CACHE_ROOT/cooperative_moe"
    if [ "$src" != "$CACHE_ROOT/cooperative_moe" ]; then
        install -m 644 "$src/runtime.py" "$src/cooperative_moe.so" "$CACHE_ROOT/cooperative_moe/"
        src="$CACHE_ROOT/cooperative_moe"
    fi
    dest="$WORKER_VLLM_CACHE/cooperative_moe"
    worker_ssh "mkdir -p '$dest'"
    scp -q -o BatchMode=yes "$src/runtime.py" "$src/cooperative_moe.so" "${WORKER_SSH}:${dest}/"
    log "cooperative MoE runtime staged on worker (${dest})"
}

# ------------------------------- launch ------------------------------------
# Drop clean page cache and cycle swap on both nodes while no engine runs.
# Needs passwordless sudo (sudo -n); otherwise warns and continues. The
# in-container patch_cold_load_uma.py cannot drop caches (no CAP_SYS_ADMIN)
# and only rescales the InstantTensor budget, so this host step is what keeps
# the 164 GiB load at the NVMe ceiling instead of a double-digit io_depth.
host_memory_hygiene() {
    [ "$GLM53_HOST_MEM_HYGIENE" = "1" ] || { log "host memory hygiene skipped (GLM53_HOST_MEM_HYGIENE=0)"; return 0; }
    local cmd='sync; echo 1 > /proc/sys/vm/drop_caches; if swapon --noheadings --show=USED 2>/dev/null | grep -qvE "^\s*0B?\s*$"; then swapoff -a && swapon -a; fi; echo "MemFree=$(awk "/MemFree/{print int(\$2/1048576)}" /proc/meminfo)GiB swap_used=$(swapon --noheadings --show=USED 2>/dev/null | tr -d " " | paste -sd, -)"'
    # After a teardown the driver returns ~80 GiB of weights asynchronously;
    # vLLM's startup check reads cuda free (== MemFree on UMA) against
    # GPU_MEM_UTIL x total, so wait until MemFree clears that bar (+1 GiB).
    # The gate has to be the number vLLM reads: cudaMemGetInfo free, which
    # trails host MemFree by a few GiB while the previous context unwinds.
    local need_mib i free_mib
    need_mib=$(awk -v u="$GPU_MEM_UTIL" '/MemTotal/{printf "%d", $2*u/1024 + 1536}' /proc/meminfo)
    for i in $(seq 1 45); do
        free_mib=$(docker run --rm --gpus all --entrypoint python3 "$IMAGE" -c 'import torch; print(torch.cuda.mem_get_info()[0]//1048576)' 2>/dev/null || awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
        [ "${free_mib:-0}" -ge "$need_mib" ] && break
        [ "$i" = 1 ] && log "waiting for CUDA free ($((free_mib/1024)) GiB) to reach $((need_mib/1024)) GiB (GPU_MEM_UTIL x total + 1.5 GiB) ..."
        sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null || true
        sleep 2
    done
    log "cuda free before launch: $((free_mib/1024)) GiB (need $((need_mib/1024)))"
    local out
    if out="$(sudo -n sh -c "$cmd" 2>/dev/null)"; then
        log "host hygiene head: $out"
    else
        warn "head: passwordless sudo unavailable — page cache not dropped (cold load may run below the NVMe ceiling)"
    fi
    if out="$(worker_ssh "sudo -n sh -c '$cmd'" 2>/dev/null)"; then
        log "host hygiene worker: $out"
    else
        warn "worker: passwordless sudo unavailable — page cache not dropped"
    fi
}

launch_cluster() {
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || true
    worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 || true
    host_memory_hygiene

    mkdir -p "$CACHE_ROOT" "$TRITON_HOST_CACHE" "$TILELANG_HOST_CACHE"
    worker_ssh "mkdir -p '$WORKER_VLLM_CACHE' '$WORKER_TRITON_CACHE' '$WORKER_TILELANG_CACHE'"
    scp -q -o BatchMode=yes "$WORKER_SCRIPT" "${WORKER_SSH}:/tmp/${CONTAINER_WORKER}.sh"
    [ -f "$CHAT_TEMPLATE_HOST" ] || die "missing chat template: $CHAT_TEMPLATE_HOST"
    scp -q -o BatchMode=yes "$CHAT_TEMPLATE_HOST" "${WORKER_SSH}:/tmp/glm53-chat_template.jinja"
    [ -f "$VIDEO_PATCH_HOST" ] || die "missing $VIDEO_PATCH_HOST"
    scp -q -o BatchMode=yes "$VIDEO_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_glm_video_placeholders.py"
    [ -f "$STOP_PATCH_HOST" ] || die "missing $STOP_PATCH_HOST"
    scp -q -o BatchMode=yes "$STOP_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_suppress_stops_in_reasoning.py"
    [ -f "$TOOLCHOICE_PATCH_HOST" ] || die "missing $TOOLCHOICE_PATCH_HOST"
    scp -q -o BatchMode=yes "$TOOLCHOICE_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_tool_choice_none.py"
    [ -f "$SCHED_PATCH_HOST" ] || die "missing $SCHED_PATCH_HOST"
    scp -q -o BatchMode=yes "$SCHED_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_scheduler_decode_floor.py"
    [ -f "$DRAFTER_PATCH_HOST" ] || die "missing $DRAFTER_PATCH_HOST"
    scp -q -o BatchMode=yes "$DRAFTER_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_glm5_drafter_group.py"
    [ -f "$APC_PATCH_HOST" ] || die "missing $APC_PATCH_HOST"
    scp -q -o BatchMode=yes "$APC_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_hybrid_prefix_hit.py"
    [ -f "$PERGROUP_PATCH_HOST" ] || die "missing $PERGROUP_PATCH_HOST"
    scp -q -o BatchMode=yes "$PERGROUP_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_apc_per_group_retention.py"
    [ -f "$NOSTORE_PATCH_HOST" ] || die "missing $NOSTORE_PATCH_HOST"
    scp -q -o BatchMode=yes "$NOSTORE_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_apc_no_store.py"
    [ -f "$KVCAP_PATCH_HOST" ] || die "missing $KVCAP_PATCH_HOST"
    scp -q -o BatchMode=yes "$KVCAP_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_kv_capacity_log.py"
    [ -f "$XGRAMMAR_PATCH_HOST" ] || die "missing $XGRAMMAR_PATCH_HOST"
    scp -q -o BatchMode=yes "$XGRAMMAR_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_xgrammar_termination.py"
    [ -f "$CACHE_RESET_PATCH_HOST" ] || die "missing $CACHE_RESET_PATCH_HOST"
    scp -q -o BatchMode=yes "$CACHE_RESET_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_cache_reset.py"
    [ -f "$COLD_LOAD_PATCH_HOST" ] || die "missing $COLD_LOAD_PATCH_HOST"
    scp -q -o BatchMode=yes "$COLD_LOAD_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_cold_load_uma.py"
    [ -f "$KPOOL_TAIL_PATCH_HOST" ] || die "missing $KPOOL_TAIL_PATCH_HOST"
    scp -q -o BatchMode=yes "$KPOOL_TAIL_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_kpool_tail_slotmap.py"
    [ -f "$SPINWAIT_PATCH_HOST" ] || die "missing $SPINWAIT_PATCH_HOST"
    scp -q -o BatchMode=yes "$SPINWAIT_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_spinwait.py"
    [ -f "$ADAPTIVE_K_PATCH_HOST" ] || die "missing $ADAPTIVE_K_PATCH_HOST"
    scp -q -o BatchMode=yes "$ADAPTIVE_K_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_adaptive_k.py"
    [ -f "$DENSE_FP8_PATCH_HOST" ] || die "missing $DENSE_FP8_PATCH_HOST"
    scp -q -o BatchMode=yes "$DENSE_FP8_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_dense_fp8.py"
    [ -f "$DEFAULT_TOKENS_PATCH_HOST" ] || die "missing $DEFAULT_TOKENS_PATCH_HOST"
    scp -q -o BatchMode=yes "$DEFAULT_TOKENS_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_default_max_new_tokens.py"
    scp -q -o BatchMode=yes "$EXL3_OVERLAY_HOST" "${WORKER_SSH}:/tmp/glm53-exl3.py"
    _glm53_stage_coop_runtime_worker

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
        -e "GLM53_APC_NO_STORE=$GLM53_APC_NO_STORE"
        -e "GLM53_KV_CAPACITY_LOG=$GLM53_KV_CAPACITY_LOG"
        -e "GLM53_FAIR_PREFILL_CHUNK=$GLM53_FAIR_PREFILL_CHUNK"
        -e "GLM53_FAIR_PREFILL_SHARE=$GLM53_FAIR_PREFILL_SHARE"
        -e "GLM53_FAIR_PREFILL_MAX_INTERVAL_MS=$GLM53_FAIR_PREFILL_MAX_INTERVAL_MS"
        -e "GLM53_FAIR_PREFILL_MAX_STEP_MS=$GLM53_FAIR_PREFILL_MAX_STEP_MS"
        -e "GLM53_FAIR_PREFILL_MAX_CHUNKS=$GLM53_FAIR_PREFILL_MAX_CHUNKS"
        # Cache-only opt-in; independent VLLM_SERVER_DEV_MODE retains precedence.
        -e "GLM53_EXPOSE_CACHE_RESET=$GLM53_EXPOSE_CACHE_RESET"
        -e "GLM53_DEFAULT_REASONING_EFFORT=${GLM53_DEFAULT_REASONING_EFFORT-}"
        -e "GLM53_INDEXER_WORKSPACE=$GLM53_INDEXER_WORKSPACE"
        -e "GLM53_SPINWAIT_MS=$GLM53_SPINWAIT_MS"
        -e "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
        -e "TILELANG_CACHE_DIR=$TILELANG_CACHE_DIR"
        -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
        -e "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
        -e "FLASHINFER_CUDA_ARCH_LIST=$FLASHINFER_CUDA_ARCH_LIST"
        -e FLASHINFER_DISABLE_VERSION_CHECK=1
        # Overridable: the hidden-state KV connector (training windows) refuses
        # expandable_segments; pass PYTORCH_CUDA_ALLOC_CONF= to disable.
        -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}"
        -e "VLLM_ENGINE_READY_TIMEOUT_S=$READY_TIMEOUT"
        # py-cpuinfo JSON-parses empty output on Grace/aarch64; the usage
        # thread then dumps JSONDecodeError. Stats are off on this private kit.
        -e VLLM_NO_USAGE_STATS=1
        -e DO_NOT_TRACK=1
        -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=$CG_ESTIMATE"
    )
    if [ -n "${NCCL_NCHANNELS:-}" ]; then
        [[ "$NCCL_NCHANNELS" =~ ^[1-9][0-9]*$ ]] || die "NCCL_NCHANNELS must be a positive integer (got ${NCCL_NCHANNELS})"
        nccl_common+=(
            -e "NCCL_MIN_NCHANNELS=$NCCL_NCHANNELS"
            -e "NCCL_MAX_NCHANNELS=$NCCL_NCHANNELS"
        )
        log "NCCL channels pinned MIN=MAX=${NCCL_NCHANNELS} (both ranks)"
    fi
    log "per-request prefix-cache no-store flag: GLM53_APC_NO_STORE=${GLM53_APC_NO_STORE} (both ranks)"
    log "boot KV-capacity breakdown log: GLM53_KV_CAPACITY_LOG=${GLM53_KV_CAPACITY_LOG} (both ranks)"
    # Global sparse retention is implemented by the pinned vLLM runtime.  Full
    # attention remains dense; Mamba managers use this value.  Keep this an
    # explicit deployer setting and forward it identically to both ranks.
    if [ -n "${GLM53_APC_RETENTION_INTERVAL:-}" ]; then
        nccl_common+=(-e "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=$GLM53_APC_RETENTION_INTERVAL")
        log "global prefix-cache retention interval: ${GLM53_APC_RETENTION_INTERVAL} (both ranks)"
    fi
    # Per-group APC retention for the DFlash2 drafter SWA group (overlay patch_apc_per_group_retention.py).
    # "" = inherit global, 0 = boundaries only, N = multiple of the scheduler block.
    if [ -n "${GLM53_APC_RETENTION_INTERVAL_SWA:-}" ]; then
        nccl_common+=(-e "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA=$GLM53_APC_RETENTION_INTERVAL_SWA")
        log "drafter (SWA) prefix-cache retention interval: ${GLM53_APC_RETENTION_INTERVAL_SWA} (both ranks)"
    fi

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
    local -a serve_env_names=()
    local v
    for v in SERVED_MODEL_NAME PORT TP NNODES HEAD_IP MASTER_PORT QUANTIZATION \
             MAX_MODEL_LEN GPU_MEM_UTIL MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
             LONG_PREFILL_TOKEN_THRESHOLD \
             KV_CACHE_DTYPE LOAD_FORMAT PREFIX_MATCH_UNIT MTP_TOKENS SPEC_METHOD DFLASH_TOKENS DFLASH_MODEL_DIR \
             DFLASH_DRAFT_TP \
             LANGUAGE_MODEL_ONLY SKIP_MM_PROFILING \
             MM_IMAGE_TOKENS VIDEO_NUM_FRAMES MM_PROCESSOR_CACHE_GB \
             LIMIT_MM CHAT_TEMPLATE ENFORCE_EAGER EXL3_FUSED_MOE EXL3_MOE_ROW_TILE EXL3_TEMP_ROWS_FUSED \
             EXL3_FAT_SORTED EXL3_FAT_BATCHED EXL3_FAT_KERNEL EXL3_FAT_GROUPED \
             DEFAULT_MAX_NEW_TOKENS MODEL_DIR EXTRA_ARGS \
             ABLIT ABLIT_METHOD ABLIT_DIRECTION ABLIT_LAYERS ABLIT_ALPHA ABLIT_INCLUDE_MTP \
             GLM53_ADAPTIVE_K GLM53_ADAPTIVE_K_SET GLM53_ADAPTIVE_K_ALPHA GLM53_ADAPTIVE_K_MARGIN \
             GLM53_ADAPTIVE_K_MIN_STEPS GLM53_ADAPTIVE_K_SATURATE GLM53_ADAPTIVE_K_HIST GLM53_DENSE_FP8 \
             GLM53_EXL3_MOE_FAST \
             GLM53_COOP_GEOMETRY; do
        serve_env+=" -e $v='${!v:-}'"
        serve_env_names+=("$v")
    done

    # Extra container env for diagnostics (space-separated NAME=VALUE list, e.g.
    # GLM53_EXTRA_ENV="VLLM_DEBUG_WORKSPACE=1"). Forwarded to both ranks as -e pairs.
    # A name this launch already forwards — every entry of nccl_common and
    # serve_env, the per-rank NCCL_*/VLLM_HOST_IP block — is rejected, as are the
    # launcher's namespaces and conditionally-forwarded knobs: docker takes the
    # last duplicate -e, so a same-named entry would silently override the knob
    # and skip its range check. Values: [A-Za-z0-9_./:@,+=-]* only (no spaces,
    # quotes, globs or shell metacharacters — the worker command line is built as
    # shell text). Only names are logged; values may carry credentials, so a
    # rejected entry is reported by position only and is never echoed.
    if [ -n "${GLM53_EXTRA_ENV:-}" ]; then
        local _kv _name _value _entry _names="" _owned=" " _idx=0
        # Read the owned set off the arguments this launch builds, so a knob added
        # to either list cannot be shadowed here without a second list to keep in sync.
        for _entry in "${nccl_common[@]}" "${serve_env_names[@]}" \
                      NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA \
                      NCCL_IB_GID_INDEX VLLM_HOST_IP VLLM_API_KEY LD_PRELOAD; do
            [ "$_entry" = "-e" ] && continue
            _owned="$_owned ${_entry%%=*} "
        done
        set -f
        for _kv in $GLM53_EXTRA_ENV; do
            _idx=$((_idx + 1))
            # The word split above delivers a whitespace-containing value as
            # fragments, so the raw token and the unvalidated name can both carry
            # part of a credential: neither is interpolated into a rejection.
            case "$_kv" in *=*) ;; *) die "GLM53_EXTRA_ENV entry $_idx must be NAME=VALUE";; esac
            _name="${_kv%%=*}"; _value="${_kv#*=}"
            [[ "$_name" =~ ^[A-Z_][A-Z0-9_]*$ ]] || die "GLM53_EXTRA_ENV entry $_idx: name must match [A-Z_][A-Z0-9_]*"
            [[ "$_value" =~ ^[A-Za-z0-9_./:@,+=-]*$ ]] || die "GLM53_EXTRA_ENV: unsafe value for $_name (allowed: A-Z a-z 0-9 _ . / : @ , + = -)"
            case "$_name" in
                NCCL_*|HF_*|GLM53_*|EXL3_*|FLASHINFER_*|PATH|PYTHONPATH|VLLM_PREFIX_CACHE_RETENTION_INTERVAL*)
                    die "GLM53_EXTRA_ENV: $_name is launcher-owned; set it through its own knob";;
            esac
            case "$_owned" in
                *" $_name "*) die "GLM53_EXTRA_ENV: $_name is launcher-owned; set it through its own knob";;
            esac
            nccl_common+=(-e "$_kv"); _names="$_names $_name"
        done
        set +f
        log "extra container env (both ranks):${_names}"
    fi

    local worker_nccl="" e quoted_env
    for e in "${nccl_common[@]}"; do
        [ "$e" = "-e" ] && continue
        printf -v quoted_env '%q' "$e"
        worker_nccl+=" -e $quoted_env"
    done

    # The worker is headless and serves no API, so do not propagate the API
    # credential into its remote docker command or container environment.

    log "starting worker on ${WORKER_SSH} (NCCL if=${WORKER_CX7_IF} hca=${WORKER_CX7_IB}) ..."
    worker_ssh "docker run -d --name '$CONTAINER_WORKER' \
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v '$(_hf_mount)' \
        -v '$WORKER_VLLM_CACHE:/root/.cache/vllm' \
        -v '$WORKER_TRITON_CACHE:/root/.triton/cache' \
        -v '$WORKER_TILELANG_CACHE:/root/.tilelang/cache' \
        -v '/tmp/${CONTAINER_WORKER}.sh:/start.sh:ro' \
        -v '/tmp/glm53-chat_template.jinja:${CHAT_TEMPLATE}:ro' \
        -v '/tmp/patch_glm_video_placeholders.py:/opt/glm53/patch_glm_video_placeholders.py:ro' \
        -v '/tmp/patch_suppress_stops_in_reasoning.py:/opt/glm53/patch_suppress_stops_in_reasoning.py:ro' \
        -v '/tmp/patch_scheduler_decode_floor.py:/opt/glm53/patch_scheduler_decode_floor.py:ro' \
        -v '/tmp/patch_tool_choice_none.py:/opt/glm53/patch_tool_choice_none.py:ro' \
        -v '/tmp/patch_glm5_drafter_group.py:/opt/glm53/patch_glm5_drafter_group.py:ro' \
        -v '/tmp/patch_hybrid_prefix_hit.py:/opt/glm53/patch_hybrid_prefix_hit.py:ro' \
        -v '/tmp/patch_apc_per_group_retention.py:/opt/glm53/patch_apc_per_group_retention.py:ro' \
        -v '/tmp/patch_apc_no_store.py:/opt/glm53/patch_apc_no_store.py:ro' \
        -v '/tmp/patch_kv_capacity_log.py:/opt/glm53/patch_kv_capacity_log.py:ro' \
        -v '/tmp/patch_xgrammar_termination.py:/opt/glm53/patch_xgrammar_termination.py:ro' \
        -v '/tmp/patch_cache_reset.py:/opt/glm53/patch_cache_reset.py:ro' \
        -v '/tmp/patch_cold_load_uma.py:/opt/glm53/patch_cold_load_uma.py:ro' \
        -v '/tmp/patch_kpool_tail_slotmap.py:/opt/glm53/patch_kpool_tail_slotmap.py:ro' \
        -v '/tmp/patch_spinwait.py:/opt/glm53/patch_spinwait.py:ro' \
        -v '/tmp/patch_adaptive_k.py:/opt/glm53/patch_adaptive_k.py:ro' \
        -v '/tmp/patch_dense_fp8.py:/opt/glm53/patch_dense_fp8.py:ro' \
        -v '/tmp/patch_default_max_new_tokens.py:/opt/glm53/patch_default_max_new_tokens.py:ro' \
        -v '/tmp/glm53-exl3.py:/opt/glm53/exl3.py:ro' \
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
    VLLM_API_KEY="$VLLM_API_KEY" docker run -d --name "$CONTAINER_HEAD" \
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
        -v "$TOOLCHOICE_PATCH_HOST:/opt/glm53/patch_tool_choice_none.py:ro" \
        -v "$DRAFTER_PATCH_HOST:/opt/glm53/patch_glm5_drafter_group.py:ro" \
        -v "$APC_PATCH_HOST:/opt/glm53/patch_hybrid_prefix_hit.py:ro" \
        -v "$PERGROUP_PATCH_HOST:/opt/glm53/patch_apc_per_group_retention.py:ro" \
        -v "$NOSTORE_PATCH_HOST:/opt/glm53/patch_apc_no_store.py:ro" \
        -v "$KVCAP_PATCH_HOST:/opt/glm53/patch_kv_capacity_log.py:ro" \
        -v "$XGRAMMAR_PATCH_HOST:/opt/glm53/patch_xgrammar_termination.py:ro" \
        -v "$CACHE_RESET_PATCH_HOST:/opt/glm53/patch_cache_reset.py:ro" \
        -v "$COLD_LOAD_PATCH_HOST:/opt/glm53/patch_cold_load_uma.py:ro" \
        -v "$KPOOL_TAIL_PATCH_HOST:/opt/glm53/patch_kpool_tail_slotmap.py:ro" \
        -v "$SPINWAIT_PATCH_HOST:/opt/glm53/patch_spinwait.py:ro" \
        -v "$ADAPTIVE_K_PATCH_HOST:/opt/glm53/patch_adaptive_k.py:ro" \
        -v "$DENSE_FP8_PATCH_HOST:/opt/glm53/patch_dense_fp8.py:ro" \
        -v "$DEFAULT_TOKENS_PATCH_HOST:/opt/glm53/patch_default_max_new_tokens.py:ro" \
        -v "$EXL3_OVERLAY_HOST:/opt/glm53/exl3.py:ro" \
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
        -e DEFAULT_MAX_NEW_TOKENS="$DEFAULT_MAX_NEW_TOKENS" \
        -e LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-}" \
        -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" \
        -e LOAD_FORMAT="${LOAD_FORMAT:-}" \
        -e PREFIX_MATCH_UNIT="${PREFIX_MATCH_UNIT:-}" \
        -e MTP_TOKENS="$MTP_TOKENS" \
        -e SPEC_METHOD="$SPEC_METHOD" \
        -e DFLASH_TOKENS="${DFLASH_TOKENS:-7}" \
        -e DFLASH_MODEL_DIR="${DFLASH_MODEL_DIR:-}" \
        -e DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP:-}" \
        -e LANGUAGE_MODEL_ONLY="$LANGUAGE_MODEL_ONLY" \
        -e SKIP_MM_PROFILING="$SKIP_MM_PROFILING" \
        -e LIMIT_MM="$LIMIT_MM" \
        -e MM_IMAGE_TOKENS="${MM_IMAGE_TOKENS:-}" \
        -e VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:-}" \
        -e MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-}" \
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
        -e GLM53_ADAPTIVE_K="$GLM53_ADAPTIVE_K" \
        -e GLM53_ADAPTIVE_K_SET="$GLM53_ADAPTIVE_K_SET" \
        -e GLM53_ADAPTIVE_K_ALPHA="$GLM53_ADAPTIVE_K_ALPHA" \
        -e GLM53_ADAPTIVE_K_MARGIN="$GLM53_ADAPTIVE_K_MARGIN" \
        -e GLM53_ADAPTIVE_K_MIN_STEPS="$GLM53_ADAPTIVE_K_MIN_STEPS" \
        -e GLM53_ADAPTIVE_K_SATURATE="$GLM53_ADAPTIVE_K_SATURATE" \
        -e GLM53_ADAPTIVE_K_HIST="$GLM53_ADAPTIVE_K_HIST" \
        -e GLM53_DENSE_FP8="$GLM53_DENSE_FP8" \
        -e GLM53_EXL3_MOE_FAST="$GLM53_EXL3_MOE_FAST" \
        -e GLM53_COOP_GEOMETRY="$GLM53_COOP_GEOMETRY" \
        -e MODEL_DIR="$MODEL_DIR" \
        -e VLLM_API_KEY \
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
    # 9>&-: the follower must not inherit the lifecycle lock fd. Bash passes
    # `exec 9>` descriptors through exec, so a follower that somehow outlives
    # this shell (SIGKILL, no trap) would keep the flock held.
    docker logs -f --tail 0 "$CONTAINER_HEAD" 2>&1 9>&- &
    logpid=$!

    local elapsed=0 healthy=0 exited=0 dead_side="" worker_fail=0 head_fail=0
    while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
        if curl -fsS -m 5 "$url" >/dev/null 2>&1; then healthy=1; break; fi
        # Keep the inspect result out of a grep -q pipeline. With pipefail,
        # grep can close early and make a running container look dead.
        # Same 3-strike window as the worker: one transient docker miss must
        # not abort a multi-minute weight load.
        if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null || true)" = "true" ]; then
            head_fail=0
        else
            head_fail=$((head_fail + 1))
            if [ "$head_fail" -ge 3 ]; then
                if docker inspect "$CONTAINER_HEAD" >/dev/null 2>&1; then
                    log "head container not running during startup (3 consecutive checks)"
                else
                    log "head container missing during startup (removed by concurrent stop/restart?)"
                fi
                exited=1; dead_side="head"; break
            fi
        fi
        # A dead worker rank can never make the head healthy — fail fast with
        # the log dump instead of polling for the full READY_TIMEOUT (issue
        # #22, item 4). Transient ssh/docker hiccups are tolerated; only
        # three consecutive non-running answers (~30 s) count as a dead
        # worker.
        if [ "$(worker_ssh "docker inspect -f '{{.State.Running}}' '$CONTAINER_WORKER' 2>/dev/null" || true)" = "true" ]; then
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
    local mt_line="mt_default=off (stock model/server limits)"
    [ -n "${DEFAULT_MAX_NEW_TOKENS:-}" ] && mt_line="mt_default=${DEFAULT_MAX_NEW_TOKENS}"
    local ablit="off (checkpoint weights unchanged)"
    [ "$ABLIT" = "1" ] && ablit="ON method=${ABLIT_METHOD} direction=${ABLIT_DIRECTION} layers=${ABLIT_LAYERS} alpha=${ABLIT_ALPHA}"
    log "  features   : tools=glm47+auto, reasoning=glm45, spec=${spec}, vision=${vision}, ${mt_line}, ablit=${ablit}"
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
start_unlocked() {
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
    log "exl3: fat_kernel=${EXL3_FAT_KERNEL} fat_grouped=${EXL3_FAT_GROUPED} temp_rows_fused=${EXL3_TEMP_ROWS_FUSED} mnbt=${MAX_NUM_BATCHED_TOKENS} max_num_seqs=${MAX_NUM_SEQS} draft_tp=${DFLASH_DRAFT_TP}"
    log "mixed-prefill: policy=${GLM53_MIXED_PREFILL_CHUNK} fair_chunk=${GLM53_FAIR_PREFILL_CHUNK} share=${GLM53_FAIR_PREFILL_SHARE} interval_ms=${GLM53_FAIR_PREFILL_MAX_INTERVAL_MS} max_step_ms=${GLM53_FAIR_PREFILL_MAX_STEP_MS} max_chunks=${GLM53_FAIR_PREFILL_MAX_CHUNKS} long_prefill=${LONG_PREFILL_TOKEN_THRESHOLD:-}"

    launch_cluster
    if wait_for_health; then
        post_ready_warmup
        on_ready
        return 0
    fi
    collect_failure_logs
    echo "---- last 60 lines of head log ($LOGDIR/head.log) ----"
    tail -n 60 "$LOGDIR/head.log" || true
    echo "---- last 40 lines of worker log ($LOGDIR/worker.log) ----"
    tail -n 40 "$LOGDIR/worker.log" || true
    die "server did not become healthy — full logs in $LOGDIR/"
}

start() {
    with_cluster_lock
    start_unlocked
}

stop_containers() {
    log "stopping head container ..."
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || log "  (no head container was running)"
    log "stopping worker container on ${WORKER_SSH} ..."
    worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 \
        || log "  (no worker container was running)"
    if [ "${NFS_SHARE:-0}" = "1" ]; then
        log "removing the worker NFS volume (the exporter stays up) ..."
        nfs_unmount_workers
    fi
    log "stopped."
}

# ------------------------------- stop --------------------------------------
stop() {
    with_cluster_lock_for_stop
    stop_containers
    rm -f "$CLUSTER_LOCK_PID" || true
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
        start|restart) validate_numeric_config; configure_capture_sizes; validate_overlay_artifacts ;;
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
        restart)
            with_cluster_lock
            stop_containers
            start_unlocked
            ;;
        status)   status ;;
        logs)     shift || true; logs "$@" ;;
        share)    [ "${NFS_SHARE:-0}" = "1" ] || die "NFS_SHARE=0 in .env — set NFS_SHARE=1 to share the head HF cache over NFS"
                  nfs_share_weights ;;
        -h|--help|help) usage ;;
        *) usage; exit 1 ;;
    esac
}

main "$@"
