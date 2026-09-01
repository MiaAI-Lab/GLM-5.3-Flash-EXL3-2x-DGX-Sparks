#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# GLM-5.3-Flash quant-fidelity (KLD) run on cmp170hx.
#
#   MODE=anchor   reproduce the published FP8 row from the downloaded suite. No
#                 capture, no card hand-off - this is the gate that says the
#                 scorer speaks the suite's language. Run it first, always.
#   MODE=capture  teacher-force the sealed windows through our own AWQ engine and
#                 write a lane. Takes all four cards, so it refuses to start while
#                 a serving worker owns them (stop it through the orchestrator).
#   MODE=score    score our lane against the sealed reference lane.
#   MODE=self     reference lane against itself; mean KLD must be exactly 0.
#
# Engine flags mirror launch-glm.sh for the AWQ profile (PP partition
# 14,12,12,7, no --moe-backend, TRITON_MLA_SPARSE + sparse_mla_force_mqa) with two
# deliberate differences: speculative decoding is OFF (a draft tower would put its
# own quantization into the hidden states) and the context is 2048 tokens, so
# max-model-len is small and KV profiling is fast. Both are recorded in the lane's
# capture-receipt.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Scoring runs on the host and needs torch + safetensors; the vllm26 conda env has
# both (host torch 2.11 for sm80, the same build the KLD anchor was validated on).
if [ -z "${PY:-}" ]; then
  if [ -x /home/kk/miniconda3/envs/vllm26/bin/python ]; then
    PY=/home/kk/miniconda3/envs/vllm26/bin/python
  else
    PY=python3
  fi
fi
IMAGE="${IMAGE:-vllm/vllm-backport:cmp170hx}"
SUITE="${SUITE:-/srv/models/fidelity-suites/GLM-5.3-Flash-fidelity-suite-v1}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/srv/models/wtdcode/GLM-5.3-Flash-AWQ-W4A16}"
MODEL_CONTAINER_PATH="${MODEL_CONTAINER_PATH:-/models/GLM-5.3-Flash-AWQ-W4A16}"
LANE="${LANE:-$SUITE/cmp170hx-awq-w4a16}"
DUMP_ROOT="${DUMP_ROOT:-/srv/models/fidelity-suites/.capture-dumps}"
CACHE_DIR="${CACHE_DIR:-/opt/vllm-backport/vllm_cache}"
SERVED_NAME="${SERVED_NAME:-glm-5.3-flash-awq}"
TP="${TP:-1}"
PP="${PP:-4}"
PP_PARTITION="${PP_PARTITION:-14,12,12,7}"
GMU="${GMU:-0.95}"
CONTEXTS="${CONTEXTS:-}"                      # e.g. "0,1,2" ; empty = whole lane
LIMIT="${LIMIT:-}"                             # e.g. 64 contexts, for a smoke run
# Production parity flags that are not plain engine kwargs in launch-glm.sh.
# Verified against this build: attention_config reaches VllmConfig
# (sparse_mla_force_mqa=True), enable_flashinfer_autotune matches
# --no-enable-flashinfer-autotune, max_num_batched_tokens matches 8192.
# Deliberate, disclosed differences: no speculative-config (MTP would put the
# draft tower's quantization into the hidden states), prefix caching off (a shared
# 2048-token prefix would make later contexts read cached states instead of
# recomputing them), max_model_len 4096 instead of 524288 (KV profile only).
PARITY_JSON='{"enable_flashinfer_autotune": false, "max_num_batched_tokens": 8192, "enable_prefix_caching": false, "attention_config": {"sparse_mla_force_mqa": true}}'
EXTRA_ENGINE_JSON="${EXTRA_ENGINE_JSON:-$PARITY_JSON}"
DEVICE="${DEVICE:-cuda}"                        # cuda|cpu; the head is 1.27 GiB bf16
REFERENCE_LANE="${REFERENCE_LANE:-reference-bf16-shard0}"
# The published replay is bf16 matmul, no float32 upcast - see README "dtype".
DTYPE="${DTYPE:-bfloat16}"
FP8_ROW="${FP8_ROW:-0.028103897727130314}"     # reports/report-fp8-vs-bf16.json
ANCHOR_TOLERANCE="${ANCHOR_TOLERANCE:-2e-3}"
MODE="${MODE:?MODE=anchor|capture|score|self}"

if [ ! -d "$SUITE/suite" ]; then
  echo "suite not fetched: $SUITE (run fetch-fidelity-suite.sh SELECT=metadata first)" >&2
  exit 1
fi

running="$(sudo docker ps --format '{{.Names}}' | tr '\n' ' ')"
require_free_cards() {
  if [ -n "${running// /}" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "container(s) holding the cards: $running" >&2
    echo "stop the worker through the orchestrator first, or set FORCE=1 to proceed anyway." >&2
    exit 1
  fi
}

case "$MODE" in
  self)
    # The suite against itself: every metric must be identically zero. Catches a
    # broken scorer, not a bad lane. Runs on CPU.
    "$PY" "$HERE/score_hidden_kld.py" --suite "$SUITE" \
      --reference-lane "$REFERENCE_LANE" --candidate-lane "$REFERENCE_LANE" \
      --device cpu --limit "${LIMIT:-8}" \
      --expect-mean 0.0 --expect-relative 0 \
      --out "${OUT:-$HERE/../glm53-kld-runs/self-canary.json}"
    ;;

  anchor)
    # The published FP8 row, reproduced from the shipped lanes. The published
    # global mean only applies to the whole published scope; at partial scope the
    # gate is the per-context comparison, which the scorer enforces either way.
    "$PY" "$HERE/score_hidden_kld.py" --suite "$SUITE" \
      --reference-lane "$REFERENCE_LANE" --candidate-lane as-served-fp8-shard0 \
      --device "$DEVICE" ${LIMIT:+--limit "$LIMIT"} \
      --offset-audit ${EXPECT_MEAN:+--expect-mean "$FP8_ROW"} \
      ${EXPECT_MEAN:+--expect-relative "$ANCHOR_TOLERANCE"} \
      --compare-report "$SUITE/reports/report-fp8-vs-bf16.json" \
      --dtype "$DTYPE" --chunk "${CHUNK:-128}" \
      --out "${OUT:-$HERE/../glm53-kld-runs/fp8-anchor.json}"
    ;;

  capture)
    require_free_cards
    mkdir -p "$DUMP_ROOT" "$LANE"
    # Touch the credential cache without a password prompt (sudo -v wants a tty on
    # this box; sudoers is NOPASSWD, so a plain -n probe is enough to fail early).
    sudo -n true || { echo "sudo -n failed; run with a tty or configure NOPASSWD" >&2; exit 1; }
    sudo docker run --rm \
      --name glm53-kld-capture \
      --gpus all \
      -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
      --privileged --ipc=host \
      -v "$MODEL_HOST_PATH:$MODEL_CONTAINER_PATH:ro" \
      -v "$SUITE:/suite" \
      -v "$DUMP_ROOT:/work/dumps" \
      -v "$HERE:/kld:ro" \
      -v "$CACHE_DIR:/root/.cache/vllm" \
      -v "$CACHE_DIR/triton:/root/.cache/triton" \
      -v "$CACHE_DIR/tilelang:/root/.cache/tilelang" \
      -e PYTHONPATH=/kld \
      -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
      -e GLM_KLD_DUMP_DIR=/work/dumps \
      -e GLM_KLD_DUMP_CONTEXT_ROWS=2048 \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -e VLLM_ATTENTION_BACKEND=TRITON_MLA_SPARSE \
      -e VLLM_PP_LAYER_PARTITION="$PP_PARTITION" \
      -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
      -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
      -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
      -e HF_HUB_OFFLINE=1 \
      -e PYTHONUNBUFFERED=1 \
      --entrypoint python3 \
      "$IMAGE" /kld/capture_hidden_states.py \
        --suite /suite \
        --model "$MODEL_CONTAINER_PATH" \
        --out "/suite/${LANE#"$SUITE"/}" \
        --dump-dir /work/dumps \
        --served-model-name "$SERVED_NAME" \
        --tensor-parallel-size "$TP" \
        --pipeline-parallel-size "$PP" \
        --enable-expert-parallel \
        --gpu-memory-utilization "$GMU" \
        ${CONTEXTS:+--indices "$CONTEXTS"} \
        ${LIMIT:+--limit "$LIMIT"} \
        ${EXTRA_ENGINE_JSON:+--extra-engine-json "$EXTRA_ENGINE_JSON"}
    # The container writes as root; without this the host scorer cannot read its own lane.
    sudo chown -R "$(id -u):$(id -g)" "$LANE" "$DUMP_ROOT"
    ;;

  score)
    [ -d "$LANE" ] || { echo "no lane at $LANE (run MODE=capture)" >&2; exit 1; }
    "$PY" "$HERE/score_hidden_kld.py" --suite "$SUITE" \
      --reference-lane "$REFERENCE_LANE" --candidate-lane "$LANE" \
      --candidate-label "${CANDIDATE_LABEL:-cmp170hx-awq-w4a16}" \
      --device "$DEVICE" --offset-audit \
      --dtype "$DTYPE" --chunk "${CHUNK:-128}" \
      --out "${OUT:-$HERE/../glm53-kld-runs/awq-row.json}"
    # No --compare-report here on purpose: that report is the FP8 lane's own row, and
    # comparing a different candidate against it is the measurement, not a validity
    # check. The FP8 anchor (MODE=anchor) is where that gate belongs.
    ;;

  *)
    echo "unknown MODE=$MODE" >&2
    exit 2
    ;;
esac
