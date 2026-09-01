#!/usr/bin/env bash
# hf-mirror throttles per connection: individual files stall and fail, so a single
# pass never finishes. Loop the verified-fetcher (it skips files whose sha256
# already matches) until the distributed shard is complete, then reproduce the
# published FP8 row over all 512 contexts.
#
# Runs alongside the production worker on purpose: the scorer only needs ~3 GiB on
# one card. It never touches the capture path.
set -uo pipefail
cd /opt/vllm-backport/bench/glm53-kld
SUITE=/srv/models/fidelity-suites/GLM-5.3-Flash-fidelity-suite-v1
count() { ls "$SUITE/$1"/hidden_*.safetensors 2>/dev/null | wc -l; }
tok() { ls "$SUITE/suite/tokens" 2>/dev/null | wc -l; }

for round in $(seq 1 400); do
  echo "== round $round $(date +%H:%M:%S) tokens=$(tok) ref=$(count reference-bf16-shard0) fp8=$(count as-served-fp8-shard0)"
  if [ "$(tok)" -lt 512 ]; then
    SELECT=tokens TOKENS_UPTO=512 PARALLEL=4 bash fetch-fidelity-suite.sh
  elif [ "$(count reference-bf16-shard0)" -lt 512 ]; then
    SELECT=ref PARALLEL=4 bash fetch-fidelity-suite.sh
  elif [ "$(count as-served-fp8-shard0)" -lt 512 ]; then
    SELECT=fp8 PARALLEL=4 bash fetch-fidelity-suite.sh
  else
    echo "suite complete: tokens=$(tok) ref=$(count reference-bf16-shard0) fp8=$(count as-served-fp8-shard0)"
    break
  fi
  sleep 20
done

bash fetch-fidelity-suite.sh --verify
MODE=anchor DEVICE=cuda CHUNK=128 \
  OUT=/opt/vllm-backport/bench/glm53-kld-runs/fp8-anchor-shard0-512.json \
  bash capture-glm-awq.sh
echo "CHAIN DONE $(date -Is)"
