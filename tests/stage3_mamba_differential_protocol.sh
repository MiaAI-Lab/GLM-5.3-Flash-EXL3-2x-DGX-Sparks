#!/usr/bin/env bash
# Stage-3 mamba cross-boot differential test — the EXACT live protocol.
#
# STATUS: DESIGNED IN STAGE 2, NOT RUN. This script refuses to execute
# without GLM53_STAGE3_LIVE=1 and prints the protocol instead. Stage 2 ships
# it so the stage-3 acceptance gate is pinned in advance (PLAN-KV-OFFLOAD
# §3.1/§11 stage 3; Codex OFFLOAD2 finding 14: the host-side harness in
# tests/_mamba_differential.py is an ABI/codec comparator, NOT evidence —
# THIS protocol, run live, is the evidence instrument).
#
# Prerequisites (all owner-gated): stage-2 live receipts green on the probe;
# mamba group restore implemented (stage 3 code, NOT stage 2); an idle
# server window on both Sparks; PR-PAUSE absent.
set -euo pipefail

HEAD_SSH="${HEAD_SSH:-blockbrain@100.68.104.95}"
API="${GLM53_API:-http://127.0.0.1:8000/v1}"
STORE_DIR="${GLM53_KV_OFFLOAD_DIR:-$HOME/glm53-kv-offload}"
OUT_DIR="${GLM53_STAGE3_OUT:-$HOME/stage3-mamba-differential}"
BOUNDARIES=(1 2 8)          # park boundaries k (tokens = (k+1) * 3584)
CONTINUATIONS=(16 256 2048) # continuation lengths per boundary
PROMPT_TOKENS=$(( (8 + 1) * 3584 + 1500 ))  # covers the deepest boundary + tail

protocol() {
    cat <<'EOF'
================= STAGE-3 MAMBA CROSS-BOOT DIFFERENTIAL PROTOCOL =================
Step 0 — STATE-INDEX CONVENTION RECEIPT (the open stage-0 R13 residual; gate
  for everything below):
  - In-container, instrument the mamba align-mode manager on ONE boundary
    materialization: dump the (block_id, state_index, accepted-token offset)
    triple and the raw conv/temporal bytes at snapshot time, per rank.
  - Verify the snapshot bytes equal the block's bytes the connector's store
    path gathers (byte-diff == 0) — the "the page is the state" assumption
    made real, or refuted HERE before any cross-boot claim.

Step 1 — BOOT A PARK:
  - Boot the pair with GLM53_KV_OFFLOAD=1 GLM53_KV_OFFLOAD_RESTORE=0.
  - Drive the park probe (deterministic token prompt, temp 0, fresh
    cache_salt) to PROMPT_TOKENS; force store-cascade completion (request
    finish); confirm boundary manifests for every k in BOUNDARIES on BOTH
    ranks' NVMe stores.
  - Capture per-rank state fixtures: copy each parked boundary's four mamba
    chunk files + manifest to $OUT_DIR/bootA/r{0,1}/ (files are content-
    addressed; the copy is the frozen boot-A state).
  - Record the uninterrupted CONTROL: continue the same conversation to each
    continuation length, logging per-token logprobs (temp 0, logprobs on).

Step 2 — RESTART (the cross-boot cut):
  - ./stop.sh && ./start.sh restart (same recipe SHA, same knobs, RESTORE=1).
  - Confirm namespace hash unchanged (same-config boot) in the writer-up log.

Step 3 — BOOT B RESTORE + DIFFERENTIAL:
  For each k in BOUNDARIES:
  - Issue the SAME prompt truncated to the boundary + 1 token (forces the
    lookup at the deepest manifested boundary <= k).
  - Verify the restore receipts: WAITING_FOR_REMOTE_KVS entered, disk load
    job on both ranks, cache_blocks re-entry, zero load errors.
  - BIT-DIFFERENTIAL: copy the restored blocks' bytes (in-container dump of
    the mamba state blocks after restore) and run
      python3 tests/_mamba_differential.py-style compare_state_bits
    against $OUT_DIR/bootA fixtures → per-(group, layer, tensor) mismatch
    counts. Gate: ZERO mismatches (the ABI holds bit-exact) BEFORE any
    behavioral claim.
  For each continuation length c in CONTINUATIONS:
  - Generate c tokens at temp 0 with logprobs; compare against the boot-A
    control per token: max |delta logprob| must sit inside the probe-v3
    measured noise band.
  - Needle-at-depth: query a fact planted before the boundary; must match.
  - FRESH-SESSION CONTROL: a new cache_salt session of the same prompt must
    produce control-equal output (proves no cross-contamination).

Step 4 — FAULT MATRIX (plan T4, group x rank x {normal, preemption, reuse}):
  - For each mamba group g in {2,3,4,5} x rank r in {0,1}: corrupt/remove
    that (g, r) chunk file; re-run the restore; REQUIRED: global miss or
    invalid-block recompute, fresh-session control green, NEVER a served
    half-restore. Repeat one cell under forced preemption and one under
    block-reuse pressure (small pool).

Step 5 — RECEIPTS: p50/p95 restore timing table (replaces plan §9's
  hypothesis rows), all logs + fixtures + comparator outputs archived under
  $OUT_DIR; STAGE3-RECEIPTS.md written from this protocol's numbered steps.
==================================================================================
EOF
}

if [ "${GLM53_STAGE3_LIVE:-0}" != "1" ]; then
    echo "stage3_mamba_differential_protocol.sh: GLM53_STAGE3_LIVE=1 not set."
    echo "This protocol is DESIGNED (stage 2) but NOT runnable yet: mamba"
    echo "restore is stage-3 code. Printing the protocol instead:"
    echo
    protocol
    exit 0
fi

echo "GLM53_STAGE3_LIVE=1: refusing anyway — mamba group restore is not"
echo "implemented in this branch (stage 2 is g0-only). Remove this guard"
echo "only in the stage-3 branch that implements it."
exit 2
