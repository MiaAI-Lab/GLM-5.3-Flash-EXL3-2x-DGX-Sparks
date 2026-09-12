# GB10 UVM livelock: prevention, detection, recovery

Field notes from running this kit's lineage on DGX Spark (GB10, unified
memory). Two incidents on a 4x TP=4 deployment (2026-08-26 unclean node
reboots; 2026-08-31 mid-serve wedge) plus the recovery that worked. The
failure class is the one tonyd2wild's OPEN-PROBLEMS #4 describes — this doc
adds the part that was unknown there: it is survivable without a power cycle.

## The failure

If the kernel pages vLLM memory out (or back in) under memory pressure, the
UVM driver can enter an unrecoverable spin:

- every engine step stalls; `shm_broadcast.py` logs
  `No available shared memory broadcast block found in 60 seconds` every 60 s
- `nvidia-smi` shows **~96% GPU utilization at idle wattage (~12–20 W)** —
  busy-looking, doing nothing
- the stuck request stays in `num_requests_running` forever;
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` does not fire
- worst case the node drops off SSH (sshd cannot fork) and needs a power cycle

**`/health` stays 200 the whole time.** vLLM's V1 check_health only reads an
errored flag; it never probes workers. Do not use `/health` as a liveness
signal for this failure class.

## Prevention

```bash
# /etc/sysctl.d/95-glm53-uvm.conf
vm.swappiness = 0
```

- **Keep the swap file.** With no swap at all, allocation spikes kill the
  worker outright (marlin repack, big prefill bursts).
- **Cycle residual swap before every start** (`swapoff -a && swapon -a`
  while containers are down). `swappiness=0` stops new page-outs but does
  nothing about pages already swapped: our 08-31 wedge was triggered by a
  ~236k-token prefill touching pages parked in swap days earlier.
- A boot-window page-cache flusher (unconditional `drop_caches` loop) keeps
  MemFree honest for the NVRM allocator during load.

## Detection (works while /health lies)

Poll `/metrics` every 60 s and alert when both hold for N consecutive ticks
(N=10 is calm):

- `num_requests_running + num_requests_waiting > 0`, and
- `prompt_tokens_total + generation_tokens_total` frozen.

Summing prompt+generation keeps long chunked prefills from false-alarming.
Stand down while the service is loading or the container is young.

## Recovery (18 minutes, no machine-room trip)

1. Capture logs first (`docker logs --tail 300` per node).
2. Graceful stop will hang — go straight to `docker kill` on every node.
3. Cycle swap on every node (RAM is free now): `swapoff -a && swapon -a`.
4. `nvidia-smi` may **still read 96%** with nothing running. That reading is
   cosmetic after the kill — verify with a trivial CUDA probe before assuming
   the driver is wedged:

   ```bash
   docker run --rm --gpus all --entrypoint python3 <serving-image> \
     -c "import torch; a=torch.ones(2048,2048,device=0); print('CUDA_OK', float((a@a).sum()))"
   ```

   `CUDA_OK` → just restart the stack; the utilization counter resets with
   the next real workload. Probe hangs → that node needs a reboot.
5. Restart. Wait for MemFree to settle first if the teardown just returned
   ~80 GiB of weights (the vLLM startup pre-check is the real gate).

## Related: speculative-acceptance sanity after any graph-enabled boot

vllm#53030 can silently pin per-position acceptance at exactly 1.00 under
CUDA graphs. `scripts/spec-accept-gate.sh` checks the decay curve from
`/metrics` once >100 drafts have accumulated; healthy looks like
0.85 → 0.70 → 0.56 → … monotone decay, and an exact 1.00 at pos0 is the bug,
not a win.
