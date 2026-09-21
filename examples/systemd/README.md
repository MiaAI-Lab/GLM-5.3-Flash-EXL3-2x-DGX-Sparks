# Reboot persistence (systemd)

`./start.sh` is a one-shot launcher. It `docker run -d` both ranks (restart
policy **no**), polls `/health`, then **exits**. Docker will not bring the
pair back after a power cycle. These units make systemd own that lifecycle
without fighting `start.sh`.

This is optional. It does not change launcher defaults.

## Why systemd, not Docker `--restart`

Tensor-parallel rank 0 cannot boot unless rank 1 is already watching. A
per-container `unless-stopped` policy races the two nodes and can resurrect
the old healthy head container before `ExecStart` replaces it. systemd
starts the worker watcher first, then the head, then waits on the **new**
container exit.

Measured on an independent 2× GB10 pair after a dual-node power cycle
(2026-09-20): both units enabled; worker then head; containers 0 Docker
restarts; no OOM; RoCE up; `/health` 200; TP=2 DFlash2 warmup 24/24.

## Layout

| Piece | Role |
|---|---|
| `env.sh` | paths, SSH target, container names, port, served model id |
| `glm53-worker.service` | rank 1 watcher (`docker wait`, `Restart=no`) |
| `glm53-head.service` | rank 0: coordinate worker → `./start.sh` → `docker wait` |
| `coordinate-glm53-worker.sh` | SSH + `systemctl restart` worker unit, host keys required |
| `launch-glm53.sh` | rank 0: `./start.sh`; rank 1: wait for the worker container |
| `run-glm53-systemd.sh` | run the launcher, then `docker wait` so the unit tracks the container |
| `wait-glm53-health.sh` | `ExecStartPost`: `/health` then `/v1/models` (HTTP, not log greps) |
| `90-glm53.conf` | `vm.swappiness=0` on both nodes; swap must still exist |
| `glm53-worker-sudoers` | one command on the worker: restart this unit |

Do not set `TAIL=1` under these units. `start.sh` must exit after `/health`
so `docker wait` can take over.

## Install

On **both** nodes:

```bash
sudo install -d -o "$USER" -g "$USER" /home/spark/glm53-deploy
cp examples/systemd/* /home/spark/glm53-deploy/
# edit env.sh, both units (User=/paths/Conflicts=), sudoers, sysctl
sudo cp /home/spark/glm53-deploy/90-glm53.conf /etc/sysctl.d/90-glm53.conf
sudo sysctl --system
```

On the **worker** only:

```bash
sudo cp /home/spark/glm53-deploy/glm53-worker-sudoers /etc/sudoers.d/glm53-worker
sudo visudo -c -f /etc/sudoers.d/glm53-worker
sudo install -m 644 /home/spark/glm53-deploy/glm53-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable glm53-worker.service
```

On the **head** only:

```bash
sudo install -m 644 /home/spark/glm53-deploy/glm53-head.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now glm53-head.service
```

Head → worker SSH must already work with `BatchMode=yes` and
`StrictHostKeyChecking=yes` (the same assumption as `./start.sh`). Do not
disable host-key checking to make boot succeed.

`TimeoutStartSec=35min` on both units covers a cached cold start plus HTTP
readiness. Weight download + first InstantTensor load can exceed that —
stage `./download.sh` and a successful `./start.sh` **before** enabling the
units.

## Production image pin

`IMAGE=ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`
moves. A live pair that should survive reboot **and** tag drift pins the
digest and skips pull/rebuild:

```bash
docker inspect --format '{{index .RepoDigests 0}}' \
  ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor
```

```
IMAGE=ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:<digest>
SKIP_PULL=1
SKIP_BUILD=1
```

Do **not** `git pull` on the live serving checkout. `start.sh` rebuilds when
the `glm53.recipe.stamp` drifts (see [Existing installs](../../README.md#existing-installs-pull-the-instanttensor-image)).
Diff a separate clone, then cut over.

One production pair on 2026-09-16 ran
`sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d`
at checkout `bc68f310f8d5`. That is a receipt, not the current `main` tag.

## Cold boot on GB10 (issue #205)

Independent reproduction after this kit's own ~164 GiB rsync, InstantTensor
default, `MAX_MODEL_LEN=850000`:

```
RuntimeError: buffer_size (1268776960 B) exceeds device memory budget (693237760 B)
```

On GB10, `cudaMemGetInfo` treats page cache as **used** while
`MemAvailable` still passes `start.sh` preflight. Same class as #205.

Workaround that booted this pair (not a new default):

1. Drop the weight files from page cache on both nodes (`echo 3 > /proc/sys/vm/drop_caches` with `CAP_SYS_ADMIN`, or reboot).
2. `CG_ESTIMATE=0` — keep CUDA graphs, drop the over-deducted estimate (#204).
3. `GPU_MEM_UTIL=0.87` — **do not** raise the shipped default. `0.85` exists
   because a 256k prefill at `0.87` with zero `MemAvailable` crashed a head
   on 2026-09-06. InstantTensor at 850k needed the extra KV on this kit
   after the cache drop.

The in-tree loader/hygiene fix is [#230](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/230)
(`GLM53_HOST_MEM_HYGIENE`, InstantTensor budget on UMA). Until that lands,
drop cache before the first `docker run` of a session that just rsynced
weights. Packed host RAM (~119/121 GiB) after a successful EXL3 boot is
expected, not an OOM.

## Readiness

`/health` on this pair was HTTP 200 about **4–6 minutes** after the head
unit started (cached weights, pinned image). DFlash2 / sampler warmup
continues after that. Boot watches must match HTTP + `/v1/models`, not a
log line, and should allow **10–12 minutes** before declaring failure.

`ExecStartPost` can run before Docker has created the new container. The
wait script treats that absence as “not healthy yet”. systemd still fails
the unit if the launcher process exits.

## RoCE

Pin `HEAD_CX7_IF/IB` and `WORKER_CX7_IF/IB` per kit. This pair used a
private CX7 fabric with `NCCL_IB_GID_INDEX=3` valid on **both** ranks.
If the nodes need different GID indices, set `HEAD_GID` / `WORKER_GID`.
An all-zero GID kills a rank ~60 s in (`ibv_modify_qp` errno 61). See
[Running on a different 2×Spark kit](../../README.md#running-on-a-different-2spark-kit).

## Out of scope

Client routing and desktop SSH tunnels do not belong in these units. The API
remains whatever `PORT` / `HEAD_IP` `start.sh` already published.
