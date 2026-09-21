# Switchless four-Spark CX7 ring (TP=4)

Opt-in path for `./start-tp4.sh` when four GB10 nodes are cabled as a DAC ring
with **no RoCE switch**. Switched / default TP=4 is unchanged unless
`NCCL_SWITCHLESS_RING_ONLY=1`.

This is the GLM port of the DeepSeek ring work:

- [DeepSeek PR #19](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/19)
  (`carlosduque-incoxe`) — launcher integration, overlay-without-`LD_PRELOAD`,
  all-rank preflight.
- [DeepSeek PR #3](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/3)
  (`@Saolence`) — original ring integration.
- [FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring) —
  NCCL `SWITCHLESS_RING_ONLY` / skip-tree patches.

It ports the **transport layer**, not SGLang-specific DeepSeek knobs, into this
vLLM TP=4 launcher.

## Why stock NCCL dies on a ring

A 4-node ring has no direct fabric path between opposite ranks (rank0 ↔ rank2).
NCCL still builds a *tree* in addition to the ring. RoCE queue-pair setup does
not follow the IP route, so `ncclTransportTreeConnect` fails with an unhandled
system error after `Connected all rings`.

The patched host library plus `NCCL_SWITCHLESS_RING_ONLY=1` keeps Ring and
skips tree/PAT connect. Image NCCL is not that library.

## Physical cable example

Four DACs, no switch, no diagonal links. Each Spark has two QSFP ports:

```text
  rank0                         rank1
  CX7-0 -------- cable 1 ------- CX7-1
  CX7-1                         CX7-0
    |                             |
  cable 4                       cable 2
    |                             |
  CX7-0                         CX7-1
  CX7-1 -------- cable 3 ------- CX7-0
  rank3                         rank2
```

Give **one /24 per DAC** (subnet-aware routing). Keep vLLM Gloo / SSH on a
separate all-rank management LAN (`NCCL_SOCKET_IFNAME` / `*_CX7_IF`). Jumbo
9000 on the fabric. Do not reuse a production pair's `172.31.200.0/31`.

Addresses below are illustrative fabric-only ranges — do not commit fleet
management IPs:

```text
cable 1: 172.31.201.0/24
cable 2: 172.31.202.0/24
cable 3: 172.31.203.0/24
cable 4: 172.31.204.0/24
```

Inspect HCAs with `ibdev2netdev`. Do not infer HCA names from Linux NIC names.
Pick the RoCEv2 `::ffff:<ip>` GID on **both** cable-facing ports (`HEAD_GID` /
`WORKER*_GID`).

## Image and NCCL (do not rebuild)

Pin the public InstantTensor image; the tag is mutable:

```text
ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor
digest sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d
id     sha256:ef9f5013c41adf93a5171abb809283be0f202b88fd14098e4434337715614c62
platform linux/arm64
```

Copy the **same** trusted host library that already carries
`SWITCHLESS_RING_ONLY` onto every rank (this kit used NCCL 2.30.7):

```text
$HOME/nccl-2.30.7/libnccl.so.2.30.7
```

Provenance from the DeepSeek ring PRs: NCCL commit
`73cf112295c33aee2b895f329f592f2a9b4b0f97`; sparkring blob
`f4853e84334eaa3f980dce69a12660d8f1774d7c`, file
`spark_transport/nccl/nccl-2.30.7-dual-pci-domain.patch`.

`doctor-ring` checks the marker and that the SHA-256, image id and in-image pip
NCCL path match on all four ranks. That is consistency, not authenticity.

The launcher bind-mounts that one `.so` **over** the image pip path discovered
at preflight. It does **not** use `LD_PRELOAD` in ring mode (two NCCL runtimes
visible is the DeepEP abort on the sibling recipe).

## Enable

```bash
cp -n .env.tp4.example .env.tp4
# merge the opt-in block from .env.tp4.ring.example
# fill four management IPs, management IFs, two HCAs and GID per rank
./start-tp4.sh doctor-ring          # no containers replaced
./start-tp4.sh start                # refuses if any GPU is occupied
./start-tp4.sh status
```

`restart` is rejected in ring mode: run `doctor-ring`, then explicit `stop`
and `start`. Preflight runs again immediately before `docker rm`. A failure
after that point can still leave a partial boot; recover with `stop`.

Commissioning baseline in `.env.tp4.ring.example` is conservative: 128k, 4
seqs, DFlash off, GMU 0.75. It is not the measured production profile.

## What this kit measured

See [tp4-switchless-ring-results.md](tp4-switchless-ring-results.md).

Safe production on this 4× 200 GbE ring (2026-09-16):

| Knob | Value | Why |
|---|---|---|
| `MAX_MODEL_LEN` | `262144` | 512k + DFlash hung on ~32k cold prefill |
| `MAX_NUM_SEQS` | `8` | C8 now scales; 4 left C8 queued |
| `GPU_MEM_UTIL` | `0.75` | `0.85` produced `NV_ERR_NO_MEMORY` here |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | same as switched TP=4 default |
| `SPEC_METHOD` / `DFLASH_TOKENS` | `dflash` / `3` | k=3 as in switched TP=4 |
| `DFLASH_DRAFT_TP` | `4` | `1` was slower on this ring (C4 145→130, C8 179→176) |
| `NCCL_NCHANNELS` | `8` | 4 channels left C8 stuck near C4 |
| mixed prefill | `fair` | inherited; not re-tuned |

Do **not** combine DFlash with 512k/1M on this ring. Do not advertise switched
TP=4 1M numbers as ring numbers.

## Logs that prove the ring, not the tree

With `NCCL_DEBUG=INFO` (commissioning) or `WARN` (production), look for the
patched library skipping tree/PAT. Opposite ranks must not attempt a direct
tree QP.

## Credits

NCCL patch: FujitsuPolycom/sparkring. Ring integration: @Saolence. GLM port
and four-node GB10 measurements: this PR.
