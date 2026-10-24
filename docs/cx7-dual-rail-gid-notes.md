# CX7 preflight: a stable-privacy IPv6 address can silently break dual rail (2026-09-28)

Field report, second 2×Spark kit (`spark-d931`/`spark-ea54`, recipe `943912c`).

## Symptom

A previously working **dual-rail** configuration refused to start after a reboot:

```
[glm53-exl3] worker GID index 3 is EMPTY on rocep1s0f0
[glm53-exl3] ERROR: set NCCL_IB_GID_INDEX (same index both ranks) or HEAD_GID/WORKER_GID
              (per rank) in .env to populated indices
```

`.env` was the shipped dual-rail form (`#*_CX7_IB=rocep1s0f0,roceP2p1s0f0`, `HEAD_GID=3`,
`WORKER_GID=3`) and the same file had started fine before the reboot.

## Root cause: the IPv4 GID is pushed down the table by an extra IPv6 entry

`show_gids` on the worker, before:

```
rocep1s0f0 1 0 fe80::4ebb:47ff:fe80:ea55  v1  enp1s0f0np0    <- MAC-derived link-local
rocep1s0f0 1 1 fe80::4ebb:47ff:fe80:ea55  v2  enp1s0f0np0
rocep1s0f0 1 2 fe80::d4d1:b2a1:94ce:6b30  v1  enp1s0f0np0    <- NetworkManager stable-privacy
rocep1s0f0 1 4 fe80::d4d1:b2a1:94ce:6b30  v2  enp1s0f0np0       link-local (extra pair!)
rocep1s0f0 1 5 ::ffff:10.0.0.2            v1  enp1s0f0np0    <- IPv4 pushed to 5/6
rocep1s0f0 1 6 ::ffff:10.0.0.2            v2  enp1s0f0np0
```

The head, and the worker's *other* HCA, each carry only one link-local, so their IPv4 lands at
the expected `2 (v1) / 3 (v2)`:

```
head   rocep1s0f0    idx2 10.0.0.1 v1   idx3 10.0.0.1 v2
head   roceP2p1s0f0  idx2 10.0.1.1 v1   idx3 10.0.1.1 v2
worker roceP2p1s0f0  idx2 10.0.1.2 v1   idx3 10.0.1.2 v2
worker rocep1s0f0    idx2 (empty)       idx3 (empty)      <- the one that moved
```

Consequence for a dual-rail `NCCL_IB_HCA`: the per-rank index must be populated on **every**
listed device, and after the shift there is **no single index** valid on both worker devices
(5/6 on rail 1, 2/3 on rail 2). Single-rail still worked only because `WORKER_GID=6` happened to
be right for the one listed device.

Why it appeared only after a reboot: the stable-privacy address is NM's default
(`ipv6.addr-gen-mode=stable-privacy`), and whether it exists at GID-table build time depends on
the link/address ordering at boot — i.e. the table layout is not stable across boots.

## Fix (deterministic, survives reboots)

```bash
# on the affected node (needs root)
con=$(nmcli -g NAME,DEVICE connection show --active | awk -F: '$2=="enp1s0f0np0"{print $1; exit}')
sudo nmcli connection modify "$con" ipv6.addr-gen-mode eui64   # one link-local, not two

# rebuild the GID table from the CURRENT addresses:
pci=$(readlink -f /sys/class/infiniband/rocep1s0f0/device | xargs basename)
echo "$pci" | sudo tee /sys/bus/pci/drivers/mlx5_core/unbind
sleep 3
echo "$pci" | sudo tee /sys/bus/pci/drivers/mlx5_core/bind
# verify
show_gids | grep rocep1s0f0
```

Result — `idx2 v1 / idx3 v2 = ::ffff:10.0.0.2`, matching the other three device instances, so
`WORKER_GID=3` works for both worker HCAs again.

Notes:

* A flush+re-add of the address is **not** enough — we tried it and NM restored the same
  layout; the table only moved after the driver rebind.
* If the node is reachable only over the rail being rebound, keep a second path (the second
  rail in a dual-rail setup is exactly that) and run the rebind detached with a self-heal loop
  that re-adds the address if NM does not.
* The launcher's own hint is accurate and worth reading literally: an all-zero entry kills that
  rank ~60 s in with `ibv_modify_qp` errno 61.

## Companion issue: the second rail's IP can vanish

On the head, rail 2 is configured by a one-shot unit whose ExecStart is error-ignoring
(`ip addr add … ; ip link set … up`). After a link flap the address was gone while the unit
still reported `active (exited)`, which fails the same preflight. A 2-minute guard that
re-adds `10.0.1.1/24` when missing removes that failure mode (a unit-level `Restart=on-failure`
plus `BindsTo` the interface would work equally well).

## Is dual rail worth it? (same-machine A/B, both directions)

Raw link measurements (`ib_write_bw -D 6`, 65536 B messages):

| rail | device | subnet | measured | line rate |
|---|---|---|---:|---|
| rail 1 | `rocep1s0f0` | 10.0.0.x | **108.94 Gb/s** | 200 Gb/s (4X HDR) |
| rail 2 | `roceP2p1s0f0` | 10.0.1.x | **108.91 Gb/s** | 200 Gb/s (4X HDR) |

Serving A/B, same kit/config, only `NCCL_IB_HCA` changed:

| workload | dual rail | single rail | delta |
|---|---:|---:|---:|
| cold prefill, 21.2k-token prompt (3 runs each) | **14.10 / 14.13 / 14.21 s** | 14.81 / 14.84 / 15.03 s | **+5.0 %** |
| structured ×1 | 83.46 tok/s | 83.58 tok/s | neutral |
| prose ×1 | 38.61 tok/s | 38.07 tok/s | neutral |
| prose ×8 aggregate | 123.79 tok/s | 124.28 tok/s | neutral |

Interpretation: decode's per-layer allreduce moves tens of KB to ~0.5 MB per step, so it is
**latency**-bound and extra rails do not help; the prefill chunk's allreduce moves ~58 MB per
layer at 7168 tokens, which is **bandwidth**-bound, so the second rail halves that term. Net:
prefill-only gain, no decode cost — worth enabling, provided the two failure modes above are
handled.

## Suggestion for the repo

* `README.md`/`.env.example` could state the invariant explicitly: *the configured per-rank GID
  index must be populated on every device in `<rank>_CX7_IB`, and NM's default
  `ipv6.addr-gen-mode` can break that after a reboot on a dual-rail kit.*
* A one-line preflight addition would turn a hard refusal into an actionable message:
  when the configured index is empty on some device, also print
  `nmcli -g ipv6.addr-gen-mode connection show <con>` for that interface.
