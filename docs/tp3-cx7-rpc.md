# TP3: move worker RPC off management Wi-Fi onto the CX7 triangle

Long image-containing conversations can pause on every tool-result turn even
with high prefix/processor cache hit rates and NCCL using RoCE. The multiprocess
executor broadcasts `NewRequestData`, including `mm_features`, to remote
workers over TCP. Processor cache hits restore those tensors in EngineCore;
they do not suppress the subsequent worker broadcast. A hundreds-of-MB RPC
over Wi-Fi can hold up the first model step while decode remains fast.

`shm_broadcast.py`'s 60-second warning is also used when EngineCore waits for
its local worker's response. It is not proof of exhausted shared memory or a
60-second sleep. Check socket byte counters and request timing before applying
this diagnosis to another deployment.

## Existing settings, separate transports

No vLLM patch or new environment variable is needed. `start-tp3.sh` already
passes the following settings:

| Setting | Purpose |
| --- | --- |
| `HEAD_HOST_IP`, `WORKER_HOST_IP`, `WORKER2_HOST_IP` | Per-rank `VLLM_HOST_IP`, including TCP worker message-queue endpoints |
| `SOCKET_IFNAME` / per-rank `*_SOCKET_IFNAME` | Gloo and NCCL bootstrap interface |
| `*_CX7_IB`, GID and NCCL settings | GPU collectives over RoCE |
| `HEAD_IP`, worker SSH targets | Existing rendezvous/management connections; do not change for this procedure |

Keep the working Gloo/bootstrap, RoCE, SSH and API configuration. Change only
the advertised worker addresses after establishing bidirectional TCP
reachability. `VLLM_HOST_IP` need not be on `GLOO_SOCKET_IFNAME`; it is not the
API's `--host` bind option. The API can remain available on management.

## Worked triangle

These are example addresses and port names. Substitute your actual cable
mapping, interfaces and NetworkManager profile names. Inspect `ip -br -4 addr`,
`ip -4 route`, `ip -4 rule` and `nmcli connection show --active` on every node.
Keep a separate subnet per cable and the existing working MTU/RoCE setup.

| Cable | Endpoint 1 | Endpoint 2 |
| --- | --- | --- |
| head ↔ worker1 | head f0: `192.168.100.10/24` | worker1 f1: `192.168.100.11/24` |
| head ↔ worker2 | head f1: `192.168.102.10/24` | worker2 f0: `192.168.102.12/24` |
| worker1 ↔ worker2 | worker1 f0: `192.168.103.11/24` | worker2 f1: `192.168.103.12/24` |

Advertise one existing address per node:

```bash
# .env.tp3 on the head (keep copies on other nodes consistent)
HEAD_HOST_IP=192.168.100.10
WORKER_HOST_IP=192.168.100.11
WORKER2_HOST_IP=192.168.102.12
# Leave the existing SOCKET_IFNAME / per-rank overrides unchanged.
```

The head already has a connected route to each worker. The other nodes need
three /32 routes to reach a peer's advertised address via that peer's directly
attached cable address:

```bash
# On worker1:
sudo ip route add 192.168.102.12/32 via 192.168.103.12 dev enp1s0f0np0

# On worker2:
sudo ip route add 192.168.100.10/32 via 192.168.102.10 dev enp1s0f0np0
sudo ip route add 192.168.100.11/32 via 192.168.103.11 dev enp1s0f1np1
```

Inspect existing routes first; do not overwrite a conflicting route. These
commands intentionally fail if the route already exists. Each next hop is the
destination node itself, not a transit router: Linux accepts its own local IP
on the other interface. No bridge, bond, additional cable, switch, NAT or new
IP-forwarding setting is required. Source-bound traffic must also have a valid
return route. The measured setup used loose reverse-path filtering (`rp_filter=2`);
strict source validation, policy routing and host firewalls need checking on
other hosts. Verify TCP, not only ICMP, before claiming success.

### Persist without cycling live interfaces

Back up each affected profile first. For NetworkManager-managed CX7 profiles
called `glm-ring-port0` and `glm-ring-port1`, append only these routes:

```bash
# On worker1:
sudo nmcli connection modify glm-ring-port0 +ipv4.routes '192.168.102.12/32 192.168.103.12'

# On worker2:
sudo nmcli connection modify glm-ring-port0 +ipv4.routes '192.168.100.10/32 192.168.102.10'
sudo nmcli connection modify glm-ring-port1 +ipv4.routes '192.168.100.11/32 192.168.103.11'
```

`nmcli connection modify` updates the saved profile; the `ip route add` commands
above update the current kernel routes. Do not run `connection down/up` or a
blanket `netplan apply` while the serving cluster is using CX7. Use the owning
network manager's equivalent persistent route configuration on other systems.
Read back `nmcli -g ipv4.routes connection show <profile>` and the kernel routes.

## Validate, redeploy, measure

1. Back up `.env.tp3`, record the image/weights and save the existing routes.
   Collect a baseline before changing advertised endpoints. Do not tune caches,
   image resolution, batching or model settings during the comparison.
2. On each node, use `ip route get <peer-advertised-IP> from <local-advertised-IP>`
   for both peers. Every route should use the correct CX7 cable. Test all six
   directed pairs with `ping -I <local-advertised-IP> <peer-advertised-IP>`.
3. Drain requests and stop the serving containers gracefully, head first then
   workers. `docker stop --timeout -1 <name>` avoids Docker's forced-stop timer;
   this does not override vLLM's own child-process shutdown behavior. Save
   shutdown logs. The launcher's `stop`/`restart` uses `docker rm -f`, so use
   explicit graceful stops first when preserving in-flight work matters.
4. Set the three `*_HOST_IP` values, then run `./start-tp3.sh` on the head.
   Keep the same image/weights. Startup will recreate the stopped containers.
   Wait for health and the launcher's normal warmup to finish.
5. Confirm each container's `VLLM_HOST_IP` individually (do not dump environment
   secrets). The head log's `mq_connect_ip` should be its CX7 address. Inspect
   `ss -tin` during an image-containing follow-up: the high-byte-count worker
   sockets must now use CX7 addresses/routes, not management Wi-Fi. NCCL should
   still report IB/RoCE. Test ordinary generation and a tool-call/result round trip.

For a bounded synthetic comparison, run this probe **on the head** before and
after redeployment. It requires Pillow and `ss`, sends actual inference work,
and reads the API key from `VLLM_API_KEY` without printing it. Run on a quiet
server; other traffic can contaminate socket-byte deltas. Save output privately:
it includes network endpoints. Do not commit your `.env` or inspection dumps.

```bash
# Before: substitute the workers' current management addresses.
python3 tests/bench_tp3_multimodal_rpc.py --base-url http://127.0.0.1:8888 \
  --peer <worker1-management-IP> --peer <worker2-management-IP> \
  --phase management > before.jsonl

# After: the example advertised addresses above.
python3 tests/bench_tp3_multimodal_rpc.py --base-url http://127.0.0.1:8888 \
  --peer 192.168.100.11 --peer 192.168.102.12 --phase cx7 > after.jsonl
```

The default fixture has 12 distinct generated images and three requests: one
cold request followed by two tool-result turns retaining all images. Compare
the `image_history_sha256`, per-turn TTFT, tool correctness and dominant socket
`bytes_acked` deltas. Cached follow-ups isolate broadcast cost better than cold
prefill. TTFT here is time to the first nonempty content or tool-call delta;
the streaming parser can buffer tokens before exposing that delta. These
small synthetic requests do not qualify full-context quality or
prove that every source of latency has disappeared.

### Measured example (2026-09-21)

A sequential before/after comparison on three GB10 nodes used recipe/image
`775a58b`, the same EXL3 target and DFlash2 draft, TP=3/EP=3, 1M max context,
GPU utilization 0.80, MNBT 7168, four max sequences, FP8 KV and
`MM_IMAGE_TOKENS=2048`. The only serving configuration changes were the three
advertised addresses; three host routes were added and persisted. Gloo/NCCL
bootstrap stayed on management Wi-Fi; NCCL collectives stayed on RoCE. No
image rebuild or model/kernel/cache change was made.

The probe above sent 12 distinct 1536×1024 PNGs and two tool-result follow-ups.
Image-history SHA-256 matched between runs:
`22602ab99f5854f00c71230347f7a91f0e18a8e26f2851ff6a815d621d241889`.
Prompt-token counts matched: 24,599 / 24,630 / 24,661. All six requests returned
the expected `record_check(value=42)` call. Results are one observation per
turn, not medians or a statistical throughput benchmark.

| Request | Wi-Fi TTFT | CX7 TTFT | TCP bytes acknowledged per worker, Wi-Fi → CX7 |
| --- | ---: | ---: | ---: |
| Initial image prompt | 95.131 s | 37.917 s | 230,011,545 → 230,010,073 |
| Tool follow-up 1 | 56.730 s | 2.298 s | 230,002,700 → 230,003,490 |
| Tool follow-up 2 | 61.676 s | 7.888 s | 230,005,308 → 230,005,986 |

Byte deltas cover the whole request on the dominant worker-broadcast socket;
small differences include scheduling/decode RPC metadata. The same ~230 MB
bulk input reached each worker in both configurations. After redeployment,
those sockets used CX7 addresses with PMTU 9000. All six directed,
source-bound TCP connectivity checks passed. The 60-second broadcast warning
reproduced before and did not occur in the candidate run. The remaining 7.9 s
follow-up includes logged DFlash replay/prefill work; this fix does not remove
that work or promise a fixed TTFT.

The normal launcher warmup completed. Health, 1M context metadata, three
concurrent short generations, automatic tool calling and `tool_choice=none`
checks also passed. Persistent NetworkManager route entries were read back;
host reboot behavior was not exercised. Full 1M-context inference, prolonged
soak, other topologies and concurrent image workloads were not tested.

## Rollback

Drain and gracefully stop the candidate, restore the saved `.env.tp3` on all
nodes, and launch the original advertised-address configuration. Once no
serving socket uses the new routes, remove exactly the entries added above:

```bash
# Worker1:
sudo ip route del 192.168.102.12/32 via 192.168.103.12 dev enp1s0f0np0
sudo nmcli connection modify glm-ring-port0 -ipv4.routes '192.168.102.12/32 192.168.103.12'

# Worker2:
sudo ip route del 192.168.100.10/32 via 192.168.102.10 dev enp1s0f0np0
sudo ip route del 192.168.100.11/32 via 192.168.103.11 dev enp1s0f1np1
sudo nmcli connection modify glm-ring-port0 -ipv4.routes '192.168.100.10/32 192.168.102.10'
sudo nmcli connection modify glm-ring-port1 -ipv4.routes '192.168.100.11/32 192.168.103.11'
```

Do not flush route tables or remove pre-existing routes. The procedure leaves
NCCL's working fabric addresses, management access and default routes intact.
