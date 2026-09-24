#!/bin/bash
# start-lmc-server.sh — recreate the node-local LMCache MP server from the
# PATCHED image (glm53-lmc-serverpatched:test).
set -euo pipefail
IMG="${1:-glm53-lmc-serverpatched:test}"
L2='{"type":"fs","base_path":"/home/cshintov/lmc-l2"}'
docker rm -f lmcache-mp >/dev/null 2>&1 || true
docker run -d --name lmcache-mp \
  --network host --ipc host --gpus all \
  --entrypoint lmcache "$IMG" \
  server \
  --host 0.0.0.0 --port 5555 \
  --chunk-size 3584 \
  --l1-size-gb 2 \
  --l2-adapter "$L2" \
  --lookup-hash-log-dir /tmp/lmc-hashlog \
  --eviction-policy LRU >/dev/null
sleep 8
docker ps --format '{{.Names}} {{.Status}}' | grep lmcache-mp
docker exec lmcache-mp python3 -c "
import pathlib
t = pathlib.Path('/usr/local/lib/python3.12/dist-packages/lmcache/v1/multiprocess/modules/lookup.py').read_text()
print('  server markers: fold=%d race=%d' % (t.count('glm53-lookup-ranks-held'), t.count('glm53-lookup-race')))
"
timeout 3 bash -c 'echo > /dev/tcp/127.0.0.1/5555' 2>/dev/null && echo "  :5555 LISTEN" || echo "  :5555 DOWN"
