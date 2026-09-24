#!/bin/bash
# lmc-ab-boot.sh — clean A/B on ONE fresh boot (the only way to get a truly cold
# in-process cache: /reset_prefix_cache fails ("Failed to reset KV cache even
# when all the running requests are done") while LMCache holds the L1 read locks).
#
#   arm A: prompt that has NEVER been stored -> cold prefill baseline
#   arm B: prime prompt (in L1/L2)           -> external (LMCache) hit
# Arm A runs FIRST so the fresh APC is cold for it; arm A's prompt is not in
# LMCache, so it cannot be external-served either.
RES=/tmp/lmc-ab-boot.txt
say(){ printf '\n=== %s ===\n' "$*" | tee -a "$RES"; }
: > "$RES"

met(){ curl -s --max-time 5 http://127.0.0.1:8899/metrics 2>/dev/null | grep -E "^vllm:$1" | sed 's/.*} //' | head -1; }
srv(){ curl -s --max-time 5 http://127.0.0.1:8080/metrics 2>/dev/null | grep -E "^lmcache_mp_$1" | sed 's/.*} //' | head -1; }

say "wait for health=200 (fresh boot => APC cold)"
h=000
for i in $(seq 1 100); do
  h=$(curl -s -o /dev/null -w "%{http_code}" --max-time 4 http://127.0.0.1:8899/health 2>/dev/null)
  [ $((i % 5)) = 0 ] && echo "t=$((i*20))s health=$h" | tee -a "$RES"
  [ "$h" = "200" ] && break
  sleep 20
done
[ "$h" = "200" ] || { echo "BOOT TIMEOUT" | tee -a "$RES"; exit 1; }
echo "  UP $(date +%H:%M:%S)" | tee -a "$RES"
sleep 25

# arm A prompt: new seed AND a disjoint vocabulary => no shared prefix with
# anything previously stored, and never sent before.
python3 - <<'PY' > /tmp/ab-coldprompt.txt
import random
random.seed(555003)
W=["zephyr","obsidian","tundra","basalt","quartzite","lichen","marrow","cistern",
   "lattice","verdant","sable","cinder","fathom","gossamer","harrier","jetty",
   "kelp","loam","marl","nimbus"]
print(" ".join(random.choice(W) for _ in range(46000)))
PY

arm(){ # $1=tag $2=file
  local tag="$1" f="$2" a0 e0 l0 t0
  a0=$(met prefix_cache_hits_total); e0=$(met external_prefix_cache_hits_total); t0=$(srv lookup_requested_tokens_total)
  python3 - "$tag" "$f" <<'PY' 2>&1 | tee -a "$RES"
import json,sys,urllib.request,time
tag,src=sys.argv[1],sys.argv[2]
ctx=open(src).read()
p={"model":"GLM-5.3-Flash-EXL3","messages":[{"role":"user","content":ctx+"\n\nReply OK."}],
   "temperature":0,"max_tokens":16,"stream":False,"chat_template_kwargs":{"enable_thinking":False}}
req=urllib.request.Request("http://127.0.0.1:8899/v1/chat/completions",
    data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
t0=time.monotonic()
with op.open(req,timeout=3600) as r: o=json.loads(r.read().decode())
w=time.monotonic()-t0
print("%s prompt=%d wall=%.2fs"%(tag,o["usage"]["prompt_tokens"],w))
PY
  sleep 8
  python3 -c "
a=float('$(met prefix_cache_hits_total)' or 0)-float('${a0:-0}' or 0)
e=float('$(met external_prefix_cache_hits_total)' or 0)-float('${e0:-0}' or 0)
t=float('$(srv lookup_requested_tokens_total)' or 0)-float('${t0:-0}' or 0)
print('  delta: APC_hits=%d EXTERNAL_hits=%d srv_lookup_req=%d'%(a,e,t))
print('  served_by: ' + ('LMCache(external)' if e>0 else ('vLLM APC' if a>0 else 'COLD prefill')))" | tee -a "$RES"
}

say "ARM A — never-stored prompt, fresh boot (cold baseline)"
arm "A-cold" /tmp/ab-coldprompt.txt
say "ARM B — prime prompt, in L1/L2 from earlier boots (external expected)"
arm "B-external" /tmp/prime-prompt.txt

say "VERDICT"
python3 - <<'PY' | tee -a "$RES"
import re
t=open("/tmp/lmc-ab-boot.txt").read()
def g(tag):
    m=re.search(r"%s prompt=(\d+) wall=([\d.]+)s"%re.escape(tag),t); return (int(m.group(1)),float(m.group(2))) if m else (None,None)
pa,wa=g("A-cold"); pb,wb=g("B-external")
if pa and pb:
    ra,rb=pa/wa,pb/wb
    print("  A cold prefill : %d tok / %.2fs = %6.0f tok/s"%(pa,wa,ra))
    print("  B external hit : %d tok / %.2fs = %6.0f tok/s"%(pb,wb,rb))
    print("  => prefill-phase speedup: %.1fx  (saved %.1fs on a %d-token prompt)"%(rb/ra, wa*(pb/pa)-wb, pb))
else:
    print("  incomplete:",pa,pb)
PY
echo "done: $RES"
