#!/bin/bash
# lmc-ab-verified.sh — A/B with a VERIFIED APC reset and per-arm counter deltas,
# so we can prove which cache served each arm instead of guessing from wall time.
#
#   Arm A: prompt NEVER stored anywhere  -> cold prefill (true baseline)
#   Arm B: prompt stored in L2          -> external (LMCache) hit
# APC is force-reset (+reset_running_requests) before each arm, and the reset
# result plus both counter families are recorded per arm.
RES=/tmp/lmc-ab-verified.txt
say(){ printf '\n=== %s ===\n' "$*" | tee -a "$RES"; }
: > "$RES"

met(){ curl -s --max-time 5 http://127.0.0.1:8899/metrics 2>/dev/null | grep -E "^vllm:$1" | sed 's/.*} //' | head -1; }
srv(){ curl -s --max-time 5 http://127.0.0.1:8080/metrics 2>/dev/null | grep -E "^lmcache_mp_$1" | sed 's/.*} //' | head -1; }

reset_apc(){
  local out
  out=$(curl -s -X POST --max-time 40 "http://127.0.0.1:8899/reset_prefix_cache?reset_running_requests=true&reset_external=false")
  echo "  reset -> $out" | tee -a "$RES"
  sleep 6
}

arm(){ # $1=tag $2=promptfile
  local tag="$1" f="$2"
  local ah0 eq0 lr0
  ah0=$(met prefix_cache_hits_total); eq0=$(met external_prefix_cache_hits_total); lr0=$(srv lookup_requested_tokens_total)
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
u=o["usage"]
print("%s prompt=%d wall=%.2fs"%(tag,u["prompt_tokens"],w))
PY
  sleep 8
  local ah1 eq1 lr1
  ah1=$(met prefix_cache_hits_total); eq1=$(met external_prefix_cache_hits_total); lr1=$(srv lookup_requested_tokens_total)
  python3 -c "
a=float('${ah1:-0}' or 0)-float('${ah0:-0}' or 0)
e=float('${eq1:-0}' or 0)-float('${eq0:-0}' or 0)
l=float('${lr1:-0}' or 0)-float('${lr0:-0}' or 0)
print('  delta APC_hits=%d  EXTERNAL_hits=%d  srv_lookup_req=%d'%(a,e,l))
print('  served_by: ' + ('LMCache(external)' if e>0 else ('vLLM APC' if a>0 else 'COLD')))" | tee -a "$RES"
}

python3 - <<'PY' > /tmp/ab-never.txt
import random
random.seed(20260924)
W=["north","south","east","west","ridge","valley","harbor","delta","summit","meadow",
   "cedar","willow","granite","amber","ivory","cobalt","quartz","ember","falcon","otter"]
print(" ".join(random.choice(W) for _ in range(46000)))
PY

say "$(date +%H:%M:%S) verified A/B (no restart; force APC reset between arms)"

say "ARM A — never-stored prompt, APC force-reset (true cold baseline)"
reset_apc
arm "A-cold" /tmp/ab-never.txt

say "ARM B — stored prime prompt, APC force-reset (external hit expected)"
reset_apc
arm "B-external" /tmp/prime-prompt.txt

say "VERDICT"
python3 - <<'PY' | tee -a "$RES"
import re
t=open("/tmp/lmc-ab-verified.txt").read()
def w(tag):
    m=re.search(r"%s prompt=(\d+) wall=([\d.]+)s"%re.escape(tag),t)
    return (int(m.group(1)),float(m.group(2))) if m else (None,None)
pa,wa=w("A-cold"); pb,wb=w("B-external")
if pa and pb:
    print("  A cold     : %d tok / %.2fs = %.0f tok/s"%(pa,wa,pa/wa))
    print("  B external : %d tok / %.2fs = %.0f tok/s"%(pb,wb,pb/wb))
    print("  speedup (normalised by length): %.1fx"%(wa/pb*pa/wa*1.0 if False else (wa/pa)/(wb/pb)))
else:
    print("  incomplete")
PY
echo "done: $RES"
