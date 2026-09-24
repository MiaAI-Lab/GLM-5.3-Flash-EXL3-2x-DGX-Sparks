#!/bin/bash
# lmc-apc-reset-proof.sh — decisive external-hit proof with NO extra restarts.
#
# Idea: /reset_prefix_cache clears ONLY vLLM's in-process APC while the LMCache
# L2/L1 data (and the serving process) stays untouched. So:
#   1. wait health=200 (this boot has GLM53_EXPOSE_CACHE_RESET=1)
#   2. prime the prompt  -> APC fills, external store happens
#   3. POST /reset_prefix_cache  -> APC cold again, LMCache data RETAINED
#   4. resend identical prompt -> any hit MUST come from LMCache
RES=/tmp/lmc-apc-proof.txt
PRIME=/tmp/prime-prompt.txt
say(){ printf '\n=== %s ===\n' "$*" | tee -a "$RES"; }
: > "$RES"

srv(){ curl -s --max-time 5 http://127.0.0.1:8080/metrics 2>/dev/null | grep -E "^lmcache_mp_$1" | sed 's/.*} //' | head -1; }
vllm(){ curl -s --max-time 5 http://127.0.0.1:8899/metrics 2>/dev/null | grep -E "^vllm:$1" | sed 's/.*} //' | head -1; }

send(){ python3 - "$1" "$PRIME" <<'PY' 2>&1 | tee -a "$RES"
import json,sys,urllib.request,time
tag=sys.argv[1]; ctx=open(sys.argv[2]).read()
p={"model":"GLM-5.3-Flash-EXL3","messages":[{"role":"user","content":ctx+"\n\nReply OK."}],
   "temperature":0,"max_tokens":16,"stream":False,"chat_template_kwargs":{"enable_thinking":False}}
req=urllib.request.Request("http://127.0.0.1:8899/v1/chat/completions",
    data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
t0=time.monotonic()
with op.open(req,timeout=1800) as r: o=json.loads(r.read().decode())
print("%s prompt=%d wall=%.1fs"%(tag,o["usage"]["prompt_tokens"],time.monotonic()-t0))
PY
}

say "wait for health=200"
h=000
for i in $(seq 1 100); do
  h=$(curl -s -o /dev/null -w "%{http_code}" --max-time 4 http://127.0.0.1:8899/health 2>/dev/null)
  [ $((i % 5)) = 0 ] && echo "t=$((i*20))s health=$h" | tee -a "$RES"
  [ "$h" = "200" ] && break
  sleep 20
done
[ "$h" = "200" ] || { echo "BOOT TIMEOUT" | tee -a "$RES"; exit 1; }
echo "UP $(date +%H:%M:%S)" | tee -a "$RES"
sleep 20

say "dev/cache route mounted?"
curl -s -o /dev/null -w "  /reset_prefix_cache -> %{http_code}\n" -X POST --max-time 15 "http://127.0.0.1:8899/reset_prefix_cache" | tee -a "$RES"

say "STEP 1: prime (APC fills + external store)"
send "PRIME"
sleep 8
echo "  srv l1_write=$(srv l1_write_chunks_total) lookup_req=$(srv lookup_requested_tokens_total)" | tee -a "$RES"
echo "  ext_hits=$(vllm external_prefix_cache_hits_total)" | tee -a "$RES"

say "STEP 2: reset vLLM APC (reset_external=false -> LMCache data RETAINED)"
curl -s -X POST --max-time 30 "http://127.0.0.1:8899/reset_prefix_cache?reset_external=false" | tee -a "$RES"; echo | tee -a "$RES"
sleep 5
echo "  srv L2 files still present: $(docker exec lmcache-mp sh -c 'ls /home/cshintov/lmc-l2 | wc -l')" | tee -a "$RES"
echo "  ext_hits before resend=$(vllm external_prefix_cache_hits_total)" | tee -a "$RES"

say "STEP 3: resend identical prompt (APC cold -> hit must be EXTERNAL)"
send "RESEND"
sleep 8

say "RESULT"
H=$(vllm external_prefix_cache_hits_total); LH=$(srv lookup_hit_tokens_total); LR=$(srv lookup_requested_tokens_total)
echo "  EXT_HITS=$H" | tee -a "$RES"
echo "  srv lookup_req=$LR hit=$LH l1_read=$(srv l1_read_chunks_total)" | tee -a "$RES"
say "server traces"
docker logs lmcache-mp 2>&1 | grep -E "LMCTRACE2|LMCTRACE RESULT" | tail -14 | tee -a "$RES"
say "VERDICT"
python3 -c "
h=float('${H:-0}' or 0); lh=float('${LH:-0}' or 0); lr=float('${LR:-0}' or 0)
print('  external_prefix_cache_hits_total =', h)
print('  server lookup_req / hit_tokens   =', lr, '/', lh)
ok = (h>0 or lh>0)
print('  ==> EXTERNAL KV REUSE ' + ('PROVEN' if ok else 'NOT PROVEN'))
print('  ==> connector consulted LMCache: ' + ('YES' if lr>0 else 'NO'))
" | tee -a "$RES"
echo "done: $RES"
