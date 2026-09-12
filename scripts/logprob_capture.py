#!/usr/bin/env python3
"""Capture prompt logprobs (top-20 per position) for fixed texts via /v1/completions (echo, max_tokens=1).
usage: logprob_capture.py OUT.json   |   logprob_capture.py --compare A.json B.json"""
import json,sys,math,urllib.request
TEXTS={
 "prose_silk": open('/home/mia/NewModels/glm-5.3-flash-sm120/logs/overnight-decode-20260907T224521Z/B0/decode-short.json') and None,
}
def load_texts():
    d=json.load(open('/home/mia/NewModels/glm-5.3-flash-sm120/logs/overnight-decode-20260907T224521Z/B0/decode-short.json'))
    out={}
    for cls,rec in d["classes"].items():
        for r in rec["runs"]:
            if r["run"]==0 and r.get("text"): out[f"{cls}{r['prompt_index']}"]=r["text"][:6000]
    hist="\n".join(f"Entry {i}: node NODE{i%7} reported checksum CK-{i:06d} after the maintenance window; the operator logged temperature {40+(i*7)%23} C, fan duty {30+(i*13)%60} percent, and no faults." for i in range(60))
    out["log60"]=hist
    return out
def capture(out):
    res={}
    for name,text in load_texts().items():
        body={"model":"GLM-5.3-Flash-EXL3","prompt":text,"max_tokens":1,"temperature":0,"echo":True,"logprobs":1,"prompt_logprobs":20}
        req=urllib.request.Request("http://127.0.0.1:8888/v1/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        d=json.load(urllib.request.urlopen(req,timeout=600))
        pl=d["choices"][0].get("prompt_logprobs") or []
        res[name]=pl
        nll=[-max(v["logprob"] for v in pos.values() if v.get("rank")==1) for pos in pl[1:] if pos]
        print(name, "positions", len(pl), "mean top1 logprob", round(-sum(nll)/max(1,len(nll)),4))
    json.dump(res,open(out,"w"))
def compare(a,b):
    A=json.load(open(a)); B=json.load(open(b))
    for name in A:
        pa,pb=A[name],B[name]; kls=[]; nlla=[]; nllb=[]; agree=0; n=0
        for x,y in zip(pa[1:],pb[1:]):
            if not x or not y: continue
            # actual token = the one whose entry carries the prompt token; vLLM marks it with rank of the actual token; use union of top-20 for KL
            ax={k:math.exp(v["logprob"]) for k,v in x.items()}; by={k:math.exp(v["logprob"]) for k,v in y.items()}
            keys=set(ax)&set(by)
            if not keys: continue
            za=sum(ax[k] for k in keys); zb=sum(by[k] for k in keys)
            kl=sum((ax[k]/za)*math.log((ax[k]/za)/(by[k]/zb)) for k in keys); kls.append(kl)
            ta=max(x.items(),key=lambda kv:kv[1]["logprob"])[0]; tb=max(y.items(),key=lambda kv:kv[1]["logprob"])[0]; agree+= ta==tb; n+=1
        print(f"{name:12s} positions={n} mean KL(A||B, top-20 support)={sum(kls)/max(1,len(kls)):.4f} nats  argmax agreement={agree/max(1,n):.3f}")
if sys.argv[1]=="--compare": compare(sys.argv[2],sys.argv[3])
else: capture(sys.argv[1])
