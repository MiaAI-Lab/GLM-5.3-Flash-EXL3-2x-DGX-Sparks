#!/usr/bin/env python3
"""Sampled long generations (served defaults) + one long-prompt request; flags non-ASCII garbage and repetition."""
import json,sys,re,urllib.request
out=sys.argv[1]; res=[]
def call(msgs,max_tokens,extra=None):
    body={"model":"GLM-5.3-Flash-EXL3","messages":msgs,"max_tokens":max_tokens}; body.update(extra or {})
    req=urllib.request.Request("http://127.0.0.1:8888/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(req,timeout=1800)); m=d['choices'][0]['message']; return d, (m.get('reasoning_content') or m.get('reasoning') or ''), (m.get('content') or '')
hist="\n".join(f"Entry {i}: node NODE{i%7} reported checksum CK-{i:06d} after the maintenance window; the operator logged temperature {40+(i*7)%23} C, fan duty {30+(i*13)%60} percent, and no faults." for i in range(150))
cases=[("sampled-2000-essay",[{"role":"user","content":"Write a long, detailed essay on the history and economics of the Silk Road, at least 1500 words."}],2000,None),
       ("sampled-2000-story",[{"role":"user","content":"Write a long short story about a magazine editor who plays chess and poker."}],2000,None),
       ("sampled-longprompt-5k",[{"role":"user","content":hist+"\n\nWrite a thorough narrative summary of the log above in several paragraphs."}],800,None),
       ("temp0-2000",[{"role":"user","content":"Write a long, detailed essay on the history and economics of the Silk Road, at least 1500 words."}],2000,{"temperature":0})]
for tag,msgs,mt,extra in cases:
    d,rs,txt=call(msgs,mt,extra); allt=rs+txt
    bad=re.findall(r'[^\x00-\x7F–—‘’“”…°é§]',allt); words=allt.split(); rep=len(words)-len(set(words[-300:])) if words else 0
    print(f"{tag:24s} prompt={d['usage']['prompt_tokens']} completion={d['usage']['completion_tokens']} non-ascii={len(bad)} sample={''.join(bad)[:30]!r}"); print('    tail:',allt[-140:].replace('\n',' '))
    res.append({"tag":tag,"usage":d['usage'],"non_ascii":len(bad),"reasoning":rs,"content":txt})
json.dump(res,open(out,"w"),indent=1); print("DONE")
