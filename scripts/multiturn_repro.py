#!/usr/bin/env python3
"""Multi-turn chat with the served sampling defaults (temperature 1.0, thinking on); counts non-ASCII garbage per turn."""
import json,sys,re,urllib.request
out=sys.argv[1]; max_tokens=int(sys.argv[2]) if len(sys.argv)>2 else 700
msgs=[]; log=[]; total_bad=0
qs=["Tell me about the history of poker in a few paragraphs.","Now compare that with the history of chess, and mention how both spread to the Soviet Union.","Write a short story that mixes the two games, with a magazine editor as the main character.","Summarize everything we discussed as bullet points."]
for i,q in enumerate(qs):
    msgs.append({"role":"user","content":q})
    body={"model":"GLM-5.3-Flash-EXL3","messages":msgs,"max_tokens":max_tokens}
    req=urllib.request.Request("http://127.0.0.1:8888/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(req,timeout=900)); m=d['choices'][0]['message']; txt=m.get('content') or ''; rs=m.get('reasoning_content') or m.get('reasoning') or ''
    msgs.append({"role":"assistant","content":txt})
    bad=re.findall(r'[^\x00-\x7F–—‘’“”…°é]',txt+rs); total_bad+=len(bad)
    print(f"turn{i} prompt_tokens={d['usage']['prompt_tokens']} completion={d['usage']['completion_tokens']} non-ascii={len(bad)} sample={''.join(bad)[:40]!r}"); print('   ',(txt or rs)[-160:].replace('\n',' '))
    log.append({"turn":i,"usage":d['usage'],"reasoning":rs,"content":txt})
json.dump(log,open(out,"w"),indent=1); print("TOTAL non-ascii", total_bad)
