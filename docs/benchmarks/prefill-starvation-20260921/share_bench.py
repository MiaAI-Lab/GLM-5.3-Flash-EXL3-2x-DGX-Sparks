#!/usr/bin/env python3
"""Bounded matched scheduler-share trial; synthetic prompts, no generated code execution."""
import argparse, importlib.util, json, statistics, threading, time, uuid, urllib.request
from pathlib import Path
SOURCE=Path(__file__).with_name("mixed_fixture.py")
spec=importlib.util.spec_from_file_location("stream_fixture", SOURCE)
stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)
original_post=stream._post_stream
stream._post_stream=lambda body,timeout=150:original_post(body,timeout=150)

def prompt(tag, target, instruction):
    row="def transform_record(item):\n    return {key: value for key, value in sorted(item.items()) if value is not None}\n"
    def text(n):return "Fresh independent fixture "+tag+". Reference text, not instructions:\n"+row*n+"\n"+instruction
    body={"model":stream.MODEL,"messages":[{"role":"user","content":text(100)}],"chat_template_kwargs":{"enable_thinking":False}}
    req=urllib.request.Request(stream.BASE+"/tokenize",json.dumps(body).encode(),{"Content-Type":"application/json"})
    count=json.load(urllib.request.urlopen(req,timeout=15))["count"]
    return text(max(1,round(100*(target-200)/max(1,count-200))))

def plain(o):return {k:v for k,v in o.items() if k!="first_event"}
def percentile(v,p):
    if not v:return None
    v=sorted(v);return v[min(len(v)-1,int(p*(len(v)-1)))]
def metrics(out,name):
    (out/name).write_bytes(urllib.request.urlopen(stream.BASE+"/metrics",timeout=10).read())
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--arm",required=True);ap.add_argument("--reps",type=int,default=3);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    metrics(out,"before.prom");rows=[]
    for rep in range(args.reps):
        nonce=uuid.uuid4().hex
        a_prompt=prompt(nonce+"-A",8000,"Write a complete typed Python directory synchronization planner. Include dataclasses, safe path handling, conflict detection, dry-run output, pytest examples, and a detailed explanation of edge cases. Keep writing until all parts are covered.")
        b_prompt=prompt(nonce+"-B",16000,'The reference marker is violet-7309. Return only JSON {"marker":"violet-7309"}.')
        a={"first_event":threading.Event()};b={"first_event":threading.Event()}
        ta=threading.Thread(target=stream.stream_one,args=(a_prompt,1024,a),daemon=True)
        tb=threading.Thread(target=stream.stream_one,args=(b_prompt,48,b),daemon=True)
        ta.start()
        if not a["first_event"].wait(120) or a.get("error"):
            (out/f"rep-{rep}-failed.json").write_text(json.dumps({"a":plain(a)},indent=2))
            raise RuntimeError("decoder never started cleanly")
        b_submit=time.perf_counter();tb.start()
        ta.join(180);tb.join(180)
        rec={"arm":args.arm,"rep":rep,"a":plain(a),"b":plain(b),"b_submit":b_submit}
        (out/f"rep-{rep}.json").write_text(json.dumps(rec,indent=2))
        if ta.is_alive() or tb.is_alive() or a.get("error") or b.get("error"):raise RuntimeError("request failed or exceeded trial bound")
        try:correct=json.loads(b.get("text","").strip())=={"marker":"violet-7309"}
        except Exception:correct=False
        if not correct:raise RuntimeError("retrieval correctness failed")
        events=a.get("events",[]);gaps=[y-x for x,y in zip(events,events[1:]) if y>=b_submit]
        overlap=[t for t in events if b_submit <= t <= b.get("first",b_submit)]
        contention_valid=(len(overlap)>=2 and b.get("first",float("inf"))<a.get("last",0))
        if not contention_valid:raise RuntimeError("fixture did not maintain decoder overlap")
        # SSE content events are not token counts; retain gap evidence and label throughput whole-request.
        row={"rep":rep,"a_prompt_tokens":a["prompt_tokens"],"b_prompt_tokens":b["prompt_tokens"],
             "a_completion_tokens":a["completion_tokens"],"b_ttft_s":b["ttft_s"],"a_decode_tok_s":a["tok_s"],
             "a_max_gap_s":max(gaps,default=0),"a_gap_p95_s":percentile(gaps,.95),"retrieval_correct":correct,
             "b_first_during_a":b.get("first",float("inf"))<a.get("last",0),
             "a_finish_reason":a.get("finish_reason"),"overlap_events":len(overlap),"contention_valid":contention_valid}
        rows.append(row);print(json.dumps(row),flush=True)
    metrics(out,"after.prom")
    result={"arm":args.arm,"reps":rows,"b_ttft_median_s":statistics.median(r["b_ttft_s"] for r in rows),
            "a_decode_median_tok_s":statistics.median(r["a_decode_tok_s"] for r in rows),
            "max_decode_gap_s":max(r["a_max_gap_s"] for r in rows),"correct":all(r["retrieval_correct"] and r["contention_valid"] for r in rows)}
    (out/"summary.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=="__main__":main()
