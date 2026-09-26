#!/usr/bin/env python3
"""Two bounded eight-request bursts; preserve per-request streaming evidence."""
import argparse,json,statistics,subprocess,sys,threading,time,uuid
from pathlib import Path
from share_bench import stream,prompt,plain,metrics
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);summaries=[]
    # Match the three mixed repetitions that precede each eight-lane burst.
    if out.name=="lane4-burst":
        subprocess.run([sys.executable,str(Path(__file__).with_name("share_bench.py")),"--arm","lane4-prime","--out",str(out.parent/"lane4-prime"),"--reps","3"],check=True,timeout=900)
        from health_counters import healthy_arm
        assert healthy_arm(out.parent/"lane4-prime"),"lane4 warm-up health counters failed"
    metrics(out,"before.prom")
    for rep in range(2):
        prompts=[prompt(uuid.uuid4().hex,8000,"Write a detailed Python code review checklist with concrete examples covering correctness, security, concurrency, testing, typing, performance, and error handling. Explain each example.") for _ in range(8)]
        barrier=threading.Barrier(9);results=[{"first_event":threading.Event()} for _ in prompts]
        def go(i):
            barrier.wait();stream.stream_one(prompts[i],256,results[i])
        threads=[threading.Thread(target=go,args=(i,),daemon=True) for i in range(8)]
        for t in threads:t.start()
        started=time.perf_counter();barrier.wait()
        deadline=started+240
        for t in threads:t.join(max(0,deadline-time.perf_counter()))
        (out/f"rep-{rep}.json").write_text(json.dumps([plain(r) for r in results],indent=2))
        assert not any(t.is_alive() for t in threads),"burst timeout"
        assert all(not r.get("error") and r.get("ttft_s") is not None and r.get("completion_tokens",0)>0 for r in results),"burst request failed"
        ttfts=sorted(r["ttft_s"] for r in results);makespan=max(r["t0"]+r["wall_s"] for r in results)-started
        summary={"rep":rep,"ttft_median_s":statistics.median(ttfts),"ttft_p95_s":ttfts[-1],
                 "makespan_s":makespan,"aggregate_tok_s":sum(r["completion_tokens"] for r in results)/makespan,
                 "prompt_tokens":[r["prompt_tokens"] for r in results],"completion_tokens":[r["completion_tokens"] for r in results]}
        summaries.append(summary);print(json.dumps(summary),flush=True)
    metrics(out,"after.prom")
    s={"reps":summaries,**{k:statistics.median(r[k] for r in summaries) for k in ["ttft_median_s","ttft_p95_s","makespan_s","aggregate_tok_s"]}}
    (out/"summary.json").write_text(json.dumps(s,indent=2))
if __name__=="__main__":main()
