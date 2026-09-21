from pathlib import Path
def counters(text):
    return {l.rsplit(" ",1)[0]:float(l.rsplit(" ",1)[1]) for l in text.splitlines()
            if l.startswith(("vllm:num_preemptions_total","vllm:request_success_total")) and not l.startswith("#")}
def healthy_arm(label):
    a=counters((Path(label)/"before.prom").read_text());b=counters((Path(label)/"after.prom").read_text())
    assert a and b,"missing health counters"
    for key,value in a.items():
        if key not in b or b[key]<value:return False
    for key in set(a)|set(b):
        if "preemptions" in key or 'finished_reason="error"' in key or 'finished_reason="abort"' in key:
            if b.get(key,0)!=a.get(key,0):return False
    return True