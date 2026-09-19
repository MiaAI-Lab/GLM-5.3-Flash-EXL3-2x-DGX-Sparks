#!/usr/bin/env python3
"""In-image checks for overlay/kvoffload/nvme_direct.py: _write_all/_read_all
through a pipe (forces short writes/reads) and NvmeDirectManager.lookup
returning MISS after the file is removed (no stale _exists HIT).
Run inside the serving image (needs vllm.v1.kv_offload.base)."""
import os, sys, numpy as np, importlib.util, threading, tempfile
spec=importlib.util.spec_from_file_location('nd', __import__('os').environ.get('GLM53_NVME_DIRECT_PY', __import__('pathlib').Path(__file__).resolve().parent.parent / 'overlay/kvoffload/nvme_direct.py')); nd=importlib.util.module_from_spec(spec); spec.loader.exec_module(nd)
data=np.arange(3_000_000,dtype=np.uint8)
r,w=os.pipe()
def writer(): nd._write_all(w, data.data); os.close(w)
t=threading.Thread(target=writer); t.start()
out=np.zeros_like(data); got=nd._read_all(r, out.data); t.join(); os.close(r)
assert got==len(data) and np.array_equal(out,data), got
print("pipe round-trip OK:", got, "bytes")
d=tempfile.mkdtemp(); m=nd.NvmeDirectManager(d)
k=bytes(b"\x01"*40); p=m._path(k); os.makedirs(os.path.dirname(p)); open(p,"wb").write(b"x")
from vllm.v1.kv_offload.base import LookupResult
assert m.lookup(k,None)==LookupResult.HIT; os.unlink(p)
assert m.lookup(k,None)==LookupResult.MISS and k not in m._exists
print("manager: deleted file -> MISS OK")
