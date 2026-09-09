# SPDX-License-Identifier: Apache-2.0
"""Exact native gather + Hadamard oracle. Run only with inference stopped."""
import importlib.util
from pathlib import Path
import torch
import exllamav3_ext as ext

spec=importlib.util.spec_from_file_location(
    "gather_had",Path(__file__).resolve().parents[1]/"overlay/exl3_gather_hadamard.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def reference(x,indices,scale):
    gathered=torch.index_select(x,0,indices)
    out=torch.empty_like(gathered)
    if indices.numel():ext.had_r_128(gathered,out,scale,None,1.0)
    return out


def same(a,b):
    torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
    mask=~torch.isnan(b)
    assert torch.equal(a[mask].view(torch.int16),b[mask].view(torch.int16)), "FP16 bits differ"


def check(x,indices,scale,*,graph=False):
    rows,hidden=indices.numel(),x.shape[1]
    storage=torch.full((rows,hidden+32),123.,device=x.device,dtype=torch.float16)
    out=storage[:,:hidden]
    originals=[v.clone() for v in (x,indices,scale)]
    mod.gather_hadamard(x,indices,scale,out)
    same(out,reference(x,indices,scale))
    for a,b in zip((x,indices,scale),originals):same(a,b) if a.dtype==torch.float16 else torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.all(storage[:,hidden:]==123.)
    if graph:
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):mod.gather_hadamard(x,indices,scale,out)
        for v in [-0.,.125,-.25]:
            x.fill_(v)
            scale.fill_(1.)
            indices.copy_(torch.arange(rows,device=x.device)%x.shape[0])
            out.fill_(42.)
            g.replay()
            same(out,reference(x,indices,scale))
            assert torch.all(storage[:,hidden:]==123.)


def main():
    torch.manual_seed(72319)
    count=0
    for rows,hidden in [(0,128),(1,128),(3,256),(7,4096),(129,4096),
                        (257,1024),(1024,4096),(7168,4096),(513,8192),(5,384)]:
        for magnitude in [.001,1.,100.]:
            total=max(513,rows+13)
            xstore=torch.randn(total,hidden+16,device='cuda',dtype=torch.float16)*magnitude
            x=xstore[:,:hidden]
            indices=torch.randint(total,(rows*2,),device='cuda')[::2]
            scale=torch.randn(hidden,device='cuda',dtype=torch.float16)*magnitude
            check(x,indices,scale,graph=rows in (7,257))
            count+=1
    for repeated in [False,True]:
        x=torch.randn(17,256,device='cuda',dtype=torch.float16)
        indices=torch.zeros(35,device='cuda',dtype=torch.long) if repeated else torch.arange(16,-1,-1,device='cuda')
        check(x,indices,torch.ones(256,device='cuda',dtype=torch.float16),graph=True)
        count+=1
    values=[float('nan'),float('inf'),-float('inf'),0.,-0.,2**-24,-2**-24,
            2**-14,1.,-1.,65504.,-65504.,100.,-100.]
    for value in values:
        x=torch.randn(5,256,device='cuda',dtype=torch.float16)
        scale=torch.ones(256,device='cuda',dtype=torch.float16)
        x[2,17]=value
        indices=torch.tensor([4,2,0,2],device='cuda')
        check(x,indices,scale)
        scale[53]=value
        check(x,indices,scale)
        count+=2
    for v in [-0.,2**-24,-2**-24]:
        check(torch.full((2,128),v,device='cuda',dtype=torch.float16),
              torch.tensor([1,0],device='cuda'),torch.ones(128,device='cuda',dtype=torch.float16))
        count+=1
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        check(torch.randn(13,1024,device='cuda',dtype=torch.float16),
              torch.tensor([12,0,4,3],device='cuda'),torch.randn(1024,device='cuda',dtype=torch.float16))
    torch.cuda.current_stream().wait_stream(stream)
    count+=1
    torch.cuda.synchronize()
    print(f'PASS {count} exact gather+Hadamard cases including bits, graph updates, strides and current stream')


if __name__=='__main__':main()
