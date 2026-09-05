#!/usr/bin/env python3
"""Release consumed prefill logits before the next indexer chunk allocates them.

Python evaluates a rebinding's RHS while the previous value is still live.
Without an explicit release, consecutive prefill chunks can overlap two logits
allocations, despite the memory profiler accounting for only one. Top-k is the
last consumer; same-stream CUDA allocator reuse preserves its execution order.
No chunk sizes, logits computation, or selected indices are changed.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import stat
import tempfile

TARGET = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py')
ANCHOR = ('            if index_kpool > 1:\n'
          '                pool_ids = pool_topk.to(torch.int64)\n')
INSERT = ('            # [glm53-release-prefill-logits] Top-k is the final consumer.\n'
          '            del logits\n\n')


def patched_source(source: str) -> str:
    ast.parse(source)
    if INSERT in source:
        if source.count(INSERT + ANCHOR) != 1 or source.count(INSERT) != 1:
            raise RuntimeError('inconsistent prefill logits lifetime patch')
        return source
    if '[glm53-release-prefill-logits]' in source or source.count(ANCHOR) != 1:
        raise RuntimeError('prefill logits lifetime anchor drift')
    patched = source.replace(ANCHOR, INSERT + ANCHOR, 1)
    # Verify that the patch only adds the intended lifetime operation.
    before, after = ast.parse(source), ast.parse(patched)
    class RemoveRelease(ast.NodeTransformer):
        removed = 0
        def visit_Delete(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'logits'):
                self.removed += 1
                return None
            return node
    strip = RemoveRelease()
    after = strip.visit(after)
    if strip.removed != 1 or ast.dump(before) != ast.dump(after):
        raise RuntimeError('unexpected AST change in logits lifetime patch')
    return patched


def apply(path: Path = TARGET) -> None:
    original = path.read_text()
    candidate = patched_source(original)
    if candidate == original:
        return
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(candidate)
        os.chmod(temp, stat.S_IMODE(path.stat().st_mode))
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    for pyc in (path.parent / '__pycache__').glob(path.stem + '.*.pyc'):
        pyc.unlink()


if __name__ == '__main__':
    apply()
    print('prefill logits lifetime patch verified')
