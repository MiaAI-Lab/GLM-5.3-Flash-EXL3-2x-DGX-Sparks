#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow auto/safetensors loader staging and bounded local shard read-ahead.

Motivated by Alexbob0/glm53-flash-vllm-upstream-sm121 loadclone measurements,
revision bc3891aed74a1f4ccd679e5205ab9bd2605cf283. The implementation below
retains the upstream iterator rather than replacing weight_utils.py. Its mmap
block remains byte-identical so PR #230 (bc97cbf1d5ab06df778b79b99efa92997bb4f153)
can apply before OR after this patch, including that patch's verification.

GLM53_LOAD_CLONE=1 stages auto/lazy mmap tensors; 0 disables optional staging, never the
non-4KiB page safety fallback (GLM53_COLD_LOAD_STAGE_MMAP remains its switch).
GLM53_LOAD_PREFETCH=0 disables added read-ahead; 1..16 specifies the maximum
number of submitted shards (including the current shard) and worker threads.
Nonlazy strategies and recognized network filesystems retain stock prefetch.
No InstantTensor changes. Close a partially consumed generator explicitly.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

TARGET = Path(os.environ.get(
    "GLM53_WEIGHT_UTILS_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py",
))
MARK = "# [glm53-loadclone:v2]"
NAME = "safetensors_weights_iterator"
PRIVATE = "_glm53_original_safetensors_weights_iterator"
LOOP = "    for st_file in tqdm(\n        sorted_files,\n"
NEW_LOOP = "    for st_file in tqdm(\n        _glm53_prefetcher.files(sorted_files, safetensors_load_strategy, is_net_fs),\n        total=len(sorted_files),\n"
MMAP_PREFIX = (
    '            with safe_open(st_file, framework="pt") as f:\n'
    '                for name in f.keys():  # noqa: SIM118\n'
    '                    if should_skip_weight(name, local_expert_ids):\n'
    '                        continue\n'
    '                    param = f.get_tensor(name)\n'
)
MMAP_PR230 = (
    '                    # [glm53-cold-load-uma:v1] cuMemcpyHtoDAsync wedges on GB10 when the\n'
    '                    # source is a file-backed 64 KiB-page mapping: stage the\n'
    '                    # tensor off the mmap into anonymous memory first.\n'
    '                    if _GLM53_UMA_STAGE_MMAP:\n'
    '                        param = param.clone()\n'
)
MMAP_YIELD = '                    yield name, param'

HELPERS = '''# [glm53-loadclone:v2]
def _glm53_load_options():
    import os

    clone = os.environ.get("GLM53_LOAD_CLONE", "1")
    raw_depth = os.environ.get("GLM53_LOAD_PREFETCH", "0")
    if clone not in ("0", "1"):
        raise ValueError("GLM53_LOAD_CLONE must be 0 or 1")
    if not raw_depth.isascii() or not raw_depth.isdecimal():
        raise ValueError("GLM53_LOAD_PREFETCH must be an integer in [0, 16]")
    depth = int(raw_depth)
    if not 0 <= depth <= 16:
        raise ValueError("GLM53_LOAD_PREFETCH must be an integer in [0, 16]")
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        page_size = 4096
    safety = page_size != 4096 and os.environ.get("GLM53_COLD_LOAD_STAGE_MMAP", "1") == "1"
    return clone == "1", safety, depth


class _Glm53ShardPrefetch:
    """One bounded shard window, owned and joined by the public iterator."""

    def __init__(self, depth):
        import threading

        self.depth = depth
        self.stop = threading.Event()
        self.pool = None
        self.pending = {}
        self.consumed = []
        self.current = -1
        self.released = 0
        self.advice_enabled = True

    def read(self, path):
        # One reusable 1 MiB buffer per active worker, not per queued shard.
        # Cancellation is cooperative between reads. A kernel-blocked read cannot
        # be interrupted by Python: close waits for it rather than leaking threads.
        if self.stop.is_set():
            return
        with open(path, "rb", buffering=0) as stream:
            buffer = bytearray(1 << 20)
            while not self.stop.is_set() and stream.readinto(buffer):
                pass

    def release(self, path):
        import os

        if not self.advice_enabled:
            return
        try:
            with open(path, "rb", buffering=0) as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, NotImplementedError, OSError):
            self.advice_enabled = False
            logger.warning_once("glm53: checkpoint cache advice unavailable; disabling for this load")

    def release_before(self, index):
        # A new tensor, not a new filename, proves both param locals advanced.
        # Fully filtered shards can leave the preceding tensor alive across gaps.
        for previous in range(self.released, index):
            self.release(self.consumed[previous])
        self.released = index

    def files(self, files, strategy, is_net_fs):
        if strategy not in (None, "lazy") or is_net_fs or not files:
            yield from files
            return
        from concurrent.futures import ThreadPoolExecutor

        if self.depth:
            self.pool = ThreadPoolExecutor(
                max_workers=min(self.depth, len(files)), thread_name_prefix="glm53-load-prefetch"
            )
        next_index = 0
        for index, path in enumerate(files):
            if self.depth:
                # Includes the current shard: outstanding reads never exceed depth.
                while next_index < min(index + self.depth, len(files)):
                    self.pending[next_index] = self.pool.submit(self.read, files[next_index])
                    next_index += 1
                self.pending.pop(index).result()
            self.consumed.append(path)
            self.current = index
            yield path

    def close(self):
        self.stop.set()
        for future in self.pending.values():
            future.cancel()
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
        # Retry all yielded shards after native close, including trailing filtered
        # shards and pages that a clone-disabled consumer may since have released.
        # Future prefetched shards were never yielded and must remain cached.
        for path in self.consumed:
            self.release(path)
        self.consumed.clear()


'''


def _function(src: str, name: str) -> ast.FunctionDef:
    matches = [node for node in ast.parse(src).body
               if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} function, got {len(matches)}")
    return matches[0]


def prepare(src: str) -> str:
    """Fail closed on partial edits; preserve every unrelated function byte."""
    if MARK in src:
        if src.count(HELPERS) != 1 or src.count(NEW_LOOP) != 1:
            raise ValueError("partial loadclone patch or helper drift")
        public = _function(src, NAME)
        lines = src.splitlines(keepends=True)
        actual = "".join(lines[public.lineno - 1:public.end_lineno])
        if actual.rstrip("\n") != _wrapper():
            raise ValueError("loadclone public wrapper drift")
        private = _function(src, PRIVATE)
        body = "".join(lines[private.lineno - 1:private.end_lineno])
        if not any(body.rstrip("\n").endswith(MMAP_PREFIX + stage + MMAP_YIELD)
                   for stage in ("", MMAP_PR230)):
            raise ValueError("safetensors mmap staging drift or conflicting clone patch")
        return src
    if "# [glm53-loadclone:v1]" in src or PRIVATE in src or "_Glm53ShardPrefetch" in src:
        raise ValueError("unmarked/legacy loadclone patch")
    node = _function(src, NAME)
    lines = src.splitlines(keepends=True)
    original = "".join(lines[node.lineno - 1:node.end_lineno])
    if not any(original.rstrip("\n").endswith(MMAP_PREFIX + stage + MMAP_YIELD)
               for stage in ("", MMAP_PR230)):
        raise ValueError("safetensors mmap staging drift or conflicting clone patch")
    if original.count(LOOP) != 1:
        raise ValueError("safetensors shard loop drift")
    # Validate the API before replacing it with an explicit, inspectable wrapper.
    expected = ast.parse(_wrapper()).body[0].args
    if ast.dump(node.args) != ast.dump(expected):
        raise ValueError("safetensors iterator signature drift")
    private = original.replace(f"def {NAME}(\n", f"def {PRIVATE}(\n    _glm53_prefetcher,\n", 1)
    private = private.replace(LOOP, NEW_LOOP, 1)
    out = "".join(lines[:node.lineno - 1]) + HELPERS + _wrapper() + "\n\n\n" + private + "".join(lines[node.end_lineno:])
    compile(out, str(TARGET), "exec")
    return out


def _wrapper() -> str:
    return '''def safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    safetensors_load_strategy: str | None = None,
    local_expert_ids: set[int] | None = None,
    *,
    safetensors_prefetch_num_threads: int = DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS,
    safetensors_prefetch_block_size: int = DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stage mmap tensors and optionally prefetch bounded local shards."""
    clone, safety, depth = _glm53_load_options()
    prefetcher = _Glm53ShardPrefetch(depth)
    iterator = _glm53_original_safetensors_weights_iterator(
        prefetcher, hf_weights_files, use_tqdm_on_load, safetensors_load_strategy,
        local_expert_ids,
        safetensors_prefetch_num_threads=safetensors_prefetch_num_threads,
        safetensors_prefetch_block_size=safetensors_prefetch_block_size,
    )
    seen = -1
    try:
        # PR #230 owns staging when its module-level flag is true. Do not clone
        # again, nor weaken its non-4KiB safety when optional clone is disabled.
        stage = ((safety or (clone and safetensors_load_strategy in (None, "lazy")))
                 and safetensors_load_strategy not in ("eager", "torchao")
                 and not globals().get("_GLM53_UMA_STAGE_MMAP", False))
        for name, param in iterator:
            if prefetcher.current != seen:
                prefetcher.release_before(prefetcher.current)
                seen = prefetcher.current
            yield name, param.clone() if stage else param
    finally:
        param = None
        try:
            iterator.close()
        finally:
            prefetcher.close()'''


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="require an already-patched target without writing")
    args = parser.parse_args()
    src = TARGET.read_text()
    out = prepare(src)
    if args.check and out != src:
        raise SystemExit(f"[glm53-loadclone] {TARGET}: missing loader overlay; regenerate before mounting read-only")
    if out != src:
        # A failed write must not leave the installed loader truncated.
        temporary = TARGET.with_name(f".{TARGET.name}.glm53-loadclone.tmp")
        try:
            temporary.write_text(out)
            temporary.chmod(TARGET.stat().st_mode)
            os.replace(temporary, TARGET)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"[glm53-loadclone] {TARGET}: {'already patched' if out == src else 'patched'}")


if __name__ == "__main__":
    main()
