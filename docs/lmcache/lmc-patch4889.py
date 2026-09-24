#!/usr/bin/env python3
"""Backport of the LMCache#4889 fix (`_complete_kernel_pages`) into the
installed LMCache inside the GLM image.

Why: with GLM-5.3-Flash the attention group is allocated with a *padded*
physical pool, so the pool's kernel-page count is not always a whole multiple
of the logical/kernel block ratio. Upstream LMCache (v0.5.4 and `dev`) raises:

    ValueError: kernel page count 7098 is not a multiple of the
                logical/kernel block ratio 56

and refuses to register the group, so the engine cannot start at all.
LMCache#4889 ("preserve opaque group identity and page geometry") fixes this
by *trimming* the incomplete kernel-page tail instead of raising, because those
trailing pages are not addressable by the scheduler and must simply be excluded
from registration.

This shim reproduces that behaviour at both call sites (`_SubpagedAttentionViewEdit`
and `_SubpagedMLAAttentionViewEdit`); GLM-5.3-Flash hits the MLA one. It is a
backport of the upstream fix, not a novel change.

Idempotent: re-running is a no-op.
"""
import re
import sys

TARGET = ("/usr/local/lib/python3.12/dist-packages/lmcache/"
          "integration/vllm/kv_cache_group_edits.py")

RAISE = """        if num_kernel_pages % ratio != 0:
            raise ValueError(
                f"kernel page count {num_kernel_pages} is not a multiple of "
                f"the logical/kernel block ratio {ratio}"
            )
"""

TRIM = """        # Backport of LMCache#4889 `_complete_kernel_pages`: a padded
        # physical pool can leave an incomplete kernel-page tail that the
        # scheduler cannot address, so exclude it from registration rather
        # than refusing the whole group.
        num_blocks = num_kernel_pages // ratio
        if num_blocks == 0:
            raise ValueError(
                f"kernel page count {num_kernel_pages} cannot hold one "
                f"logical page requiring {ratio} kernel pages"
            )
        num_kernel_pages = num_blocks * ratio
        kv_cache = kv_cache[:num_kernel_pages]
"""

MARK = "Backport of LMCache#4889"


def main() -> int:
    with open(TARGET) as fh:
        src = fh.read()

    if MARK in src:
        print("already patched; no change")
        return 0

    n = src.count(RAISE)
    if n == 0:
        print("anchor not found; refusing", file=sys.stderr)
        return 1

    src = src.replace(RAISE, TRIM)

    # Each patched method ended with a recomputation of num_blocks from the
    # untrimmed page count; drop the duplicates introduced after the trim.
    src = src.replace("        num_kernel_pages = num_blocks * ratio\n"
                      "        kv_cache = kv_cache[:num_kernel_pages]\n",
                      "        num_kernel_pages = num_blocks * ratio\n"
                      "        kv_cache = kv_cache[:num_kernel_pages]\n", 1)
    # Remove the now-redundant later `num_blocks = num_kernel_pages // ratio`
    # lines only where they directly follow the trim block's usage.
    src = re.sub(
        r"(num_kernel_pages = num_blocks \* ratio\n"
        r"        kv_cache = kv_cache\[:num_kernel_pages\]\n)"
        r"(\s+)(num_blocks = num_kernel_pages // ratio\n)",
        r"\1", src)

    with open(TARGET, "w") as fh:
        fh.write(src)
    print(f"patched {n} call site(s) in {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
