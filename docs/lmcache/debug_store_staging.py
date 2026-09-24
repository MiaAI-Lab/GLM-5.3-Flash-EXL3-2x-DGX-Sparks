"""Apply the store-staging fix + LMCDBG logging to LMCache's connector + metadata.

Run INSIDE the image container:  python3 debug_store_staging.py
Grep LMCDBG in the serve logs for the diagnosis.

REAL FIX (glm53-staging-skip-excluded-groups):
GetStoreMetadata bounds the storable prefix with
    min over ALL engine groups of allocated_blocks[i] * tokens_per_block[i]
Non-prefix-cachable groups (GLM-5.3-Flash's KpoolTailSpec ring buffer, block_size=4)
hold a FIXED-SIZE allocation (1 block) that never grows with the sequence, so the
min() caps allocated_tokens at 4 for every request: num_chunks = 4 // 3584 = 0 and
STORE never fires (lmcache:num_stored_tokens stays 0 forever).
Fix: skip groups whose kv_cache_spec.participates_in_prefix_caching is False in
that bound only. Their block-id positions stay in the op (the server's group ids
keep vLLM numbering with an unused hole; the server ignores the position).
"""
from pathlib import Path
import ast

BASE = Path("/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm")

# ---------------------------------------------------------------- connector
LOOKUP_ANCHOR = '        """\n        tracker = self._get_or_create_request_tracker(request)'
LOOKUP_NEW = (
    '        """\n'
    '        if __import__("os").environ.get("LMC_DEBUG") == "1": logger.warning("LMCDBG lookup-phase req=%s status=%s", getattr(request, "request_id", "?"), str(request.status))\n'
    "        tracker = self._get_or_create_request_tracker(request)"
)
ADAPTER_ANCHOR = (
    "        self._ensure_heartbeat_started()\n\n"
    "        if not self.is_healthy:\n            return"
)
ADAPTER_NEW = (
    "        self._ensure_heartbeat_started()\n"
    '        if __import__("os").environ.get("LMC_DEBUG") == "1": logger.warning("LMCDBG maybe_submit is_healthy=%s req=%s", self.is_healthy, request_id)\n'
    "        if not self.is_healthy:\n"
    '            logger.warning("LMCDBG SKIPPED LOOKUP: adapter unhealthy")\n'
    "            return"
)
# 1) build the participating-group index list next to _group_tokens_per_block
GROUPS_ANCHOR = (
    "        self._group_tokens_per_block: list[int] = [\n"
    "            group.kv_cache_spec.block_size for group in vllm_groups\n"
    "        ] or [vllm_config.cache_config.block_size]"
)
GROUPS_NEW = (
    GROUPS_ANCHOR
    + "\n"
    + "        # [glm53-staging-skip-excluded-groups] Engine groups whose spec\n"
    + "        # does not participate in prefix caching (scratch / ring buffers)\n"
    + "        # must be skipped by the store staging bound: their block\n"
    + "        # allocation is fixed-size and does not grow with the sequence.\n"
    + "        self._participating_group_idxs: list[int] = (\n"
    + "            [\n"
    + "                engine_group_idx\n"
    + "                for engine_group_idx, group in enumerate(vllm_groups)\n"
    + "                if getattr(group.kv_cache_spec,\n"
    + "                           \"participates_in_prefix_caching\", True)\n"
    + "            ]\n"
    + "            if vllm_groups\n"
    + "            else list(range(len(self._group_tokens_per_block)))\n"
    + "        )"
)
# 2) pass the list at both GetStoreMetadata callsites (identical text, count 2)
CALLSITE_OLD = (
    "            r_meta = LMCacheMPRequestMetadata.GetStoreMetadata(\n"
    "                request_tracker,\n"
    "                lmcache_tokens_per_chunk,\n"
    "                self._group_tokens_per_block,\n"
    "            )"
)
CALLSITE_NEW = (
    "            r_meta = LMCacheMPRequestMetadata.GetStoreMetadata(\n"
    "                request_tracker,\n"
    "                lmcache_tokens_per_chunk,\n"
    "                self._group_tokens_per_block,\n"
    "                participating_group_idxs=self._participating_group_idxs,\n"
    "            )"
)

# ----------------------------------------------------------------- metadata
META_ANCHOR = "        num_engine_groups = len(group_tokens_per_block)"
META_NEW = (
    META_ANCHOR
    + "\n"
    + "        import sys as _s\n"
    + "        if __import__(\"os\").environ.get(\"LMC_DEBUG\") == \"1\": print('LMCDBG meta groups=%d tpn=%s part=%s alloc=%s' % (num_engine_groups, list(group_tokens_per_block), sorted(participating_group_idxs), dict(tracker.num_allocated_blocks())), file=_s.stderr)"
)
# 3) GetStoreMetadata signature gains the participating-group list
SIG_OLD = (
    "    def GetStoreMetadata(\n"
    "        tracker: LMCacheMPRequestTracker,\n"
    "        lmcache_tokens_per_chunk: int,\n"
    "        group_tokens_per_block: list[int],\n"
    "    ) -> \"LMCacheMPRequestMetadata | None\":"
)
SIG_NEW = (
    "    def GetStoreMetadata(\n"
    "        tracker: LMCacheMPRequestTracker,\n"
    "        lmcache_tokens_per_chunk: int,\n"
    "        group_tokens_per_block: list[int],\n"
    "        participating_group_idxs: list[int] | None = None,\n"
    "    ) -> \"LMCacheMPRequestMetadata | None\":"
)
# 4) the staging min() skips non-participating groups
MIN_OLD = (
    "        allocated_tokens = (\n"
    "            min(\n"
    "                allocated_lengths.get(engine_group_idx, 0)\n"
    "                * group_tokens_per_block[engine_group_idx]\n"
    "                for engine_group_idx in range(num_engine_groups)\n"
    "            )\n"
    "            if num_engine_groups > 0\n"
    "            else 0\n"
    "        )"
)
MIN_NEW = (
    "        # [glm53-staging-skip-excluded-groups] Non-prefix-cachable engine\n"
    "        # groups (e.g. GLM-5.3-Flash's KpoolTailSpec ring buffer) hold a\n"
    "        # fixed-size allocation that never grows with the sequence, so\n"
    "        # including them in this bound capped allocated_tokens at a few\n"
    "        # tokens and silently disabled STORE for every request. Bound the\n"
    "        # storable prefix by participating groups only; their block-id\n"
    + "        # positions stay verbatim in the op (the server's group ids keep\n"
    + "        # vLLM numbering; unused positions carry no tensors and are ignored).\n"
    + "        _participating = (\n"
    + "            participating_group_idxs\n"
    + "            if participating_group_idxs is not None\n"
    + "            else list(range(num_engine_groups))\n"
    + "        )\n"
    + "        allocated_tokens = (\n"
    + "            min(\n"
    + "                allocated_lengths.get(engine_group_idx, 0)\n"
    + "                * group_tokens_per_block[engine_group_idx]\n"
    + "                for engine_group_idx in _participating\n"
    + "            )\n"
    + "            if _participating\n"
    + "            else 0\n"
    + "        )"
)
# 5) debug prints AFTER num_chunks is computed (the earlier version referenced
#    num_chunks before assignment and crashed the engine: UnboundLocalError)
STAGE_ANCHOR = "        num_chunks = num_staging_tokens // lmcache_tokens_per_chunk"
STAGE_NEW = (
    STAGE_ANCHOR
    + "\n"
    + "        if __import__(\"os\").environ.get(\"LMC_DEBUG\") == \"1\": print('LMCDBG staging part=%s alloc=%s tpn=%s min_avail=%d staged=%d chunks=%d' % (_participating, dict(tracker.num_allocated_blocks()), list(group_tokens_per_block), min_available_tokens, num_staging_tokens, num_chunks), file=_s.stderr)\n"
    + "        if num_chunks < 1 and __import__(\"os\").environ.get(\"LMC_DEBUG\") == \"1\":\n"
    + "            print('LMCDBG STAGING ZERO: alloc=%s alltok=%d computed=%d stored=%d' % (dict(tracker.num_allocated_blocks()), len(tracker.all_token_ids), computed_tokens, tracker.num_stored_tokens), file=_s.stderr)"
)


def once(text, old, new, tag, expect=1):
    n = text.count(old)
    if n != expect:
        raise RuntimeError(f"{tag}: expected {expect} anchor(s), found {n}")
    return text.replace(old, new)


def patch_file(path, edits, mark):
    text = path.read_text()
    if mark in text:
        print(f"  {path.name}: already patched")
        return
    for old, new, tag, expect in edits:
        text = once(text, old, new, tag, expect)
    path.write_text(text)
    print(f"  {path.name}: patched ({len(edits)} edits)")


if __name__ == "__main__":
    base = BASE

    pc = base / "lmcache_mp_connector.py"
    patch_file(
        pc,
        [
            (LOOKUP_ANCHOR, LOOKUP_NEW, "gNNMT", 1),
            (GROUPS_ANCHOR, GROUPS_NEW, "cgroups", 1),
            (CALLSITE_OLD, CALLSITE_NEW, "ccallsite", 2),
        ],
        "LMCDBG",
    )

    pa = base / "vllm_multi_process_adapter.py"
    patch_file(
        pa,
        [(ADAPTER_ANCHOR, ADAPTER_NEW, "hbgate", 1)],
        "LMCDBG",
    )

    pm = base / "lmcache_mp_metadata.py"
    patch_file(
        pm,
        [
            (SIG_OLD, SIG_NEW, "msig", 1),
            (MIN_OLD, MIN_NEW, "mmin", 1),
            (META_ANCHOR, META_NEW, "meta-ng", 1),
            (STAGE_ANCHOR, STAGE_NEW, "mstage", 1),
        ],
        "LMCDBG",
    )

    for f in ("lmcache_mp_connector.py", "vllm_multi_process_adapter.py", "lmcache_mp_metadata.py"):
        ast.parse((base / f).read_text())
    print("patch OK (fix + debug logging; syntax verified on all 3 files)")