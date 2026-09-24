"""Fix LMCache server defect 8: the lookup fold requires every kv_rank shard
and every object group, but node-partitioned stores hold exactly one rank's
shards (as single merged group-0 objects) per server.

Run INSIDE the image container:  python3 lmc-fix-fold-topology.py
"""
from pathlib import Path
import ast

BASE = Path("/usr/local/lib/python3.12/dist-packages/lmcache/v1/distributed")
LOOKUP = Path(
    "/usr/local/lib/python3.12/dist-packages/lmcache/v1/multiprocess/modules/lookup.py"
)

HELD_COMMENT = (
    "        # [glm53-lookup-ranks-held] Expand only over the (kv_rank,\n"
    "        # object_group_id) combinations this server actually stores. The\n"
    "        # store path is node-local (each rank's shards land on its own\n"
    "        # node's server) and writes single merged objects (one group id)\n"
    "        # per chunk, while fold() below requires EVERY expanded rank shard\n"
    "        # and intersects ALL object groups - a grid no node-partitioned\n"
    "        # store can ever satisfy, which zeroed every external lookup hit.\n"
    "        # Reduced expansion only when the held set forms a dense grid.\n"
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
    l1 = BASE / "l1_manager.py"
    patch_file(
        l1,
        [
            (
                "    def register_listener(self, listener: L1ManagerListener) -> None:\n",
                "    def distinct_object_key_shapes(\n"
                "        self, model_name: str\n"
                "    ) -> set[tuple[int, int]]:\n"
                "        \"\"\"[glm53-lookup-ranks-held] Distinct (kv_rank, object_group_id)\n"
                "        combinations this server's L1 index holds for a model.\"\"\"\n"
                "        return {\n"
                "            (key.kv_rank, key.object_group_id)\n"
                "            for key in self._objects\n"
                "            if key.model_name == model_name\n"
                "        }\n"
                "\n"
                "    def register_listener(self, listener: L1ManagerListener) -> None:\n",
                "l1-held",
                1,
            )
        ],
        "glm53-lookup-ranks-held",
    )

    fsa = BASE / "l2_adapters" / "fs_l2_adapter.py"
    patch_file(
        fsa,
        [
            (
                "    def __init__(self, config: FSL2AdapterConfig):\n",
                "    def stored_object_key_shapes(self) -> set[tuple[int, int]]:\n"
                "        \"\"\"[glm53-lookup-ranks-held] Distinct (kv_rank, object_group_id)\n"
                "        combinations among the object files stored under base_path.\"\"\"\n"
                "        shapes: set[tuple[int, int]] = set()\n"
                "        try:\n"
                "            names = os.listdir(self._config.base_path)\n"
                "        except OSError:\n"
                "            return shapes\n"
                "        for name in names:\n"
                "            key = _filename_to_object_key(name)\n"
                "            if key is not None:\n"
                "                shapes.add((key.kv_rank, key.object_group_id))\n"
                "        return shapes\n"
                "\n"
                "    def __init__(self, config: FSL2AdapterConfig):\n",
                "fs-held",
                1,
            )
        ],
        "glm53-lookup-ranks-held",
    )

    sm = BASE / "storage_manager.py"
    patch_file(
        sm,
        [
            (
                "    def _combine_found(\n",
                "    def held_object_key_shapes(\n"
                "        self, model_name: str\n"
                "    ) -> set[tuple[int, int]]:\n"
                "        \"\"\"[glm53-lookup-ranks-held] The (kv_rank, object_group_id) grid\n"
                "        this server can serve for a model: L1-resident objects union\n"
                "        L2-stored objects. Empty when this server stores nothing.\"\"\"\n"
                "        shapes = self._l1_manager.distinct_object_key_shapes(model_name)\n"
                "        for adapter in self._l2_adapters.values():\n"
                "            method = getattr(adapter, \"stored_object_key_shapes\", None)\n"
                "            if method is not None:\n"
                "                shapes |= method()\n"
                "        return shapes\n"
                "\n"
                "    def _combine_found(\n",
                "sm-held",
                1,
            )
        ],
        "glm53-lookup-ranks-held",
    )

    lu = LOOKUP
    patch_file(
        lu,
        [
            # L1: held-set expansion in the key builder
            (
                "        per_group = ipc_key_to_object_keys(key, chunk_hashes, list(range(num_groups)))\n",
                HELD_COMMENT
                + "        held = self._ctx._storage_manager.held_object_key_shapes(\n"
                "            key.model_name\n"
                "        )\n"
                "        if held:\n"
                "            held_groups = sorted({group for _rank, group in held})\n"
                "            held_ranks = sorted({rank for rank, _group in held})\n"
                "            if len(held) == len(held_groups) * len(held_ranks):\n"
                "                return [\n"
                "                    ObjectKey(\n"
                "                        chunk_hash=chunk_hashes[j],\n"
                "                        model_name=key.model_name,\n"
                "                        kv_rank=kv_rank,\n"
                "                        object_group_id=group_id,\n"
                "                        cache_salt=key.cache_salt,\n"
                "                    )\n"
                "                    for j in range(len(chunk_hashes))\n"
                "                    for group_id, kv_rank in held\n"
                "                ]\n"
                "        per_group = ipc_key_to_object_keys(key, chunk_hashes, list(range(num_groups)))\n",
                "lu-expand",
                1,
            ),
            # L2: reduced attn_desc / group_layout_descs / job world_size
            (
                "        obj_keys = self._chunk_major_object_keys(key, chunk_hashes)\n"
                "\n"
                "        group_layout_descs = self._ctx.layout_desc_registry.find_group_layout_descs(\n"
                "            model_name, world_size\n"
                "        )\n",
                "        obj_keys = self._chunk_major_object_keys(key, chunk_hashes)\n"
                "\n"
                "        group_layout_descs = self._ctx.layout_desc_registry.find_group_layout_descs(\n"
                "            model_name, world_size\n"
                "        )\n"
                "        # [glm53-lookup-ranks-held] Reduce the attention/grid dimensions\n"
                "        # to the held set when the expansion was reduced, so the fold's\n"
                "        # stride and rank count match the keys actually scanned.\n"
                "        held_shapes = self._ctx._storage_manager.held_object_key_shapes(\n"
                "            model_name\n"
                "        )\n"
                "        grid_size = (\n"
                "            len(chunk_hashes) * attn_desc.num_object_groups * world_size\n"
                "        )\n"
                "        if held_shapes and len(obj_keys) < grid_size:\n"
                "            held_groups = sorted({group for _rank, group in held_shapes})\n"
                "            held_ranks = sorted({rank for rank, _group in held_shapes})\n"
                "            if held_groups and held_ranks:\n"
                "                attn_desc = AttnWindowDesc(\n"
                "                    num_chunks_in_sw=[\n"
                "                        attn_desc.num_chunks_in_sw[g] for g in held_groups\n"
                "                    ],\n"
                "                    world_size=len(held_ranks),\n"
                "                )\n"
                "                group_layout_descs = {\n"
                "                    g: desc\n"
                "                    for g, desc in group_layout_descs.items()\n"
                "                    if g in held_groups\n"
                "                }\n"
                "                world_size = len(held_ranks)\n",
                "lu-reduced-spec",
                1,
            ),
        ],
        "glm53-lookup-ranks-held",
    )

    for f in (
        BASE / "l1_manager.py",
        BASE / "l2_adapters" / "fs_l2_adapter.py",
        BASE / "storage_manager.py",
        LOOKUP,
    ):
        ast.parse(f.read_text())
    print("fold-topology fix OK (syntax verified on all 4 files)")