"""Fix defect 8b: my held-set key expansion unpacked (kv_rank, object_group_id)
in the WRONG ORDER.

`held_object_key_shapes()` returns ``(key.kv_rank, key.object_group_id)`` tuples
(see l1_manager.distinct_object_key_shapes and
fs_l2_adapter.stored_object_key_shapes). The expansion loop in
``LookupModule._chunk_major_object_keys`` wrote ``for group_id, kv_rank in held``,
which bound group_id := kv_rank and kv_rank := object_group_id.

Measured effect (LMCTRACE2, 2026-09-24): lookup built
``ObjectKey(kv_rank=0, object_group_id=33554944)`` while both the L1 index and the
L2 files hold ``kv_rank=33554944 (rank 0 of ws=2), object_group_id=0``. Every scan
therefore missed: ``l1_present=0 absent=0``, ``found_bits=0``, 0 hits.

Run INSIDE the server container. Idempotent.
"""
from pathlib import Path
import ast

LOOKUP = Path(
    "/usr/local/lib/python3.12/dist-packages/lmcache/v1/multiprocess/modules/lookup.py"
)

OLD = "                    for group_id, kv_rank in held\n"
NEW = (
    "                    # [glm53-held-unpack] `held` holds (kv_rank,\n"
    "                    # object_group_id) tuples - unpack in that order.\n"
    "                    for kv_rank, group_id in held\n"
)

t = LOOKUP.read_text()
if "glm53-held-unpack" in t:
    print("already fixed")
else:
    n = t.count(OLD)
    assert n == 1, f"anchor count {n}"
    LOOKUP.write_text(t.replace(OLD, NEW, 1))
    ast.parse(LOOKUP.read_text())
    print("held-unpack fix applied")
