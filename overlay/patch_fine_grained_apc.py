#!/usr/bin/env python3
"""[glm53-fgapc] Fine-grained APC for the GLM-5.3 hybrid: exempt the KpoolTail
scratch manager from the partial-hash-hit veto.

Upstream disables fine-grained prefix-cache hits whenever ANY cache manager
lacks `supports_fine_grained_hash_lookup` and is not already hash-grained.
On this hybrid the only such manager is KpoolTailManager — a one-block
circular scratch per request that (a) never has block hashes computed,
(b) already opts out of the hybrid hit min (KpoolTailSpec), and (c) whose
group is reseeded fresh when the cached window does not cover the hit. Its
veto therefore zeroes a capability the MLA + mamba(align) managers do
support, and every follow-up re-prefills up to a full 3584-token hybrid
block. Exempting it lets the MLA/mamba hit reconcile at hash granularity.

On by default (fix, not feature). GLM53_FINE_GRAINED_APC=0 reverts to
block-aligned-only hits. Idempotent via the [glm53-fgapc] marker;
ast.parse-validated after edit.
"""

import ast
import os
import sys

MARKER = "[glm53-fgapc]"

VETO_OLD = """            unsupported_partial_hit_managers = {
                type(manager).__name__
                for manager in self.single_type_managers
                if not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
            }"""

VETO_NEW = """            unsupported_partial_hit_managers = {
                type(manager).__name__
                for manager in self.single_type_managers
                if not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
                # [glm53-fgapc] KpoolTail is a 1-block/req scratch: it never
                # sees block hashes and already opts out of the hybrid hit
                # min, so its block-aligned-only lookup must not veto
                # fine-grained hits for the managers that do participate.
                and type(manager).__name__ != "KpoolTailManager"
            }"""

DIAG_OLD = """        self.verify_and_split_kv_cache_groups()"""

DIAG_NEW = """        logger.info(
            # [glm53-fgapc]
            "[glm53-fgapc] partial_hash=%s hash_block=%s managers=%s",
            self.enable_partial_hash_hits,
            hash_block_size,
            sorted(
                (
                    type(manager).__name__,
                    manager.block_size,
                    bool(
                        getattr(
                            manager, "supports_fine_grained_hash_lookup", False
                        )
                    ),
                )
                for manager in self.single_type_managers
            ),
        )
        self.verify_and_split_kv_cache_groups()"""


def patch_file(path: str, dry_run: bool = False) -> int:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    if MARKER in text:
        print(f"[patch_fine_grained_apc] {path}: already patched; no-op.")
        return 0

    if VETO_OLD not in text:
        raise AssertionError(
            f"{path}: partial-hash veto block not found (upstream layout "
            "changed?); refusing to guess."
        )
    text = text.replace(VETO_OLD, VETO_NEW, 1)

    if text.count(DIAG_OLD) != 1:
        raise AssertionError(
            f"{path}: expected exactly one verify_and_split call site, "
            f"found {text.count(DIAG_OLD)}."
        )
    text = text.replace(DIAG_OLD, DIAG_NEW, 1)

    try:
        ast.parse(text, filename=path)
    except SyntaxError as e:
        raise AssertionError(f"POST-EDIT ast.parse FAILED for {path}: {e}") from e

    if dry_run:
        print(f"[patch_fine_grained_apc] DRY RUN -- {path} not written.")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[patch_fine_grained_apc] {path}: fine-grained APC exemption applied.")
    return 1


def main() -> int:
    # On by default (this is a fix, not a feature): the KpoolTail veto zeroes
    # fine-grained hits the MLA/mamba(align) managers support. Set
    # GLM53_FINE_GRAINED_APC=0 to revert to block-aligned-only hits.
    if os.environ.get("GLM53_FINE_GRAINED_APC", "1") == "0":
        print(
            "[patch_fine_grained_apc] GLM53_FINE_GRAINED_APC=0 — "
            "skipping (block-aligned-only upstream behavior)."
        )
        return 0

    candidates = [
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py",
    ]
    patched = 0
    for path in candidates:
        if os.path.isfile(path):
            patched += patch_file(path, dry_run=os.environ.get("DRY_RUN") == "1")
    if patched == 0:
        print("[patch_fine_grained_apc] no candidate files found; nothing to do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
