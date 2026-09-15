#!/usr/bin/env python3
"""Opt-in hash-block prefix-cache lookup for participating KV cache groups.

KpoolTailManager opts out of prefix caching but vetoes fine lookups in the
pinned coordinator. Scope that veto to participating groups, and verify that
every scratch alignment divides hash_block_size before enabling fine hits.
GLM53_FINEGRAINED_APC accepts exactly 0/1 and defaults to 0.

Patch-owned source regions are validated canonically before accepting an
existing patch or atomically replacing a pristine file. Unsupported drift fails
without writing. See docs/DESIGN-apc-fine-grained-hits.md for the contract and
verification limits.
"""

from __future__ import annotations

import ast
import os
import sys
import textwrap
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_KV_COORDINATOR_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py",
    )
)
MARK = "# [glm53-finegrained-apc]"
# Runtime message tag for the refusals raised at coordinator init.
RUNTIME_TAG = "[glm53-apc-finegrained]"
HELPER_BEGIN = "# [glm53-finegrained-apc] helper-begin"
HELPER_END = "# [glm53-finegrained-apc] helper-end"
HELPER_NEEDLE = "def _validate_prefix_cache_retention_interval(\n"
IMPORT_ANCHOR = "from abc import ABC, abstractmethod\n"

# Where the helper block goes. ``patch_hybrid_prefix_hit.py`` inserts its own
# helper before the same upstream needle, so "insert before the needle" makes
# the composed file depend on which patch ran first. Anchoring ahead of a
# sibling glm53 helper when one is already present makes the two patches
# byte-commutative in either order (host test A4).
HELPER_ANCHORS = (
    "\ndef _glm53_inner_kv_spec(spec):\n",  # patch_hybrid_prefix_hit.py
    HELPER_NEEDLE,
)


def helper_anchor(text: str) -> str | None:
    for anchor in HELPER_ANCHORS:
        if anchor in text:
            return anchor
    return None

HELPER = '''
# [glm53-finegrained-apc] helper-begin
GLM53_FG_TAG = "[glm53-apc-finegrained]"


class Glm53FineGrainedAPCError(RuntimeError):
    """Fine enablement requires a verifiable, compatible scratch alignment."""


def _glm53_strict_int(value):
    """Accept integers or signed ASCII decimal strings without coercion or trimming."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = value[1:] if value[:1] in ("+", "-") else value
        if digits and digits.isascii() and digits.isdigit():
            return int(value)
        return None
    return None


# Every attribute that can speak for a scratch group's required hit alignment,
# in the order they are reported.  ``fine_grained_hit_alignment`` is the
# forward-compatible capability; the rest are the observable facts it must not
# contradict.
GLM53_ALIGNMENT_ATTRS = (
    "fine_grained_hit_alignment",
    "block_size",
    "index_kpool",
    "kpool",
)


def _glm53_scratch_alignment(manager, spec):
    """Return (alignment, source), or (None, reason) if unverifiable.

    Collect all exposed alignment attributes and require agreement. Without an
    explicit capability, both spec.block_size and manager.block_size are required.
    A capability cannot override contradictory manager/spec values.
    """
    candidates = []
    for owner, owner_name in ((manager, "manager"), (spec, "spec")):
        for attr in GLM53_ALIGNMENT_ATTRS:
            raw = getattr(owner, attr, None)
            if raw is None:
                continue
            parsed = _glm53_strict_int(raw)
            if parsed is None:
                return None, (
                    f"{owner_name}.{attr}={raw!r} is not an integer "
                    f"({type(raw).__name__}); an alignment that cannot be "
                    "parsed exactly cannot be verified"
                )
            candidates.append((f"{owner_name}.{attr}", parsed))

    labels = {label for label, _ in candidates}
    if not any(label.endswith("fine_grained_hit_alignment") for label in labels):
        if "spec.block_size" not in labels:
            return None, (
                f"{type(spec).__name__} exposes neither block_size nor "
                "fine_grained_hit_alignment"
            )
        if "manager.block_size" not in labels:
            return None, (
                f"{type(manager).__name__} exposes no block_size to cross-check"
            )

    first_label, first_value = candidates[0]
    for label, value in candidates[1:]:
        if value != first_value:
            return None, (
                f"{label}={value} disagrees with {first_label}={first_value} "
                f"({type(spec).__name__} / {type(manager).__name__}); "
                "contradictory alignment sources are unverifiable, and an "
                "explicit fine_grained_hit_alignment capability is never taken "
                "as authoritative over them"
            )

    return first_value, " == ".join(label for label, _ in candidates)


def _glm53_finegrained_enabled(value):
    """Parse the exact 0/1 flag; reject invalid values instead of guessing."""
    if value == "1":
        return True
    if value == "0":
        return False
    raise Glm53FineGrainedAPCError(
        f"{GLM53_FG_TAG} GLM53_FINEGRAINED_APC={value!r} is not a valid "
        "kill-switch value. It must be exactly '1' (fine-grained prefix-cache "
        "hits ON) or '0' (OFF, the default when unset; block-aligned hits). "
        "Values such as 'true', 'yes', "
        "'01', ' 0' or '' are REFUSED rather than guessed, because guessing "
        "wrong silently changes the KV-cache hit alignment. Refusing to start. "
        "Set GLM53_FINEGRAINED_APC=0 to restore the upstream gate."
    )


def _glm53_finegrained_hit_gate(managers, kv_cache_groups, hash_block_size):
    """Return (enable, blockers, scratch) for the participating lookup gate.

    Participating managers must support hash-granular lookup or already have the
    hash block size. Their blockers select the coarse fallback. Non-participating
    scratch groups must instead allow resuming with empty per-request state:
    hash_block_size must be divisible by each verified scratch alignment.

    Collect scratch faults until the participating blockers are known. Raise only
    if fine hits would otherwise be enabled with unsafe or unverifiable scratch;
    a coarse fallback makes that invariant unreachable. The scratch result maps
    manager names to verified alignments or tolerated-fault diagnostics.
    """
    managers = list(managers)
    groups = list(kv_cache_groups)
    if len(managers) != len(groups):
        raise Glm53FineGrainedAPCError(
            f"{GLM53_FG_TAG} manager/group cardinality mismatch: "
            f"{len(managers)} single-type managers vs {len(groups)} KV cache "
            "groups. zip() would silently truncate the fine-grained-hit "
            "validation and skip real blockers, so this layout cannot be "
            "validated. Refusing to start. Set GLM53_FINEGRAINED_APC=0 to "
            "restore the upstream (all-managers, block-aligned) gate."
        )
    if not managers:
        raise Glm53FineGrainedAPCError(
            f"{GLM53_FG_TAG} no KV cache managers to validate; refusing to "
            "enable fine-grained prefix-cache hits. Set "
            "GLM53_FINEGRAINED_APC=0 to restore the upstream gate."
        )
    if _glm53_strict_int(hash_block_size) is None or not isinstance(
        hash_block_size, int
    ) or hash_block_size <= 0:
        raise Glm53FineGrainedAPCError(
            f"{GLM53_FG_TAG} hash_block_size={hash_block_size!r} is not a "
            "positive integer; the fine-grained hit alignment is undefined. "
            "Refusing to start. Set GLM53_FINEGRAINED_APC=0 to restore the "
            "upstream gate."
        )

    blockers = []
    scratch = {}
    faults = []
    for index, (manager, group) in enumerate(zip(managers, groups)):
        name = type(manager).__name__
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            raise Glm53FineGrainedAPCError(
                f"{GLM53_FG_TAG} KV cache group {index} (manager {name}) has "
                "no kv_cache_spec; the fine-grained-hit invariants cannot be "
                "verified. Refusing to start. Set GLM53_FINEGRAINED_APC=0 to "
                "restore the upstream gate."
            )
        if getattr(spec, "participates_in_prefix_caching", True):
            block_size = getattr(manager, "block_size", None)
            supports_fine = getattr(
                manager, "supports_fine_grained_hash_lookup", False
            )
            if not supports_fine and block_size != hash_block_size:
                blockers.append(
                    f"{name}"
                    f"(participating, block_size={block_size}, "
                    f"block-aligned lookups only)"
                )
            continue

        alignment, source = _glm53_scratch_alignment(manager, spec)
        if alignment is None:
            scratch[name] = f"UNVERIFIABLE ({source})"
            faults.append(
                "cannot verify the hit-alignment requirement of "
                f"non-participating scratch group {index} {name} "
                f"({type(spec).__name__}): {source}. A fine-grained hit could "
                "resume mid-pool and leave un-recomputed raw K/gate entries."
            )
            continue
        if alignment <= 0 or hash_block_size % alignment != 0:
            scratch[name] = (
                f"UNSAFE (requires hit alignment {alignment}, verified from "
                f"{source}; hash_block_size={hash_block_size} is not a "
                "multiple of it)"
            )
            faults.append(
                f"scratch group {index} {name} ({type(spec).__name__}) "
                f"requires hit alignment {alignment} (verified from {source}), "
                f"but hash_block_size={hash_block_size} is not a multiple of "
                "it. Every reachable fine-grained hit boundary H would satisfy "
                f"H % {alignment} != 0 for some H, resuming mid-pool and "
                "leaving un-recomputed raw K/gate entries that "
                "index_kpool_always_select_tail would then compress."
            )
            continue
        scratch[name] = alignment

    if blockers:
        # Mixed layout, matrix rows 3 and 4: a PARTICIPATING manager cannot
        # answer a fine lookup, so fine-grained hits are off and the alignment
        # stays at scheduler_block_size -- upstream's own behaviour. No
        # fine-grained hit is taken, so an unsafe scratch group (if any) is
        # unreachable, and a safe fallback must not be escalated into a boot
        # failure. The fault text still travels in `scratch` for the receipt.
        return False, blockers, scratch
    if faults:
        # Matrix row 2: nothing else stops fine-grained hits here, so this
        # refusal is the only thing between this layout and an unsafe mid-pool
        # resume.
        raise Glm53FineGrainedAPCError(
            f"{GLM53_FG_TAG} " + " ".join(faults) + " Fine-grained "
            "prefix-cache hits would otherwise be ENABLED on this layout and "
            "no participating manager forces the safe fallback. Refusing to "
            "start. Set GLM53_FINEGRAINED_APC=0 to fall back to block-aligned "
            "(scheduler_block_size) hits."
        )
    return True, blockers, scratch


# [glm53-finegrained-apc] helper-end


'''

GATE_OLD = """        if self.enable_partial_hash_hits:
            unsupported_partial_hit_managers = {
                type(manager).__name__
                for manager in self.single_type_managers
                if not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
            }
            if unsupported_partial_hit_managers:
                self.enable_partial_hash_hits = False
                logger.warning_once(
                    "Disabling fine-grained prefix-cache hits because these KV "
                    "cache managers require block-aligned lookups: %s.",
                    ", ".join(sorted(unsupported_partial_hit_managers)),
                )
"""

GATE_NEW = """        if self.enable_partial_hash_hits:
            # [glm53-finegrained-apc] Scope lookup compatibility to participating
            # groups; verify scratch alignment separately before enabling.
            if _glm53_finegrained_enabled(
                os.environ.get("GLM53_FINEGRAINED_APC", "0")
            ):
                _glm53_ok, _glm53_blockers, _glm53_scratch = (
                    _glm53_finegrained_hit_gate(
                        self.single_type_managers,
                        kv_cache_config.kv_cache_groups,
                        hash_block_size,
                    )
                )
                _glm53_why = (
                    "every participating manager can answer a "
                    "hash_block_size-granular lookup, and every "
                    "non-participating scratch alignment divides it"
                    if _glm53_ok
                    else "participating managers require block-aligned "
                    "lookups: " + ", ".join(sorted(_glm53_blockers))
                )
            else:
                _glm53_ok = False
                _glm53_blockers = ["GLM53_FINEGRAINED_APC=0 (kill switch)"]
                _glm53_scratch = {}
                _glm53_why = (
                    "GLM53_FINEGRAINED_APC=0 (kill switch): upstream "
                    "all-managers veto restored"
                )
            self.enable_partial_hash_hits = _glm53_ok
            # The scheduler-owning coordinator reports effective alignment
            # and the scratch checks behind its enable/fallback decision.
            if _glm53_ok:
                logger.info(  # [glm53-finegrained-apc]
                    "[glm53-apc-finegrained] "
                    "Fine-grained prefix-cache hits ENABLED: reason=%s; "
                    "effective alignment=%s tokens (hash_block_size; "
                    "scheduler_block_size=%s); "
                    "scratch groups checked: %s.",
                    _glm53_why,
                    hash_block_size,
                    self.scheduler_block_size,
                    _glm53_scratch or {},
                )
            else:
                logger.warning(  # [glm53-finegrained-apc]
                    "[glm53-apc-finegrained] "
                    "Fine-grained prefix-cache hits DISABLED: reason=%s; "
                    "effective alignment=%s tokens (scheduler_block_size; "
                    "hash_block_size=%s); "
                    "scratch groups checked: %s.",
                    _glm53_why,
                    self.scheduler_block_size,
                    hash_block_size,
                    _glm53_scratch or {},
                )
                logger.warning_once(
                    "Disabling fine-grained prefix-cache hits because these KV "
                    "cache managers require block-aligned lookups: %s.",
                    ", ".join(sorted(_glm53_blockers)),
                )
"""


def has_os_import(text: str) -> bool:
    # The retention overlay adds an inline marker to its os import. Recognize
    # that real import without inserting a duplicate in one application order.
    try:
        body = ast.parse(text).body
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Import)
               and any(alias.name == "os" and alias.asname in (None, "os")
                       for alias in node.names)
               for node in body)


def patched_problems(text: str) -> list[str]:
    """Accept only the canonical owned regions at their intended scope.

    Exact regions include comments and diagnostics; AST checks ensure those
    bytes are executable module helpers and the Hybrid coordinator init gate,
    rather than text in a string or an unrelated scope. Check all bindings of
    owned names so a later definition, assignment or import cannot replace a
    validated helper. This is source-drift detection, not a Python sandbox.
    """
    problems: list[str] = []
    for label, region in (("helper", HELPER.strip()), ("gate", GATE_NEW.rstrip())):
        if text.count(region) != 1:
            problems.append(f"canonical {label} region missing, changed or duplicated")
    for marker in (HELPER_BEGIN, HELPER_END):
        if text.count(marker) != 1:
            problems.append(f"{marker!r} must appear exactly once")
    if "unsupported_partial_hit_managers" in text:
        problems.append("upstream all-managers veto still present")
    try:
        tree = ast.parse(text)
        compile(tree, str(P), "exec")
    except (SyntaxError, ValueError) as exc:
        return problems + [f"does not compile: {exc}"]

    expected = ast.parse(HELPER).body
    owned_names = {
        node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        else node.targets[0].id
        for node in expected
    }
    canonical_nodes = []
    for definition in expected:
        matches = [node for node in tree.body
                   if ast.dump(node) == ast.dump(definition)]
        if len(matches) != 1:
            problems.append("canonical module helper definition missing or duplicated")
        canonical_nodes.extend(matches)
    allowed = {id(node) for root in canonical_nodes for node in ast.walk(root)}
    for node in ast.walk(tree):
        if id(node) in allowed:
            continue
        bound = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Import):
            bound.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            bound.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        if bound & owned_names:
            problems.append(f"owned helper name rebound outside canonical region: {sorted(bound & owned_names)}")

    expected_gate = ast.parse(textwrap.dedent(GATE_NEW)).body[0]
    coordinators = [node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "HybridKVCacheCoordinator"]
    inits = [node for cls in coordinators for node in cls.body
             if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
    gates = [node for init in inits for node in init.body
             if ast.dump(node) == ast.dump(expected_gate)]
    if len(coordinators) != 1 or len(inits) != 1 or len(gates) != 1:
        problems.append("canonical gate missing from HybridKVCacheCoordinator.__init__")
    if not any(isinstance(node, ast.Import)
               and any(alias.name == "os" and alias.asname in (None, "os")
                       for alias in node.names) for node in tree.body):
        problems.append("missing module import os")
    return problems


def pristine_problems(text: str) -> list[str]:
    """Anchor drift detection, run before anything is mutated."""
    problems: list[str] = []
    n = text.count(GATE_OLD)
    if n != 1:
        problems.append(f"expected one partial-hit-gate target, found {n}")
    n = text.count(HELPER_NEEDLE)
    if n != 1:
        problems.append(f"expected one helper insert point, found {n}")
    anchor = helper_anchor(text)
    if anchor is None:
        problems.append("no helper insert anchor found")
    elif text.count(anchor) != 1:
        problems.append(
            f"helper insert anchor {anchor!r} is not unique "
            f"(found {text.count(anchor)})"
        )
    if not has_os_import(text) and text.count(IMPORT_ANCHOR) != 1:
        problems.append(
            f"import anchor not unique (found {text.count(IMPORT_ANCHOR)})"
        )
    for stray in (
        HELPER_BEGIN,
        HELPER_END,
        "def _glm53_strict_int(",
        "def _glm53_scratch_alignment(",
        "def _glm53_finegrained_enabled(",
        "def _glm53_finegrained_hit_gate(",
        "class Glm53FineGrainedAPCError(",
    ):
        if stray in text:
            problems.append(
                f"{stray!r} already present without {MARK} "
                "(partial or foreign patch state)"
            )
    return problems


def atomic_write(path: Path, text: str) -> None:
    """Write via a temp file in the same directory + os.replace.

    An interrupted or failing run must never leave a half-patched coordinator:
    either the old bytes or the new bytes, never a mixture.
    """
    tmp = path.with_name(path.name + ".glm53-finegrained.tmp")
    try:
        tmp.write_text(text)
        try:
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        except OSError:
            pass
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()

    # A pre-existing marker is not proof of a complete patch: validate it.
    if MARK in text:
        problems = patched_problems(text)
        if problems:
            raise SystemExit(
                f"{P}: {MARK} is present but the patch is INCOMPLETE - "
                "refusing to leave a half-patched coordinator in place: "
                + "; ".join(problems)
            )
        print(f"{P.name}: {MARK} already present and complete - skipping")
        return 0

    problems = pristine_problems(text)
    if problems:
        raise SystemExit(f"{P}: " + "; ".join(problems))

    # The patched gate reads os.environ; upstream does not import os here
    # (checked against the live file: only abc/collections/typing + vllm).
    if not has_os_import(text):
        text = text.replace(IMPORT_ANCHOR, "import os\n" + IMPORT_ANCHOR, 1)

    anchor = helper_anchor(text)
    text = text.replace(anchor, HELPER + anchor, 1)
    text = text.replace(GATE_OLD, GATE_NEW, 1)

    # Validate the full result BEFORE it touches the filesystem.
    problems = patched_problems(text)
    if problems:
        raise SystemExit(
            f"{P}: refusing to write an invalid patched file: "
            + "; ".join(problems)
        )

    atomic_write(P, text)

    # And validate what actually landed on disk.
    problems = patched_problems(P.read_text())
    if problems:
        raise SystemExit(
            f"{P}: post-write validation failed: " + "; ".join(problems)
        )

    print(
        f"patched {P.name} (fine-grained APC: veto scoped to participating "
        f"groups; scratch groups verified for alignment divisibility, "
        f"refusing to start if violated)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
