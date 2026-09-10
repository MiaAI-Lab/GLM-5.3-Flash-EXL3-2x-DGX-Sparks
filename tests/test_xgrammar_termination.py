#!/usr/bin/env python3
"""Regression tests for the vLLM #52805/#53046 XGrammar backports."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_xgrammar_termination.py",
        ROOT / "overlay" / "patch_xgrammar_termination.py",
    )
    if p.is_file()
)
INSTALLED_BACKEND = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/"
    "backend_xgrammar.py"
)
INSTALLED_MANAGER = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/"
    "__init__.py"
)
BACKEND_MARK = (
    "# [glm53-xgrammar-termination] Source-exact vLLM 12f64b39 backport."
)
MANAGER_MARK = (
    "# [glm53-xgrammar-reasoning] Source-exact vLLM c6e19b3 backport."
)

# Exact vLLM 487ecf187 patch anchors, embedded in a dependency-free harness.
PINNED_BACKEND_FIXTURE = '''class _Logger:
    def error(self, *args):
        pass


logger = _Logger()


class XgrammarGrammar:
    def __init__(self, matcher):
        self.matcher = matcher
        self.num_processed_tokens = 0
        self._is_terminated = False

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

    def fill_bitmask(self, bitmask, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        return self._is_terminated

    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()
'''

PINNED_MANAGER_FIXTURE = '''class StructuredOutputManager:
    def advance_speculative_draft(
        self,
        grammar,
        token: int,
        post_reasoning_end_in_window: bool,
    ) -> int:
        advance_grammar = True
        req_id = "req"
        scheduled_spec_decode_tokens = {req_id: [token]}
        state_advancements = 0
        if grammar is not None:
            for req_id in scheduled_spec_decode_tokens:
                for token in [token]:
                    if advance_grammar and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
                            raise AssertionError(
                                (token, req_id, scheduled_spec_decode_tokens)
                            )
        return state_advancements
'''


class FakeMatcher:
    """Small matcher that terminates on token 99 and detects over-advance."""

    def __init__(self):
        self.accepted: list[int] = []
        self.calls_after_termination = 0

    def accept_token(self, token: int) -> bool:
        if self.is_terminated():
            self.calls_after_termination += 1
            return False
        if token == -1:
            return False
        self.accepted.append(token)
        return True

    def is_terminated(self) -> bool:
        return bool(self.accepted and self.accepted[-1] == 99)

    def rollback(self, count: int) -> None:
        if count:
            del self.accepted[-count:]

    def reset(self) -> None:
        self.accepted.clear()

    def fill_next_token_bitmask(self, bitmask, idx: int) -> None:
        pass


class FakeGrammar:
    """Grammar whose invalid tokens must be validated, never advanced."""

    def __init__(self, valid_token: int):
        self.valid_token = valid_token
        self.validated: list[list[int]] = []
        self.accepted: list[list[int]] = []

    def is_terminated(self) -> bool:
        return False

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        self.validated.append(tokens)
        return tokens if tokens == [self.valid_token] else []

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        if tokens != [self.valid_token]:
            raise AssertionError("invalid speculative draft reached accept_tokens")
        self.accepted.append(tokens)
        return True


def run_patch(
    backend: Path,
    manager: Path,
    *,
    ok: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_XGRAMMAR_BACKEND_PY"] = str(backend)
    env["GLM53_XGRAMMAR_MANAGER_PY"] = str(manager)
    proc = subprocess.run(
        [sys.executable, str(PATCH)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if ok and proc.returncode != 0:
        raise AssertionError(proc.stdout + proc.stderr)
    if not ok and proc.returncode == 0:
        raise AssertionError("patch unexpectedly accepted a drifted/partial target")
    return proc


def assert_backend_behavior(source: str) -> None:
    namespace: dict[str, object] = {}
    exec(compile(source, "patched_backend_fixture.py", "exec"), namespace)
    grammar_cls = namespace["XgrammarGrammar"]

    matcher = FakeMatcher()
    grammar = grammar_cls(matcher)
    assert grammar.accept_tokens("req", [7, 99, 8])
    assert matcher.accepted == [7, 99]
    assert matcher.calls_after_termination == 0
    assert grammar.num_processed_tokens == 2
    assert grammar.is_terminated()
    assert grammar.accept_tokens("req", [8])
    assert matcher.accepted == [7, 99]
    assert matcher.calls_after_termination == 0

    grammar.reset()
    assert matcher.accepted == []
    assert grammar.num_processed_tokens == 0
    assert not grammar.is_terminated()

    assert grammar.validate_tokens([7, 99, 8]) == [7, 99]
    assert matcher.accepted == []
    assert matcher.calls_after_termination == 0
    assert not grammar.is_terminated()

    assert grammar.accept_tokens("req", [99])
    assert grammar.validate_tokens([8]) == []
    assert matcher.calls_after_termination == 0


def assert_manager_behavior(source: str) -> None:
    namespace: dict[str, object] = {}
    exec(compile(source, "patched_manager_fixture.py", "exec"), namespace)
    manager = namespace["StructuredOutputManager"]()

    grammar = FakeGrammar(valid_token=7)
    assert manager.advance_speculative_draft(grammar, 8, True) == 0
    assert grammar.validated == [[8]]
    assert grammar.accepted == []

    assert manager.advance_speculative_draft(grammar, 7, True) == 1
    assert grammar.validated == [[8], [7]]
    assert grammar.accepted == [[7]]

    # Drafts outside the mid-window reasoning boundary retain the original
    # direct-advance path.
    direct = FakeGrammar(valid_token=9)
    assert manager.advance_speculative_draft(direct, 9, False) == 1
    assert direct.validated == []
    assert direct.accepted == [[9]]


def test_fixture() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        backend = Path(tmp) / "backend_xgrammar.py"
        manager = Path(tmp) / "__init__.py"
        backend.write_text(PINNED_BACKEND_FIXTURE)
        manager.write_text(PINNED_MANAGER_FIXTURE)
        run_patch(backend, manager)
        patched_backend = backend.read_text()
        patched_manager = manager.read_text()
        assert patched_backend.count(BACKEND_MARK) == 1
        assert patched_manager.count(MANAGER_MARK) == 1
        assert "Tokens after termination are ignored." in patched_backend
        assert (
            "if self.matcher.is_terminated():\n                    break"
            in patched_backend
        )
        assert (
            "self.matcher.reset()\n        self.num_processed_tokens = 0"
            in patched_backend
        )
        assert "accepted = bool(grammar.validate_tokens([token]))" in patched_manager
        assert_backend_behavior(patched_backend)
        assert_manager_behavior(patched_manager)

        run_patch(backend, manager)
        assert backend.read_text() == patched_backend
        assert manager.read_text() == patched_manager

        # Exact merged behavior is accepted when a newer image already has it.
        backend.write_text(
            patched_backend.replace(f"    {BACKEND_MARK}\n", "", 1)
        )
        manager.write_text(
            patched_manager.replace(f"                    {MANAGER_MARK}\n", "", 1)
        )
        upstream_backend = backend.read_text()
        upstream_manager = manager.read_text()
        run_patch(backend, manager)
        assert backend.read_text() == upstream_backend
        assert manager.read_text() == upstream_manager


def test_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        backend = Path(tmp) / "backend_xgrammar.py"
        manager = Path(tmp) / "__init__.py"

        drifted_backend = PINNED_BACKEND_FIXTURE.replace(
            "Returns False if the FSM failed to advance.",
            "Returns False on failure.",
            1,
        )
        backend.write_text(drifted_backend)
        manager.write_text(PINNED_MANAGER_FIXTURE)
        run_patch(backend, manager, ok=False)
        assert backend.read_text() == drifted_backend
        assert manager.read_text() == PINNED_MANAGER_FIXTURE

        # The backend is not written when the second-file preflight fails.
        drifted_manager = PINNED_MANAGER_FIXTURE.replace(
            "accepted = grammar.accept_tokens(req_id, [token])",
            "accepted = grammar.accept_tokens(req_id, tuple([token]))",
            1,
        )
        backend.write_text(PINNED_BACKEND_FIXTURE)
        manager.write_text(drifted_manager)
        run_patch(backend, manager, ok=False)
        assert backend.read_text() == PINNED_BACKEND_FIXTURE
        assert manager.read_text() == drifted_manager

        partial_backend = PINNED_BACKEND_FIXTURE.replace(
            "    def accept_tokens",
            f"    {BACKEND_MARK}\n    def accept_tokens",
            1,
        )
        backend.write_text(partial_backend)
        manager.write_text(PINNED_MANAGER_FIXTURE)
        run_patch(backend, manager, ok=False)
        assert backend.read_text() == partial_backend
        assert manager.read_text() == PINNED_MANAGER_FIXTURE

        partial_manager = PINNED_MANAGER_FIXTURE.replace(
            "                    if advance_grammar",
            f"                    {MANAGER_MARK}\n"
            "                    if advance_grammar",
            1,
        )
        backend.write_text(PINNED_BACKEND_FIXTURE)
        manager.write_text(partial_manager)
        run_patch(backend, manager, ok=False)
        assert backend.read_text() == PINNED_BACKEND_FIXTURE
        assert manager.read_text() == partial_manager


def test_installed_copy_if_present() -> None:
    backend_source = Path(
        os.environ.get("GLM53_XGRAMMAR_BACKEND_PY_SRC", INSTALLED_BACKEND)
    )
    manager_source = Path(
        os.environ.get("GLM53_XGRAMMAR_MANAGER_PY_SRC", INSTALLED_MANAGER)
    )
    if not backend_source.is_file() or not manager_source.is_file():
        return
    with tempfile.TemporaryDirectory() as tmp:
        backend = Path(tmp) / "backend_xgrammar.py"
        manager = Path(tmp) / "__init__.py"
        backend.write_bytes(backend_source.read_bytes())
        manager.write_bytes(manager_source.read_bytes())
        run_patch(backend, manager)
        patched_backend = backend.read_text()
        patched_manager = manager.read_text()
        compile(patched_backend, str(backend), "exec")
        compile(patched_manager, str(manager), "exec")
        assert "Tokens after termination are ignored." in patched_backend
        assert "self._is_terminated = False" in patched_backend
        assert "accepted = bool(grammar.validate_tokens([token]))" in patched_manager
        run_patch(backend, manager)


def test_recipe_wiring_if_present() -> None:
    start = ROOT / "start.sh"
    dockerfile = ROOT / "Dockerfile"
    if not start.is_file() or not dockerfile.is_file():
        return
    launcher = start.read_text()
    image = dockerfile.read_text()
    assert 'XGRAMMAR_PATCH_HOST="${XGRAMMAR_PATCH_HOST:-' in launcher
    # Both ranks apply the one pinned list (GLM53_OVERLAY_ORDER) that
    # write_inner_scripts emits into the head and worker inner scripts.
    order = launcher[launcher.index("GLM53_OVERLAY_ORDER=(") : launcher.index(")", launcher.index("GLM53_OVERLAY_ORDER=("))]
    assert "\n    patch_xgrammar_termination.py\n" in order
    assert 'emit_overlay_block >> "$HEAD_SCRIPT"' in launcher
    assert 'emit_overlay_block >> "$WORKER_SCRIPT"' in launcher
    assert (
        "-v '/tmp/patch_xgrammar_termination.py:"
        "/opt/glm53/patch_xgrammar_termination.py:ro'" in launcher
    )
    assert (
        '-v "$XGRAMMAR_PATCH_HOST:'
        '/opt/glm53/patch_xgrammar_termination.py:ro"' in launcher
    )
    assert "scp -q -o BatchMode=yes \"$XGRAMMAR_PATCH_HOST\"" in launcher
    assert "COPY overlay/patch_xgrammar_termination.py" in image
    assert "RUN python3 /opt/glm53/patch_xgrammar_termination.py" in image
    assert "python3 /opt/glm53/test_xgrammar_termination.py" in image


def main() -> int:
    test_fixture()
    test_fail_closed()
    test_installed_copy_if_present()
    test_recipe_wiring_if_present()
    print("xgrammar speculative patches OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
