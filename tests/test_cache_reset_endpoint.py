#!/usr/bin/env python3
"""Regression tests for the #31 cache-reset endpoint exposure."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_cache_reset.py",
        ROOT / "overlay" / "patch_cache_reset.py",
    )
    if p.is_file()
)
INSTALLED_API_SERVER = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/"
    "api_server.py"
)
MARK = "# [glm53-cache-reset]"
FLAG = "GLM53_EXPOSE_CACHE_RESET"

# Exact vLLM 487ecf187 build_app dev-mode block, embedded in a
# dependency-free harness. The harness injects fake `envs` / vLLM modules so
# the patched source can be exec'd and its routing behavior asserted.
PINNED_API_SERVER_FIXTURE = '''def build_app(app):
    if envs.VLLM_SERVER_DEV_MODE:
        from vllm.entrypoints.serve import register_vllm_dev_api_routers

        register_vllm_dev_api_routers(app)
'''


def run_patch(
    target: Path,
    *,
    ok: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_API_SERVER_PY"] = str(target)
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


def run_build_app(
    source: str,
    *,
    dev_mode: bool,
    flag: str | None,
    calls: list[str],
) -> None:
    """Exec the fixture's build_app with fake vLLM modules and flag env."""
    serve = types.ModuleType("vllm.entrypoints.serve")
    serve.register_vllm_dev_api_routers = lambda app: calls.append("dev")
    api_router = types.ModuleType(
        "vllm.entrypoints.serve.dev.cache.api_router"
    )
    api_router.attach_router = lambda app: calls.append("cache")

    saved = sys.modules.get("vllm")
    saved_flag = os.environ.get(FLAG)
    try:
        sys.modules["vllm"] = types.ModuleType("vllm")
        sys.modules["vllm.entrypoints"] = types.ModuleType("vllm.entrypoints")
        sys.modules["vllm.entrypoints.serve"] = serve
        sys.modules["vllm.entrypoints.serve.dev"] = types.ModuleType(
            "vllm.entrypoints.serve.dev"
        )
        sys.modules["vllm.entrypoints.serve.dev.cache"] = types.ModuleType(
            "vllm.entrypoints.serve.dev.cache"
        )
        sys.modules["vllm.entrypoints.serve.dev.cache.api_router"] = api_router
        if flag is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = flag
        namespace: dict[str, object] = {
            "os": os,
            "envs": types.SimpleNamespace(VLLM_SERVER_DEV_MODE=dev_mode),
        }
        exec(compile(source, "patched_api_server_fixture.py", "exec"), namespace)
        namespace["build_app"](object())
    finally:
        if saved is not None:
            sys.modules["vllm"] = saved
        else:
            sys.modules.pop("vllm", None)
        for name in (
            "vllm.entrypoints",
            "vllm.entrypoints.serve",
            "vllm.entrypoints.serve.dev",
            "vllm.entrypoints.serve.dev.cache",
            "vllm.entrypoints.serve.dev.cache.api_router",
        ):
            sys.modules.pop(name, None)
        if saved_flag is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = saved_flag


def test_fixture() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "api_server.py"
        target.write_text(PINNED_API_SERVER_FIXTURE)
        run_patch(target)
        patched = target.read_text()
        assert patched.count(MARK) == 1
        assert 'elif os.getenv("GLM53_EXPOSE_CACHE_RESET", "0") == "1":' in patched
        assert "from vllm.entrypoints.serve.dev.cache.api_router import" in patched
        assert "attach_cache_reset_router(app)" in patched
        compile(patched, "patched_fixture.py", "exec")

        # Flag semantics: the router mounts only when the flag is "1", full
        # dev mode keeps precedence, and anything else stays stock.
        calls: list[str] = []
        run_build_app(patched, dev_mode=False, flag="1", calls=calls)
        assert calls == ["cache"], calls
        calls = []
        run_build_app(patched, dev_mode=False, flag=None, calls=calls)
        assert calls == [], calls
        calls = []
        run_build_app(patched, dev_mode=False, flag="0", calls=calls)
        assert calls == [], calls
        calls = []
        run_build_app(patched, dev_mode=True, flag="1", calls=calls)
        assert calls == ["dev"], calls

        run_patch(target)
        assert target.read_text() == patched

        # Exact merged behavior is accepted when a newer image already has it
        # (marker stripped, elif kept): the patch skips instead of failing.
        merged = patched.replace("        # [glm53-cache-reset] Expose only\n"
                                 , "        #\n", 1)
        target.write_text(merged)
        run_patch(target)
        assert target.read_text() == merged


def test_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "api_server.py"

        drifted = PINNED_API_SERVER_FIXTURE.replace(
            "register_vllm_dev_api_routers(app)",
            "register_vllm_dev_api_routers(app, strict=True)",
            1,
        )
        target.write_text(drifted)
        run_patch(target, ok=False)
        assert target.read_text() == drifted

        partial = PINNED_API_SERVER_FIXTURE.replace(
            "    if envs.VLLM_SERVER_DEV_MODE:",
            f"    {MARK} Expose only the cache-reset dev routes.\n"
            "    if envs.VLLM_SERVER_DEV_MODE:",
            1,
        )
        target.write_text(partial)
        run_patch(target, ok=False)
        assert target.read_text() == partial

        duplicated = PINNED_API_SERVER_FIXTURE + PINNED_API_SERVER_FIXTURE
        target.write_text(duplicated)
        run_patch(target, ok=False)
        assert target.read_text() == duplicated


def test_installed_copy_if_present() -> None:
    source = Path(os.environ.get("GLM53_API_SERVER_PY_SRC", INSTALLED_API_SERVER))
    if not source.is_file():
        return
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "api_server.py"
        target.write_bytes(source.read_bytes())
        run_patch(target)
        patched = target.read_text()
        compile(patched, str(target), "exec")
        assert patched.count(MARK) == 1
        assert "attach_cache_reset_router(app)" in patched
        run_patch(target)


def test_recipe_wiring_if_present() -> None:
    start = ROOT / "start.sh"
    dockerfile = ROOT / "Dockerfile"
    if not start.is_file() or not dockerfile.is_file():
        return
    launcher = start.read_text()
    image = dockerfile.read_text()
    assert 'CACHE_RESET_PATCH_HOST="${CACHE_RESET_PATCH_HOST:-' in launcher
    assert 'GLM53_EXPOSE_CACHE_RESET="${GLM53_EXPOSE_CACHE_RESET:-1}"' in launcher
    # forwarded through the shared container env block (head + worker)
    assert '-e "GLM53_EXPOSE_CACHE_RESET=$GLM53_EXPOSE_CACHE_RESET"' in launcher
    assert launcher.count("python3 /opt/glm53/patch_cache_reset.py") == 2
    assert (
        "-v '/tmp/patch_cache_reset.py:"
        "/opt/glm53/patch_cache_reset.py:ro'" in launcher
    )
    assert (
        '-v "$CACHE_RESET_PATCH_HOST:'
        '/opt/glm53/patch_cache_reset.py:ro"' in launcher
    )
    assert 'scp -q -o BatchMode=yes "$CACHE_RESET_PATCH_HOST"' in launcher
    assert "COPY overlay/patch_cache_reset.py" in image
    assert "RUN python3 /opt/glm53/patch_cache_reset.py" in image
    assert "python3 /opt/glm53/test_cache_reset_endpoint.py" in image


def main() -> int:
    test_fixture()
    test_fail_closed()
    test_installed_copy_if_present()
    test_recipe_wiring_if_present()
    print("cache-reset endpoint patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
