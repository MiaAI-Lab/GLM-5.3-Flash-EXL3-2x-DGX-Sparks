#!/usr/bin/env python3
"""Configure vLLM SpinCondition's reader spin window in milliseconds.

``GLM53_SPINWAIT_MS=stock`` (or an unset variable) preserves vLLM's one-second
default. A positive integer from 1 through 1000 selects that many milliseconds.
The launcher validates and canonicalizes the value before stopping a running
cluster; this patch validates again inside each container and fails closed on
source drift.

The deployment value 16 was selected by a frozen TP=2 sweep at MNBT=2048:
median decode was +0.95% versus stock while active EngineCore CPU fell 85.3%.
The 2 ms candidate lost 1.68% decode in its paired test; 32 and 64 ms both lost
about 1.8% while consuming more CPU than 16 ms.
"""
from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path

ENV_NAME = "GLM53_SPINWAIT_MS"
MAX_MS = 1000
TARGET = Path(
    os.environ.get(
        "GLM53_SPINWAIT_TARGET",
        "/usr/local/lib/python3.12/dist-packages/vllm/distributed/"
        "device_communicators/shm_broadcast.py",
    )
)
STOCK_LINE = "        busy_loop_s: float = 1,\n"


def parse_spinwait_ms(raw: str | None) -> int | None:
    """Return milliseconds, or ``None`` for stock behavior."""
    if raw is None or raw == "stock":
        return None
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise ValueError(
            f"{ENV_NAME} must be 'stock' or a base-10 integer from 1 to "
            f"{MAX_MS} (got: {raw!r})"
        )
    value = int(raw, 10)
    if not 1 <= value <= MAX_MS:
        raise ValueError(
            f"{ENV_NAME} must be between 1 and {MAX_MS} milliseconds "
            f"(got: {raw!r})"
        )
    return value


def seconds_literal(milliseconds: int) -> str:
    if milliseconds == 1000:
        return "1"
    return f"0.{milliseconds:03d}".rstrip("0")


def configured_line(milliseconds: int) -> str:
    return f"        busy_loop_s: float = {seconds_literal(milliseconds)},\n"


def prepare(source: str, milliseconds: int | None) -> tuple[str, str]:
    """Return patched source and an action; reject ambiguous source states."""
    stock_count = source.count(STOCK_LINE)
    if milliseconds is None:
        if stock_count != 1:
            raise ValueError(
                f"stock anchor found {stock_count} times (expected exactly 1)"
            )
        return source, "stock"

    replacement = configured_line(milliseconds)
    if replacement == STOCK_LINE:
        if stock_count != 1:
            raise ValueError(
                f"stock anchor found {stock_count} times (expected exactly 1)"
            )
        return source, "already stock-equivalent (1000 ms)"

    replacement_count = source.count(replacement)
    if stock_count == 1 and replacement_count == 0:
        return source.replace(STOCK_LINE, replacement, 1), "patched"
    if stock_count == 0 and replacement_count == 1:
        return source, "already present"
    raise ValueError(
        "spinwait source is ambiguous or drifted: "
        f"stock={stock_count}, configured={replacement_count}"
    )


def replace_file(target: Path, source: str) -> None:
    temp = target.with_name(f".{target.name}.glm53-spinwait.tmp")
    try:
        temp.write_text(source)
        os.chmod(temp, stat.S_IMODE(target.stat().st_mode))
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{target.stem}*.pyc"):
            pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    unknown = [arg for arg in argv[1:] if arg != "--preflight"]
    if unknown:
        raise SystemExit(f"unknown arguments: {' '.join(unknown)}")
    preflight_only = "--preflight" in argv[1:]

    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    try:
        milliseconds = parse_spinwait_ms(os.environ.get(ENV_NAME))
        source = TARGET.read_text()
        patched, action = prepare(source, milliseconds)
        compile(patched, str(TARGET), "exec")
    except ValueError as exc:
        raise SystemExit(f"spinwait preflight failed: {exc}") from exc

    selected = "stock" if milliseconds is None else f"{milliseconds} ms"
    if preflight_only:
        print(f"{TARGET.name}: spinwait preflight OK ({selected}; {action})")
        return 0
    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    print(f"{TARGET.name}: spinwait {selected} ({action})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
