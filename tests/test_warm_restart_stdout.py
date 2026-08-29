#!/usr/bin/env python3
"""Keep launcher JSON clean when the image imports the video overlay at startup."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_json_command_substitutions_skip_sitecustomize() -> None:
    source = (ROOT / "start.sh").read_text()
    marker = '$(python3 -S -c \'import json,os'
    assert source.count(marker) == 2


def test_import_time_overlay_never_writes_stdout() -> None:
    path = ROOT / "overlay" / "patch_glm_video_placeholders.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert calls
    for call in calls:
        file_keywords = [keyword for keyword in call.keywords if keyword.arg == "file"]
        assert len(file_keywords) == 1, f"stdout print at line {call.lineno}"
        target = file_keywords[0].value
        assert (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "sys"
            and target.attr == "stderr"
        ), f"print at line {call.lineno} must target sys.stderr"


if __name__ == "__main__":
    test_json_command_substitutions_skip_sitecustomize()
    test_import_time_overlay_never_writes_stdout()
    print("warm-restart stdout regression OK")
