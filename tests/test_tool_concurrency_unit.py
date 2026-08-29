import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pytest
import json
from unittest.mock import MagicMock, patch
from test_tool_concurrency import stream_tool_request, make_padded_prompt, TEST_TOOLS


def test_make_padded_prompt():
    prompt = make_padded_prompt("TEST", filler_words=10, task="Do something")
    assert "SESSION TEST UNIQUE TSET." in prompt
    assert "the the the the the the the the the the" in prompt
    assert "Do something" in prompt


def test_tool_definitions_schema_validity():
    assert len(TEST_TOOLS) >= 2
    for t in TEST_TOOLS:
        assert t["type"] == "function"
        fn = t["function"]
        assert "name" in fn
        assert "parameters" in fn
        assert "required" in fn["parameters"]
        assert len(fn["parameters"]["required"]) > 0


import io

def test_stream_tool_request_detects_blank_args():
    # Simulate SSE response where tool_calls arguments is empty "{}" (Issue #10 bug)
    sse_data = (
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"terminal_execute","arguments":"{}"}}]}}]}\n\n'
        b'data: {"choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":20}}\n\n'
        b'data: [DONE]\n\n'
    )

    class MockHTTPResponse:
        def __init__(self, data):
            self.status = 200
            self._bio = io.BytesIO(data)
        def read(self, size=-1):
            return self._bio.read(size)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(sse_data)):
        out = stream_tool_request(prompt="test", max_tokens=64)
        assert out["http_status"] == 200
        assert out["num_tool_calls"] == 1
        assert out["blank_args_count"] == 1
        assert out["tool_calls"][0]["is_blank"] is True


def test_stream_tool_request_valid_args_reconstruction():
    # Simulate well-formed SSE chunks
    sse_data = (
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"terminal_execute","arguments":"{\\"command\\": "}}]}}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"echo hello\\""}}]}}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}\n\n'
        b'data: {"choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":50,"completion_tokens":15}}\n\n'
        b'data: [DONE]\n\n'
    )

    class MockHTTPResponse:
        def __init__(self, data):
            self.status = 200
            self._bio = io.BytesIO(data)
        def read(self, size=-1):
            return self._bio.read(size)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(sse_data)):
        out = stream_tool_request(prompt="test", max_tokens=64)
        assert out["http_status"] == 200
        assert out["num_tool_calls"] == 1
        assert out["blank_args_count"] == 0
        assert out["tool_calls"][0]["is_blank"] is False
        assert out["tool_calls"][0]["parsed_arguments"] == {"command": "echo hello"}
