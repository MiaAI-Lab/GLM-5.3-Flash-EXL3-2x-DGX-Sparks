import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from test_tool_concurrency import main, stream_tool_request


class MockStreamResponse:
    """Minimal urlopen-compatible response over a fixed SSE byte string."""

    def __init__(self, data: bytes):
        self.status = 200
        self._bio = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self._bio.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _stream(sse_data: bytes) -> dict:
    with patch("urllib.request.urlopen", return_value=MockStreamResponse(sse_data)):
        return stream_tool_request(prompt="test", max_tokens=64)


def test_stream_tool_request_detects_blank_args():
    # Simulate SSE response where tool_calls arguments is empty "{}" (Issue #10 bug)
    sse_data = (
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"terminal_execute","arguments":"{}"}}]}}]}\n\n'
        b'data: {"choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":20}}\n\n'
        b'data: [DONE]\n\n'
    )

    out = _stream(sse_data)
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

    out = _stream(sse_data)
    assert out["http_status"] == 200
    assert out["num_tool_calls"] == 1
    assert out["blank_args_count"] == 0
    assert out["tool_calls"][0]["is_blank"] is False
    assert out["tool_calls"][0]["parsed_arguments"] == {"command": "echo hello"}


def test_missing_usage_response_still_writes_receipt(tmp_path, capsys):
    """A stream that ends without a usage block must not abort receipt emission."""
    sse_data = (
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"terminal_execute","arguments":"{\\"command\\": \\"echo hi\\"}"}}]}}]}\n\n'
        b'data: {"choices":[{"finish_reason":"tool_calls"}]}\n\n'
        b'data: [DONE]\n\n'
    )
    receipt_path = tmp_path / "receipt.json"
    argv = ["test_tool_concurrency.py", "--concurrency", "1", "--out", str(receipt_path)]

    with patch("urllib.request.urlopen", return_value=MockStreamResponse(sse_data)):
        with patch.object(sys, "argv", argv):
            rc = main()

    assert rc == 0
    lane = json.loads(receipt_path.read_text(encoding="utf-8"))["results"][0]
    assert lane["usage"] is None
    assert lane["tok_s"] is None
    assert "Decode: unavailable" in capsys.readouterr().out
