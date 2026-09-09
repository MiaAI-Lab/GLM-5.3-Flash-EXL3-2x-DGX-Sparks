#!/usr/bin/env python3
"""Regression checks for GLM-5.3 chat-template reasoning controls."""

import json
import unittest
from pathlib import Path

from jinja2 import Environment


def _tojson(value, ensure_ascii=False, indent=None, **_kwargs):
    """vLLM's renderer supplies a tojson accepting ensure_ascii; bare Jinja2
    does not. Register the same shape so the template renders standalone."""
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)


def _environment() -> Environment:
    env = Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = _tojson
    return env


TEMPLATE = Path(__file__).parents[1] / "files" / "chat_template.jinja"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    }
]

CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Hi. How can I help?"},
    {"role": "user", "content": "and 3+3?"},
]


def render_generation_prompt(**kwargs: object) -> str:
    template = _environment().from_string(TEMPLATE.read_text())
    return template.render(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        add_generation_prompt=True,
        **kwargs,
    )


def render_conversation(**kwargs: object) -> str:
    template = _environment().from_string(TEMPLATE.read_text())
    return template.render(add_generation_prompt=True, **kwargs)


class ChatTemplateTests(unittest.TestCase):
    def test_thinking_defaults_on(self) -> None:
        rendered = render_generation_prompt()
        self.assertIn("<|system|>Reasoning Effort: Max", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered)

    def test_thinking_can_be_disabled(self) -> None:
        rendered = render_generation_prompt(enable_thinking=False)
        # Thinking off = closed think block AND no Reasoning Effort directive.
        # Emitting the directive with an empty think block (2026-08-30..09-09,
        # done to keep the prefix cache stable across a thinking toggle) made
        # the model reason in the answer channel: spark-bench code 98 -> 60,
        # structured 97 -> 58, syntax errors and fenced JSON at temperature 0.3.
        self.assertNotIn("Reasoning Effort", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think></think>"), rendered)

    def test_thinking_alias_matches_parser_behavior(self) -> None:
        rendered = render_generation_prompt(thinking=False)
        self.assertNotIn("Reasoning Effort", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think></think>"), rendered)

    def test_explicit_thinking_preserves_reasoning_effort(self) -> None:
        rendered = render_generation_prompt(
            enable_thinking=True,
            reasoning_effort="low",
        )
        self.assertIn("<|system|>Reasoning Effort: Low", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered)


class PrefixStabilityTests(unittest.TestCase):
    """What a thinking toggle costs in prefix cache, and what it must not cost.

    The Reasoning Effort head line is gated on thinking, so toggling thinking
    changes token ~2 and re-prefills the prompt (accepted: quality first —
    agent traffic does not toggle mid-conversation). Everything after the head
    line must still be byte-identical between the two shapes.
    """

    def _assert_same_after_head(self, tools) -> None:
        on = render_conversation(
            messages=CONVERSATION, tools=tools, enable_thinking=True
        )
        off = render_conversation(
            messages=CONVERSATION, tools=tools, enable_thinking=False
        )
        head = "<|system|>Reasoning Effort: Max"
        self.assertIn(head, on)
        self.assertNotIn("Reasoning Effort", off)
        self.assertEqual(off, on.replace(head, "", 1) + "</think>")

    def test_toggle_only_differs_by_head_line_without_tools(self) -> None:
        self._assert_same_after_head(None)

    def test_toggle_only_differs_by_head_line_with_tools(self) -> None:
        self._assert_same_after_head(TOOLS)

    def test_effort_levels_share_the_prompt_up_to_the_effort_word(self) -> None:
        low = render_conversation(
            messages=CONVERSATION, tools=TOOLS, enable_thinking=True,
            reasoning_effort="low",
        )
        high = render_conversation(
            messages=CONVERSATION, tools=TOOLS, enable_thinking=True,
            reasoning_effort="high",
        )
        self.assertEqual(len(_common_prefix(low, high)), len("[gMASK]<sop><|system|>Reasoning Effort: "))


def _common_prefix(a: str, b: str) -> str:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return a[:index]


if __name__ == "__main__":
    unittest.main()
