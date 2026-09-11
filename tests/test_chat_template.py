#!/usr/bin/env python3
"""Regression checks for GLM-5.3 chat-template reasoning controls.

Contract: the `<|system|>Reasoning Effort: <Low|High|Max>` directive is emitted
exactly once, immediately before the LAST user message, and only when thinking
is on and a generation prompt is requested. Everything before that point is
byte-identical across thinking off / low / high / max, so toggling thinking or
changing the effort per request keeps the vLLM prefix cache (block hashes are
chained from token 0; the old head placement at char ~39 re-prefilled the
whole conversation on every change).
"""

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
DIRECTIVE = "<|system|>Reasoning Effort: "

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

TOOL_CONVERSATION = [
    {"role": "user", "content": "multiply 6 by 7"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "multiply", "arguments": {"a": 6, "b": 7}},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "42"},
    {"role": "user", "content": "thanks, and 8 by 9?"},
]

SHAPES = ((CONVERSATION, None), (CONVERSATION, TOOLS), (TOOL_CONVERSATION, TOOLS))


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


def _assert_directive_before_last_user(test, rendered: str, word: str) -> None:
    """Exactly one directive, and it sits immediately before the final <|user|>."""
    test.assertEqual(rendered.count(DIRECTIVE), 1, rendered)
    last_user = rendered.rfind("<|user|>")
    test.assertTrue(
        rendered[:last_user].endswith(f"{DIRECTIVE}{word}"),
        repr(rendered[max(0, last_user - 60):last_user + 20]),
    )


class ChatTemplateTests(unittest.TestCase):
    def test_thinking_defaults_on(self) -> None:
        rendered = render_generation_prompt()
        _assert_directive_before_last_user(self, rendered, "Max")
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
        _assert_directive_before_last_user(self, rendered, "Low")
        self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered)

    def test_directive_sits_before_last_user_in_every_shape(self) -> None:
        for messages, tools in SHAPES:
            for effort, word in (("low", "Low"), ("high", "High"), ("max", "Max")):
                rendered = render_conversation(
                    messages=messages, tools=tools, enable_thinking=True,
                    reasoning_effort=effort,
                )
                _assert_directive_before_last_user(self, rendered, word)
                self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered)

    def test_history_render_carries_no_directive(self) -> None:
        # add_generation_prompt=False is how a history is re-rendered (e.g. by a
        # training/eval pipeline); the directive is a generation-time control.
        template = _environment().from_string(TEMPLATE.read_text())
        rendered = template.render(
            messages=CONVERSATION, tools=None, add_generation_prompt=False,
            enable_thinking=True, reasoning_effort="high",
        )
        self.assertNotIn("Reasoning Effort", rendered)

    def test_no_user_message_falls_back_to_tail(self) -> None:
        rendered = render_conversation(
            messages=[{"role": "system", "content": "Say hi."}], tools=None,
            enable_thinking=True, reasoning_effort="high",
        )
        self.assertTrue(rendered.endswith(f"{DIRECTIVE}High<|assistant|><think>"), rendered)


class PrefixStabilityTests(unittest.TestCase):
    """What a thinking toggle or an effort change costs in prefix cache: only
    the last user turn.

    vLLM chains prefix-cache block hashes forward from token 0, so any
    divergence near the head invalidates the whole prompt. Every mode must
    therefore share the prompt byte-for-byte up to the last user message.
    """

    def _modes(self, messages, tools) -> dict[str, str]:
        return {
            "off": render_conversation(messages=messages, tools=tools, enable_thinking=False),
            "low": render_conversation(messages=messages, tools=tools, enable_thinking=True, reasoning_effort="low"),
            "high": render_conversation(messages=messages, tools=tools, enable_thinking=True, reasoning_effort="high"),
            "max": render_conversation(messages=messages, tools=tools, enable_thinking=True, reasoning_effort="max"),
            "default": render_conversation(messages=messages, tools=tools),
        }

    def test_all_modes_share_the_prompt_up_to_the_last_user_message(self) -> None:
        for messages, tools in SHAPES:
            modes = self._modes(messages, tools)
            off = modes["off"]
            cut = off.rfind("<|user|>")
            self.assertGreater(cut, 0)
            for name, rendered in modes.items():
                self.assertTrue(
                    rendered.startswith(off[:cut]),
                    f"{name} diverges from off at char {len(_common_prefix(off, rendered))} of {cut}",
                )

    def test_off_is_on_with_the_directive_removed(self) -> None:
        for messages, tools in SHAPES:
            modes = self._modes(messages, tools)
            for name in ("low", "high", "max"):
                on = modes[name]
                stripped = on.replace(f"{DIRECTIVE}{name.capitalize()}", "", 1)
                self.assertEqual(modes["off"], stripped + "</think>", name)

    def test_effort_levels_share_the_prompt_up_to_the_effort_word(self) -> None:
        low = render_conversation(
            messages=CONVERSATION, tools=TOOLS, enable_thinking=True,
            reasoning_effort="low",
        )
        high = render_conversation(
            messages=CONVERSATION, tools=TOOLS, enable_thinking=True,
            reasoning_effort="high",
        )
        shared = len(_common_prefix(low, high))
        self.assertEqual(shared, low.index(DIRECTIVE) + len(DIRECTIVE))
        # ... and that point is after the whole history, just before the last user turn
        self.assertGreater(shared, low.index("Hi. How can I help?"))


def _common_prefix(a: str, b: str) -> str:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return a[:index]


if __name__ == "__main__":
    unittest.main()
