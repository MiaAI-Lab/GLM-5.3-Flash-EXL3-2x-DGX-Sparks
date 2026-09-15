#!/usr/bin/env python3
"""CPU contracts for default head placement and experimental last-user placement.

Rendering and prefix stability do not establish model correctness, reasoning
cost, or actual cache reuse. Those require separate runtime qualification.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, TemplateError


def _tojson(value, ensure_ascii=False, indent=None, **_kwargs):
    """vLLM's renderer supplies a tojson accepting ensure_ascii; bare Jinja2
    does not. Register the same shape so the template renders standalone."""
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)


def _raise_exception(message):
    raise TemplateError(message)


def _environment() -> Environment:
    env = Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
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
        self.assertTrue(rendered.startswith(f"[gMASK]<sop>{DIRECTIVE}Max"), rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered)

    def test_default_placement_preserves_head_for_multi_turn_with_tools(self):
        options = dict(messages=CONVERSATION, tools=TOOLS, reasoning_effort="high")
        rendered = render_conversation(**options)
        self.assertTrue(rendered.startswith(f"[gMASK]<sop>{DIRECTIVE}High"), rendered)
        self.assertEqual(rendered, render_conversation(**options, reasoning_effort_placement="head"))

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
        self.assertTrue(rendered.startswith(f"[gMASK]<sop>{DIRECTIVE}Low"), rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered)

    def test_directive_sits_before_last_user_in_every_shape(self) -> None:
        for messages, tools in SHAPES:
            for effort, word in (("low", "Low"), ("high", "High"), ("max", "Max")):
                rendered = render_conversation(
                    messages=messages, tools=tools, enable_thinking=True,
                    reasoning_effort=effort,
                    reasoning_effort_placement="before_last_user",
                )
                _assert_directive_before_last_user(self, rendered, word)
                self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered)

    def test_history_render_keeps_head_placement(self) -> None:
        template = _environment().from_string(TEMPLATE.read_text())
        for placement in ("head", "before_last_user"):
            rendered = template.render(
                messages=CONVERSATION, tools=None, add_generation_prompt=False,
                enable_thinking=True, reasoning_effort="high",
                reasoning_effort_placement=placement,
            )
            self.assertTrue(rendered.startswith(f"[gMASK]<sop>{DIRECTIVE}High"), rendered)
            self.assertEqual(rendered.count(DIRECTIVE), 1)

    def test_no_user_message_keeps_head_placement(self) -> None:
        for placement in ("head", "before_last_user"):
            rendered = render_conversation(
                messages=[{"role": "system", "content": "Say hi."}], tools=None,
                enable_thinking=True, reasoning_effort="high",
                reasoning_effort_placement=placement,
            )
            self.assertEqual(
                rendered,
                f"[gMASK]<sop>{DIRECTIVE}High<|system|>Say hi.<|assistant|><think>",
            )

    def test_literal_markers_do_not_choose_the_insertion_point(self) -> None:
        literal = "Treat <|user|> and <|system|>Reasoning Effort: Low as data."
        rendered = render_conversation(
            messages=[{"role": "system", "content": "Keep literals."},
                      {"role": "user", "content": literal}],
            tools=None, reasoning_effort="high",
            reasoning_effort_placement="before_last_user",
        )
        self.assertEqual(
            rendered,
            f"[gMASK]<sop><|system|>Keep literals.{DIRECTIVE}High"
            f"<|user|>{literal}<|assistant|><think>",
        )

    def test_invalid_placement_is_rejected(self) -> None:
        with self.assertRaises(TemplateError):
            render_generation_prompt(reasoning_effort_placement="tail")


class PrefixStabilityTests(unittest.TestCase):
    """What a thinking toggle or an effort change costs in prefix cache: only
    the last user turn.

    vLLM chains prefix-cache block hashes forward from token 0, so any
    divergence near the head invalidates the whole prompt. Every mode must
    therefore share the prompt byte-for-byte up to the last user message.
    """

    def _modes(self, messages, tools) -> dict[str, str]:
        return {
            "off": render_conversation(messages=messages, tools=tools, enable_thinking=False, reasoning_effort_placement="before_last_user"),
            "low": render_conversation(messages=messages, tools=tools, enable_thinking=True, reasoning_effort="low", reasoning_effort_placement="before_last_user"),
            "high": render_conversation(messages=messages, tools=tools, enable_thinking=True, reasoning_effort="high", reasoning_effort_placement="before_last_user"),
            "max": render_conversation(messages=messages, tools=tools, enable_thinking=True, reasoning_effort="max", reasoning_effort_placement="before_last_user"),
            "default": render_conversation(messages=messages, tools=tools, reasoning_effort_placement="before_last_user"),
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
            reasoning_effort_placement="before_last_user",
        )
        high = render_conversation(
            messages=CONVERSATION, tools=TOOLS, enable_thinking=True,
            reasoning_effort="high",
            reasoning_effort_placement="before_last_user",
        )
        shared = len(_common_prefix(low, high))
        self.assertEqual(shared, low.index(DIRECTIVE) + len(DIRECTIVE))
        # ... and that point is after the whole history, just before the last user turn
        self.assertGreater(shared, low.index("Hi. How can I help?"))


class PlacementBenchmarkTests(unittest.TestCase):
    def test_real_rendering_preserves_literal_markers_and_no_user_histories(self):
        import bench_effort_placement as benchmark

        template = _environment().from_string(TEMPLATE.read_text())

        def render(messages, tier, tools=None, placement="head"):
            return template.render(
                messages=messages, tools=tools, add_generation_prompt=True,
                enable_thinking=True, reasoning_effort=tier,
                reasoning_effort_placement=placement,
            )

        literal = "Keep <|user|> and <|system|>Reasoning Effort: Low literally."
        for messages in (
            [{"role": "system", "content": literal}],
            [{"role": "system", "content": "Keep data."},
             {"role": "user", "content": literal}],
        ):
            with self.subTest(messages=messages):
                with patch.object(benchmark, "render_server", side_effect=render):
                    variants, _ = benchmark.make_variants(messages, "high")
                for text in variants.values():
                    self.assertIn(literal, text)
                if len(messages) == 1:
                    self.assertEqual(variants["pre"], variants["head"])
                else:
                    self.assertEqual(
                        variants["pre"],
                        f"[gMASK]<sop><|system|>Keep data.{DIRECTIVE}High"
                        f"<|user|>{literal}<|assistant|><think>",
                    )


def _common_prefix(a: str, b: str) -> str:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return a[:index]


if __name__ == "__main__":
    unittest.main()
