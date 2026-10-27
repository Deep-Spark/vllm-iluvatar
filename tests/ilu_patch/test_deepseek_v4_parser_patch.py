# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek V4/V4.1 DSML compatibility patch."""

from __future__ import annotations

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.registered_adapters import (
    DeepSeekV4ParserReasoningAdapter,
    DeepSeekV4ParserToolAdapter,
    DeepSeekV41ParserReasoningAdapter,
    DeepSeekV41ParserToolAdapter,
)
from vllm_iluvatar.ilu_patch.general_patches import patch_deepseek_v4_parser

_TOOL = {
    "type": "function",
    "function": {
        "name": "weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    },
}
_NAMED_TOOL_CHOICE = {
    "type": "function",
    "function": {"name": "weather"},
}
_V41_INVOKE = (
    '<｜DSML｜ invoke name="weather">'
    '<｜DSML｜ parameter name="city" string="true">'
    "北京"
    "</｜DSML｜ parameter>"
    "</｜DSML｜ invoke>"
)
_V4_INVOKE = (
    '<｜DSML｜invoke name="weather">'
    '<｜DSML｜parameter name="city" string="true">'
    "北京"
    "</｜DSML｜parameter>"
    "</｜DSML｜invoke>"
)


class _Tokenizer:
    def get_vocab(self):
        return {}

    def decode(self, _token_ids):
        return ""


class _V4Parser(DelegatingParser):
    reasoning_parser_cls = DeepSeekV4ParserReasoningAdapter
    tool_parser_cls = DeepSeekV4ParserToolAdapter


class _V41Parser(DelegatingParser):
    reasoning_parser_cls = DeepSeekV41ParserReasoningAdapter
    tool_parser_cls = DeepSeekV41ParserToolAdapter


@pytest.fixture(scope="module", autouse=True)
def _apply_patch():
    patch_deepseek_v4_parser._apply()


def _request(
    *,
    tools: bool,
    tool_choice="auto",
    stream: bool = False,
):
    kwargs = {
        "messages": [{"role": "user", "content": "weather"}],
        "stream": stream,
    }
    if tools:
        kwargs.update(tools=[_TOOL], tool_choice=tool_choice)
    return ChatCompletionRequest(**kwargs)


def _parse(parser_cls, model_output, *, tools, tool_choice="auto", stream=False):
    request = _request(tools=tools, tool_choice=tool_choice, stream=stream)
    parser = parser_cls(_Tokenizer(), tools=request.tools)
    if stream:
        delta = parser.parse_delta(
            model_output,
            [],
            request,
            prompt_token_ids=[],
            finished=True,
        )
        return (
            getattr(delta, "reasoning", None),
            getattr(delta, "content", None),
            getattr(delta, "tool_calls", None) or [],
        )
    return parser.parse(model_output, request, enable_auto_tools=True)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "wrapper",
    [
        "<｜DSML｜ toolcalls>",
        "<｜DSML｜ tool_calls>",
        "<｜DSML｜ tool>",
        "<｜DSML｜  calls>",
    ],
)
def test_v41_wrapper_variants_are_parsed(wrapper, stream):
    output = f"</think>{wrapper}{_V41_INVOKE}</｜DSML｜ calls>"

    reasoning, content, tool_calls = _parse(
        _V41Parser,
        output,
        tools=True,
        stream=stream,
    )

    assert reasoning is None
    assert content is None
    assert tool_calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "parser_cls,invoke",
    [(_V4Parser, _V4_INVOKE), (_V41Parser, _V41_INVOKE)],
)
def test_bare_invoke_ends_reasoning(parser_cls, invoke, stream):
    reasoning, content, tool_calls = _parse(
        parser_cls,
        invoke,
        tools=True,
        stream=stream,
    )

    assert reasoning is None
    assert content is None
    assert tool_calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "tool_choice",
    [
        pytest.param(None, id="null"),
        pytest.param("auto", id="auto"),
        pytest.param("none", id="none"),
        pytest.param("required", id="required"),
        pytest.param(_NAMED_TOOL_CHOICE, id="named"),
    ],
)
def test_tools_suppress_dsml_for_every_tool_choice(tool_choice, stream):
    reasoning, content, _ = _parse(
        _V41Parser,
        _V41_INVOKE,
        tools=True,
        tool_choice=tool_choice,
        stream=stream,
    )

    assert "DSML" not in (reasoning or "")
    assert "DSML" not in (content or "")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "output",
    [
        f"</think><｜DSML｜tool_calls>{_V4_INVOKE}</｜DSML｜tool_calls>",
        "</think><｜DSML｜ toolc",
        "</think>answer before <｜DSML｜unknown> answer after",
    ],
)
def test_unknown_dsml_is_suppressed_when_tools_are_present(output, stream):
    reasoning, content, tool_calls = _parse(
        _V41Parser,
        output,
        tools=True,
        stream=stream,
    )

    assert "DSML" not in (reasoning or "")
    assert "DSML" not in (content or "")
    assert not tool_calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "output",
    [
        f"</think><｜DSML｜ calls>{_V41_INVOKE}</｜DSML｜ calls>",
        f"</think><｜DSML｜ toolcalls>{_V41_INVOKE}</｜DSML｜ calls>",
        f"</think><｜DSML｜tool_calls>{_V4_INVOKE}</｜DSML｜tool_calls>",
        _V41_INVOKE,
        "</think><｜DSML｜ toolc",
    ],
)
def test_no_tools_preserves_dsml_content(output, stream):
    reasoning, content, tool_calls = _parse(
        _V41Parser,
        output,
        tools=False,
        stream=stream,
    )

    assert reasoning is None
    assert "DSML" in content
    assert not tool_calls


@pytest.mark.parametrize("stream", [False, True])
def test_tool_choice_none_suppresses_calls_without_leaking(stream):
    output = f"</think><｜DSML｜ toolcalls>{_V41_INVOKE}</｜DSML｜ calls>"

    reasoning, content, tool_calls = _parse(
        _V41Parser,
        output,
        tools=True,
        tool_choice="none",
        stream=stream,
    )

    assert reasoning is None
    assert content is None
    assert not tool_calls
