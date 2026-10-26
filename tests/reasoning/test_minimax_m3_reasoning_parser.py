# SPDX-License-Identifier: Apache-2.0
"""Prompt markers belong to the current assistant turn only."""

import pytest
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser

from vllm_iluvatar.ilu_register.reasoning import register_reasoning_parsers
from vllm_iluvatar.reasoning.minimax_m3_reasoning_parser import (
    IluMiniMaxM3ReasoningParser,
)


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("system <mm:think> example </mm:think>]~b]ai\n", False),
        ("]~b]ai\n<mm:think>old</mm:think>]~b]user\nnext]~b]ai\n", False),
        ("]~b]ai\n</mm:think>", True),
        ("]~b]ai\n<mm:think>reasoning", False),
        ("]~b]ai\n<mm:think>reasoning</mm:think>answer", True),
        ("]~b]ai\n</mm:think><mm:think>new", False),
    ],
)
def test_current_turn(prompt, expected):
    parser = IluMiniMaxM3ReasoningParser.__new__(IluMiniMaxM3ReasoningParser)
    parser._decode_text = lambda ids: prompt
    assert parser.is_reasoning_end([1, 2]) is expected


def test_non_chat_input_delegates(monkeypatch):
    parser = IluMiniMaxM3ReasoningParser.__new__(IluMiniMaxM3ReasoningParser)
    parser._decode_text = lambda ids: "plain completion"
    calls = []

    def original(self, ids):
        calls.append(ids)
        return True

    monkeypatch.setattr(MiniMaxM3ReasoningParser, "is_reasoning_end", original)
    assert parser.is_reasoning_end([3]) is True
    assert calls == [[3]]


def test_registry_replaces_only_minimax(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        ReasoningParserManager, "reasoning_parsers", {"other": sentinel}
    )
    register_reasoning_parsers()
    register_reasoning_parsers()
    assert ReasoningParserManager.reasoning_parsers == {
        "other": sentinel,
        "minimax_m3": IluMiniMaxM3ReasoningParser,
    }
