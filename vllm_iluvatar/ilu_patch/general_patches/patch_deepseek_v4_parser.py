# SPDX-License-Identifier: Apache-2.0
"""Harden DeepSeek-V4/V4.1 DSML parsing while preserving no-tools output."""

from __future__ import annotations

from dataclasses import replace
from functools import cache, wraps
from typing import Any, Callable

from vllm_iluvatar.ilu_patch import general_patch as p

_PATCH_MARKER = "_ilu_preserve_dsml_without_tools_patched"
_CONFIG_PATCH_MARKER = "_ilu_dsml_leak_guard_patched"
_DSML_PREFIX = "<｜DSML｜"
_V41_TOOL_START_VARIANTS = (
    "<｜DSML｜ toolcalls>",
    "<｜DSML｜ tool_calls>",
    "<｜DSML｜ tool>",
    "<｜DSML｜  calls>",
)


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="preserve DeepSeek-V4/V4.1 DSML content when tools are omitted",
    )


def _configure_passthrough(adapter: Any, request: Any) -> None:
    adapter._parser_engine.skip_tool_parsing = not bool(
        getattr(request, "tools", None)
    )


def _patch_parser_config(
    parser_cls: type,
    *,
    tool_start_variants: tuple[str, ...] = (),
) -> None:
    if parser_cls.__dict__.get(_CONFIG_PATCH_MARKER, False):
        return

    from vllm.parser.engine.events import EventType
    from vllm.parser.engine.parser_engine_config import ParserState, Transition

    original = parser_cls.parser_config

    @cache
    def patched(thinking: bool = False):
        config = original(thinking=thinking)
        terminals = dict(config.terminals)
        tool_start = terminals["TOOL_START"]
        spellings = tool_start if isinstance(tool_start, tuple) else (tool_start,)
        terminals["TOOL_START"] = tuple(
            dict.fromkeys((*spellings, *tool_start_variants))
        )
        # A generic DSML prefix is a fail-closed fallback for malformed or
        # cross-version markup. Known longer terminals still win lexer matching.
        terminals["DSML_PREFIX"] = _DSML_PREFIX

        transitions = dict(config.transitions)
        transitions[(ParserState.REASONING, "INVOKE_PREFIX")] = Transition(
            ParserState.TOOL_NAME,
            (EventType.REASONING_END, EventType.TOOL_CALL_START),
        )
        transitions[(ParserState.REASONING, "DSML_PREFIX")] = Transition(
            ParserState.TOOL_PREAMBLE,
            (EventType.REASONING_END,),
        )
        transitions[(ParserState.CONTENT, "DSML_PREFIX")] = Transition(
            ParserState.TOOL_PREAMBLE,
        )
        transitions[(ParserState.TOOL_PREAMBLE, "DSML_PREFIX")] = Transition(
            ParserState.TOOL_PREAMBLE,
        )
        transitions[(ParserState.TOOL_BETWEEN, "DSML_PREFIX")] = Transition(
            ParserState.TOOL_BETWEEN,
        )
        return replace(config, terminals=terminals, transitions=transitions)

    parser_cls.parser_config = staticmethod(patched)
    setattr(parser_cls, _CONFIG_PATCH_MARKER, True)


def _wrap_adjust_request(original: Callable) -> Callable:
    @wraps(original)
    def wrapped(adapter: Any, request: Any):
        _configure_passthrough(adapter, request)
        return original(adapter, request)

    return wrapped


def _wrap_extract_tool_calls(original: Callable) -> Callable:
    @wraps(original)
    def wrapped(adapter: Any, model_output: str, request: Any):
        _configure_passthrough(adapter, request)
        return original(adapter, model_output, request)

    return wrapped


def _wrap_extract_tool_calls_streaming(original: Callable) -> Callable:
    @wraps(original)
    def wrapped(
        adapter: Any,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request: Any,
    ):
        _configure_passthrough(adapter, request)
        return original(
            adapter,
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )

    return wrapped


def _apply() -> None:
    from vllm.parser.deepseek_v4 import DeepSeekV4Parser
    from vllm.parser.deepseek_v41 import DeepSeekV41Parser
    # V4.1 adapters come from a separate ``make_adapters`` call, so they do not
    # inherit the V4 wrappers even though both share the DSML grammar.
    from vllm.parser.engine.registered_adapters import (
        DeepSeekV4ParserToolAdapter,
        DeepSeekV41ParserToolAdapter,
    )

    _patch_parser_config(DeepSeekV4Parser)
    _patch_parser_config(
        DeepSeekV41Parser,
        tool_start_variants=_V41_TOOL_START_VARIANTS,
    )

    for adapter_cls in (DeepSeekV4ParserToolAdapter, DeepSeekV41ParserToolAdapter):
        if getattr(adapter_cls, _PATCH_MARKER, False):
            continue

        adapter_cls.adjust_request = _wrap_adjust_request(
            adapter_cls.adjust_request
        )
        adapter_cls.extract_tool_calls = _wrap_extract_tool_calls(
            adapter_cls.extract_tool_calls
        )
        adapter_cls.extract_tool_calls_streaming = (
            _wrap_extract_tool_calls_streaming(
                adapter_cls.extract_tool_calls_streaming
            )
        )
        setattr(adapter_cls, _PATCH_MARKER, True)

