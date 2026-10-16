# SPDX-License-Identifier: Apache-2.0
"""Preserve DeepSeek-V4/V4.1 DSML markup for legacy requests without tools."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from vllm_iluvatar.ilu_patch import general_patch as p

_PATCH_MARKER = "_ilu_preserve_dsml_without_tools_patched"


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
    # V4.1 adapters come from a separate ``make_adapters`` call, so they do not
    # inherit the V4 wrappers even though both share the DSML grammar.
    from vllm.parser.engine.registered_adapters import (
        DeepSeekV4ParserToolAdapter,
        DeepSeekV41ParserToolAdapter,
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

