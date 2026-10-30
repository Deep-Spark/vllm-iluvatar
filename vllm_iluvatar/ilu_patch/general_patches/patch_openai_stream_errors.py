# SPDX-License-Identifier: Apache-2.0
"""Report streaming errors for requests that fail before their first token.

Corresponds to: vllm/entrypoints/openai/chat_completion/serving.py,
                vllm/entrypoints/openai/completion/serving.py
Target   : OpenAIServingChat.chat_completion_stream_generator,
           OpenAIServingCompletion.completion_stream_generator
Baseline : v0.30.1rc0 (6919d781a6)
Why      : both generators skip an output that has no text, no token ids and
           no earlier tokens (the chunked-prefill case) before they reach
           ``_raise_if_error``. A request that finishes with
           ``finish_reason='error'`` before its first token, such as a PD
           decode whose remote KV lease expired, therefore ends as an HTTP 200
           stream with neither content nor an error.
How      : wrap ``result_generator`` so ``_raise_if_error`` sees every output
           before the community loop does; the generators' own
           ``except GenerationError`` branch then emits the error event and
           ``[DONE]``. Installed through a post-import hook, so only processes
           that import the OpenAI serving modules are touched.
Scope    : streaming chat completions and completions.
Remove   : when community vLLM checks the error finish reason before the
           empty-chunk skip.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import AsyncIterator, Callable
from functools import partial, wraps
from typing import Any

from vllm_iluvatar.ilu_patch import general_patch as p

_PATCH_MARKER = "_ilu_stream_error_before_first_token"

_TARGETS = (
    (
        "vllm.entrypoints.openai.chat_completion.serving",
        "OpenAIServingChat",
        "chat_completion_stream_generator",
    ),
    (
        "vllm.entrypoints.openai.completion.serving",
        "OpenAIServingCompletion",
        "completion_stream_generator",
    ),
)


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="report finish_reason='error' before the first streamed token",
    )


async def raise_on_error_outputs(
    serving: Any,
    result_generator: AsyncIterator[Any],
    request_id: str,
) -> AsyncIterator[Any]:
    """Pass engine outputs through, raising before any errored one."""
    try:
        async for item in result_generator:
            # Completions yield (prompt_idx, RequestOutput) pairs.
            res = item[1] if isinstance(item, tuple) else item
            for output in res.outputs:
                serving._raise_if_error(output.finish_reason, request_id)
            yield item
    finally:
        aclose = getattr(result_generator, "aclose", None)
        if aclose is not None:
            await aclose()


def make_stream_generator(original: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(original)
    missing = {"result_generator", "request_id"}.difference(signature.parameters)
    if missing:
        raise RuntimeError(
            f"{original.__qualname__} no longer takes {sorted(missing)}; "
            "re-sync patch_openai_stream_errors.py"
        )

    @wraps(original)
    def _ilu_stream_generator(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.arguments["result_generator"] = raise_on_error_outputs(
            self,
            bound.arguments["result_generator"],
            bound.arguments["request_id"],
        )
        return original(*bound.args, **bound.kwargs)

    setattr(_ilu_stream_generator, _PATCH_MARKER, True)
    return _ilu_stream_generator


def _patch_class(module: Any, cls_name: str, method: str) -> None:
    cls = getattr(module, cls_name)
    original = getattr(cls, method)
    if getattr(original, _PATCH_MARKER, False):
        return
    if not callable(getattr(cls, "_raise_if_error", None)):
        raise RuntimeError(
            f"{cls_name}._raise_if_error is gone; "
            "re-sync patch_openai_stream_errors.py"
        )
    setattr(cls, method, make_stream_generator(original))


def _apply() -> None:
    from vllm_iluvatar.ilu_patch import import_hook

    for module_name, cls_name, method in _TARGETS:
        callback = partial(_patch_class, cls_name=cls_name, method=method)
        module = sys.modules.get(module_name)
        if module is None:
            import_hook.register(module_name, callback)
        else:
            callback(module)
