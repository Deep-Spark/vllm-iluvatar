# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI streaming error patch."""

from __future__ import annotations

import asyncio
import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.v1.engine.async_llm import AsyncLLM
from vllm_iluvatar.ilu_patch.general_patches import patch_openai_stream_errors

_MODEL = "test-model"
_DONE = "data: [DONE]\n\n"
_ERROR = "Internal server error"


@dataclass
class _HFConfig:
    model_type: str = "any"


@dataclass
class _ModelConfig:
    task = "generate"
    runner_type = "generate"
    model = _MODEL
    tokenizer = _MODEL
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = _HFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@pytest.fixture(scope="module", autouse=True)
def _apply_patch():
    patch_openai_stream_errors._apply()


def _serving_args() -> tuple[MagicMock, OpenAIServingModels, OnlineRenderer]:
    engine = MagicMock(spec=AsyncLLM)
    engine.errored = False
    engine.model_config = _ModelConfig()
    engine.input_processor = MagicMock()
    engine.io_processor = MagicMock()
    engine.renderer = MagicMock()
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[BaseModelPath(name=_MODEL, model_path=_MODEL)],
    )
    online_renderer = OnlineRenderer(
        model_config=engine.model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    return engine, models, online_renderer


def _output(token_ids: list[int], finish_reason: str | None) -> RequestOutput:
    return RequestOutput(
        request_id="test-id",
        prompt="Test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="Hello" if token_ids else "",
                token_ids=token_ids,
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=finish_reason,
            )
        ],
        finished=finish_reason is not None,
    )


async def _yield_all(items):
    for item in items:
        yield item


def _collect(stream) -> list[str]:
    async def run():
        return [chunk async for chunk in stream]

    return asyncio.run(run())


def _stream_chat(*outputs: RequestOutput) -> list[str]:
    engine, models, online_renderer = _serving_args()
    serving = OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        online_renderer=online_renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    request = ChatCompletionRequest(
        model=_MODEL,
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=True,
        stream_options={"include_usage": True},
    )
    return _collect(
        serving.chat_completion_stream_generator(
            request,
            _yield_all(outputs),
            "test-id",
            _MODEL,
            [],
            None,
            RequestResponseMetadata(request_id="test-id"),
        )
    )


def _stream_completion(*outputs: RequestOutput) -> list[str]:
    engine, models, online_renderer = _serving_args()
    serving = OpenAIServingCompletion(
        engine, models, online_renderer=online_renderer, request_logger=None
    )
    request = CompletionRequest(
        model=_MODEL,
        prompt="Test prompt",
        max_tokens=10,
        stream=True,
        stream_options={"include_usage": True},
    )
    return _collect(
        serving.completion_stream_generator(
            request,
            [{"prompt_token_ids": [1, 2, 3]}],
            _yield_all((0, output) for output in outputs),
            "test-id",
            0,
            _MODEL,
            1,
            None,
            RequestResponseMetadata(request_id="test-id"),
        )
    )


_STREAMS = pytest.mark.parametrize(
    "stream", [_stream_chat, _stream_completion], ids=["chat", "completion"]
)


@_STREAMS
def test_error_before_first_token_is_reported(stream):
    chunks = stream(_output([], "error"))
    assert any(_ERROR in chunk for chunk in chunks), chunks
    assert chunks[-1] == _DONE


@_STREAMS
def test_error_after_tokens_is_reported(stream):
    chunks = stream(_output([100], None), _output([101], "error"))
    assert any("Hello" in chunk for chunk in chunks), chunks
    assert any(_ERROR in chunk for chunk in chunks), chunks
    assert chunks[-1] == _DONE


@_STREAMS
def test_successful_stream_has_no_error(stream):
    chunks = stream(_output([100], "stop"))
    assert any("Hello" in chunk for chunk in chunks), chunks
    assert not any(_ERROR in chunk for chunk in chunks), chunks
    assert chunks[-1] == _DONE


def test_apply_is_idempotent():
    chat = OpenAIServingChat.chat_completion_stream_generator
    completion = OpenAIServingCompletion.completion_stream_generator
    patch_openai_stream_errors._apply()
    assert OpenAIServingChat.chat_completion_stream_generator is chat
    assert OpenAIServingCompletion.completion_stream_generator is completion


def test_signature_drift_fails_fast():
    async def drifted(self, request, results, rid):
        yield ""

    with pytest.raises(RuntimeError, match="re-sync"):
        patch_openai_stream_errors.make_stream_generator(drifted)


def test_missing_raise_if_error_fails_fast():
    class Serving:
        async def chat_completion_stream_generator(
            self, request, result_generator, request_id
        ):
            yield ""

    module = types.SimpleNamespace(OpenAIServingChat=Serving)
    with pytest.raises(RuntimeError, match="re-sync"):
        patch_openai_stream_errors._patch_class(
            module, "OpenAIServingChat", "chat_completion_stream_generator"
        )
