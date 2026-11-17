# SPDX-License-Identifier: Apache-2.0
"""Register the delegating checkpoint loader through vLLM's public API."""

from vllm.model_executor.model_loader import register_model_loader

from vllm_iluvatar.model_executor.model_loader.ilu_default_loader import (
    IluDefaultModelLoader,
)


def register_model_loaders():
    for load_format in ("auto", "safetensors", "pt", "npcache"):
        register_model_loader(load_format)(IluDefaultModelLoader)
