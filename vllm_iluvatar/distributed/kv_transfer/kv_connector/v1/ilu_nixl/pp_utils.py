# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP split helpers for Ilu NIXL metadata."""

from __future__ import annotations

from vllm.distributed.utils import get_pp_indices


def get_num_hidden_layers(vllm_config) -> int:
    """Return the model layer count used by vLLM's PP partitioning."""
    model_config = vllm_config.model_config
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if hf_text_config is not None and hasattr(hf_text_config, "num_hidden_layers"):
        return hf_text_config.num_hidden_layers
    return model_config.hf_config.num_hidden_layers


def infer_pp_split(vllm_config) -> str:
    """Return comma-separated layer counts for every PP stage.

    This mirrors vLLM's default partitioning when VLLM_PP_LAYER_PARTITION is not
    set, and respects the env override when it is set because get_pp_indices()
    uses the same source of truth as make_layers().
    """
    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    num_hidden_layers = get_num_hidden_layers(vllm_config)
    parts: list[str] = []
    for pp_rank in range(pp_size):
        start, end = get_pp_indices(num_hidden_layers, pp_rank, pp_size)
        parts.append(str(end - start))
    return ",".join(parts)
