# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar MLA attention compatibility layer.

Corresponds to: vllm/model_executor/layers/attention/mla_attention.py

Target   : ``MLAAttention`` instances created by the plugin MLA wrapper.
Baseline : main_6e448d0
Why      : Iluvatar MLA backends apply rope/cache updates inside the backend
           ``forward_full`` path and need speculative-decode decode rows expanded
           to token granularity. Keep those deltas out of community vLLM.
How      : instance-level method binding for ``MLAAttention``. Community
           behavior is used when the ILU full-forward/projection hooks are
           absent.
Scope    : Iluvatar ``FLASHMLA`` / sparse MLA plugin backends only.
Remove   : when upstream exposes backend-owned rope/cache-update hooks and
           token-level speculative MLA decode metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MethodType
from typing import TYPE_CHECKING

import torch
import vllm.envs as envs
import vllm.model_executor.layers.attention.mla_attention as _mla

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention


_COMMUNITY_FORWARD = _mla.MLAAttention.forward
_COMMUNITY_PROCESS_WEIGHTS_AFTER_LOADING = (
    _mla.MLAAttention.process_weights_after_loading
)


def wrap_determine_chunked_prefill_workspace_size(
    community_func: Callable[["VllmConfig"], int],
) -> Callable[["VllmConfig"], int]:
    """Cap workspace rows at ``VLLM_ILU_MLA_CHUNKED_PREFILL_MAX_WS_SIZE``."""

    def determine_chunked_prefill_workspace_size(vllm_config: "VllmConfig") -> int:
        original_size = community_func(vllm_config)
        max_ws = envs.VLLM_ILU_MLA_CHUNKED_PREFILL_MAX_WS_SIZE
        if max_ws > 0:
            return min(original_size, max_ws)
        return original_size

    return determine_chunked_prefill_workspace_size



def install_ilu_mla_attention_overrides(
    layer: MLAAttention,
    rotary_emb: torch.nn.Module | None,
) -> None:
    """Install Iluvatar-only MLA hooks on a single community MLAAttention."""
    layer.rotary_emb = rotary_emb
    layer.use_int8_mla = envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == "INT8"
    if layer.use_int8_mla and not hasattr(layer, "kv_cache_scale"):
        layer.kv_cache_scale = torch.tensor([])
    layer.forward = MethodType(_ilu_mla_attention_forward, layer)
    layer.process_weights_after_loading = MethodType(
        _ilu_process_weights_after_loading,
        layer,
    )


def _ilu_mla_attention_forward(
    self: MLAAttention,
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    position: torch.Tensor | None = None,
    output_shape: torch.Size | None = None,
    q_dcp_replicated: torch.Tensor | None = None,
) -> torch.Tensor:
    forward_full = getattr(self.impl, "forward_full", None)
    if not callable(forward_full):
        return _COMMUNITY_FORWARD(
            self,
            q,
            kv_c_normed,
            k_pe,
            output_shape=output_shape,
            q_dcp_replicated=q_dcp_replicated,
        )

    assert position is not None and getattr(self, "rotary_emb", None) is not None, (
        "Iluvatar MLA forward_full requires positions and rotary_emb; "
        "falling back to community MLAAttention.forward is not supported for "
        f"{self.impl.__class__.__name__}."
    )
    assert q_dcp_replicated is None, (
        "q_dcp_replicated is not supported by the Iluvatar MLA full-forward path"
    )
    attn_metadata = _get_mla_attn_metadata(self.layer_name)
    output = torch.zeros(
        output_shape if output_shape is not None else q.shape,
        dtype=q.dtype,
        device=q.device,
    )
    kv_cache_scale = (
        self.kv_cache_scale if getattr(self, "use_int8_mla", False) else None
    )
    return forward_full(
        self,
        q,
        kv_c_normed,
        k_pe,
        self.kv_cache,
        attn_metadata,
        output,
        position,
        kv_cache_scale,
    )


def _get_mla_attn_metadata(layer_name: str):
    forward_context: _mla.ForwardContext = _mla.get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    if isinstance(attn_metadata_raw, dict):
        return attn_metadata_raw[layer_name]
    if isinstance(attn_metadata_raw, list):
        # list[dict[str, AttentionMetadata]]: used in speculative decoding where
        # [0] is the base-model metadata dict.
        return attn_metadata_raw[0][layer_name]
    return attn_metadata_raw


def _ilu_process_weights_after_loading(
    self: MLAAttention,
    act_dtype: torch.dtype,
) -> None:
    setup_projections = getattr(self.impl, "setup_projections", None)
    if callable(setup_projections) and setup_projections(self, act_dtype) is True:
        return

    _COMMUNITY_PROCESS_WEIGHTS_AFTER_LOADING(self, act_dtype)
    if getattr(self, "rotary_emb", None) is not None:
        if hasattr(self, "W_UV"):
            self.impl.W_UV = self.W_UV
        if hasattr(self, "W_UK_T"):
            self.impl.W_UK_T = self.W_UK_T
