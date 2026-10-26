# SPDX-License-Identifier: Apache-2.0
"""Restore MTP-layer attention config for ``Step3p5DecoderLayer``.

Corresponds to: vllm/model_executor/models/step3p5.py ::
``Step3p5DecoderLayer.__init__``

Community code indexes ``layer_types[layer_idx]`` without bounds checking or
an MTP fallback when ``layer_idx >= num_hidden_layers``. MTP draft blocks
(e.g. layers 45-47 on Step-3.7) therefore miss ``attention_other_setting``
head overrides and pick the wrong SWA pattern via ``layer_idx % 2``.

This wrapper temporarily extends ``hf_config.layer_types`` (and optionally
``partial_rotary_factors``) for MTP indices before delegating to the community
``__init__``, then restores the original lists.

Baseline : v0.31.0 (341bf7d4d1)
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index


def _extend_mtp_layer_config(config: Any, layer_idx: int) -> tuple[Any, Any]:
    """Extend layer_types / partial_rotary_factors for an MTP layer index."""
    attention_other = getattr(config, "attention_other_setting", None)
    orig_layer_types = getattr(config, "layer_types", None)
    orig_partial = getattr(config, "partial_rotary_factors", None)

    if layer_idx < config.num_hidden_layers or attention_other is None:
        return orig_layer_types, orig_partial

    attn_type = attention_other["attention_type"]
    layer_types = list(orig_layer_types) if orig_layer_types else []
    needed_len = layer_idx + 1
    if len(layer_types) < needed_len:
        layer_types.extend([attn_type] * (needed_len - len(layer_types)))
        config.layer_types = layer_types

    partial = list(orig_partial) if orig_partial else None
    if partial is not None and len(partial) < needed_len:
        fill = partial[-1] if partial else 1.0
        partial.extend([fill] * (needed_len - len(partial)))
        config.partial_rotary_factors = partial

    return orig_layer_types, orig_partial


def _restore_mtp_layer_config(
    config: Any,
    orig_layer_types: Any,
    orig_partial: Any,
) -> None:
    if orig_layer_types is None:
        if hasattr(config, "layer_types"):
            delattr(config, "layer_types")
    else:
        config.layer_types = orig_layer_types

    if orig_partial is None:
        if hasattr(config, "partial_rotary_factors"):
            delattr(config, "partial_rotary_factors")
    else:
        config.partial_rotary_factors = orig_partial


def make_decoder_layer_init(
    orig: Callable[..., None],
) -> Callable[..., None]:
    """Wrap ``Step3p5DecoderLayer.__init__`` with MTP config semantics."""

    @functools.wraps(orig)
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        config = vllm_config.model_config.hf_config
        layer_idx = extract_layer_index(prefix)
        orig_lt, orig_pr = _extend_mtp_layer_config(config, layer_idx)
        try:
            orig(self, vllm_config, prefix)
        finally:
            _restore_mtp_layer_config(config, orig_lt, orig_pr)

    return __init__