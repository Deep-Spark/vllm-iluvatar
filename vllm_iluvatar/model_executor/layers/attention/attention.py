# SPDX-License-Identifier: Apache-2.0
"""CoreX compatibility wrappers for :mod:`vllm` attention layers.

Corresponds to: vllm/model_executor/layers/attention/attention.py
Baseline     : main_61ac3680210
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from functools import wraps
from typing import Any


def _ixformer_torch_ops_enabled() -> bool:
    """Read the same torch-op switch used by the legacy attention path."""
    if os.environ.get("IXFORMER_USE_TORCH_OPS") == "1":
        return True
    try:
        ixf_config = importlib.import_module("ixformer.core.config")
        return bool(ixf_config.IXFORMER_USE_TORCH_OPS)
    except Exception:
        return False


def make_init(original_init: Callable[..., Any]) -> Callable[..., Any]:
    """Restore the opaque custom-op boundary for ixformer torch ops."""

    @wraps(original_init)
    def _ilu_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if _ixformer_torch_ops_enabled():
            # [ILU override] Attention.__init__ - delta: do not trace the
            # mutable ixformer attention path inside a piecewise FX graph.
            # The registered unified_attention_with_output custom op is the
            # required graph boundary, matching the pre-migration behavior.
            self.use_direct_call = False

    return _ilu_init


def make_forward(original_forward: Callable[..., Any]) -> Callable[..., Any]:
    """Avoid the CoreX Dynamo-incompatible ``torch.Size`` fallback.

    Community ``Attention.forward`` constructs ``torch.Size`` only when the
    caller omits ``output_shape``. A regular tuple is accepted by
    ``torch.empty`` and preserves the same output allocation semantics.
    """

    @wraps(original_forward)
    def _ilu_forward(
        self: Any,
        query: Any,
        key: Any,
        value: Any,
        output_shape: tuple[int, int] | None = None,
        output_dtype: Any = None,
    ) -> Any:
        if output_shape is None:
            # [ILU override] Attention.forward - delta: pass a tuple so CoreX
            # TorchDynamo never traces ``torch.Size(...)`` in community code.
            output_shape = (
                query.shape[0],
                self.num_heads * self.head_size_v,
            )
        return original_forward(
            self,
            query,
            key,
            value,
            output_shape=output_shape,
            output_dtype=output_dtype,
        )

    return _ilu_forward


def make_process_weights_after_loading(
    original_process_weights_after_loading: Callable[..., Any],
) -> Callable[..., Any]:
    """Let the Iluvatar flash-attention impl bind external FP8 scales."""

    @wraps(original_process_weights_after_loading)
    def _ilu_process_weights_after_loading(self: Any, act_dtype: Any) -> None:
        scale_path = os.environ.get("VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH", "")
        if (
            scale_path
            and os.path.isfile(scale_path)
            and getattr(self.impl, "supports_ilu_external_fp8_scales", False)
        ):
            handled = self.impl.process_weights_after_loading(
                act_dtype,
                layer=self,
            )
            if handled:
                # The community method would reset scales to 1.0 when the
                # checkpoint has no official KV-cache quant method.
                return

        original_process_weights_after_loading(self, act_dtype)

    return _ilu_process_weights_after_loading
