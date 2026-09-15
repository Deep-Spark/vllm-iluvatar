# SPDX-License-Identifier: Apache-2.0
"""KV cache scale cleanup for ILU int8 quantization.

Patches the V2 KV-cache teardown path so the ILU K-scale storage (which
shares the community KV backing buffer) is released alongside the KV cache:

* ``clear_layer_kv_caches`` (vllm.v1.worker.utils) -- also detach
  ``layer.kv_cache_scale`` installed by ``bind_kv_cache_scale``. This is the
  layer-level fix and covers EVERY caller with one patch: the two
  ``clear_layer_kv_caches(...)`` calls inside ``GPUModelRunner.shutdown`` plus
  the one in ``cudagraph_utils``. The community body is not copied (no
  fingerprint to maintain): the iterable is materialised, the original is
  called, then a second pass detaches ``kv_cache_scale``.

The runner-level ``self.kv_scale_caches`` list does not exist in V2 yet (the
ILU scale wiring into the real V2 path is pending); its clear will be added
in that wiring patch once ``self.kv_scale_caches`` is created, not here.

Target   : vllm.v1.worker.utils.clear_layer_kv_caches
Scope    : only active when VLLM_ILU_FLASH_ATTN_QUANT_TYPE != "" or
           VLLM_ILU_MLA_ATTN_QUANT_TYPE != ""
Remove   : when upstream supports int8 kv cache quantization.
"""

from __future__ import annotations

from functools import wraps

import torch

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="wrap clear_layer_kv_caches for kv scale cleanup",
    )


def _apply() -> None:
    import vllm.envs as envs

    if (
        envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == ""
        and envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == ""
    ):
        return

    from vllm.v1.worker import utils as worker_utils

    # --- clear_layer_kv_caches: also detach layer.kv_cache_scale -------------
    # Covers every call site (shutdown x2, cudagraph_utils x1) with one patch.
    # The community body is NOT copied -- materialise the iterable, call the
    # original (clears kv_cache + impl._k_scale_cache/_v_scale_cache), then do
    # a second pass for the ILU ``layer.kv_cache_scale`` installed by
    # ``bind_kv_cache_scale``. No fingerprint needed since no body is duplicated.
    _orig_clear = worker_utils.clear_layer_kv_caches

    @wraps(_orig_clear)
    def clear_layer_kv_caches(layers):
        layers = list(layers)
        _orig_clear(layers)
        for layer in layers:
            if hasattr(layer, "kv_cache_scale"):
                kv_scale = layer.kv_cache_scale
                layer.kv_cache_scale = (
                    torch.tensor([]) if isinstance(kv_scale, torch.Tensor) else []
                )

    p.rebind_module_func(worker_utils, "clear_layer_kv_caches", clear_layer_kv_caches)
