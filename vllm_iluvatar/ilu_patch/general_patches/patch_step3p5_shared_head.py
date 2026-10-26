# SPDX-License-Identifier: Apache-2.0
"""Keep Step3.5 MTP per-layer shared heads.

Corresponds to: vllm/v1/worker/gpu/spec_decode/eagle/utils.py
Target   : load_eagle_model
Baseline : v0.31.0 (341bf7d4d1)
Why      : V2 replaces every ``layer.shared_head.head`` with the target
           ``lm_head`` when the draft has no top-level ``lm_head``. Step3.5
           checkpoints store a distinct head on each MTP layer. Sharing the
           target head changes the draft argmax.
How      : wrap ``load_eagle_model``. When ``get_model`` returns a draft
           with ``has_own_shared_head``, remember each layer head, let the
           community loader run, then put those heads back if it replaced
           them. ``rebind_module_func`` installs the wrapper so a prior
           ``from eagle.utils import load_eagle_model`` sees it.
Scope    : draft models that set ``has_own_shared_head``. Eagle, DeepSeek,
           and DSpark do not, so their heads are still shared.
Remove   : when upstream stops replacing per-layer shared heads, or grows
           the same flag.
"""

from __future__ import annotations

from typing import Any

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="keep Step3.5 per-layer shared_head during eagle load",
    )


def _checkpoint_heads(model: object) -> list[tuple[Any, Any]]:
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        return []
    values = getattr(layers, "values", None)
    items = values() if callable(values) else layers
    saved: list[tuple[Any, Any]] = []
    for layer in items:
        shared_head = getattr(layer, "shared_head", None)
        if shared_head is None or not hasattr(shared_head, "head"):
            continue
        saved.append((shared_head, shared_head.head))
    return saved


def _restore_heads(saved: list[tuple[Any, Any]]) -> None:
    for shared_head, head in saved:
        if getattr(shared_head, "head", None) is head:
            continue
        if hasattr(shared_head, "head"):
            del shared_head.head
        shared_head.head = head


def _wrap_load_eagle_model(original: Any, eagle_utils: Any) -> Any:
    def load_eagle_model(target_model: Any, vllm_config: Any) -> Any:
        saved: list[tuple[Any, Any]] = []
        real_get_model = eagle_utils.get_model

        def get_model(*args: Any, **kwargs: Any) -> Any:
            model = real_get_model(*args, **kwargs)
            if getattr(model, "has_own_shared_head", False):
                saved.extend(_checkpoint_heads(model))
            return model

        # load_eagle_model resolves get_model from this module's globals.
        eagle_utils.get_model = get_model
        try:
            eagle = original(target_model, vllm_config)
        finally:
            eagle_utils.get_model = real_get_model
        _restore_heads(saved)
        return eagle

    return load_eagle_model


def _apply() -> None:
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

    # [ILU override] load_eagle_model — delta: restore checkpoint
    # shared_head.head when the draft sets has_own_shared_head.
    p.rebind_module_func(
        eagle_utils,
        "load_eagle_model",
        _wrap_load_eagle_model(eagle_utils.load_eagle_model, eagle_utils),
    )