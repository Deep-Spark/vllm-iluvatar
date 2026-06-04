"""Patch ``vllm._custom_ops`` with the iluvatar implementations.

``vllm._custom_ops`` is imported very early (platform init) and is referenced
by many modules as ``from vllm import _custom_ops as ops`` -- holders keep the
*same module object*, so we cannot use the override finder. Instead we mutate
that live module in place.

Two layers:

1. **Full splat (preferred).** Load the internal fork's ``_custom_ops.py``
   (shipped as :mod:`vllm_iluvatar._ilu_custom_ops`) and copy every function
   it *defines* onto the live ``vllm._custom_ops``. This brings the full set
   of ilu ops (``rms_norm_qk``, ``poly_norm``, ``awq_gemm``,
   ``custom_gptq_marlin_gemm``, ``quant_kv``, ``reshape_and_cache_flash_mix``,
   ``router_gemm_bf16_fp32``, ``silu_and_mul_with_clamp``,
   ``topk_hash_softplus_sqrt``, ``dsa_convert_req_index_to_global_index``, ...).
   It requires ``ixformer`` (imported at the top of that file), so it only
   succeeds on a real iluvatar runtime.

2. **Fallback.** If the full splat fails (e.g. CPU box without ixformer), fall
   back to the four hand-written ops in :mod:`vllm_iluvatar.ops` so the import
   path and the smoke test still work.
"""

from __future__ import annotations

import types

_PATCHED = False


def _full_splat(_custom_ops, logger) -> bool:
    """Load the internal _custom_ops and splat its functions. Returns success."""
    import importlib.util
    import os
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "_ilu_custom_ops.py"
    if not src.is_file():
        return False

    spec = importlib.util.spec_from_file_location(
        "vllm_iluvatar._ilu_custom_ops", os.fspath(src)
    )
    if spec is None or spec.loader is None:
        return False
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # may raise if ixformer missing

    count = 0
    for name, val in vars(module).items():
        if name.startswith("_"):
            continue
        # Only copy functions DEFINED in the internal _custom_ops (skip its
        # imported names like torch / ixf_ops), to avoid clobbering unrelated
        # attributes on the live module.
        if isinstance(val, types.FunctionType) and val.__module__ == module.__name__:
            setattr(_custom_ops, name, val)
            count += 1

    logger.info("vllm-iluvatar: splatted %d functions from internal _custom_ops", count)
    return count > 0


def _fallback(_custom_ops, logger) -> None:
    """Minimal four-op patch (works without ixformer at import time)."""
    from vllm_iluvatar import ops as ilu_ops

    _custom_ops.silu_and_mul_with_clamp = ilu_ops.silu_and_mul_with_clamp
    _custom_ops.router_gemm_bf16_fp32 = ilu_ops.router_gemm_bf16_fp32
    _custom_ops.topk_hash_softplus_sqrt = ilu_ops.topk_hash_softplus_sqrt
    _custom_ops.dsa_convert_req_index_to_global_index = (
        ilu_ops.dsa_convert_req_index_to_global_index
    )
    logger.info(
        "vllm-iluvatar: patched vllm._custom_ops (fallback: 4 ops "
        "silu_and_mul_with_clamp, router_gemm_bf16_fp32, "
        "topk_hash_softplus_sqrt, dsa_convert_req_index_to_global_index)"
    )


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    import vllm._custom_ops as _custom_ops
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    ok = False
    try:
        ok = _full_splat(_custom_ops, logger)
    except Exception:
        logger.warning(
            "vllm-iluvatar: full _custom_ops splat failed (ixformer missing?); "
            "falling back to 4-op patch",
            exc_info=True,
        )

    if not ok:
        try:
            _fallback(_custom_ops, logger)
        except Exception:
            logger.exception("vllm-iluvatar: _custom_ops fallback patch failed")

    # Register the meta/fake impl for router_gemm only if the matching op
    # exists (mirrors the upstream guarded register_fake block).
    try:
        import torch
        from torch.library import register_fake

        if hasattr(torch.ops, "_moe_C") and hasattr(
            torch.ops._moe_C, "router_gemm_bf16_fp32"
        ):

            @register_fake("_moe_C::router_gemm_bf16_fp32")
            def _router_gemm_bf16_fp32_fake(
                input: "torch.Tensor", weight: "torch.Tensor"
            ) -> "torch.Tensor":
                return torch.empty(
                    input.shape[0],
                    weight.shape[0],
                    dtype=torch.float32,
                    device=input.device,
                )

    except Exception:  # pragma: no cover - meta-impl is optional
        logger.debug("vllm-iluvatar: skipped register_fake for router_gemm")
