"""Patch ``vllm._custom_ops`` with iluvatar-specific implementations.

We install (or replace) four attributes on :mod:`vllm._custom_ops`:

* ``silu_and_mul_with_clamp`` (NEW): the ilu MLP/activation path needs this
  but stock vLLM only exposes ``torch.ops._C.silu_and_mul_with_clamp``
  (CUDA / not built on iluvatar). We route to the ixformer fused op.
* ``router_gemm_bf16_fp32`` (NEW): ``bf16 x bf16 -> fp32`` GEMM with fp32
  accumulate, used by the V4 router. Routes to
  ``ixformer.inference.functions.gemm_bf16_bf16_fp32(..., format="TN")``.
* ``topk_hash_softplus_sqrt`` (REPLACED): upstream calls
  ``torch.ops._moe_C.topk_softplus_sqrt`` which does not exist on
  iluvatar. Reimplemented in pure PyTorch (sqrt(softplus(x)) + optional
  hash routing + topk + renormalize + scale).
* ``dsa_convert_req_index_to_global_index`` (DISPATCH FIX): the upstream
  iluvatar_main version dispatches to ``ops.dsa_convert_req_index_to_global_index``
  but the kernel actually lives at ``ixf_ops.dsa_convert_req_index_to_global_index``.

All replacements are imported from ``vllm_iluvatar.ops`` (the implementations
live there so they can also be called directly from our own model code if
needed).
"""

from __future__ import annotations


def apply() -> None:
    import vllm._custom_ops as _custom_ops
    from vllm.logger import init_logger

    from vllm_iluvatar import ops as ilu_ops

    logger = init_logger(__name__)

    _custom_ops.silu_and_mul_with_clamp = ilu_ops.silu_and_mul_with_clamp
    _custom_ops.router_gemm_bf16_fp32 = ilu_ops.router_gemm_bf16_fp32
    _custom_ops.topk_hash_softplus_sqrt = ilu_ops.topk_hash_softplus_sqrt
    _custom_ops.dsa_convert_req_index_to_global_index = (
        ilu_ops.dsa_convert_req_index_to_global_index
    )

    # Register fake meta-impl only if torch.ops._moe_C exposes the matching
    # op (mirrors the upstream guarded ``register_fake`` block).
    try:
        import torch
        from torch.library import register_fake

        if hasattr(torch.ops, "_moe_C") and hasattr(
            torch.ops._moe_C, "router_gemm_bf16_fp32"
        ):

            @register_fake("_moe_C::router_gemm_bf16_fp32")
            def _router_gemm_bf16_fp32_fake(
                input: torch.Tensor, weight: torch.Tensor
            ) -> torch.Tensor:
                return torch.empty(
                    input.shape[0],
                    weight.shape[0],
                    dtype=torch.float32,
                    device=input.device,
                )

    except Exception:  # pragma: no cover - meta-impl is optional
        logger.debug("vllm-iluvatar: skipped register_fake for router_gemm")

    logger.info(
        "vllm-iluvatar: patched vllm._custom_ops "
        "(silu_and_mul_with_clamp, router_gemm_bf16_fp32, "
        "topk_hash_softplus_sqrt, dsa_convert_req_index_to_global_index)"
    )
