# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 on the official ZhipuAI FP8 weights (``GlmMoeDsaForCausalLM``).

The W4A8 chain in :mod:`.model` is reused wholesale; this module only swaps in
the FP8 attention variant and fixes up the two places where the official FP8
checkpoint's tensor set differs from the converted W4A8 one:

1. ``indexer.wk`` ships as FP8 (``weight`` + 128x128 ``weight_scale_inv``)
   while its fusion partner ``weights_proj`` stays BF16. The fused
   ``wk_weights_proj`` GEMM is an unquantised BF16 layer with no slot for a
   scale, so ``wk`` is dequantised to BF16 while streaming the checkpoint
   (:func:`dequant_indexer_wk`). Keeping the fusion is worth it: ``wk`` is
   tiny (tens of MiB across all layers) and splitting it back into two GEMMs
   would cost a kernel launch per layer in the indexer hot path.

2. ``kv_b_proj`` has no ``mla_bmm_w8a16`` entry to route it to the W4A8
   grouped-Marlin method, so the absorbed up-proj is rebuilt as BF16 -- see
   :mod:`.fp8_attention`.

Everything else the FP8 checkpoint needs is already handled by
``IluFp8Config``: dense linears requantise 128x128 block scale to per-channel
FP8 at load time, and MoE experts convert to ix30 FP4.

Registered under the community arch name ``GlmMoeDsaForCausalLM``, which
overrides vLLM's own (NV-kernel) implementation.
"""

from collections.abc import Iterable

import torch
from vllm.logger import init_logger

from vllm_iluvatar.models.glm_moe_dsa.fp8_attention import IluGlmMoeDsaFp8MLAAttention
from vllm_iluvatar.models.glm_moe_dsa.model import (
    IluGlmMoeDsaDecoderLayer,
    IluGlmMoeDsaForCausalLM,
    IluGlmMoeDsaModel,
)

logger = init_logger(__name__)

_FP8_BLOCK = 128
_WK_WEIGHT_SUFFIX = ".indexer.wk.weight"
_WK_SCALE_SUFFIX = ".indexer.wk.weight_scale_inv"


def _block_dequant(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block: int = _FP8_BLOCK,
) -> torch.Tensor:
    """Dequantise a 128x128 block-scaled FP8 weight to BF16."""
    n, k = weight.shape
    expected = ((n + block - 1) // block, (k + block - 1) // block)
    if tuple(scale_inv.shape) != expected:
        raise ValueError(
            f"block scale shape {tuple(scale_inv.shape)} != {expected} "
            f"expected for a {n}x{k} weight at block size {block}"
        )
    expanded = (
        scale_inv.to(torch.float32)
        .repeat_interleave(block, dim=0)
        .repeat_interleave(block, dim=1)[:n, :k]
    )
    return (weight.to(torch.float32) * expanded).to(torch.bfloat16)


def dequant_indexer_wk(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Fold ``indexer.wk.weight_scale_inv`` into its weight, yielding BF16.

    Rewrites the stream so the downstream loader never sees the scale: by the
    time a name reaches ``stacked_params_mapping`` it looks exactly like the
    BF16 ``wk`` the W4A8 checkpoint ships. Weight and scale are paired
    whichever order they arrive in, since nothing guarantees they share a
    safetensors shard.
    """
    pending_weight: dict[str, torch.Tensor] = {}
    pending_scale: dict[str, torch.Tensor] = {}
    n_folded = 0

    for name, tensor in weights:
        if name.endswith(_WK_SCALE_SUFFIX):
            base = name[: -len(_WK_SCALE_SUFFIX)]
            weight = pending_weight.pop(base, None)
            if weight is None:
                pending_scale[base] = tensor
            else:
                n_folded += 1
                yield base + _WK_WEIGHT_SUFFIX, _block_dequant(weight, tensor)
            continue

        if name.endswith(_WK_WEIGHT_SUFFIX):
            base = name[: -len(_WK_WEIGHT_SUFFIX)]
            if tensor.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                # Already BF16 (e.g. a layer left out of the FP8 conversion).
                yield name, tensor
                continue
            scale = pending_scale.pop(base, None)
            if scale is None:
                pending_weight[base] = tensor
            else:
                n_folded += 1
                yield name, _block_dequant(tensor, scale)
            continue

        yield name, tensor

    if pending_weight or pending_scale:
        raise ValueError(
            "[GLM-5.2 FP8] unpaired indexer.wk tensors after streaming the "
            f"checkpoint: {sorted(pending_weight)} have no scale, "
            f"{sorted(pending_scale)} have no weight."
        )
    logger.info("[GLM-5.2 FP8] dequantised %d indexer.wk weights to BF16", n_folded)


class IluGlmMoeDsaFp8DecoderLayer(IluGlmMoeDsaDecoderLayer):
    mla_attention_cls = IluGlmMoeDsaFp8MLAAttention

    # The W4A8 layer fuses each RMSNorm with an INT8 quant and hands the
    # ``(q, scales, ...)`` triple straight to the next GEMM. None of the FP8
    # consumers can unpack that -- ``BlockScaledMMLinearKernel.apply_weights``
    # calls ``x.view()`` on it -- so the fusion is off and every layer takes
    # the plain-BF16 path. Re-fusing RMSNorm with *FP8* quant is D3 in the
    # plan: a performance item for stage E, not needed to run.
    fuse_norm_quant = False


class IluGlmMoeDsaFp8Model(IluGlmMoeDsaModel):
    decoder_layer_cls = IluGlmMoeDsaFp8DecoderLayer

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(dequant_indexer_wk(weights))


class IluGlmMoeDsaFp8ForCausalLM(IluGlmMoeDsaForCausalLM):
    """Top-level causal-LM for the official GLM-5.2 FP8 weights."""

    model_cls = IluGlmMoeDsaFp8Model


def register_model() -> None:
    """Point the community ``GlmMoeDsaForCausalLM`` arch at the ilu FP8 impl.

    The official ZhipuAI FP8 checkpoint declares the community arch name, so
    unlike the converted W4A8 weights (``IluGlmMoeDsaForCausalLM``) there is no
    ilu-specific name to key off -- this override is what keeps the load off
    vLLM's NV-kernel path.

    The MTP draft is the opposite case: it gets its own ``IluGlmMoeDsaFp8MTPModel``
    name because the community router sends ``glm_moe_dsa`` drafts to the
    *generic* ``DeepSeekMTPModel``, shared with real DeepSeek checkpoints, so
    overriding that name would hijack them. ``config.speculative``'s
    ``_ILU_MTP_ARCH_MAP`` re-points the draft here instead.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "GlmMoeDsaForCausalLM",
        "vllm_iluvatar.models.glm_moe_dsa.fp8_model:IluGlmMoeDsaFp8ForCausalLM",
    )
    ModelRegistry.register_model(
        "IluGlmMoeDsaFp8MTPModel",
        "vllm_iluvatar.models.glm_moe_dsa.fp8_mtp:IluGlmMoeDsaFp8MTP",
    )
