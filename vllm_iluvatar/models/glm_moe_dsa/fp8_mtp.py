# SPDX-License-Identifier: Apache-2.0
"""MTP draft model for GLM-5.2 on the official ZhipuAI FP8 weights.

Thin subclasses of the W4A8 chain in :mod:`.mtp`, mirroring how
:mod:`.fp8_model` relates to :mod:`.model`. Only two things differ:

* ``mtp_block`` is an :class:`IluGlmMoeDsaFp8DecoderLayer`, so the draft runs
  the same FP8 attention (BF16 absorbed up-proj, no INT8 norm+quant fusion) as
  the target. Everything else about the draft -- the standard DeepSeek MTP
  layout, ``index_share_for_mtp_iteration`` via ``set_skip_topk`` /
  ``compact_topk_indices``, the ``lm_head`` / ``embed_tokens`` ties -- is
  inherited unchanged.

* ``indexer.wk`` is block-FP8 in this checkpoint (including the MTP layer's own
  indexer) while its fusion partner ``weights_proj`` is BF16, so the weight
  stream is folded to BF16 before the loader sees it, exactly as the target
  does.

Two things that *look* like they need overriding and do not:

* ``eh_proj`` is BF16 here (the W4A8 converter quantised it to INT8). No change
  needed: the checkpoint lists ``model.layers.78.eh_proj`` in
  ``modules_to_not_convert``, so ``IluFp8Config`` hands it an
  ``UnquantizedLinearMethod`` and the ``quant_config``-bound ``ReplicatedLinear``
  loads a plain BF16 weight. Same for ``mlp.gate`` and the norms.

* The MTP layer index (78) is past the end of ``indexer_types`` (78 entries,
  layers 0-77), but ``IluGlmMoeDsaMLAAttention`` already forces
  ``skip_topk=False`` for ``layer_id >= num_hidden_layers``, so the draft builds
  its own ``full`` indexer -- which is what the checkpoint ships for layer 78.

Registered as ``IluGlmMoeDsaFp8MTPModel``. It needs its own arch name rather
than overriding the community one: for ``glm_moe_dsa`` the community router
rewrites the draft to the *generic* ``DeepSeekMTPModel``, which real DeepSeek
checkpoints also use, so overriding it would hijack them. See
``vllm_iluvatar.config.speculative._ILU_MTP_ARCH_MAP``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from vllm_iluvatar.models.glm_moe_dsa.fp8_model import (
    IluGlmMoeDsaFp8DecoderLayer,
    dequant_indexer_wk,
)
from vllm_iluvatar.models.glm_moe_dsa.mtp import (
    IluGlmMoeDsaMTP,
    IluGlmMoeDsaMultiTokenPredictor,
    IluGlmMoeDsaMultiTokenPredictorLayer,
)


class IluGlmMoeDsaFp8MultiTokenPredictorLayer(IluGlmMoeDsaMultiTokenPredictorLayer):
    decoder_layer_cls = IluGlmMoeDsaFp8DecoderLayer


class IluGlmMoeDsaFp8MultiTokenPredictor(IluGlmMoeDsaMultiTokenPredictor):
    predictor_layer_cls = IluGlmMoeDsaFp8MultiTokenPredictorLayer


class IluGlmMoeDsaFp8MTP(IluGlmMoeDsaMTP):
    """Top-level GLM-5.2 FP8 MTP draft (``IluGlmMoeDsaFp8MTPModel``)."""

    predictor_cls = IluGlmMoeDsaFp8MultiTokenPredictor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The draft loader is handed the whole checkpoint and filters by spec
        # layer, so this also walks the 78 target-layer ``wk`` tensors it will
        # then discard. They are small and the fold is a couple of CPU ops
        # each; keeping one shared implementation is worth more than skipping
        # them.
        return super().load_weights(dequant_indexer_wk(weights))
