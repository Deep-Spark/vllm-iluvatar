# SPDX-License-Identifier: Apache-2.0
"""MTP draft model for GLM-5.2 on the official ZhipuAI FP8 weights.

Thin subclasses of the W4A8 chain in :mod:`.mtp`, mirroring how
:mod:`.fp8_model` relates to :mod:`.model`. The draft-specific differences are:

* ``mtp_block`` is an :class:`IluGlmMoeDsaFp8DecoderLayer`, so the draft runs
  the same FP8 attention (absorbed up-proj, no INT8 norm+quant fusion) as
  the target. Everything else about the draft -- the standard DeepSeek MTP
  layout, ``index_share_for_mtp_iteration`` via ``set_skip_topk`` /
  ``compact_topk_indices``, the ``lm_head`` / ``embed_tokens`` ties -- is
  inherited unchanged.

* ``indexer.wk`` is block-FP8 in this checkpoint (including the MTP layer's own
  indexer) while its fusion partner ``weights_proj`` is BF16, so the weight
  stream is folded to BF16 before the loader sees it, exactly as the target
  does.

* Optional draft-only per-channel FP8 LM head reduces speculative head weight
  traffic. Conversion runs after loading, and execution reuses the existing
  FP8 Linear method. Target and tied source weights remain unchanged.

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

import copy
from collections.abc import Iterable

import ixformer.inference.functions as ixf_ops
import torch
import vllm.envs as envs
from vllm.logger import init_logger

from vllm_iluvatar.ilu_util.arch import is_ivcore30
from vllm_iluvatar.model_executor.layers.quantization.configs.fp8 import IluFp8Config
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.linear.native_to_fp8 import (
    IluFp8PerChannelOnlineLinearMethod,
)
from vllm_iluvatar.models.glm_moe_dsa.fp8_model import (
    IluGlmMoeDsaFp8DecoderLayer,
    dequant_indexer_wk,
)
from vllm_iluvatar.models.glm_moe_dsa.mtp import (
    IluGlmMoeDsaMTP,
    IluGlmMoeDsaMultiTokenPredictor,
    IluGlmMoeDsaMultiTokenPredictorLayer,
)

logger = init_logger(__name__)


class _Fp8DraftLMHeadProxy:
    """Loaded draft runtime weights; the original tied head is never mutated.

    A plain proxy keeps these derived weights out of the checkpoint/module
    loading pass. LogitsProcessor consumes its TP metadata and Linear method.
    """

    def __init__(self, head: torch.nn.Module, quant_config: IluFp8Config) -> None:
        self.tp_size = head.tp_size
        self.shard_indices = head.shard_indices
        self.output_size_per_partition, self.input_size_per_partition = (
            head.weight.shape
        )
        self.weight, scale = ixf_ops.fp8_per_token_dynamic_scaled(head.weight.data)
        self.weight_scale = scale.reshape(-1)
        # The unsharded GLM vocabulary favors GEMM at the draft batch size;
        # retain the ordinary Linear selection for smaller TP shards.
        self._can_use_gemm = self.tp_size > 1
        method_config = copy.copy(quant_config)
        method_config.weight_block_size = None
        method_config.gemm_format = "TN"
        self.quant_method = IluFp8PerChannelOnlineLinearMethod(method_config)


class IluGlmMoeDsaFp8MultiTokenPredictorLayer(IluGlmMoeDsaMultiTokenPredictorLayer):
    decoder_layer_cls = IluGlmMoeDsaFp8DecoderLayer
    _fp8_head_proxy: _Fp8DraftLMHeadProxy | None = None


class IluGlmMoeDsaFp8MultiTokenPredictor(IluGlmMoeDsaMultiTokenPredictor):
    predictor_layer_cls = IluGlmMoeDsaFp8MultiTokenPredictorLayer

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        layer_idx = self.mtp_start_layer_idx + spec_step_idx % self.num_mtp_layers
        layer = self.layers[str(layer_idx)]
        proxy = layer._fp8_head_proxy
        if proxy is not None and self.logits_processor.head_dtype in (
            None,
            hidden_states.dtype,
        ):
            return self.logits_processor(proxy, layer.shared_head(hidden_states))
        return super().compute_logits(hidden_states, spec_step_idx)


class IluGlmMoeDsaFp8MTP(IluGlmMoeDsaMTP):
    """Top-level GLM-5.2 FP8 MTP draft (``IluGlmMoeDsaFp8MTPModel``)."""

    predictor_cls = IluGlmMoeDsaFp8MultiTokenPredictor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The draft loader is handed the whole checkpoint and filters by spec
        # layer, so this also walks the 78 target-layer ``wk`` tensors it will
        # then discard. They are small and the fold is a couple of CPU ops
        # each; keeping one shared implementation is worth more than skipping
        # them.
        loaded = super().load_weights(dequant_indexer_wk(weights))
        if envs.VLLM_ILU_GLM_MTP_FP8_LM_HEAD and is_ivcore30():
            for layer in self.model.layers.values():
                head = layer.shared_head.head
                weight = getattr(head, "weight", None)
                if (
                    weight is not None
                    and weight.is_cuda
                    and weight.dtype in (torch.bfloat16, torch.float16)
                    and weight.ndim == 2
                    and weight.shape[0] % 32 == 0
                    and weight.shape[1] % 64 == 0
                ):
                    layer._fp8_head_proxy = _Fp8DraftLMHeadProxy(
                        head, self.quant_config
                    )
                    logger.info_once(
                        "GLM FP8 MTP: built draft-only FP8 lm_head %s using FP8 Linear",
                        tuple(weight.shape),
                    )
        return loaded
