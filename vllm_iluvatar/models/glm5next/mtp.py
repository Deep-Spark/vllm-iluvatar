# SPDX-License-Identifier: Apache-2.0
"""GLM5Next MTP draft on Iluvatar."""

import copy

import ixformer.inference.functions as ixfops
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.models.glm5next.common.mtp import (
    Glm5NextMTP,
    Glm5NextMultiTokenPredictor,
    Glm5NextMultiTokenPredictorLayer,
)
from vllm.platforms import current_platform

from .attention import IluGlm5NextMLAAttention, glm5next_topk_buffer_width
from .model import _GLM5_PACKED_MODULES_MAPPING, IluGlm5NextDecoderLayer


class IluGlm5NextEhProj(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        assert linear.bias is None
        self.weight = linear.weight
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ixfops.linear(x, self.weight)


class IluGlm5NextMultiTokenPredictorLayer(Glm5NextMultiTokenPredictorLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super(Glm5NextMultiTokenPredictorLayer, self).__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        topk_tokens = config.index_topk
        assert topk_tokens is not None
        kpool = config.index_kpool
        assert kpool is not None
        buffer_width = glm5next_topk_buffer_width(vllm_config, topk_tokens, kpool)
        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            buffer_width,
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.shared_head = SharedHead(
            config=config,
            prefix=prefix,
            defer_lm_head=True,
        )
        layer_idx = int(prefix.rsplit(".", 1)[-1])
        self.mtp_block = IluGlm5NextDecoderLayer(
            vllm_config=vllm_config,
            config=config,
            layer_idx=layer_idx,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            is_mtp_layer=True,
        )
        self.eh_proj = IluGlm5NextEhProj(self.eh_proj)


class IluGlm5NextMultiTokenPredictor(Glm5NextMultiTokenPredictor):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm5NextMultiTokenPredictor, self).__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): IluGlm5NextMultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self._mtp_layers = list(self.layers.values())
        self._mtp_mla_attns = []
        for layer in self._mtp_layers:
            self_attn = layer.mtp_block.self_attn
            assert isinstance(self_attn, IluGlm5NextMLAAttention)
            self._mtp_mla_attns.append(self_attn.mla_attn)
        self.logits_processor = LogitsProcessor(config.vocab_size)


class IluGlm5NextMTP(Glm5NextMTP, SupportsPP):
    packed_modules_mapping = _GLM5_PACKED_MODULES_MAPPING

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm5NextMTP, self).__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = IluGlm5NextMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.set_moe_parameters()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )

    def load_weights(self, weights):
        if get_pp_group().world_size > 1:
            mtp_idx = self.model.mtp_start_layer_idx
            weights = (
                (
                    f"model.layers.{mtp_idx}.embed_tokens.weight"
                    if w[0]
                    in (
                        "model.language_model.embed_tokens.weight",
                        "model.embed_tokens.weight",
                    )
                    else w[0],
                    *w[1:],
                )
                for w in weights
            )
        return super().load_weights(weights)


def prepare_draft_head_w8a8(draft: IluGlm5NextMTP) -> None:
    """Pack private MTP heads after vLLM has finished sharing target weights.

    vLLM aliases the target head *module*, not only its parameter. Copy the
    module dictionaries before replacing its quantization method and weights.
    The target head remains BF16 and the existing local argmax/TP reduction
    route is unchanged. Called once after loading, before graph capture.
    """
    from vllm.logger import init_logger

    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes.ilu_compressed_tensors_w8a8_int8 import (
        IluW8A8OnlineLinearMethod,
    )

    class LoadedWeights:
        def process_weights_after_loading(self, layer):
            # The original method has already run during model loading.
            pass

    for layer in draft.model._mtp_layers:
        original = layer.shared_head.head
        if isinstance(original.quant_method, IluW8A8OnlineLinearMethod):
            continue
        if original.weight.dtype != torch.bfloat16:
            raise ValueError("Draft-only W8A8 requires a loaded BF16 target head")
        head = copy.copy(original)
        head._parameters = original._parameters.copy()
        head._buffers = original._buffers.copy()
        head._modules = original._modules.copy()
        head._non_persistent_buffers_set = original._non_persistent_buffers_set.copy()
        source_weight = original.weight
        source_pointer = source_weight.data_ptr()
        method = IluW8A8OnlineLinearMethod(LoadedWeights())
        method.process_weights_after_loading(head)
        head.quant_method = method
        if (
            original.weight is not source_weight
            or original.weight.data_ptr() != source_pointer
            or original.weight.dtype != torch.bfloat16
        ):
            raise RuntimeError("Draft quantization modified the target vocabulary head")
        layer.shared_head.head = head
    init_logger(__name__).info(
        "GLM5Next draft-only W8A8 vocabulary projection ready; target head remains BF16"
    )
