# Source: vllm_iluvatar/model_executor/models/qwen3_5_apc.py at 47fb804; paired ixaite metadata and state.
# SPDX-License-Identifier: Apache-2.0
"""Dense Qwen3.5 APC construction; inherited weights/forward names stay unchanged.

Based on vLLM main_61ac3680210. The upstream dense constructor rejects all;
these plugin-owned constructors use the same modules with the APC-capable GDN
backend. No temporary config mutation is used during model construction.
"""

from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models import qwen3_5 as upstream
from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig


class IluQwen3_5Config(Qwen3_5ForConditionalGenerationConfig):
    """Keep the existing implicit align default; explicit all is a capability."""

    @staticmethod
    def verify_and_update_config(vllm_config):
        """Apply dense-only cache policy before the generic hybrid verifier."""
        cache = vllm_config.cache_config
        if cache.enable_prefix_caching and cache.mamba_cache_mode == "none":
            cache.mamba_cache_mode = "align"
        if (
            cache.mamba_cache_mode == "all"
            and vllm_config.speculative_config is not None
        ):
            raise NotImplementedError(
                "Qwen3.5 all prefix caching requires non-speculative decoding"
            )
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)


class IluQwen3_5ForCausalLM(upstream.Qwen3_5ForCausalLM):
    """Construct the standard dense language model with APC-capable OOT layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """Preserve the upstream module tree and weight-loading contract."""
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config

        scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config

        nn.Module.__init__(self)
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = upstream.Qwen3_5Model(
            vllm_config=vllm_config, prefix=upstream.maybe_prefix(prefix, "model")
        )

        if upstream.get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = upstream.ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=upstream.maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = upstream.PPMissingLayer()

        self.logits_processor = upstream.LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )


def initialize_apc_model(self, *, vllm_config: VllmConfig, prefix: str = "model"):
    """Construct the dense wrapper while keeping the requested all cache mode."""
    from vllm_iluvatar.backends.ixaite.layers.gdn import (
        require_ixaite_apc_providers,
    )

    require_ixaite_apc_providers()
    # protocols have not __init__ method, so we need to use nn.Module.__init__
    nn.Module.__init__(self)
    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config
    multimodal_config = vllm_config.model_config.multimodal_config

    self.config = config
    self.model_config = vllm_config.model_config
    self.multimodal_config = multimodal_config
    self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
    self.is_multimodal_pruning_enabled = (
        multimodal_config.is_multimodal_pruning_enabled()
    )
    self.video_pruning_rate = self.multimodal_config.video_pruning_rate
    self._tokenizer = upstream.cached_tokenizer_from_config(vllm_config.model_config)

    # attributes needed by EVS-related functions inherited from Qwen3-VL
    self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
    self.deepstack_num_level = (
        len(config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
    )
    self.visual_dim = config.vision_config.out_hidden_size
    self.multiscale_dim = self.visual_dim * self.deepstack_num_level

    with self._mark_tower_model(vllm_config, {"image", "video"}):
        self.visual = upstream.Qwen3_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=upstream.maybe_prefix(prefix, "visual"),
        )

    with self._mark_language_model(vllm_config):
        self.language_model = IluQwen3_5ForCausalLM(
            vllm_config=vllm_config,
            prefix=upstream.maybe_prefix(prefix, "language_model"),
        )

    self.make_empty_intermediate_tensors = (
        self.language_model.make_empty_intermediate_tensors
    )
