# SPDX-License-Identifier: Apache-2.0
"""GLM KDA RecoverSSM admission for the fixed vLLM V2 runner."""

from __future__ import annotations

from functools import wraps

import torch

GLM_KDA_ARCHITECTURES = frozenset(
    ("Glm5NextForCausalLM", "Glm5NextForConditionalGeneration")
)


def is_glm_recoverssm(config) -> bool:
    model = config.model_config
    return (
        model is not None
        and model.architecture in GLM_KDA_ARCHITECTURES
        and config.cache_config.use_replayssm
    )


def validate_glm_recoverssm(config):
    cache = config.cache_config
    cache.use_kda_recoverssm = False
    if config.speculative_config is None or config.num_speculative_tokens <= 0:
        raise ValueError("GLM RecoverSSM requires speculative decoding")
    if not config.use_v2_model_runner:
        raise ValueError("GLM RecoverSSM requires Model Runner V2")
    from vllm.config.compilation import CUDAGraphMode

    if config.compilation_config.cudagraph_mode not in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL_DECODE_ONLY,
    ):
        raise ValueError("GLM RecoverSSM supports eager or FULL_DECODE_ONLY graphs")
    if config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("GLM RecoverSSM requires pipeline_parallel_size=1")
    if getattr(config.parallel_config, "enable_dbo", False) or getattr(
        config.parallel_config, "enable_ubatching", False
    ):
        raise ValueError("GLM RecoverSSM does not support overlapping microbatches/DBO")
    if cache.mamba_cache_mode not in ("none", "align"):
        raise ValueError("GLM RecoverSSM supports none/align Mamba cache modes")
    if config.model_config.dtype != torch.bfloat16:
        raise ValueError("GLM RecoverSSM requires bfloat16 activations")
    if cache.mamba_cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("GLM RecoverSSM requires bfloat16 Conv history")
    if cache.mamba_ssm_cache_dtype not in ("auto", "float32", "bfloat16"):
        raise ValueError("GLM RecoverSSM requires float32/bfloat16 SSM state")
    if config.mamba_config.enable_stochastic_rounding:
        raise ValueError("GLM RecoverSSM does not support stochastic state rounding")
    from vllm.config.mamba import MambaBackendEnum

    if config.mamba_config.backend != MambaBackendEnum.TRITON:
        raise ValueError(
            "GLM RecoverSSM requires the plugin's GDN/triton backend selection"
        )
    if config.kv_transfer_config is not None and (
        config.kv_transfer_config.is_kv_transfer_instance
    ):
        raise ValueError("GLM RecoverSSM does not support KV connectors/offload")
    t = config.num_speculative_tokens + 1
    if t > 64:
        raise ValueError("GLM RecoverSSM verify capacity must be <= 64")
    text = config.model_config.hf_text_config
    if text.linear_head_dim != 128 or text.linear_conv_kernel_dim != 4:
        raise ValueError("GLM RecoverSSM requires KDA D=128 and Conv width=4")
    if not -5 <= text.linear_lower_bound < 0:
        raise ValueError("GLM RecoverSSM gate lower bound must be in [-5,0)")
    import vllm.envs as envs

    if not envs.VLLM_ILU_GDN_ATTN_BACKEND:
        raise ValueError("GLM RecoverSSM requires VLLM_ILU_GDN_ATTN_BACKEND=1")
    if not config.model_config.supports_replayssm:
        raise ValueError(
            "GLM RecoverSSM requires the Iluvatar GLM model implementation"
        )
    cache.use_kda_recoverssm = True
    return config


def make_validate_mamba_cached_kernel(original):
    @wraps(original)
    def validate_mamba_cached_kernel(self):
        if is_glm_recoverssm(self):
            return validate_glm_recoverssm(self)
        return original(self)

    return validate_mamba_cached_kernel
