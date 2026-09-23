# SPDX-License-Identifier: Apache-2.0
"""MTP draft model for HY V4 on Iluvatar.

HY4's checkpoint uses the standard DeepSeek-V3-style MTP contract
(``eh_proj`` plus one non-iHC decoder block), despite the iHC target backbone.
This module reuses the tested GLM/DeepSeek tuple-returning proposer and swaps in
HY4 attention, indexer, MoE, weight naming, and MXFP8-indexer loading.
"""

from __future__ import annotations

import copy
import typing
from collections.abc import Callable, Iterable, Iterator

import torch
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import get_pp_missing_layer_names

from vllm_iluvatar.models.glm_moe_dsa.mtp import (
    IluGlmMoeDsaMTP,
    IluGlmMoeDsaMultiTokenPredictor,
    IluGlmMoeDsaMultiTokenPredictorLayer,
)

from .model import HYV4DecoderLayer, _normalize_hyv4_config
from .weight_utils import try_load_quantized_indexer_wk


class HYV4MTPDecoderLayer(HYV4DecoderLayer):
    """HY4 decoder configured for the checkpoint's non-iHC MTP block."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        assert vllm_config.speculative_config is not None
        config = copy.copy(vllm_config.speculative_config.draft_model_config.hf_config)
        config = _normalize_hyv4_config(config)
        config.enable_ihc = False

        # The backbone arrays contain 78 entries, while the MTP module is
        # addressed as virtual layer 78. Its checkpoint has both an indexer and
        # routed experts, so append the corresponding layer kinds locally.
        config.layer_types = list(config.layer_types) + ["sparse_attention"]
        config.mlp_layer_types = list(config.mlp_layer_types) + ["moe"]

        super().__init__(
            config=config,
            vllm_config=vllm_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
        )


class HYV4MultiTokenPredictorLayer(IluGlmMoeDsaMultiTokenPredictorLayer):
    decoder_layer_cls = HYV4MTPDecoderLayer


class HYV4MultiTokenPredictor(IluGlmMoeDsaMultiTokenPredictor):
    predictor_layer_cls = HYV4MultiTokenPredictorLayer


class HYV4MTP(IluGlmMoeDsaMTP):
    """Top-level HY4 MTP draft model."""

    predictor_cls = HYV4MultiTokenPredictor

    def _translate_mtp_weight_name(self, name: str) -> str:
        prefix = "model.mtp_layers."
        if not name.startswith(prefix):
            return name
        suffix = name[len(prefix) :]
        layer_text, separator, rest = suffix.partition(".")
        if not separator:
            return name
        layer_idx = self.model.mtp_start_layer_idx + int(layer_text)
        name = f"model.layers.{layer_idx}.{rest}"
        return name.replace(".final_layernorm.", ".shared_head.norm.")

    @staticmethod
    def _load_fused_experts(
        params_dict: dict[str, torch.Tensor],
        target_name: str,
        loaded_weight: torch.Tensor,
        num_experts: int,
        shard_ids: tuple[str, ...],
    ) -> bool:
        param = params_dict[target_name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        chunks = (
            loaded_weight.chunk(2, dim=-2) if len(shard_ids) == 2 else (loaded_weight,)
        )
        loaded_any = False
        for expert_id in range(num_experts):
            for chunk, shard_id in zip(chunks, shard_ids, strict=True):
                success = weight_loader(
                    param,
                    chunk[expert_id],
                    target_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                loaded_any = loaded_any or success
        return loaded_any

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_extra: set[str] = set()
        pending_wk_fp8: dict[str, dict[str, torch.Tensor]] = {}
        pp_missing_layer_names = get_pp_missing_layer_names(self)

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        local_heads = self.config.num_attention_heads // tp_size
        head_start = local_heads * tp_rank
        head_end = local_heads * (tp_rank + 1)

        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )
        num_experts = self.config.n_routed_experts

        def translated_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for raw_name, loaded_weight in weights:
                is_mtp_weight = raw_name.startswith("model.mtp_layers.")
                if not is_mtp_weight:
                    if raw_name in ("lm_head.weight", "model.embed_tokens.weight"):
                        # Parent loader stashes these two target weights and
                        # ties them into the draft.
                        yield raw_name, loaded_weight
                    continue

                name = self._translate_mtp_weight_name(raw_name)

                layer_idx = self.model.mtp_start_layer_idx
                rewritten = self._rewrite_spec_layer_name(layer_idx, name)

                if try_load_quantized_indexer_wk(
                    rewritten,
                    loaded_weight,
                    pending_wk_fp8,
                    params_dict,
                    loaded_extra,
                    pp_missing_layer_names,
                ):
                    continue

                if ".self_attn.learnable_sink_param" in rewritten:
                    param = params_dict[rewritten]
                    with torch.no_grad():
                        param.copy_(loaded_weight[head_start:head_end].to(param.dtype))
                    loaded_extra.add(rewritten)
                    continue

                if ".mlp.experts.gate_up_proj" in rewritten:
                    target = rewritten.replace(
                        ".experts.gate_up_proj",
                        f".experts.routed_experts.{base_layer}w13_weight",
                    )
                    if self._load_fused_experts(
                        params_dict,
                        target,
                        loaded_weight,
                        num_experts,
                        ("w1", "w3"),
                    ):
                        loaded_extra.add(target)
                    continue

                if ".mlp.experts.down_proj" in rewritten:
                    target = rewritten.replace(
                        ".experts.down_proj",
                        f".experts.routed_experts.{base_layer}w2_weight",
                    )
                    if self._load_fused_experts(
                        params_dict,
                        target,
                        loaded_weight,
                        num_experts,
                        ("w2",),
                    ):
                        loaded_extra.add(target)
                    continue

                if rewritten.endswith(".mlp.gate.e_score_correction_bias"):
                    target = rewritten.replace(
                        ".gate.e_score_correction_bias", ".expert_bias"
                    )
                    param = params_dict[target]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, loaded_weight)
                    loaded_extra.add(target)
                    continue

                # The parent performs q_a+kv_a, wk+weights_proj and gate+up
                # stacking after adding ``mtp_block`` through the same rewrite.
                yield name, loaded_weight

        loaded = super().load_weights(translated_weights())
        if pending_wk_fp8:
            missing = ", ".join(sorted(pending_wk_fp8))
            raise ValueError(
                f"incomplete HY4 MTP indexer wk weight/scale pairs: {missing}"
            )
        loaded.update(loaded_extra)
        return loaded
