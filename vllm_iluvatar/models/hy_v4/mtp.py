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
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.kv_cache import KVCacheScaleParameter
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import get_pp_missing_layer_names

from vllm_iluvatar.models.glm_moe_dsa.mtp import (
    IluGlmMoeDsaMTP,
    IluGlmMoeDsaMultiTokenPredictor,
    IluGlmMoeDsaMultiTokenPredictorLayer,
)

from .model import HYV4DecoderLayer, _normalize_hyv4_config
from .weight_utils import try_load_quantized_indexer_wk

logger = init_logger(__name__)

# Attribute holding the unquantized-module prefix list, per config class:
# ``exclude_modules`` on ModelOpt* configs, ``ignored_layers`` elsewhere.
_QUANT_EXCLUSION_ATTRS = ("exclude_modules", "ignored_layers")


def _remap_mtp_quant_exclusions(
    quant_config: QuantizationConfig | None,
    mtp_start_layer_idx: int,
    num_mtp_layers: int,
) -> QuantizationConfig | None:
    """Translate checkpoint-named MTP quant exclusions to draft prefixes.

    The checkpoint names the MTP modules ``model.mtp_layers.0.*`` while the
    draft builds them under ``model.layers.<num_hidden_layers + i>.*``.
    ``is_layer_skipped`` compares prefixes literally, so an excluded MTP module
    otherwise gets a quant method while its checkpoint weight is plain BF16
    with no companion scale: the scale then keeps its uninitialized value and
    the layer silently produces garbage.
    """
    if quant_config is None:
        return None

    exclusion_attr: str | None = None
    patterns: list[str] | None = None
    for attr in _QUANT_EXCLUSION_ATTRS:
        value = getattr(quant_config, attr, None)
        if value:
            exclusion_attr = attr
            patterns = list(value)
            break
    if not patterns or exclusion_attr is None:
        return quant_config

    extra: list[str] = []
    for offset in range(num_mtp_layers):
        # The VL target rewrites ``model.`` to ``language_model.model.`` on the
        # shared quant config before the draft is built, so the checkpoint
        # prefix may already be ``language_model.model.mtp_layers.<i>.``.
        marker = f"mtp_layers.{offset}."
        draft_prefix = f"model.layers.{mtp_start_layer_idx + offset}."
        for pattern in patterns:
            stripped = pattern.rstrip("*")
            idx = stripped.find(marker)
            if idx < 0:
                continue
            trailing_stars = pattern[len(stripped) :]
            extra.append(draft_prefix + stripped[idx + len(marker) :] + trailing_stars)
    if not extra:
        return quant_config

    # The target model shares this object; copy before widening the list.
    quant_config = copy.copy(quant_config)
    setattr(quant_config, exclusion_attr, patterns + extra)
    logger.info(
        "HY4 MTP: mapped %d checkpoint-named quant exclusions (%s) onto draft "
        "module prefixes: %s",
        len(extra),
        exclusion_attr,
        ", ".join(extra),
    )
    return quant_config


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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert inputs_embeds is not None
        # M-RoPE positions are [3, num_tokens]. unsqueeze(-1) would broadcast
        # the embed to 3D and break the cat with the 2D hidden state. The
        # position-0 mask is per token; the text row is enough. The decoder
        # still gets the full 3-row positions.
        mask_positions = positions[0] if positions.ndim == 2 else positions
        inputs_embeds = torch.where(mask_positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        hidden_states = residual + hidden_states
        return hidden_states, self.shared_head(hidden_states)


class HYV4MultiTokenPredictor(IluGlmMoeDsaMultiTokenPredictor):
    predictor_layer_cls = HYV4MultiTokenPredictorLayer


class HYV4MTP(IluGlmMoeDsaMTP):
    """Top-level HY4 MTP draft model."""

    predictor_cls = HYV4MultiTokenPredictor

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        target_config = vllm_config.model_config.hf_config
        remapped = _remap_mtp_quant_exclusions(
            vllm_config.quant_config,
            target_config.num_hidden_layers,
            max(getattr(target_config, "num_nextn_predict_layers", 1), 1),
        )
        if remapped is not vllm_config.quant_config:
            vllm_config = copy.copy(vllm_config)
            vllm_config.quant_config = remapped
        super().__init__(vllm_config=vllm_config, prefix=prefix)

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
                # FP4 stores block-FP8 linear scales as ``.scale`` (uint8 E8M0).
                # Draft params are ``weight_scale_inv`` in float8_e8m0fnu. The
                # FP8 preview already used ``.weight_scale``; leave those alone
                # (MXFP4 expert scales keep their uint8 bytes for the MoE loader).
                if name.endswith(".scale") and not name.endswith(".weight_scale"):
                    name = name[: -len(".scale")] + ".weight_scale_inv"
                    if loaded_weight.dtype == torch.uint8:
                        loaded_weight = loaded_weight.view(torch.float8_e8m0fnu)

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
                    narrow_weight = loaded_weight[head_start:head_end]
                    with torch.no_grad():
                        param[: narrow_weight.shape[0]].copy_(
                            narrow_weight.to(param.dtype)
                        )
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

        # ``weight_scale`` is allocated with ``torch.empty``. A module that got
        # a quant method by mistake -- because its checkpoint-named exclusion
        # did not match the draft prefix -- therefore runs on whatever the
        # allocator returned and quietly degrades the draft instead of failing.
        # ``shared_head.head`` is exempt: the checkpoint ships no per-MTP head,
        # and the proposer replaces the whole module with the target's lm_head
        # (``_maybe_share_lm_head``) before the first draft forward.
        unscaled = sorted(
            name
            for name in params_dict
            if name.endswith(("weight_scale", "weight_scale_inv"))
            and name not in loaded
            and ".shared_head.head." not in name
        )
        if unscaled:
            raise ValueError(
                "HY4 MTP quantization scale(s) received no checkpoint value: "
                + ", ".join(unscaled)
            )
        # The GLM parent reports before HY4's sink/router/indexer additions
        # are merged. Audit the final set, not that intermediate warning.
        # KV-cache scalar placeholders have defaults applied after loading.
        # The proposer replaces the draft head with the target head.
        expected = {
            name
            for name, param in params_dict.items()
            if not isinstance(param, KVCacheScaleParameter)
            and ".shared_head.head." not in name
        }
        missing = sorted(expected - loaded)
        if missing:
            raise ValueError("HY4 MTP parameters not loaded: " + ", ".join(missing))
        logger.info(
            "HY4 MTP final load audit: %d/%d checkpoint parameters, "
            "%d HY4 additions, missing=0",
            len(loaded & expected),
            len(expected),
            len(loaded_extra),
        )
        return loaded
