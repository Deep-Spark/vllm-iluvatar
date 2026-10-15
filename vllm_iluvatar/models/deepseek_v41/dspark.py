# SPDX-License-Identifier: Apache-2.0
"""V4.1 DSpark: pure-SWA draft layers and shared target embedding/head."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from copy import copy, deepcopy
from typing import cast

import ixformer.inference.functions as ixfshared
import torch
import torch.nn as nn
from ixformer.inference.functions import deepseekv41 as ixfops  # type: ignore[import]
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

from . import mhc
from .model import IluDeepseekV41DecoderLayer


def remap_dspark_name(name: str) -> str | None:
    match = re.fullmatch(r"mtp\.([0-2])\.(.+)", name)
    if match is None:
        return None
    stage, rest = match.groups()
    if rest.startswith("confidence_head.") or rest.endswith(".bias_vl"):
        return None
    if rest.startswith("markov_head.embed."):
        return "model.markov_head.markov_w1." + rest.removeprefix("markov_head.embed.")
    if rest.startswith("markov_head.head."):
        return "model.markov_head.markov_w2." + rest.removeprefix("markov_head.head.")
    if rest.startswith(("main_proj.", "main_norm.", "norm.", "markov_head.")):
        return f"model.{rest}"
    return f"model.layers.{stage}.{rest}"


logger = init_logger(__name__)

# MoE expert scales land on the MXFP4 MoE method's ``w{1,2,3}_weight_scale``;
# every other block-FP8 linear scale is the ``weight_scale_inv`` companion.
# Same rule as fp8_mtp / fp8_model's ``_make_fp8_weights_mapper``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class IluDSparkV41Model(nn.Module):
    """Iluvatar DeepSeek-V4.1 backbone used by the DSpark draft."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_dspark_layers = config.num_nextn_predict_layers
        if self.num_dspark_layers != 3 or any(
            config.compress_ratios[self.num_hidden_layers :]
        ):
            raise ValueError("V4.1 DSpark expects three pure-SWA draft layers")

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.topk_indices_buffer = torch.full(
            (vllm_config.scheduler_config.max_num_batched_tokens, config.index_topk),
            -1,
            dtype=torch.int32,
            device=current_platform.device_type,
        )

        draft_config = deepcopy(config)
        draft_config.n_routed_experts = config.dspark_n_routed_experts
        draft_config.num_experts_per_tok = config.dspark_num_experts_per_tok
        current_vllm_config = copy(vllm_config)
        current_vllm_config.model_config = copy(vllm_config.model_config)
        current_vllm_config.model_config.hf_config = draft_config
        current_vllm_config.model_config.hf_text_config = draft_config
        self.layers = nn.ModuleList(
            [
                IluDeepseekV41DecoderLayer(
                    current_vllm_config,
                    maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    topk_indices_buffer=self.topk_indices_buffer,
                )
                for i in range(self.num_dspark_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.confidence_head = None
        self._initial_pre_mix = torch.zeros(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.hc_mult,
            dtype=torch.float32,
            device=current_platform.device_type,
        )
        self._initial_pre_mix[:, 0] = 1
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            getattr(config, "draft_vocab_size", None) or config.vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.main_norm(self.main_proj(aux_hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        # Context projection uses the same combined target states in all layers.
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                context_slot_mappings[i]
                if isinstance(context_slot_mappings, list)
                else context_slot_mappings
            )
            attn = layer.attn
            qr_kv, _ = attn.fused_wqa_wkv(main_x)
            qr, kv = qr_kv.split([attn.q_lora_rank, attn.head_dim], dim=-1)
            _, kv = ixfshared.rms_norm_qk(
                qr,
                kv,
                attn.q_norm.weight,
                attn.kv_norm.weight,
                attn.eps,
            )
            if slot_mapping is None:
                continue
            _insert_context_kv(attn, kv, context_positions, slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = (
            inputs_embeds.unsqueeze(-2).expand(-1, self.hc_mult, -1).contiguous()
        )
        pre_mix = self._initial_pre_mix[: hidden_states.shape[0]]
        residual = post_mix = res_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
            )
        hidden_states = mhc.hc_post(hidden_states, residual, post_mix, res_mix)
        return mhc.hc_pre(hidden_states, pre_mix)


def _insert_context_kv(
    attn: nn.Module,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Insert precomputed context KV into one draft layer's SWA cache."""
    swa_kv_cache = attn.swa_cache_layer.kv_cache
    dummy_q = attn._attn_out[: kv.shape[0]]
    dummy_q.zero_()
    if attn.kv_cache_dtype == "fp8_ds_mla":
        ixfops.dsv41_qkv_rope_packed_fp8_insert(
            dummy_q,
            swa_kv_cache,
            kv,
            slot_mapping,
            positions,
            attn.cos_sin,
        )
        return
    if attn.kv_cache_dtype == "fp8_e4m3":
        ixfops.dsv41_qkv_rope_fp8_pt_insert(
            dummy_q,
            swa_kv_cache,
            kv,
            slot_mapping,
            positions,
            attn.cos_sin,
            kv_scale=attn.kv_scale,
            backend="ixformer_native",
        )
        return
    # Plain bf16 window cache, same op the target attention uses.
    ixfops.dsv41_qkv_rope_insert(
        dummy_q,
        swa_kv_cache,
        kv,
        slot_mapping,
        positions,
        attn.cos_sin,
    )


class IluDSparkV41DraftModel(nn.Module):
    """Iluvatar DeepSeek-V4.1 DSpark draft model with shared target heads."""

    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        # Block-FP8 shared experts round the intermediate axis up to a whole
        # number of TP shards, same as fp8_model / fp8_mtp.
        self.pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
            and not vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.model = IluDSparkV41Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            main_x, context_positions, context_slot_mappings
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def _pad_shared_expert_weight(
        self, name: str, loaded_weight: torch.Tensor
    ) -> torch.Tensor:
        """Same pad as ``IluDeepseekV4Fp8Model._pad_shared_expert_weight``."""
        block_size = getattr(self.quant_config, "weight_block_size", None)
        assert block_size is not None
        step = 1 if name.endswith("weight_scale_inv") else block_size[0]
        dim = 1 if ".down_proj." in name else 0
        mult = get_tensor_model_parallel_world_size() * step
        pad = cdiv(loaded_weight.shape[dim], mult) * mult - loaded_weight.shape[dim]
        if pad == 0:
            return loaded_weight
        pad_shape = list(loaded_weight.shape)
        pad_shape[dim] = pad
        return torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=dim)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load only the three MTP stages; target embedding/head are shared."""
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.dspark_n_routed_experts,
        )
        # MXFP4 (expert_dtype=fp4) -> weight_scale; block-FP8 experts would
        # use weight_scale_inv. Matches fp8_mtp / community nvidia MTP.
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        for name, loaded_weight in weights:
            mapped = remap_dspark_name(name)
            if mapped is None:
                continue
            name = mapped

            # Raw-FP8 block scales arrive as ``.scale``; rewrite to the param
            # name the linear / MoE method expects.
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix

            if ".shared_experts.w2" in name:
                name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
            if self.pad_shared_expert and ".shared_experts." in name:
                loaded_weight = self._pad_shared_expert_weight(name, loaded_weight)

            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if not is_layer_param or ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales must be bitcast, not numerically converted.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        weight_loader = cast(Callable[..., bool], param.weight_loader)
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            loaded_params.add(name_mapped)
                            break
                    continue
                if "attn_sink" in name:
                    narrow_weight = loaded_weight[head_start:head_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias",
                        ".ffn.gate.e_score_correction_bias",
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        expected = {
            name
            for name, _ in self.named_parameters()
            if not name.startswith(("model.embed_tokens", "lm_head"))
        }
        missing = expected - loaded_params
        if missing:
            preview = sorted(missing)[:20]
            raise ValueError(
                "[IluDSparkV41DraftModel.load_weights] "
                f"{len(missing)} parameter(s) were not initialized from the "
                "checkpoint. Showing up to 20:\n  " + "\n  ".join(preview)
            )
        logger.info_once(
            "Iluvatar V4.1 DSpark draft model loaded: %d params", len(loaded_params)
        )
        return loaded_params


def register_model() -> None:
    from vllm import ModelRegistry

    for arch in ("DSparkV41DraftModel", "IluDSparkV41DraftModel"):
        ModelRegistry.register_model(
            arch,
            "vllm_iluvatar.models.deepseek_v41.dspark:IluDSparkV41DraftModel",
        )
