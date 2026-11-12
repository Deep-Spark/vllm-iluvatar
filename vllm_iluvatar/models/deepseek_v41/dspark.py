# SPDX-License-Identifier: Apache-2.0
"""V4.1 DSpark: pure-SWA draft layers and shared target embedding/head."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from copy import copy, deepcopy
from typing import cast

import ixformer.inference.functions as ixfshared
import torch
import torch.nn as nn
import torch.nn.functional as F
from ixformer.inference.functions.models import (
    deepseek_v41 as ixfops,  # type: ignore[import]
)
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_pp_group
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


def _local_vocab_weight(
    weight: torch.Tensor, vocab_start: int, local_vocab_size: int
) -> torch.Tensor:
    """Return this TP rank's rows from a local or replicated vocab weight."""
    if weight.shape[0] == local_vocab_size:
        return weight
    vocab_end = vocab_start + local_vocab_size
    if vocab_start < 0 or vocab_end > weight.shape[0]:
        raise ValueError(
            "Cannot select local vocabulary rows "
            f"[{vocab_start}:{vocab_end}] from weight shape {tuple(weight.shape)}."
        )
    return weight.narrow(0, vocab_start, local_vocab_size)


def _make_vocab_layer(
    factory: Callable[..., nn.Module],
    vocab_size: int,
    hidden_size: int,
    *,
    prefix: str,
    owns_weight: bool,
) -> nn.Module:
    """Keep sharing metadata without allocating a throwaway vocabulary weight."""
    with nullcontext() if owns_weight else torch.device("meta"):
        return factory(vocab_size, hidden_size, prefix=prefix)


def remap_dspark_name(name: str, load_own_embed: bool = False) -> str | None:
    """Map a checkpoint name onto this draft.

    Under pipeline parallelism the drafting rank has no target embedding, so
    ``load_own_embed=True`` claims the checkpoint's top-level ``embed.weight``.
    """
    if load_own_embed and name == "embed.weight":
        return "model.embed_tokens.weight"
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

    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", owns_embedding: bool = True
    ) -> None:
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

        self.embed_tokens = _make_vocab_layer(
            VocabParallelEmbedding,
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
            owns_weight=owns_embedding,
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
        draft_config.vision_n_layers = 0
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
        quant_x: tuple[torch.Tensor, torch.Tensor, torch.dtype] | None = None
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                context_slot_mappings[i]
                if isinstance(context_slot_mappings, list)
                else context_slot_mappings
            )
            if slot_mapping is None:
                continue
            attn = layer.attn
            w_q = getattr(attn.fused_wqa_wkv, "weight", None)
            w_s = getattr(attn.fused_wqa_wkv, "weight_scale", None)
            if (
                w_q is not None
                and w_s is not None
                and w_q.dtype == torch.int8
                and getattr(attn.fused_wqa_wkv, "input_scale", None) is None
                and getattr(attn.fused_wqa_wkv, "bias", None) is None
                and main_x.ndim == 2
            ):
                if quant_x is None:
                    x_q, x_s, _ = ops.scaled_int8_quant(
                        main_x.contiguous(), None, None, symmetric=True
                    )
                    quant_x = (x_q, x_s, main_x.dtype)
                wkv_w = getattr(attn, "_context_wkv_weight", None)
                if wkv_w is None:
                    if w_q.shape == (
                        attn.hidden_size,
                        attn.q_lora_rank + attn.head_dim,
                    ):
                        wkv_w = w_q[:, attn.q_lora_rank :].contiguous()
                        wkv_fmt = "NN"
                    else:
                        wkv_w = w_q[attn.q_lora_rank :, :].contiguous()
                        wkv_fmt = "TN"
                    wkv_s = w_s[attn.q_lora_rank :].contiguous()
                    attn._context_wkv_weight = wkv_w
                    attn._context_wkv_scale = wkv_s
                    attn._context_wkv_fmt = wkv_fmt
                kv = torch.empty(
                    (main_x.shape[0], attn.head_dim),
                    dtype=main_x.dtype,
                    device=main_x.device,
                )
                ixfshared.w8a8(
                    quant_x[0],
                    attn._context_wkv_weight,
                    quant_x[1],
                    attn._context_wkv_scale,
                    None,
                    format=attn._context_wkv_fmt,
                    output=kv,
                    out_dtype=main_x.dtype,
                )
                kv = attn.kv_norm(kv)
            else:
                qr_kv, _ = attn.fused_wqa_wkv(main_x)
                qr, kv = qr_kv.split([attn.q_lora_rank, attn.head_dim], dim=-1)
                _, kv = ixfshared.rms_norm_qk(
                    qr,
                    kv,
                    attn.q_norm.weight,
                    attn.kv_norm.weight,
                    attn.eps,
                )
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
        ixfops.qkv_rope_packed_fp8_insert(
            dummy_q,
            swa_kv_cache,
            kv,
            slot_mapping,
            positions,
            attn.cos_sin,
        )
        return
    if attn.kv_cache_dtype == "fp8_e4m3":
        ixfops.qkv_rope_fp8_pt_insert(
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
    ixfops.qkv_rope_insert(
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
    prefer_full_gather_for_small_batches = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Same Hybrid-EP budget as the target; draft MoE may init its own NVSM
        # communicator (different topk / expert count) during CUDA graph capture.
        from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
            _decode_max_tokens_per_rank_from_config,
            set_max_num_seqs,
        )

        set_max_num_seqs(_decode_max_tokens_per_rank_from_config(vllm_config))
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        # The drafter is built with pipeline_parallel_size=1, but it only runs
        # on the last PP rank. That rank's target embedding is a PPMissingLayer,
        # so the draft has to keep the checkpoint embedding instead of aliasing.
        self.load_own_embed_tokens = get_pp_group().world_size > 1
        if self.load_own_embed_tokens:
            self.has_own_embed_tokens = True
        # Block-FP8 shared experts round the intermediate axis up to a whole
        # number of TP shards, same as fp8_model / fp8_mtp.
        self.pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
            and not vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.model = IluDSparkV41Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            owns_embedding=self.load_own_embed_tokens or self.has_own_embed_tokens,
        )
        self.lm_head = _make_vocab_layer(
            ParallelLMHead,
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
            owns_weight=self.has_own_lm_head,
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

    def _compute_local_logits(
        self, lm_head: ParallelLMHead, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        logits = self.logits_processor._apply_head(lm_head, hidden_states, None)
        if self.logits_processor.soft_cap is not None:
            soft_cap = self.logits_processor.soft_cap
            logits = torch.tanh(logits / soft_cap) * soft_cap
        if self.logits_processor.scale != 1.0:
            logits = logits * self.logits_processor.scale
        return logits

    def compute_draft_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.norm(hidden_states)
        return self._compute_local_logits(self.lm_head, hidden_states)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(
            self.lm_head, self.model.norm(hidden_states)
        )

    def get_draft_vocab_shard(self) -> tuple[int, int]:
        indices = self.lm_head.shard_indices
        return indices.org_vocab_start_index, indices.num_org_vocab_padding

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def markov_bias_local(self, markov_embed: torch.Tensor) -> torch.Tensor:
        markov_w2 = self.model.markov_head.markov_w2
        indices = self.lm_head.shard_indices
        local_vocab_size = self.lm_head.num_embeddings_per_partition

        if markov_w2.weight.shape[
            0
        ] == local_vocab_size and self.logits_processor.head_dtype in (
            None,
            markov_embed.dtype,
        ):
            return self._compute_local_logits(markov_w2, markov_embed)

        weight = _local_vocab_weight(
            markov_w2.weight,
            indices.org_vocab_start_index,
            local_vocab_size,
        )
        head_dtype = self.logits_processor.head_dtype
        if head_dtype is None or head_dtype == markov_embed.dtype:
            logits = F.linear(markov_embed, weight)
        elif (
            head_dtype == torch.float32
            and (current_platform.is_cuda() or current_platform.is_rocm())
            and markov_embed.is_cuda
        ):
            flat = markov_embed.reshape(-1, markov_embed.shape[-1])
            logits = ixfshared.gemm_bf16_bf16_fp32(flat, weight, format="TN")
            logits = logits.reshape(*markov_embed.shape[:-1], -1)
        else:
            logits = F.linear(
                markov_embed.to(head_dtype),
                weight.to(head_dtype),
            )
        if self.logits_processor.soft_cap is not None:
            soft_cap = self.logits_processor.soft_cap
            logits = torch.tanh(logits / soft_cap) * soft_cap
        if self.logits_processor.scale != 1.0:
            logits = logits * self.logits_processor.scale
        return logits

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        """DSpark query block is non-causal SWA on every draft layer."""
        return [False] * len(self.model.layers)

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
            mapped = remap_dspark_name(name, load_own_embed=self.load_own_embed_tokens)
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

        # embed_tokens / lm_head are normally aliased from the target. Under PP
        # the draft owns its embedding, and a silently unloaded one produces
        # garbage drafts, so it must be covered by this check.
        aliased_prefixes = (
            ("lm_head",)
            if self.load_own_embed_tokens
            else ("model.embed_tokens", "lm_head")
        )
        expected = {
            name
            for name, _ in self.named_parameters()
            if not name.startswith(aliased_prefixes)
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
