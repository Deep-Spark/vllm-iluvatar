# SPDX-License-Identifier: Apache-2.0
"""DSpark draft model for the iluvatar *raw FP8* DeepSeek-V4.

Wired against :mod:`.model` / :mod:`.fp8_attention`, the same way :mod:`.mtp` is.

DSpark drafts a whole block in one parallel (non-causal) pass, then a low-rank
Markov head injects intra-block dependency during sequential sampling. The
proposer-facing interface is ``combine_hidden_states`` / ``forward`` /
``compute_logits`` / ``markov_*`` / ``precompute_and_store_context_kv`` /
``get_draft_kv_cache_layer_names``.

Notable points:

* the backbone layer is :class:`IluDeepseekV4Fp8DecoderLayer`, constructed
  without an ``aux_stream_dict`` (FP8 attention takes no aux streams -- same as
  :mod:`.mtp`);
* the MLA wrapper is reached through :func:`_mla_wrapper` rather than a fixed
  path: ``model._build_attention`` picks the attention stack by
  architecture, and ``IluDeepseekV4Fp8Attention`` (ivcore30) *is* the wrapper
  while ``IluDeepseekV4Attention`` (ivcore11) keeps it at ``.mla_attn``.
  ``precompute_and_store_context_kv`` / ``get_draft_kv_cache_layer_names`` go
  through it, as does :func:`_kv_cache_dtype`;
* ``load_weights`` applies the raw-FP8 ``.scale`` rewrite (``.weight_scale`` for
  E8M0 routed-expert scales, ``.weight_scale_inv`` for every other block-FP8
  scale), the E8M0 ``view(uint8)`` bitcast, and the shared-expert intermediate
  pad -- all as in :mod:`.mtp` -- on top of the DSpark ``mtp.{i}.*`` name remap.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import cast

import ixformer.inference.functions as ixfops  # type: ignore[import]
import torch
import torch.nn as nn
from vllm.config import VllmConfig, get_current_vllm_config
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

from .model import IluDeepseekV4Fp8DecoderLayer

logger = init_logger(__name__)

# MoE expert scales land on the MXFP4 MoE method's ``w{1,2,3}_weight_scale``;
# every other block-FP8 linear scale is the ``weight_scale_inv`` companion.
# Same rule as mtp / model's ``_make_fp8_weights_mapper``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")
_MTP_RE = re.compile(r"mtp\.(\d+)\.(.*)")

# Head-stack params live at model level; everything else is a decoder block.
_HEAD_PREFIXES = (
    "norm.",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
    "markov_head.",
)


def remap_dspark_name(name: str, num_hidden_layers: int) -> str | None:
    """Map a checkpoint ``mtp.{i}.*`` name to this model's parameter path.

    Returns ``None`` for weights owned by the target model, and for the
    confidence head which is not wired into inference.

    Decoder-block weights map to ``model.layers.{i}.*`` (the ModuleList
    index), not ``model.layers.{num_hidden_layers + i}.*``. The latter is
    the MTP-on-target convention; DSpark's draft is a standalone ModuleList
    of length ``n_mtp_layers``, so ``named_parameters()`` uses 0/1/2.
    ``num_hidden_layers`` is accepted for call-site compatibility and is
    unused for the layer remap (constructor prefixes still encode the
    offset for ``extract_layer_index`` / compress_ratio).
    """
    _ = num_hidden_layers  # see docstring; kept for call-site compatibility
    m = _MTP_RE.match(name)
    if m is None:
        return None
    stage = int(m.group(1))
    rest = m.group(2)
    if rest.startswith("confidence_head."):
        return None
    if rest.startswith(("main_proj.", "main_norm.")) or rest.startswith(_HEAD_PREFIXES):
        return f"model.{rest}"
    return f"model.layers.{stage}.{rest}"


class IluDSparkDeepseekV4Fp8Model(nn.Module):
    """Iluvatar FP8 DeepSeek-V4 backbone used by the DSpark draft."""

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
        self.num_dspark_layers = getattr(config, "n_mtp_layers", None) or 3

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

        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=current_platform.device_type,
        )

        current_vllm_config = get_current_vllm_config()
        # FP8 decoder layer: no aux_stream_dict (fp8 attention takes none).
        self.layers = nn.ModuleList(
            [
                IluDeepseekV4Fp8DecoderLayer(
                    current_vllm_config,
                    maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    topk_indices_buffer=self.topk_indices_buffer,
                )
                for i in range(self.num_dspark_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
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
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                context_slot_mappings[i]
                if isinstance(context_slot_mappings, list)
                else context_slot_mappings
            )
            attn = _mla_wrapper(layer.attn)
            qr_kv, _ = attn.fused_wqa_wkv(main_x)
            qr, kv = qr_kv.split([attn.q_lora_rank, attn.head_dim], dim=-1)
            _, kv = ixfops.rms_norm_qk(
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
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, input_ids)
        return ixfops.hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            rms_norm_eps=self.rms_norm_eps,
            hc_eps=self.hc_eps,
        )


def _mla_wrapper(attn: nn.Module) -> nn.Module:
    """The module owning ``swa_cache_layer`` and the pre-attention projections.

    ``model._build_attention`` picks the attention stack by architecture and
    the two put these in different places. :class:`IluDeepseekV4Fp8Attention`
    (ivcore30) is flat and owns them itself; :class:`IluDeepseekV4Attention`
    (ivcore11) keeps them on its inner MLA wrapper, one level down.
    """
    return getattr(attn, "mla_attn", attn)


def _kv_cache_dtype(attn: nn.Module) -> str:
    """``kv_cache_dtype`` of the module ``_mla_wrapper`` returned.

    The flat fp8 attention records it on itself; the ivcore11 wrapper leaves it
    on *its* inner :class:`IluDeepseekV4MLAAttention`.
    """
    dtype = getattr(attn, "kv_cache_dtype", None)
    return dtype if dtype is not None else attn.mla_attn.kv_cache_dtype


def _insert_context_kv(
    attn: nn.Module,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Insert precomputed context KV into one draft layer's SWA cache.

    ``attn`` is what :func:`_mla_wrapper` returned, not ``layer.attn``.
    """
    swa_kv_cache = attn.swa_cache_layer.kv_cache
    block_size = attn.swa_cache_layer.block_size
    dummy_q = torch.zeros(
        kv.shape[0],
        attn.n_local_heads,
        attn.head_dim,
        dtype=kv.dtype,
        device=kv.device,
    )
    if _kv_cache_dtype(attn) == "fp8_ds_mla":
        ixfops.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            dummy_q,
            kv,
            swa_kv_cache,
            slot_mapping,
            positions,
            attn.rotary_emb.cos_sin_cache,
            attn.eps,
            block_size,
        )
    else:
        ixfops.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
            dummy_q,
            kv,
            swa_kv_cache.view(swa_kv_cache.shape[0], -1),
            slot_mapping,
            positions,
            attn.rotary_emb.cos_sin_cache,
            attn.eps,
            block_size,
        )


class IluDSparkDeepseekV4Fp8ForCausalLM(nn.Module):
    """Iluvatar FP8 DeepSeek-V4 DSpark draft model with shared target heads."""

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
        # number of TP shards, same as model / mtp.
        self.pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
            and not vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.model = IluDSparkDeepseekV4Fp8Model(
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

    def _compute_local_logits(
        self, lm_head, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """TP-local logits: skip TP gather, just apply head + optional scale."""
        logits = self.logits_processor._apply_head(lm_head, hidden_states, None)
        if self.logits_processor.soft_cap is not None:
            soft_cap = self.logits_processor.soft_cap
            logits = torch.tanh(logits / soft_cap) * soft_cap
        if self.logits_processor.scale != 1.0:
            logits = logits * self.logits_processor.scale
        return logits

    def compute_draft_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute TP-local draft logits (skip TP all-gather)."""
        hidden_states = self.model.norm(hidden_states)
        return self._compute_local_logits(self.lm_head, hidden_states)

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
        """TP-local Markov bias: only compute rows for this LM-head shard."""
        import torch.nn.functional as F

        markov_w2 = self.model.markov_head.markov_w2
        local_vocab_size = self.lm_head.num_embeddings_per_partition

        if markov_w2.weight.shape[0] == local_vocab_size:
            return self._compute_local_logits(markov_w2, markov_embed)

        indices = self.lm_head.shard_indices
        weight = markov_w2.weight[
            indices.org_vocab_start_index : indices.org_vocab_start_index
            + local_vocab_size
        ]
        logits = F.linear(markov_embed, weight)
        if self.logits_processor.scale != 1.0:
            logits = logits * self.logits_processor.scale
        return logits

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [
            _mla_wrapper(layer.attn).swa_cache_layer.prefix
            for layer in self.model.layers
        ]

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
        """Load the ``mtp.{0,1,2}.*`` weights embedded in the FP8 target checkpoint.

        Combines the DSpark name remap of :mod:`.dspark` with the raw-FP8 scale
        handling of :mod:`.mtp`.
        """
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
            num_experts=self.config.n_routed_experts,
        )
        # MXFP4 (expert_dtype=fp4) -> weight_scale; block-FP8 experts would
        # use weight_scale_inv. Matches mtp / community nvidia MTP.
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        for name, loaded_weight in weights:
            mapped = remap_dspark_name(name, self.config.num_hidden_layers)
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
                "[IluDSparkDeepseekV4Fp8ForCausalLM.load_weights] "
                f"{len(missing)} parameter(s) were not initialized from the "
                "checkpoint. Showing up to 20:\n  " + "\n  ".join(preview)
            )
        logger.info_once(
            "ilu FP8 DSpark draft model loaded: %d params", len(loaded_params)
        )
        return loaded_params
