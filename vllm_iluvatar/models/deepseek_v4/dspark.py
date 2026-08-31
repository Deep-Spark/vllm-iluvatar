# SPDX-License-Identifier: Apache-2.0
"""DSpark draft model for the iluvatar W4A8 DeepSeek-V4.

Ilu counterpart of :mod:`vllm.models.deepseek_v4.nvidia.dspark`. DSpark drafts
a whole block in one parallel (non-causal) pass, then a low-rank Markov head
injects intra-block dependency during sequential sampling.

Differences from the nvidia draft:

* the backbone layer is :class:`IluDeepseekV4DecoderLayer`, whose ``forward``
  returns a single tensor (hc_post is applied inline), so the nvidia 4-tuple
  unpack and explicit ``mhc_post_tilelang`` call are dropped;
* ``hc_head`` uses ``ixfops.ds4_hc_head`` instead of the tilelang kernel;
* the offline converter already writes ``.weight_scale``, so the nvidia
  ``.scale`` rewrite is removed; ilu renames (``shared_experts.w2`` ->
  ``down_proj``, ``ffn.gate.bias`` -> ``e_score_correction_bias``) are applied
  exactly as in :mod:`.mtp`;
* constructor prefixes use ``layers.{num_hidden_layers + i}`` so
  ``extract_layer_index`` yields ``compress_ratio = 1``, but the
  ``nn.ModuleList`` parameter path is still ``model.layers.{i}`` (0/1/2),
  matching community nvidia/amd/xpu DSpark loaders — not the MTP offset.
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
from vllm.utils.multi_stream_utils import AuxStreamType

from .model import IluDeepseekV4DecoderLayer

logger = init_logger("vllm." + __name__)

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


class IluDSparkDeepseekV4Model(nn.Module):
    """Iluvatar DeepSeek-V4 backbone used by the DSpark draft."""

    # Ported from vllm/models/deepseek_v4/nvidia/dspark.py:57-198 at
    # main_6e448d0, with the Iluvatar substitutions documented above.
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
        aux_stream_dict = {AuxStreamType.Attention: torch.cuda.Stream()}

        current_vllm_config = get_current_vllm_config()
        self.layers = nn.ModuleList(
            [
                IluDeepseekV4DecoderLayer(
                    current_vllm_config,
                    maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    topk_indices_buffer=self.topk_indices_buffer,
                    aux_stream_dict=aux_stream_dict,
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
        # Mirror attention.py:788-904 (wrapper.forward + attention_impl):
        # fused_wqa_wkv -> split -> rms_norm_qk, then SWA insert. Using the
        # fused norm (not a standalone kv_norm call) keeps context KV
        # byte-identical to what the target's own forward would write.
        #
        # Some callers pass one shared 1D slot mapping. V2 may pass a
        # per-layer list when draft layers use different cache groups.
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                context_slot_mappings[i]
                if isinstance(context_slot_mappings, list)
                else context_slot_mappings
            )
            # layer.attn is IluDeepseekV4Attention; the wrapper that owns
            # swa_cache_layer / rotary_emb / fused projections is one level
            # down (same object the MTP loader reaches as attn.mla_attn.*).
            attn = layer.attn.mla_attn
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
        return ixfops.ds4_hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            rms_norm_eps=self.rms_norm_eps,
            hc_eps=self.hc_eps,
        )


def _insert_context_kv(
    attn: nn.Module,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Insert precomputed context KV into one draft layer's SWA cache."""
    # Mirrors attention.py:998-1024. Unlike that in-forward path, the proposer
    # supplies slot_mapping explicitly because no attention metadata is bound.
    swa_kv_cache = attn.swa_cache_layer.kv_cache
    block_size = attn.swa_cache_layer.block_size
    dummy_q = torch.zeros(
        kv.shape[0],
        attn.n_local_heads,
        attn.head_dim,
        dtype=kv.dtype,
        device=kv.device,
    )
    if attn.mla_attn.kv_cache_dtype == "fp8_ds_mla":
        ixfops.ds4_fused_qnorm_rope_fp8_kv_insert(
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
        ixfops.ds4_fused_qnorm_rope_kv_insert(
            dummy_q,
            kv,
            swa_kv_cache.view(swa_kv_cache.shape[0], -1),
            slot_mapping,
            positions,
            attn.rotary_emb.cos_sin_cache,
            attn.eps,
            block_size,
        )


class IluDSparkDeepseekV4ForCausalLM(nn.Module):
    """Iluvatar DeepSeek-V4 DSpark draft model with shared target heads."""

    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    # Ported from vllm/models/deepseek_v4/nvidia/dspark.py:268-344 at
    # main_6e448d0, with the Iluvatar backbone and loader below.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.model = IluDSparkDeepseekV4Model(
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
        return [
            layer.attn.mla_attn.swa_cache_layer.prefix for layer in self.model.layers
        ]

    def _remap_dspark_name(self, name: str) -> str | None:
        return remap_dspark_name(name, self.config.num_hidden_layers)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{0,1,2}.*`` weights embedded in the target checkpoint."""
        # Copied from vllm_iluvatar/.../deepseek_v4/mtp.py:284-449, with
        # only DSpark name resolution and strict parameter completeness changed.
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

        for name, loaded_weight in weights:
            mapped = remap_dspark_name(name, self.config.num_hidden_layers)
            if mapped is None:
                continue
            name = mapped

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
                if ".shared_experts.w2" in name:
                    name = name.replace(
                        ".shared_experts.w2", ".shared_experts.down_proj"
                    )
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
                "[IluDSparkDeepseekV4ForCausalLM.load_weights] "
                f"{len(missing)} parameter(s) were not initialized from the "
                "checkpoint. Showing up to 20:\n  " + "\n  ".join(preview)
            )
        logger.info_once("ilu DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params
