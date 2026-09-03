# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP (Multi-Token Prediction) draft model for the iluvatar W4A8 GLM-5.2.

GLM-5.2 is structurally DeepSeek-V3.2, so its MTP draft follows the *standard*
DeepSeek MTP layout (``enorm`` / ``hnorm`` + a single fused ``eh_proj``
(``hidden*2 -> hidden``) + ``shared_head`` + one ``mtp_block`` decoder layer) --
the algorithm mirrors the community
:mod:`vllm.model_executor.models.deepseek_mtp`, NOT DeepSeek-V4's ``hc_head`` /
``e_proj`` / ``h_proj`` variant.

Where it differs from the community ``DeepSeekMTP``:

* ``mtp_block`` is an :class:`IluGlmMoeDsaDecoderLayer` (the self-contained ilu
  MLA + DSA lightning-indexer + fine-grained MoE layer), so the draft reuses the
  exact ilu attention / W8A16 ``kv_b_proj`` / DSA kernels the target uses. The
  community ``DeepseekV2DecoderLayer`` (community attention wrapper) is avoided.
* ``eh_proj`` is a :class:`ReplicatedLinear` bound to ``quant_config`` -- the
  offline converter quantises ``...eh_proj.weight`` to INT8 W8A8 (see
  ``convert_glm52_to_w4a8.py``), so it must be a quant-aware Linear, not the
  community ``nn.Linear``.
* ``set_skip_topk`` / ``compact_topk_indices`` navigate
  ``mtp_block.self_attn.{skip_topk,topk_indices_buffer}`` directly (the ilu
  attention holds these on ``self_attn``, not on a ``.mla_attn`` wrapper), to
  implement ``index_share_for_mtp_iteration``.
* ``load_weights`` mirrors the ilu main-model loader
  (:meth:`IluGlmMoeDsaModel.load_weights`): fused ``fused_qkv_a_proj`` (q_a +
  kv_a) and ``wk_weights_proj`` (indexer wk + weights_proj) at load time, plain
  ``FusedMoE`` expert mapping, and ``kv_b_proj`` W8A16 handled by its quant
  method's ``process_weights_after_loading``. There is no FP8 indexer-wk dequant
  and no post-load q_a/kv_a fusion surgery.
"""

import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.deepseek_v2 import (
    get_spec_layer_idx_from_weight_name,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_iluvatar.models.glm_moe_dsa.model import (
    IluGlmMoeDsaDecoderLayer,
)

logger = init_logger("vllm." + __name__)


class IluGlmMoeDsaMultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str,
    ) -> None:
        super().__init__()

        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # GLM-5.2 uses the fused (V3-style) embed+hidden projection; the offline
        # converter quantises it to INT8 W8A8, so it must carry ``quant_config``.
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.eh_proj",
        )
        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        # The ilu self-contained decoder layer (MLA + DSA indexer + MoE). Passing
        # the layer prefix (ending in the numeric MTP layer index) keeps
        # ``int(prefix.split('.')[-1])`` == the MTP layer id (>= num_hidden_layers).
        self.mtp_block = IluGlmMoeDsaDecoderLayer(
            vllm_config=vllm_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
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
        hidden_states = residual + hidden_states  # pre-final-norm (logits hidden)
        # Return (logits_hidden, recycle_hidden): the first (pre-final-norm) is
        # fed to compute_logits (which applies shared_head == one final norm);
        # the second (post-final-norm) is recycled as the next spec step's
        # ``previous_hidden_states``. Matches community deepseek_mtp / SGLang.
        return hidden_states, self.shared_head(hidden_states)


class IluGlmMoeDsaMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.device = current_platform.device_type

        # GLM-5.2 is always sparse (V3.2 DSA indexer): the MTP layer's ``full``
        # indexer writes this buffer at draft step 0; steps 1+ reuse it
        # (index_share_for_mtp_iteration) via set_skip_topk / compact_topk_indices.
        topk_tokens = config.index_topk
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Map the exact layer index from weights.
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): IluGlmMoeDsaMultiTokenPredictorLayer(
                    vllm_config,
                    self.topk_indices_buffer,
                    f"{prefix}.layers.{idx}",
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
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def set_skip_topk(self, skip: bool):
        """Toggle skip_topk on all MTP layers (index_share_for_mtp_iteration).

        Draft step 0 sets skip=False so the MTP layer computes its own top-k
        indices; steps 1+ set skip=True to reuse the indices written into the
        shared buffer at step 0. The ilu attention holds ``skip_topk`` directly
        on ``self_attn`` (no ``.mla_attn`` wrapper).
        """
        for layer in self.layers.values():
            mtp_block = getattr(layer, "mtp_block", None)
            if mtp_block is None:
                continue
            self_attn = getattr(mtp_block, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "skip_topk"):
                self_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor):
        """Gather the top-k index rows at ``slot_ids`` to the front of the buffer."""
        num_slots = slot_ids.numel()
        for layer in self.layers.values():
            mtp_block = getattr(layer, "mtp_block", None)
            if mtp_block is None:
                continue
            self_attn = getattr(mtp_block, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "topk_indices_buffer"):
                topk_indices_buffer = self_attn.topk_indices_buffer
                if topk_indices_buffer is not None:
                    topk_indices_buffer[:num_slots] = topk_indices_buffer[slot_ids]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        # GLM standard MTP: apply shared_head (== final norm) then the head. No
        # V4 hc_head fold.
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits


class IluGlmMoeDsaMTP(nn.Module, SupportsPP):
    """Top-level ilu GLM-5.2 W4A8 MTP draft model (``IluGlmMoeDsaMTPModel``)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = IluGlmMoeDsaMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Mirrors IluGlmMoeDsaModel.load_weights (fused MLA q_a+kv_a, fused
        # indexer wk+weights_proj, plain FusedMoE experts), wrapped with the
        # community MTP spec-layer filter + name rewrite.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]

        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Only ``full`` layers build an indexer; drop indexer weights whose layer
        # has none built (defensive -- MTP layers force skip_topk=False so their
        # indexer is present).
        indexer_present_prefixes = {
            n.rsplit(".indexer.", 1)[0] for n in params_dict if ".indexer." in n
        }

        # GLM-5.2 ties the MTP ``shared_head.head`` to the main model's
        # ``lm_head`` (the checkpoint ships ``lm_head.weight`` but NO per-layer
        # ``shared_head.head.weight``). Stash it here and tie after the loop.
        lm_head_weight: torch.Tensor | None = None
        # GLM-5.2 also ties the MTP ``embed_tokens`` to the main model's
        # embedding (the checkpoint ships ``model.embed_tokens.weight`` but NO
        # per-MTP embedding). Under PP>1 the proposer's runtime embedding-share
        # (``_maybe_share_embeddings``, world_size==1 only) is skipped, so the
        # MTP embed_tokens would otherwise stay uninitialized (NaN) and produce
        # garbage draft tokens. Stash it here and tie after the loop.
        embed_tokens_weight: torch.Tensor | None = None

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Top-level lm_head (not under a spec layer) -> stash for the tie.
            if name == "lm_head.weight" or name.endswith(".lm_head.weight"):
                lm_head_weight = loaded_weight
                continue

            # Top-level main-model embedding (not under a spec layer) -> stash
            # for the tie. The per-MTP embed (``model.layers.{i}.embed_tokens``)
            # carries a ``.layers.`` segment and is handled by the normal path.
            if name == "model.embed_tokens.weight" or name.endswith(
                ".model.embed_tokens.weight"
            ):
                embed_tokens_weight = loaded_weight
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)

            if ".indexer." in name and (
                name.rsplit(".indexer.", 1)[0] not in indexer_present_prefixes
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Skip experts (handled below via expert_params_mapping) so we
                # don't mangle ``mlp.experts.N.gate_proj`` into gate_up_proj.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped.endswith(".bias") and name_mapped not in params_dict:
                    continue
                if name_mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                param = params_dict[name_mapped]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Per DeepSeek-V3, MTP modules share the embedding layer; only
                    # the first spec layer's shared (non ``.layers``) weights load.
                    if (
                        spec_layer != self.model.mtp_start_layer_idx
                        and ".layers" not in name
                    ):
                        continue
                    if name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        # Tie the MTP shared_head.head to the main lm_head if the checkpoint did
        # not ship a per-layer head (GLM-5.2 ties them; only lm_head.weight and
        # shared_head.norm.weight are present in the checkpoint).
        for param_name, param in params_dict.items():
            if not param_name.endswith(".shared_head.head.weight"):
                continue
            if param_name in loaded_params:
                continue
            if lm_head_weight is None:
                continue
            if is_pp_missing_parameter(param_name, self):
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, lm_head_weight)
            loaded_params.add(param_name)
            logger.info_once(
                "ilu GLM-5.2 MTP: tied %s to the target lm_head.weight",
                param_name,
            )

        # Tie the MTP embed_tokens to the main-model embedding when the
        # checkpoint has no separate MTP embed. Required under PP>1 where the
        # proposer's runtime embedding-share is skipped; otherwise MTP
        # embed_tokens stays uninitialized (NaN) -> garbage draft -> 0% accept.
        for param_name, param in params_dict.items():
            if not param_name.endswith("model.embed_tokens.weight"):
                continue
            if param_name in loaded_params:
                continue
            if embed_tokens_weight is None:
                continue
            if is_pp_missing_parameter(param_name, self):
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, embed_tokens_weight)
            loaded_params.add(param_name)
            logger.info_once(
                "ilu GLM-5.2 MTP: tied %s to the target model.embed_tokens.weight",
                param_name,
            )

        # Validate that weights were loaded for each expected MTP layer.
        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if spec_layer is not None:
                loaded_layers.add(spec_layer)
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers:
                raise ValueError(
                    f"MTP speculative decoding layer {layer_idx} weights "
                    f"missing from checkpoint. The checkpoint may have been "
                    f"quantized without including the MTP layers. Use a "
                    f"checkpoint that includes MTP layer weights, or disable "
                    f"speculative decoding."
                )

        not_loaded = set(params_dict.keys()) - loaded_params
        if not_loaded:
            logger.warning(
                "ilu GLM-5.2 MTP: %d parameter(s) were NOT loaded from the "
                "checkpoint:\n%s",
                len(not_loaded),
                "\n".join(sorted(not_loaded)),
            )
        logger.info("ilu GLM-5.2 MTP draft loaded: %d params", len(loaded_params))
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """Rewrite the weight name to match the module tree.

        Add ``.mtp_block`` for transformer-block modules of the spec layer, and
        hoist shared weights (embed_tokens) to top level.
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for the transformer layer block
            name = name.replace(
                f"model.layers.{spec_layer}.",
                f"model.layers.{spec_layer}.mtp_block.",
            )
        elif shared_weight:
            # treat shared weights as top level weights
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name
