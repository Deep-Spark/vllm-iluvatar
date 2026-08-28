# SPDX-License-Identifier: Apache-2.0
"""MTP (Multi-Token Prediction) draft model for the iluvatar *raw FP8* DeepSeek-V4.

Iluvatar counterpart of community ``vllm.models.deepseek_v4.nvidia.mtp``, wired
against the FP8 fork (``fp8_model`` / ``fp8_attention``) rather than the W4A8
``mtp.py``:

* ``mtp_block`` is an :class:`IluDeepseekV4Fp8DecoderLayer` whose ``forward``
  returns a single ``(T, hc_mult, H)`` tensor (``hc_post`` applied inline), so
  the nvidia 4-tuple unpack + explicit post is dropped -- same contract as
  W4A8 ``mtp.py``.
* ``hc_head`` uses ``ixfops.ds4_hc_head``.
* No aux streams (FP8 attention does not take ``aux_stream_dict``).
* Raw-FP8 scale rewrite (``.scale`` -> ``weight_scale`` / ``weight_scale_inv``)
  and E8M0 ``view(uint8)`` for routed-expert scales match ``fp8_model`` /
  community nvidia MTP -- the W4A8 offline converter already wrote
  ``weight_scale``, so W4A8 ``mtp.py`` skips both.
* No ``.attn.compressor.`` -> ``.attn.mla_attn.compressor.`` rename: FP8 keeps
  the compressor under ``attn.compressor``.  The Flash MTP layer itself is
  dense (``compress_ratio == 1``), so compressor weights are absent anyway.
* Registered as community ``DeepSeekV4MTPModel`` (see ``fp8_model.register_model``).
"""

from __future__ import annotations

import re
import typing
from collections.abc import Callable, Iterable

import ixformer.inference.functions as ixfops  # type: ignore[import]
import torch
import torch.nn as nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.deepseek_v2 import (
    get_spec_layer_idx_from_weight_name,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv

from .fp8_model import IluDeepseekV4Fp8DecoderLayer

logger = init_logger("vllm." + __name__)

# MoE expert scales land on Mxfp4MoEMethod's ``w{1,2,3}_weight_scale``; every
# other FP8 linear scale is Fp8LinearMethod's ``weight_scale_inv``.  Same rule
# as ``_make_fp8_weights_mapper`` / community nvidia MTP.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class IluDeepseekV4Fp8MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
    ) -> None:
        super().__init__()

        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        self.rms_norm_eps = config.rms_norm_eps

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # V4 keeps e_/h_proj separate. Raw-FP8 checkpoints store both as
        # block-FP8 (e_proj/h_proj.weight + .scale); quant_config decides.
        self.e_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.e_proj",
        )
        self.h_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.h_proj",
        )

        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )

        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        self.mtp_block = IluDeepseekV4Fp8DecoderLayer(
            vllm_config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)

        # Target stashes pre-hc_head residual as flat (T, hc_mult * D); reshape
        # to (T, hc_mult, D) -- the layout the ilu decoder layer operates on.
        previous_hidden_states = previous_hidden_states.view(
            -1, self.hc_mult, self.config.hidden_size
        )
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.h_proj(previous_hidden_states) + self.e_proj(
            inputs_embeds
        ).unsqueeze(-2)
        # The FP8 decoder layer applies hc_post inline and returns a single
        # (T, hc_mult, D) tensor -- no nvidia-style 4-tuple / explicit hc_post.
        hidden_states = self.mtp_block(hidden_states, positions, None)
        # Return the flat pre-hc_head residual so it can be re-fed as the next
        # spec step's `previous_hidden_states`. hc_head is deferred to
        # compute_logits.
        return hidden_states.flatten(1)


class IluDeepseekV4Fp8MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.device = current_platform.device_type

        topk_tokens = config.index_topk
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        self.layers = torch.nn.ModuleDict(
            {
                str(idx): IluDeepseekV4Fp8MultiTokenPredictorLayer(
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
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
        # MTP forward returns the pre-hc_head residual (T, hc_mult * D); fold it
        # back to the dense hidden state with the ilu hc_head kernel before
        # computing logits.
        hidden_states = hidden_states.view(
            -1, mtp_layer.hc_mult, mtp_layer.config.hidden_size
        )
        hidden_states = ixfops.ds4_hc_head(
            hidden_states,
            mtp_layer.hc_head_fn,
            mtp_layer.hc_head_scale,
            mtp_layer.hc_head_base,
            rms_norm_eps=mtp_layer.rms_norm_eps,
            hc_eps=mtp_layer.hc_eps,
        )
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "hidden_states": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class IluDeepseekV4Fp8MTP(nn.Module, SupportsPP):
    """FP8 MTP draft model.

    ``SupportsPP`` only gates ``ModelConfig.verify_with_parallel_config``: vLLM
    places the *whole* drafter on the last PP rank
    (``gpu_model_runner``: ``if self.speculative_config and
    get_pp_group().is_last_rank``), so this model is never split across stages
    and never exchanges PP intermediate tensors.

    The one real PP consequence is the token embedding: the proposer only
    aliases the target's ``embed_tokens`` into the draft when
    ``get_pp_group().world_size == 1`` (``llm_base_proposer``), and on the last
    PP rank the target's own ``embed_tokens`` is a ``PPMissingLayer`` anyway.
    With PP > 1 the draft therefore loads the checkpoint's top-level
    ``embed.weight`` itself (see ``load_weights``).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config

        self.load_own_embed_tokens = get_pp_group().world_size > 1

        self.pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
            and not vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.model = IluDeepseekV4Fp8MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

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
        # Weight name remapping for checkpoint compatibility.
        WEIGHT_NAME_REMAPPING: dict[str, str] = {
            ".emb.tok_emb.weight": ".embed_tokens.weight",
            ".head.weight": ".shared_head.head.weight",
            ".norm.weight": ".shared_head.norm.weight",
        }

        def _remap_weight_name(name: str) -> str:
            for old_pattern, new_pattern in WEIGHT_NAME_REMAPPING.items():
                if old_pattern in name:
                    name = name.replace(old_pattern, new_pattern)
            return name

        def _find_mtp_layer_idx(name: str) -> int:
            subnames = name.split(".")
            for subname in subnames:
                try:
                    return int(subname)
                except ValueError:
                    continue
            return 0

        # Same stacked fusion as the FP8 target. Compressor entries are
        # no-ops on the Flash MTP layer (dense, no compressor tensors).
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

        # MXFP4 (expert_dtype=fp4) -> weight_scale; block-FP8 experts would
        # use weight_scale_inv. Matches community nvidia MTP.
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        for name, loaded_weight in weights:
            # Target-side top-level token embedding. Only needed when the
            # proposer will not alias the target's embedding into the draft
            # (PP > 1); otherwise it is skipped like every other target tensor.
            if name == "embed.weight":
                if not self.load_own_embed_tokens:
                    continue
                param = params_dict["model.embed_tokens.weight"]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add("model.embed_tokens.weight")
                continue

            mtp_layer_idx = _find_mtp_layer_idx(name)
            # V4 checkpoints store MTP weights as `mtp.{i}.*`; remap to
            # `model.layers.{num_hidden_layers + i}.*` so that
            # get_spec_layer_idx_from_weight_name can identify them.
            name = name.replace(
                f"mtp.{mtp_layer_idx}.",
                f"model.layers.{self.config.num_hidden_layers + mtp_layer_idx}.",
            )

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue

            name = _remap_weight_name(name)
            name = self._rewrite_spec_layer_name(spec_layer, name)

            if spec_layer != self.model.mtp_start_layer_idx and ".layers" not in name:
                continue

            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix

            if ".shared_experts.w2" in name:
                name = name.replace(
                    ".shared_experts.w2", ".shared_experts.down_proj"
                )
            if self.pad_shared_expert and ".shared_experts." in name:
                loaded_weight = self._pad_shared_expert_weight(name, loaded_weight)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
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
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            loaded_params.add(name_mapped)
                            break
                    continue
                elif "attn_sink" in name:
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if name.endswith(".ffn.gate.bias"):
                        name = name.replace(
                            ".ffn.gate.bias",
                            ".ffn.gate.e_score_correction_bias",
                        )
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

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
                    f"missing from checkpoint. The checkpoint may have "
                    f"been quantized without including the MTP layers. "
                    f"Use a checkpoint that includes MTP layer weights, "
                    f"or disable speculative decoding."
                )

        # Strict reverse check: every parameter must have received a weight.
        # Exclude tensors the proposer replaces post-load via sharing; under
        # PP > 1 embed_tokens is not shared and must have been loaded above.
        _SHARE_EXCLUDED = {"shared_head.head"}
        if not self.load_own_embed_tokens:
            _SHARE_EXCLUDED.add("embed_tokens")
        all_params = set(dict(self.named_parameters()).keys())
        unfilled = sorted(
            name for name in all_params - loaded_params
            if not any(excl in name for excl in _SHARE_EXCLUDED)
        )
        if unfilled:
            raise ValueError(
                f"FP8 MTP: {len(unfilled)} parameter(s) were never filled "
                f"by load_weights. First 20: {unfilled[:20]}"
            )

        logger.info_once(
            "ilu FP8 MTP draft model loaded: %d params", len(loaded_params)
        )
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        and rename shared layer weights to be top level.
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "h_proj",
            "e_proj",
            "shared_head",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
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
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        elif shared_weight:
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name
