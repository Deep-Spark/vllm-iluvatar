# SPDX-License-Identifier: Apache-2.0
"""Text generation on the Flash checkpoint, with stage-local weights and paired block-FP8 expert loading."""

import json
from pathlib import Path

import ixformer.inference.functions as ops
import torch
from torch import nn
from vllm.distributed import get_dp_group, get_pp_group, get_tp_group
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsPP
from vllm.model_executor.models.utils import PPMissingLayer, make_layers, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_iluvatar.models.hy_v4.hc import HYV4HCHeadLayer, HYV4HCLayer

from .attention import FlashAttention
from .engram import FlashEngram
from .engram_lookup import map_flash_table
from .norm import sum_rms_norm
from .quantization import PACKED_LINEAR_PROJECTIONS, FlashFp8Config


class FlashMLP(nn.Module):
    def __init__(self, c, shared, prefix, quant_config):
        super().__init__()
        h = c.hidden_size
        m = (
            c.moe_intermediate_size * c.n_shared_experts
            if shared
            else c.intermediate_size
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            h,
            [m, m],
            bias=False,
            disable_tp=shared,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            m,
            h,
            bias=False,
            disable_tp=shared,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.act = SiluAndMul()
        self.post_fc2_norm = nn.Identity() if shared else RMSNorm(h, c.rms_norm_eps)

    def forward(self, x):
        return self.post_fc2_norm(self.down_proj(self.act(self.gate_up_proj(x))))


class FlashMoE(nn.Module):
    def __init__(self, c, vc, prefix, quant_config):
        super().__init__()
        h, m = c.hidden_size, c.routed_expert_hidden_size
        self.gate = nn.Linear(h, c.n_routed_experts, bias=False, dtype=torch.float32)
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(c.n_routed_experts, dtype=torch.float32)
        )
        self.top = c.num_experts_per_tok
        from vllm_iluvatar.ilu_util.arch import is_ivcore11

        self.single_row_router = (
            getattr(ops, "router_gemm_bf16_fp32_fp32", None) if is_ivcore11() else None
        )
        self.factor = c.routed_scaling_factor
        if c.n_group != 1 or c.topk_group != 1:
            raise ValueError("Unsupported grouped router configuration")
        self.experts = FusedMoEFactory(
            num_experts=c.n_routed_experts,
            top_k=self.top,
            hidden_size=m,
            intermediate_size=c.moe_intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            custom_routing_function=self.route,
            swiglu_limit=c.swiglu_limit,
        )
        self.routed_expert_down_proj = ReplicatedLinear(
            h,
            m,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.routed_expert_down_proj",
        )
        self.routed_expert_up_proj = ReplicatedLinear(
            m,
            h,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.routed_expert_up_proj",
        )
        self.shared_experts = FlashMLP(
            c, True, f"{prefix}.shared_experts", quant_config
        )
        self.combine_output_norm = RMSNorm(h, c.rms_norm_eps)

    def route(self, hidden_states, gating_output, topk, renormalize):
        return ops.moe_fused_topk_bias(
            gating_output,
            topk,
            "sigmoid",
            self.factor,
            renormalize,
            self.gate.e_score_correction_bias,
        )

    def forward(self, x):
        if (
            self.single_row_router is not None
            and x.shape[0] == 1
            and x.is_contiguous()
            and x.dtype == torch.bfloat16
        ):
            logits = self.single_row_router(x, self.gate.weight)
        else:
            logits = self.gate(x.float())
        latent = self.experts(self.routed_expert_down_proj(x), logits)
        out, _ = sum_rms_norm(
            self.routed_expert_up_proj(latent),
            self.shared_experts(x),
            self.combine_output_norm,
        )
        return out


class FlashLayer(nn.Module):
    def __init__(self, c, vc, prefix, quant_config):
        super().__init__()
        i = int(prefix.rsplit(".", 1)[1])
        self.self_attn = FlashAttention(
            c,
            i,
            vc,
            f"{prefix}.self_attn",
            quant_config=quant_config,
            with_history=i == 0,
        )
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.hc_attn_layer = HYV4HCLayer(
            c, i, prefix=f"{prefix}.hc_attn_layer", fused_fp32_projection=True
        )
        self.hc_mlp_layer = HYV4HCLayer(
            c, i, prefix=f"{prefix}.hc_mlp_layer", fused_fp32_projection=True
        )
        self.mlp = (
            FlashMoE(c, vc, f"{prefix}.mlp", quant_config)
            if c.mlp_layer_types[i] == "sparse"
            else FlashMLP(c, False, f"{prefix}.mlp", quant_config)
        )
        self.engram_layer = (
            FlashEngram(c, i) if i + 1 in (c.engram_layer_ids or ()) else None
        )

    def forward(self, positions, hidden, triplets):
        h = self.hc_attn_layer.prepare_input(hidden)
        if self.engram_layer is not None:
            h = self.engram_layer(h, triplets)
        x, post, res = self.hc_attn_layer.pre(h)
        h = self.hc_attn_layer.post(
            self.self_attn(positions, self.input_layernorm(x)), res, post
        )
        x, post, res = self.hc_mlp_layer.pre(h)
        x = self.post_attention_layernorm(x)
        y = self.mlp(x)
        return self.hc_mlp_layer.post(y, res, post)


class HYV4FlashForCausalLM(nn.Module, SupportsPP, SupportsMRoPE):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        c = vllm_config.model_config.hf_text_config
        self.config = c
        c.verify_parallel_config(vllm_config.parallel_config)
        self.quant_config = FlashFp8Config.from_text_config(c)
        pp = get_pp_group()
        self.tp = get_tp_group()
        if self.tp.world_size not in (1, 2):
            raise ValueError("Flash supports TP1 and TP2")
        if self.tp.world_size > 1 and (
            vllm_config.parallel_config.data_parallel_size > 1
            or vllm_config.speculative_config is not None
        ):
            raise ValueError("Flash DP+EP and DFly require TP1")
        self.embed_tokens = (
            VocabParallelEmbedding(c.vocab_size, c.hidden_size)
            if pp.is_first_rank
            else PPMissingLayer()
        )
        self.start, self.end, self.layers = make_layers(
            c.num_hidden_layers,
            lambda prefix: FlashLayer(c, vllm_config, prefix, self.quant_config),
            prefix=maybe_prefix(prefix, "model.layers"),
        )
        if pp.is_last_rank:
            self.hc_head = HYV4HCHeadLayer(
                c,
                c.hidden_size,
                c.hc_mult,
                c.hc_eps,
                prefix=maybe_prefix(prefix, "model.hc_head"),
                fused_fp32_projection=True,
            )
            self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
            self.lm_head = ParallelLMHead(
                c.vocab_size, c.hidden_size, prefix=maybe_prefix(prefix, "lm_head")
            )
        self.logits_processor = LogitsProcessor(c.vocab_size)

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise ValueError("The Flash text runner expects text inputs")
        return torch.arange(len(input_tokens), dtype=torch.long).repeat(3, 1), 0

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors(
            {
                "hidden_states": torch.empty(
                    batch_size,
                    self.config.hidden_size * self.config.hc_mult,
                    dtype=dtype,
                    device=device,
                ),
                "triplets": torch.empty(
                    batch_size, 3, dtype=torch.int32, device=device
                ),
            }
        )

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        pp = get_pp_group()
        if pp.is_first_rank:
            hidden = (
                self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            )
            triplets = self.layers[0].self_attn.cache.triplets(
                input_ids, self.config.engram_pad_id
            )
        else:
            hidden = intermediate_tensors["hidden_states"]
            triplets = intermediate_tensors["triplets"]
        for i in range(self.start, self.end):
            hidden = self.layers[i](positions, hidden, triplets)
        if not pp.is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden.flatten(1), "triplets": triplets}
            )
        return self.norm(self.hc_head(hidden))

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def bind_checkpoint_source(self, directory, index_file):
        self.directory = Path(directory)
        self.weight_map = json.loads((self.directory / index_file).read_text())[
            "weight_map"
        ]

    @torch.no_grad()
    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        packed_parts = {}
        shard_mapping = {
            leaf: (packed, shard)
            for packed, leaves in PACKED_LINEAR_PROJECTIONS.items()
            for shard, leaf in enumerate(leaves)
        }
        for name, value in weights:
            if name.startswith(("vit.", "model.dfly_", "dfly_")):
                continue
            key = name.removeprefix("model.")
            if key.startswith("layers."):
                layer = int(key.split(".")[1])
                if not self.start <= layer < self.end:
                    continue
                if (
                    ".compressed_indexer." in key
                    and self.layers[layer].self_attn.compressed_indexer is None
                ):
                    continue  # max_model_len <= 2048 selects every causal token
                if ".mlp.experts." in key:
                    projection = key.rsplit(".", 1)[1]
                    is_scale = projection.endswith("_scale_inv")
                    source_proj = projection.removesuffix("_scale_inv")
                    if source_proj not in ("gate_up_proj", "down_proj"):
                        raise ValueError(
                            f"Unknown Flash expert projection {projection}"
                        )
                    group = "w2" if source_proj == "down_proj" else "w13"
                    leaf = "weight_scale_inv" if is_scale else "weight"
                    dest_key = key.rsplit(".", 1)[0] + f".routed_experts.{group}_{leaf}"
                    param = params[dest_key]
                    if (
                        value.ndim != 3
                        or value.shape[0] != self.config.n_routed_experts
                    ):
                        raise ValueError(
                            f"Invalid Flash expert tensor {key}: {value.shape}"
                        )
                    # Deferred views only: source interpretation, EP filtering,
                    # TP partitioning and conversion belong to the common loader.
                    for expert in range(value.shape[0]):
                        if group == "w13":
                            if value.shape[1] % 2:
                                raise ValueError(f"Unpaired gate/up rows in {key}")
                            half = value.shape[1] // 2
                            for shard, offset in (("w1", 0), ("w3", half)):
                                view = value[expert].narrow(0, offset, half)
                                param.weight_loader(
                                    param, view, dest_key, shard, expert
                                )
                        else:
                            param.weight_loader(
                                param, value[expert], dest_key, "w2", expert
                            )
                    loaded.add(dest_key)
                    continue
                if key.endswith("engram_layer.lookup_table"):
                    self.layers[layer].engram_layer.load_map(value)
                    continue
                if key.endswith("multi_head_embedding.embedding.weight"):
                    self.layers[layer].engram_layer.owned_table = map_flash_table(
                        self.directory, self.weight_map, layer, get_dp_group()
                    )
                    continue
            if (
                key.startswith(("embed_tokens.", "hc_head.", "norm.", "lm_head."))
                and key not in params
                and key + ".weight" not in params
            ):
                continue
            if key.endswith((".hc_fn", ".hc_head_fn")):
                key += ".weight"
            parent, _, projection = key.removesuffix(".weight").rpartition(".")
            if key.endswith(".weight") and projection in shard_mapping:
                packed, shard = shard_mapping[projection]
                dest = f"{parent}.{packed}.weight"
                params[dest].weight_loader(params[dest], value, shard)
                packed_parts.setdefault(dest, set()).add(shard)
                if len(packed_parts[dest]) == len(PACKED_LINEAR_PROJECTIONS[packed]):
                    loaded.add(dest)
                continue
            if key not in params:
                raise KeyError(f"Unmapped Flash tensor {key}")
            dst = params[key]
            if hasattr(dst, "weight_loader"):
                dst.weight_loader(dst, value)
            else:
                if self.tp.world_size > 1 and ".self_attn." in key:
                    dim = FlashAttention.TP_WEIGHT_DIMS.get(
                        key.split(".self_attn.", 1)[1]
                    )
                    if dim is not None:
                        value = value.chunk(self.tp.world_size, dim=dim)[
                            self.tp.rank_in_group
                        ]
                dst.copy_(value.reshape(dst.shape))
            loaded.add(key)

        missing = params.keys() - loaded
        if missing:
            raise ValueError(f"Missing Flash parameters: {sorted(missing)}")
        for layer in self.layers[self.start : self.end]:
            if layer.engram_layer is not None:
                if (
                    layer.engram_layer.lookup_table is None
                    or layer.engram_layer.owned_table is None
                ):
                    raise ValueError("Incomplete Flash Engram storage")
                layer.engram_layer.finalize()
        return loaded


def register_model():
    from vllm import ModelRegistry

    path = "vllm_iluvatar.models.hy_v4_flash.model:HYV4FlashForCausalLM"
    for name in ("HYV4FlashForCausalLM", "HYV4FlashForConditionalGeneration"):
        ModelRegistry.register_model(name, path)
