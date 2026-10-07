# SPDX-License-Identifier: Apache-2.0
"""DSpark draft model for the iluvatar DeepSeek-V4.1.

Ilu counterpart of :mod:`vllm.models.deepseek_v41.nvidia.dspark`, and the V4.1
sibling of :mod:`vllm_iluvatar.models.deepseek_v4.dspark`. DSpark drafts a whole
block in one parallel (non-causal) pass, then a low-rank Markov head injects
intra-block dependency during sequential sampling.

Differences from the V4 ilu draft, all forced by V4.1's decoder layer:

* the layer returns the 5-tuple ``(x, residual, post_mix, res_mix, pre_mix)``
  and defers its FFN ``hc_post``, so ``forward`` has to run the same MHC chain
  :class:`~vllm_iluvatar.models.deepseek_v41.model.IluDeepseekV41Model` does
  rather than treating the layer as returning a folded hidden state;
* there is no global MHC head in the V4.1 checkpoint (no ``hc_head_*`` under
  ``mtp.*``), so the collapse is ``hc_post`` then ``hc_pre`` instead of
  ``ds4_hc_head``;
* the shared heads sit on different stages -- ``main_proj``/``main_norm`` on
  ``mtp.0``, ``norm``/``markov_head``/``confidence_head`` on ``mtp.2`` -- which
  the remap ignores, so only the decoder blocks are stage-indexed;
* the draft MoE is ``dspark_n_routed_experts`` (128) wide, not the backbone's
  384, so the layers are built against a config that carries the DSpark counts;
* context KV insert is ``kv_norm`` then the same window-cache op the target
  attention uses (``v41_qkv_rope_fp8_insert`` for ``fp8_ds_mla``, otherwise
  ``v41_qkv_rope_insert``), and reaches ``layer.attn`` directly because V4.1
  has no ``mla_attn`` wrapper.

Constructor prefixes use ``layers.{num_hidden_layers + i}`` so that
``extract_layer_index`` yields 40/41/42 and therefore ``compress_ratios[40+i]
== 0`` -- the V4.1 config ships 43 ratios for 40 layers precisely so the three
draft layers read as pure-SWA, with no compressor or indexer. The
``nn.ModuleList`` parameter path is still ``model.layers.{0,1,2}``, matching the
community DSpark loaders.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable
from typing import cast

import ixformer.inference.functions as ixfops  # type: ignore[import]
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig, get_current_vllm_config
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
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform

from vllm_iluvatar.models.deepseek_v41 import mhc
from vllm_iluvatar.models.deepseek_v41.model import IluDeepseekV41DecoderLayer

logger = init_logger("vllm." + __name__)

_MTP_RE = re.compile(r"mtp\.(\d+)\.(.*)")

# Owned by the model, not by a decoder block. Which ``mtp.{i}`` they ship under
# is not meaningful (0 for main_*, 2 for the rest), so the stage is dropped.
_HEAD_PREFIXES = (
    "norm.",
    "markov_head.",
    "main_proj.",
    "main_norm.",
    "confidence_head.",
)

# ``DSparkMarkovHead`` names its two tensors after their role in the low-rank
# product, the checkpoint after their shape.
_MARKOV_RENAME = {
    "markov_head.embed.": "markov_head.markov_w1.",
    "markov_head.head.": "markov_head.markov_w2.",
}

# Same split the target's FP8 mapper makes, minus its prefix/suffix rules: those
# rewrite ``mtp.`` and ``head.weight``, which would mangle both the stage prefix
# and ``markov_head.head.weight``.
_EXPERT_SCALE_RE = re.compile(r"(\.experts\.\d+\.w[123])\.scale$")


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


def remap_dspark_name(
    name: str,
    num_hidden_layers: int,
    load_own_embed: bool = False,
    load_confidence_head: bool = False,
) -> str | None:
    """Map a checkpoint ``mtp.{i}.*`` name to this model's parameter path.

    Returns ``None`` for weights this model does not own: anything outside
    ``mtp.*`` and the vision-language gate bias (V4.1 runs text-only).

    The confidence head is only read under adaptive verification, and the model
    only builds the module in that case, so pass ``load_confidence_head=True``
    to claim ``mtp.2.confidence_head.*``. Claiming it unconditionally would hand
    the loader a weight with no parameter to put it in.

    Under pipeline parallelism the target's ``embed_tokens`` is a
    ``PPMissingLayer`` on the drafting rank, so the draft cannot alias it and
    must own its embedding. Pass ``load_own_embed=True`` to additionally claim
    the checkpoint's top-level ``embed.weight``.

    Decoder-block weights map to ``model.layers.{i}.*`` -- the ModuleList index,
    not the ``num_hidden_layers + i`` the constructor prefix uses. That offset
    only exists so ``extract_layer_index`` can find ``compress_ratio``; the
    ModuleList itself is three layers long. ``num_hidden_layers`` is accepted for
    call-site symmetry with the V4 draft and is unused.
    """
    _ = num_hidden_layers  # see docstring
    if load_own_embed and name == "embed.weight":
        return "model.embed_tokens.weight"
    m = _MTP_RE.match(name)
    if m is None:
        return None
    stage = int(m.group(1))
    rest = m.group(2)

    if rest.endswith(".bias_vl"):
        return None
    if rest.startswith("confidence_head.") and not load_confidence_head:
        return None

    for old, new in _MARKOV_RENAME.items():
        if rest.startswith(old):
            rest = new + rest[len(old) :]
            break

    # Block scales: routed experts land on the MXFP4 method and register
    # ``w{1,2,3}_weight_scale``; every other FP8 scale is ``weight_scale_inv``.
    # Order matters -- after the experts rule the name ends in ``_scale``, so the
    # generic rule correctly misses it.
    if _EXPERT_SCALE_RE.search(rest):
        rest = _EXPERT_SCALE_RE.sub(r"\1.weight_scale", rest)
    elif rest.endswith(".scale"):
        rest = rest[: -len(".scale")] + ".weight_scale_inv"

    if rest.startswith(_HEAD_PREFIXES):
        return f"model.{rest}"
    return f"model.layers.{stage}.{rest}"


def _draft_vllm_config(vllm_config: VllmConfig) -> VllmConfig:
    """A ``VllmConfig`` whose ``model_config`` is the draft's.

    ``IluDeepseekV41DecoderLayer`` reads everything off
    ``vllm_config.model_config.hf_config``, so pointing it at the target's config
    would build 384-expert MoEs for a draft whose checkpoint holds 128. The copy
    is shallow on purpose: quant, cache, and compilation config are meant to be
    the target's.
    """
    draft_model_config = vllm_config.speculative_config.draft_model_config
    hf_config = draft_model_config.hf_config
    for field, dspark_field in (
        ("n_routed_experts", "dspark_n_routed_experts"),
        ("num_experts_per_tok", "dspark_num_experts_per_tok"),
    ):
        value = getattr(hf_config, dspark_field, None)
        if value is not None:
            setattr(hf_config, field, value)

    draft = copy.copy(get_current_vllm_config())
    draft.model_config = draft_model_config
    return draft


class IluDSparkDeepseekV41Model(nn.Module):
    """Iluvatar DeepSeek-V4.1 backbone used by the DSpark draft."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_dspark_layers = (
            getattr(config, "n_mtp_layers", None)
            or getattr(config, "num_nextn_predict_layers", None)
            or 3
        )

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

        draft_config = _draft_vllm_config(vllm_config)
        self.layers = nn.ModuleList(
            [
                IluDeepseekV41DecoderLayer(
                    draft_config,
                    maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    topk_indices_buffer=self.topk_indices_buffer,
                )
                for i in range(self.num_dspark_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            getattr(config, "draft_vocab_size", None) or config.vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

        # Per-position acceptance probability, consumed only by adaptive
        # verification to size each request's draft budget. The speculator
        # tests this attribute for None, so it has to exist either way.
        #
        # V4.1 ships a single `mtp.2.confidence_head.proj.weight` of shape
        # [1, 5376] = [1, hidden_size + dspark_markov_rank], i.e. the head
        # scores the concatenation of the per-(req, step) head hidden state and
        # that step's Markov embedding, and there is no bias tensor. Qwen3
        # DSpark builds the same head with bias=True, which would leave an
        # unloaded parameter here.
        self.confidence_head: DSparkConfidenceHead | None = None
        if vllm_config.speculative_config.enable_adaptive_verification:
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=False,
                with_markov=True,
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
        """Write the context tokens' KV into each draft layer's window cache.

        Same insert as ``IluDeepseekV41Attention._fused_qnorm_rope_kv_insert``.
        The proposer supplies ``slot_mapping`` explicitly because no attention
        metadata is bound this early, and the q half is skipped: these are
        context tokens, only ever attended *to*. An empty head axis is how
        ``v41_qkv_rope_insert`` is told not to rotate a query.

        Some callers pass one shared 1D slot mapping; a per-layer list is
        accepted for when draft layers land in different cache groups.
        """
        pos32 = context_positions.to(torch.int32)
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                context_slot_mappings[i]
                if isinstance(context_slot_mappings, list)
                else context_slot_mappings
            )
            if slot_mapping is None:
                continue
            attn = layer.attn
            qr_kv = attn.fused_wqa_wkv(main_x)[0]
            _, kv = qr_kv.split([attn.q_lora_rank, attn.head_dim], dim=-1)
            kv = attn.kv_norm(kv)
            # [T, 0, head_dim]: rotate and publish kv, leave q alone.
            empty_q = kv.new_empty((kv.size(0), 0, kv.size(1)))
            insert = (
                ixfops.v41_qkv_rope_fp8_insert
                if attn.kv_cache_dtype == "fp8_ds_mla"
                else torch.ops._C.v41_qkv_rope_insert
            )
            insert(
                empty_q,
                attn.swa_cache_layer.kv_cache,
                kv,
                slot_mapping,
                pos32,
                attn.cos_sin,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Same MHC chain as the target, over three layers and no engram."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        pre_mix = hidden_states.new_zeros(
            hidden_states.shape[0], self.hc_mult, dtype=torch.float32
        )
        pre_mix[:, 0] = 1.0

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
        # Pre-norm, like the V4 draft: ``compute_logits`` applies ``norm``.
        return mhc.hc_pre(hidden_states, pre_mix)


class IluDSparkDeepseekV41ForCausalLM(nn.Module, SupportsPP):
    """Iluvatar DeepSeek-V4.1 DSpark draft model with shared target heads.

    ``SupportsPP`` only gates ``ModelConfig.verify_with_parallel_config``.
    vLLM places the whole DSpark drafter on the last PP rank, so this model
    is never split across stages and never exchanges PP intermediate tensors.
    """

    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

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
        # Under PP the target's embed_tokens is a PPMissingLayer on the drafting
        # rank, so the loader cannot alias it in (see the PP-aware
        # load_dspark_model override) and this model owns its embedding instead.
        self.load_own_embed_tokens = get_pp_group().world_size > 1
        self.model = IluDSparkDeepseekV41Model(
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

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token.

        Both arguments are ordered (req, step) and flattened, so the head sees
        one row per drafted token.
        """
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def markov_bias_local(self, markov_embed: torch.Tensor) -> torch.Tensor:
        markov_w2 = self.model.markov_head.markov_w2
        indices = self.lm_head.shard_indices
        local_vocab_size = self.lm_head.num_embeddings_per_partition

        if markov_w2.weight.shape[0] == local_vocab_size:
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
            logits = ixfops.gemm_bf16_bf16_fp32(flat, weight, format="TN")
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

    def _remap_dspark_name(self, name: str) -> str | None:
        return remap_dspark_name(
            name,
            self.config.num_hidden_layers,
            load_own_embed=self.load_own_embed_tokens,
            load_confidence_head=self.model.confidence_head is not None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{0,1,2}.*`` weights embedded in the target checkpoint."""
        # Same shape as the V4 draft loader; the V4.1 deltas are in
        # ``remap_dspark_name`` and the fused wq_a/wkv entry below.
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
            mapped = self._remap_dspark_name(name)
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

        # embed_tokens / lm_head are normally aliased from the target and so are
        # exempt. Under PP the draft owns its embedding, and a silently
        # unloaded one produces garbage drafts rather than an error, so it must
        # be covered by this check.
        aliased_prefixes = (
            ("lm_head",)
            if self.load_own_embed_tokens
            else (
                "model.embed_tokens",
                "lm_head",
            )
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
                "[IluDSparkDeepseekV41ForCausalLM.load_weights] "
                f"{len(missing)} parameter(s) were not initialized from the "
                "checkpoint. Showing up to 20:\n  " + "\n  ".join(preview)
            )
        logger.info_once(
            "ilu V4.1 DSpark draft model loaded: %d params", len(loaded_params)
        )
        return loaded_params


def register_model() -> None:
    from vllm import ModelRegistry

    # Community speculative_config remaps V4.1 DSpark to ``DSparkV41DraftModel``.
    # Override that name so the Iluvatar decoder / mHC / attention path is used.
    for arch in ("DSparkV41DraftModel", "IluDSparkV41DraftModel"):
        ModelRegistry.register_model(
            arch,
            "vllm_iluvatar.models.deepseek_v41.dspark:IluDSparkDeepseekV41ForCausalLM",
        )
