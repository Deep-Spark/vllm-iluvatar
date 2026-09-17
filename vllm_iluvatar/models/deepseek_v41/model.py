"""DeepSeek-V4.1-Flash on Iluvatar, raw FP8 checkpoint, text-only!"""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import islice

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
)
from vllm.sequence import IntermediateTensors

from vllm_iluvatar.models.deepseek_v4.fp8_model import (
    IluDeepseekV4Fp8DecoderLayer,
    IluDeepseekV4Fp8ForCausalLM,
    IluDeepseekV4Fp8Model,
    _make_fp8_weights_mapper,
)
from vllm_iluvatar.models.deepseek_v41 import mhc
from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention
from vllm_iluvatar.models.deepseek_v41.engram import (
    EngramTokenHistory,
    IluEngram,
    build_compressed_token_map,
    build_engram_primes,
    compute_hash_multipliers,
    lookback_tokens,
)

logger = init_logger(f"vllm.{__name__}")

# Engram tables are loaded from mmap, not the vLLM weight iterator.
_ENGRAM_SKIP = re.compile(r"(^|\.)engram\.")


def _make_v41_weights_mapper() -> WeightsMapper:
    """V4 mapper as-is; engram keys are filtered out before it (see ``load_weights``)."""
    return _make_fp8_weights_mapper()


class IluDeepseekV41DecoderLayer(IluDeepseekV4Fp8DecoderLayer):
    """V4 decoder layer with V4.1 attention and optional engram write.

    Attention is swapped via ``attention_cls`` so the V4 parent never registers
    an ``fp8_ds_mla`` cache. Engram is applied to the residual before the block.
    """

    attention_cls = IluDeepseekV41Attention

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__(vllm_config, prefix, topk_indices_buffer)
        self.layer_id = extract_layer_index(prefix)
        # Set by IluDeepseekV41Model for the engram layers only.
        self.engram: IluEngram | None = None
        self._hc_fn_io_cache: dict[str, torch.Tensor] = {}

    def _hc_pre_mixes(
        self,
        x: torch.Tensor,
        name: str,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        pre_mix_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project, split, and collapse. Cache the bf16 copy of ``hc_fn``."""
        cached = self._hc_fn_io_cache.get(name)
        if cached is None or cached.dtype != x.dtype:
            cached = hc_fn.to(x.dtype).contiguous()
            self._hc_fn_io_cache[name] = cached
        return mhc.hc_pre_mixes(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            pre_mix_in,
            norm_eps=self.rms_norm_eps,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_eps=self.hc_eps,
            post_mult=self.hc_post_alpha,
            hc_fn_io=cached,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        engram_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(x, residual, post_mix, res_mix, ffn_pre)``; FFN post is delayed."""
        if residual is None:
            residual = x
        else:
            residual = mhc.hc_post(x, residual, post_mix, res_mix)

        if self.engram is not None and engram_rows is not None:
            residual = self.engram.apply(residual, engram_rows)

        attn_post, attn_comb, h, attn_pre = self._hc_pre_mixes(
            residual, "attn", self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, pre_mix
        )
        h = self.attn_norm(h)
        h = self.attn(positions, h, None)
        residual = mhc.hc_post(h, residual, attn_post, attn_comb)

        ffn_post, ffn_comb, h, ffn_pre = self._hc_pre_mixes(
            residual, "ffn", self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, attn_pre
        )
        h = self.ffn_norm(h)
        h = self.ffn(h, input_ids)
        return h, residual, ffn_post, ffn_comb, ffn_pre


class IluDeepseekV41Model(IluDeepseekV4Fp8Model):
    """V4 model plus V4.1 attention, the engram layers, and the token history they share."""

    decoder_layer_cls = IluDeepseekV41DecoderLayer

    # Shared-expert gate_up only. Attention / compressor linears stay unfused.
    stacked_params_mapping = [
        ("gate_up_proj", "w1", 0),
        ("gate_up_proj", "w3", 1),
    ]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # No global MHC head in the V4.1 checkpoint.
        del self.hc_head_fn, self.hc_head_base, self.hc_head_scale
        self._mtp_hidden_buffer = None

        config = self.config
        self.engram_layer_ids = list(getattr(config, "engram_layer_ids", ()) or ())
        self.engram_history: EngramTokenHistory | None = None
        if not self.engram_layer_ids:
            return

        model_config = vllm_config.model_config
        self.model_path = model_config.model
        self._build_engram(config, model_config)

    def _build_engram(self, config, model_config) -> None:
        """Create the shared token history and one IluEngram per engram layer.

        The compressed token map is derived from the tokenizer (about 5 s for
        129280 ids) and asserted against ``engram_compressed_vocab_size``: every
        hash multiplier is derived from that number, so a mismatch would silently
        rehash all 189 GiB of tables.
        """
        local_ids = [
            lid for lid in self.engram_layer_ids
            if not isinstance(self.layers[lid], PPMissingLayer)
        ]
        if not local_ids:
            return

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )
        token_map, compressed_vocab = build_compressed_token_map(tokenizer)
        expected = config.engram_compressed_vocab_size
        if compressed_vocab != expected:
            raise ValueError(
                f"engram compressed vocab is {compressed_vocab} but the config "
                f"declares {expected}; every hash multiplier derives from this "
                "number, so the tables would be rehashed incorrectly"
            )

        primes = build_engram_primes(
            self.engram_layer_ids,
            config.engram_max_ngram_size,
            config.engram_n_heads,
            config.engram_vocab_size,
        )
        multipliers = compute_hash_multipliers(
            self.engram_layer_ids, config.engram_max_ngram_size, compressed_vocab
        )
        pad_id = token_map[config.engram_pad_token_id]
        self.engram_history = EngramTokenHistory(
            token_map, model_config.max_model_len, self.device
        )
        self.engram_max_ngram = config.engram_max_ngram_size
        self.engram_pad_id = pad_id
        for i, layer_id in enumerate(self.engram_layer_ids):
            if layer_id not in local_ids:
                continue
            layer = self.layers[layer_id]
            layer.engram = IluEngram(
                config,
                layer_id,
                primes[i],
                multipliers[i],
                pad_id,
                self.device,
            )

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        """PP carries posted multi-stream hidden states plus the leftover ``pre_mix``."""
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "pre_mix": torch.zeros(
                    (batch_size, self.hc_mult),
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )

    def _prepare_engram_rows(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Lookback once; each local engram layer hashes its own primes."""
        if (
            input_ids is None
            or not getattr(self, "engram_history", None)
            or not getattr(self, "engram_layer_ids", None)
        ):
            return {}
        tokens = lookback_tokens(
            self.engram_history,
            input_ids,
            positions,
            self.engram_max_ngram,
            self.engram_pad_id,
        )
        rows: dict[int, torch.Tensor] = {}
        for layer_id in self.engram_layer_ids:
            layer = self.layers[layer_id]
            if isinstance(layer, PPMissingLayer) or getattr(layer, "engram", None) is None:
                continue
            rows[layer_id] = layer.engram.embed_from_tokens(tokens)
        return rows

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Embed -> local layers -> leftover FFN post -> collapse or PP send."""
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
            pre_mix = hidden_states.new_zeros(
                hidden_states.shape[0], self.hc_mult, dtype=torch.float32
            )
            pre_mix[:, 0] = 1.0
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            pre_mix = intermediate_tensors["pre_mix"]

        engram_rows = self._prepare_engram_rows(input_ids, positions)
        residual = post_mix = res_mix = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_rows.get(layer.layer_id),
            )

        hidden_states = mhc.hc_post(hidden_states, residual, post_mix, res_mix)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )
        hidden_states = mhc.hc_pre(hidden_states, pre_mix)
        return self.norm(hidden_states)

    def load_engram_weights(self) -> None:
        """Load the engram tensors that bypass the vLLM weight path."""
        for layer_id in self.engram_layer_ids:
            layer = self.layers[layer_id]
            if isinstance(layer, PPMissingLayer) or getattr(layer, "engram", None) is None:
                continue
            layer.engram.load_from_checkpoint(self.model_path)


class IluDeepseekV41ForCausalLM(IluDeepseekV4Fp8ForCausalLM):
    """DeepSeek-V4.1-Flash on Iluvatar, raw FP8 checkpoint, text-only."""

    is_attention_free = False

    model_cls = IluDeepseekV41Model
    hf_to_vllm_mapper = _make_v41_weights_mapper()

    # Text-only: skip DSpark draft, vision tower, and image-token gate bias.
    _LOAD_SKIP_SUBSTRS = ("mtp.", "vision.", "aligner.", "image_", ".bias_vl")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Drop ``engram.*`` from the iterator; tables load from mmap afterwards."""
        loaded = super().load_weights(
            (item for item in weights if not _ENGRAM_SKIP.search(item[0]))
        )
        if getattr(self.model, "engram_layer_ids", None):
            self.model.load_engram_weights()
        return loaded


def register_model() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV41ForCausalLM",
        "vllm_iluvatar.models.deepseek_v41.model:IluDeepseekV41ForCausalLM",
    )
