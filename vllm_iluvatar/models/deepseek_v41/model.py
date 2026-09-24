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
from vllm.model_executor.models.interfaces import SupportsEagle3
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
    IluEngram,
    EngramLayout,
    NgramHashState,
    _num_reqs,
    _window_metadata,
    build_compressed_token_map,
    compute_hash_multipliers,
    engram_dp_group,
)

logger = init_logger(f"vllm.{__name__}")

# Engram tables are loaded from mmap, not the vLLM weight iterator.
_ENGRAM_SKIP = re.compile(r"(^|\.)engram\.")


def default_aux_hidden_state_layers(hf_config) -> tuple[int, ...]:
    """Aux capture layers for a V4.1 DSpark draft.
    """
    return tuple(getattr(hf_config, "dspark_target_layer_ids", None) or ())


def _make_v41_weights_mapper() -> WeightsMapper:
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
        sqrsum: torch.Tensor | None = None,
        norm: torch.nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._hc_fn_io_cache.get(name)
        if cached is None or cached.dtype != x.dtype:
            cached = hc_fn.to(x.dtype).contiguous()
            self._hc_fn_io_cache[name] = cached
        can_fold = norm is not None and sqrsum is not None
        post, comb, h, pre = mhc.hc_pre_mixes(
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
            sqrsum=sqrsum,
            layer_norm_weight=norm.weight if can_fold else None,
            layer_norm_eps=self.rms_norm_eps,
        )
        if norm is not None and not can_fold:
            h = norm(h)
        return post, comb, h, pre

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(x, residual, post_mix, res_mix, ffn_pre)``; FFN post is delayed."""
        # Each post hands the pre that follows it the sum of squares of the
        # residual it just wrote, so the pre can stream instead of reducing it
        # again. sqrsum stays None wherever that chain is broken: the first
        # layer has no post, and engram rewrites the residual after one.
        sqrsum = None
        if residual is None:
            residual = x
        else:
            residual, sqrsum = mhc.hc_post_sqrsum(x, residual, post_mix, res_mix)

        if self.engram is not None and self.engram._rows is not None:
            residual = self.engram(residual, self.engram._hash_ids)
            sqrsum = None

        attn_post, attn_comb, h, attn_pre = self._hc_pre_mixes(
            residual, "attn", self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            pre_mix, sqrsum, self.attn_norm
        )
        h = self.attn(positions, h, None)
        residual, attn_sqrsum = mhc.hc_post_sqrsum(h, residual, attn_post, attn_comb)

        ffn_post, ffn_comb, h, ffn_pre = self._hc_pre_mixes(
            residual, "ffn", self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            attn_pre, attn_sqrsum, self.ffn_norm
        )
        h = self.ffn(h, input_ids)
        return h, residual, ffn_post, ffn_comb, ffn_pre


class IluDeepseekV41Model(IluDeepseekV4Fp8Model):
    """V4 model plus V4.1 attention, the engram layers, and the token history they share."""

    decoder_layer_cls = IluDeepseekV41DecoderLayer

    # Shared-expert gate_up, plus attention's wq_a/wkv (both read the same normed
    # hidden state, so they are one GEMM), plus the compressor's wkv/wgate, which
    # both read the layer's hidden state.
    stacked_params_mapping = [
        ("gate_up_proj", "w1", 0),
        ("gate_up_proj", "w3", 1),
        ("attn.fused_wqa_wkv", "attn.wq_a", 0),
        ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
        ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
    ]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # No global MHC head in the V4.1 checkpoint.
        del self.hc_head_fn, self.hc_head_base, self.hc_head_scale
        self._mtp_hidden_buffer = None

        self._bind_index_buffers(vllm_config)

        # Set by IluDeepseekV41ForCausalLM.set_aux_hidden_state_layers when a
        # DSpark draft is active.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

        config = self.config
        self.engram_layer_ids = list(getattr(config, "engram_layer_ids", ()) or ())
        self.engram_layout: EngramLayout | None = None
        self.engram_hash: NgramHashState | None = None
        if not self.engram_layer_ids:
            return

        model_config = vllm_config.model_config
        self.model_path = model_config.model
        self._build_engram(config, model_config)

    def _bind_index_buffers(self, vllm_config: VllmConfig) -> None:
        """Preallocate the selection buffers the indexers share across layers.

        The base class already owns the top-k indices buffer; these are the two
        companions it has no notion of. ``topk_lens`` because V4.1's selection is
        ragged and our kernels take the count explicitly, and the candidate list
        because the indexer at ``candidate_source_layer_id`` publishes a block
        mask that the ratio-1 indexers above it score against.

        Allocating once is not only about the per-step cost: an indexer writing
        into a buffer of a fixed address is what lets the selection cross layers
        without any layer holding a reference to another layer's tensor.
        """
        config = self.config
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        device = self.topk_indices_buffer.device
        self.topk_lens_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        blocks = getattr(config, "candidate_topk_blocks", 0)
        if getattr(config, "candidate_source_layer_id", -1) >= 0 and blocks > 0:
            self.candidate_idxs_buffer = torch.empty(
                max_tokens, blocks, dtype=torch.int32, device=device
            )
            self.candidate_lens_buffer = torch.empty(
                max_tokens, dtype=torch.int32, device=device
            )
        else:
            self.candidate_idxs_buffer = None
            self.candidate_lens_buffer = None

        for layer in self.layers:
            attn = getattr(layer, "attn", None)
            if attn is not None:
                attn.bind_index_buffers(
                    self.topk_lens_buffer,
                    self.candidate_idxs_buffer,
                    self.candidate_lens_buffer,
                )

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

        layout = EngramLayout.from_config(config)
        assert layout is not None
        self.engram_layout = layout
        multipliers = compute_hash_multipliers(
            self.engram_layer_ids, config.engram_max_ngram_size, compressed_vocab
        )
        swa = self.layers[local_ids[0]].attn.swa_cache_layer
        self.engram_hash = NgramHashState(
            layout,
            swa,
            token_map,
            multipliers,
            token_map[config.engram_pad_token_id],
            self.device,
        )
        for layer_id in self.engram_layer_ids:
            if layer_id not in local_ids:
                continue
            self.layers[layer_id].engram = IluEngram(
                config,
                layer_id,
                layout.primes[layout.layer_ids.index(layer_id)],
                multipliers[layout.layer_ids.index(layer_id)],
                token_map[config.engram_pad_token_id],
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

    def _engram_layers(self):
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            engram = getattr(layer, "engram", None)
            if engram is not None:
                yield engram

    def _prepare_engram(self, input_ids: torch.Tensor | None, positions: torch.Tensor) -> None:
        """Hash once, then each engram layer gathers its rows before the decoder loop.

        DP>1 lookups are collectives. A replica that skips the real hash still
        submits dead ids so the other replicas' rows, which live on this rank's
        shard, are still read.
        """
        ready = (
            input_ids is not None
            and self.engram_hash is not None
            and self.engram_hash.ensure_cache()
        )
        md = _window_metadata() if ready else None
        if ready and md is not None and md.slot_mapping is not None:
            n = min(input_ids.size(0), positions.size(0))
            hashes = self.engram_hash(
                input_ids[:n],
                positions[:n],
                md.query_start_loc,
                input_ids.new_zeros(n, dtype=torch.bool),
                input_ids.new_full((max(_num_reqs(md), 1), self.engram_hash.lookback_depth), -1),
                input_ids.new_zeros((max(_num_reqs(md), 1), self.engram_hash.lookback_depth), dtype=torch.bool),
                md.slot_mapping,
                md.block_table,
                md.token_to_req_indices,
                _num_reqs(md),
            )
            for engram in self._engram_layers():
                engram.prepare_embeddings(hashes[:, engram.layer_hash_index])
            return
        if engram_dp_group() is None or self.engram_layout is None:
            return
        t = 0 if input_ids is None else input_ids.size(0)
        device = self.device if input_ids is None else input_ids.device
        dummy = torch.full(
            (t, self.engram_layout.n_hash_cols),
            -1,
            dtype=torch.int64,
            device=device,
        )
        for engram in self._engram_layers():
            engram.prepare_embeddings(dummy)
            engram._rows = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | IntermediateTensors
        | tuple[torch.Tensor, list[torch.Tensor]]
    ):
        """Embed -> local layers -> leftover FFN post -> collapse or PP send.

        Returns ``(hidden_states, aux)`` instead of a bare tensor when a DSpark
        draft asked for aux layers. Like V4, aux only survives if the capture
        layers live on the last PP rank; the ilu speculative config pins draft
        PP to 1, so that is the only configuration this is exercised in.
        """
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

        self._prepare_engram(input_ids, positions)
        residual = post_mix = res_mix = None
        aux_hidden_states: list[torch.Tensor] = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer), self.start_layer
        ):
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
            )
            if (idx + 1) in self.aux_hidden_state_layers:
                # ``hidden_states`` is this layer's delayed FFN output, not the
                # hc stream, so folding it is what the next layer would do first.
                # Mean-pool the hc_mult copies to the [T, H] the DSpark combiner
                # wants -- a plain mean, not the learned ``pre_mix`` collapse.
                aux_hidden_states.append(
                    mhc.hc_post(hidden_states, residual, post_mix, res_mix).mean(dim=1)
                )

        hidden_states = mhc.hc_post(hidden_states, residual, post_mix, res_mix)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )
        hidden_states = mhc.hc_pre(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_engram_weights(self) -> None:
        """Load the engram tensors that bypass the vLLM weight path."""
        for layer_id in self.engram_layer_ids:
            layer = self.layers[layer_id]
            if isinstance(layer, PPMissingLayer) or getattr(layer, "engram", None) is None:
                continue
            layer.engram.load_from_checkpoint(self.model_path)


class IluDeepseekV41ForCausalLM(IluDeepseekV4Fp8ForCausalLM, SupportsEagle3):
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

    @property
    def get_mtp_target_hidden_states(self):
        """Absent on purpose: V4.1 has no MTP draft.

        The parent hands the drafter its pre-hc_head residual buffer, which V4.1
        never fills (``_mtp_hidden_buffer`` is None). The runner probes for this
        hook with ``hasattr`` and then indexes whatever it gets back, so
        inheriting it means a ``NoneType`` subscript; raising ``AttributeError``
        is what makes ``hasattr`` answer False for an inherited name. DSpark does
        not want the residual anyway -- it builds its own input from the aux
        hidden states, and ignores ``last_hidden_states`` whenever aux is present.
        """
        raise AttributeError(
            "IluDeepseekV41ForCausalLM has no MTP draft; DSpark reads the aux "
            "hidden states instead of the pre-hc_head residual"
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = tuple(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return default_aux_hidden_state_layers(self.config)


def register_model() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV41ForCausalLM",
        "vllm_iluvatar.models.deepseek_v41.model:IluDeepseekV41ForCausalLM",
    )
