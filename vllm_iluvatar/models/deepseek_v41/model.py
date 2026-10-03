"""DeepSeek-V4.1-Flash on Iluvatar, raw FP8 checkpoint, text-only!"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import partial
from itertools import islice

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
)
from vllm.model_executor.offloader import get_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.skip_layers import (
    producer_last_layer,
)
from vllm_iluvatar.models.deepseek_v4.fp8_model import (
    IluDeepseekV4Fp8DecoderLayer,
    IluDeepseekV4Fp8ForCausalLM,
    IluDeepseekV4Fp8Model,
    _make_fp8_weights_mapper,
)
from vllm_iluvatar.models.deepseek_v41 import mhc
from vllm_iluvatar.models.deepseek_v41.attention import (
    IluDeepseekV41Attention,
)
from vllm_iluvatar.models.deepseek_v41.engram import (
    EngramLayout,
    IluEngram,
    NgramHashState,
    _num_reqs,
    _window_metadata,
    build_compressed_token_map,
    compute_hash_multipliers,
    engram_dp_group,
)
from vllm_iluvatar.models.deepseek_v41.pp_plan import build_rank_plan, local_kv_source

logger = init_logger(f"vllm.{__name__}")

# Engram tables are loaded from mmap, not the vLLM weight iterator.
_ENGRAM_SKIP = re.compile(r"(^|\.)engram\.")


def default_aux_hidden_state_layers(hf_config) -> tuple[int, ...]:
    """Aux capture layers for a V4.1 DSpark draft."""
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
        *,
        is_kv_forwarder: bool = False,
        is_kv_sender: bool = False,
        kv_source_override: int | None = None,
    ) -> None:
        # The V4 layer constructs the attention itself and knows nothing about
        # pipeline roles, so bind them onto the callable it uses. An instance
        # attribute shadows the class one, and this runs before the base
        # constructor reaches that call.
        self.attention_cls = partial(
            IluDeepseekV41Attention,
            is_kv_forwarder=is_kv_forwarder,
            is_kv_sender=is_kv_sender,
            kv_source_override=kv_source_override,
        )
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
            residual,
            "attn",
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            pre_mix,
            sqrsum,
            self.attn_norm,
        )
        h = self.attn(positions, h, None)
        residual, attn_sqrsum = mhc.hc_post_sqrsum(h, residual, attn_post, attn_comb)

        ffn_post, ffn_comb, h, ffn_pre = self._hc_pre_mixes(
            residual,
            "ffn",
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            attn_pre,
            attn_sqrsum,
            self.ffn_norm,
        )
        h = self.ffn(h, input_ids)
        return h, residual, ffn_post, ffn_comb, ffn_pre


class IluDeepseekV41Model(IluDeepseekV4Fp8Model, EagleModelMixin):
    """V4.1 backbone: V4.1 attention, the engram layers, and the pipeline cut.

    ``supports_aux_hidden_states_over_pp`` is the vLLM 0.30.1 contract: with
    DSpark the runner calls ``verify_supports_aux_hidden_states_over_pp`` and
    ``PPHandler`` relays ``aux_hidden_states_{i}`` through middle ranks. Capture
    stays ``idx + 1`` (the id v4.1 stores), which is what
    ``EagleModelMixin._cache_aux_pp_layout`` counts with ``bisect_right``.

    The constructor is V4.1's own rather than the V4 parent's, because the
    layer list is where the pipeline plan has to be applied: which layer
    replicates a stranded KV group, which one mirrors its records for the next
    rank, and which cache every consumer should read. Weight loading is still
    the parent's -- the FP8 checkpoint layout is identical and there is nothing
    to gain from a second copy of it.
    """

    decoder_layer_cls = IluDeepseekV41DecoderLayer
    supports_aux_hidden_states_over_pp = True

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
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.parallel_config = vllm_config.parallel_config
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        # V4.1 has no global MHC head and no MTP draft, so neither the
        # ``hc_head_*`` parameters nor the residual buffer the V4 parent
        # allocates for one exist here.
        self._mtp_hidden_buffer = None
        # Filled by the inherited load_weights, read by the load verifier.
        self.consumed_ckpt_names: set[str] = set()
        # Set by IluDeepseekV41ForCausalLM.set_aux_hidden_state_layers when a
        # DSpark draft is active.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.topk_indices_buffer = torch.empty(
            max_tokens, config.index_topk, dtype=torch.int32, device=self.device
        )

        pp = get_pp_group()
        if pp.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self._build_layers(vllm_config, prefix, pp)
        self._build_transfer_buffers(max_tokens)

        if pp.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self._bind_index_buffers(max_tokens)

        self.engram_layer_ids = list(getattr(config, "engram_layer_ids", ()) or ())
        self.engram_layout: EngramLayout | None = None
        self.engram_hash: NgramHashState | None = None
        if self.engram_layer_ids:
            model_config = vllm_config.model_config
            self.model_path = model_config.model
            self._build_engram(config, model_config)
        # Flipped at the end of load_weights. The listener is up before that,
        # and it refuses a snapshot until this is set.
        self._weights_ready = False

    # ---- construction -----------------------------------------------------

    def _build_layers(self, vllm_config: VllmConfig, prefix: str, pp) -> None:
        """The local layers, each told what the pipeline cut made of it."""
        config = self.config
        kv_sources = tuple(getattr(config, "kv_source_layer_ids", None) or ())
        index_sources = tuple(getattr(config, "index_source_layer_ids", None) or ())
        compress_ratios = tuple(
            int(r) for r in (getattr(config, "compress_ratios", None) or ())
        )
        plan = build_rank_plan(
            config.num_hidden_layers,
            kv_sources,
            index_sources,
            compress_ratios,
            pp.rank_in_group,
            pp.world_size,
            # -1 unless a shortlist is actually allocated, so the plan and
            # ``_bind_index_buffers`` cannot disagree about whether there is
            # anything to carry.
            candidate_source=(
                int(getattr(config, "candidate_source_layer_id", -1))
                if int(getattr(config, "candidate_topk_blocks", 0)) > 0
                else -1
            ),
            last_layer=producer_last_layer(vllm_config),
        )
        self.pp_plan = plan
        self.start_layer, self.end_layer = plan.start_layer, plan.end_layer

        # Only the last KV source on this rank has to mirror its records: an
        # earlier one's would be overwritten before the rank finishes anyway.
        local_sources = [
            s for s in kv_sources if plan.start_layer <= s < plan.end_layer
        ]
        self.kv_sender_layer = max(local_sources) if local_sources else None
        # With no local source to refill it, the outgoing buffer is whatever
        # arrived, so a group spanning three or more ranks needs this rank to
        # pass it along.
        self.relays_kv = plan.forwarder is not None and self.kv_sender_layer is None

        def build(layer_id: int) -> nn.Module:
            real_source = None
            if (
                layer_id < len(compress_ratios)
                and compress_ratios[layer_id] > 0
                and kv_sources
            ):
                eligible = [s for s in kv_sources if s <= layer_id]
                real_source = max(eligible) if eligible else None
            return self.decoder_layer_cls(
                vllm_config,
                prefix=f"{prefix}.layers.{layer_id}",
                topk_indices_buffer=self.topk_indices_buffer,
                is_kv_forwarder=layer_id == plan.forwarder,
                is_kv_sender=layer_id == self.kv_sender_layer,
                kv_source_override=local_kv_source(layer_id, real_source, plan),
            )

        self.layers = nn.ModuleList(
            [PPMissingLayer() for _ in range(plan.start_layer)]
            + get_offloader().wrap_modules(
                build(layer_id) for layer_id in range(plan.start_layer, plan.end_layer)
            )
            + [
                PPMissingLayer()
                for _ in range(plan.end_layer, config.num_hidden_layers)
            ]
        )

        last_layer = producer_last_layer(vllm_config)
        if last_layer is not None:
            logger.info(
                "DeepSeek V4.1 PD prefiller stops after layer %s (last kv "
                "source); rank %s/%s runs [%s, %s), layers %s-%s are "
                "PPMissingLayer and their weights are never loaded",
                last_layer,
                pp.rank_in_group,
                pp.world_size,
                plan.start_layer,
                plan.end_layer,
                last_layer + 1,
                config.num_hidden_layers - 1,
            )

        if plan.transfers_kv:
            logger.info(
                "DeepSeek V4.1 pipeline rank %s/%s owns layers [%s, %s); "
                "kv forwarder=%s (replicating source %s), kv sender=%s, "
                "relays=%s, carries topk=%s",
                pp.rank_in_group,
                pp.world_size,
                plan.start_layer,
                plan.end_layer,
                plan.forwarder,
                plan.forwarded_kv_source,
                self.kv_sender_layer,
                self.relays_kv,
                plan.transfers_topk,
            )

    def _build_transfer_buffers(self, max_tokens: int) -> None:
        """Dense per-token record buffers, when the cut makes the pipeline carry KV.

        Every rank allocates them, not just the ones that send: the pipeline's
        intermediate tensors have to carry the same keys everywhere.
        """
        self.main_kv_rows: torch.Tensor | None = None
        self.indexer_kv_rows: torch.Tensor | None = None
        if not self.pp_plan.transfers_kv:
            return
        if self.start_layer == self.end_layer:
            # Truncated prefiller: this rank sits entirely past the last kv
            # source and has no layer to size the buffers from. It also has
            # nothing to send, so leaving them None is correct.
            return

        sample = self.layers[self.start_layer].attn
        self.main_kv_rows = torch.empty(
            max_tokens,
            sample.main_transfer_width,
            dtype=sample.main_transfer_dtype,
            device=self.device,
        )
        self.indexer_kv_rows = torch.empty(
            max_tokens,
            sample.indexer_transfer_width,
            dtype=sample.indexer_transfer_dtype,
            device=self.device,
        )
        if self.kv_sender_layer is not None:
            sender = self.layers[self.kv_sender_layer].attn
            sender.main_kv_rows = self.main_kv_rows
            assert sender.indexer is not None, (
                f"layer {self.kv_sender_layer} is a kv source and must own an "
                "indexer to mirror its index keys"
            )
            sender.indexer.k_row_buffer = self.indexer_kv_rows

    def _bind_index_buffers(self, max_tokens: int) -> None:
        """Preallocate the selection buffers the indexers share across layers.

        ``topk_lens`` because V4.1's selection is ragged and our kernels take
        the count explicitly, and the candidate list because the indexer at
        ``candidate_source_layer_id`` publishes a block mask that the ratio-1
        indexers above it score against.

        Allocating once is not only about the per-step cost: an indexer writing
        into a buffer of a fixed address is what lets the selection cross layers
        without any layer holding a reference to another layer's tensor.
        """
        config = self.config
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
            lid
            for lid in self.engram_layer_ids
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

    # ---- pipeline ---------------------------------------------------------

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        """The posted multi-stream hidden states, the leftover ``pre_mix``, and
        the KV records a split group has to carry."""
        tensors = {
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
        if self.pp_plan.transfers_kv:
            assert self.main_kv_rows is not None
            assert self.indexer_kv_rows is not None
            tensors["main_kv"] = torch.zeros(
                (batch_size, self.main_kv_rows.shape[1]),
                dtype=self.main_kv_rows.dtype,
                device=device,
            )
            tensors["indexer_kv"] = torch.zeros(
                (batch_size, self.indexer_kv_rows.shape[1]),
                dtype=self.indexer_kv_rows.dtype,
                device=device,
            )
        if self.pp_plan.transfers_topk:
            tensors["topk"] = torch.zeros(
                (batch_size, self.config.index_topk),
                dtype=torch.int32,
                device=device,
            )
        if self.pp_plan.transfers_candidates:
            assert self.candidate_idxs_buffer is not None
            tensors["candidates"] = torch.zeros(
                (batch_size, self.candidate_idxs_buffer.shape[1]),
                dtype=torch.int32,
                device=device,
            )
        return IntermediateTensors(tensors)

    def _accept_remote_kv(
        self,
        intermediate_tensors: IntermediateTensors,
        positions: torch.Tensor,
    ) -> None:
        """Hand a stranded group's records to this rank's forwarder."""
        forwarder = self.pp_plan.forwarder
        if forwarder is None:
            return
        main_rows = intermediate_tensors["main_kv"]
        indexer_rows = intermediate_tensors["indexer_kv"]
        topk_rows = intermediate_tensors.tensors.get("topk")
        self.layers[forwarder].attn.accept_remote_kv(
            main_rows, indexer_rows, topk_rows, positions
        )
        if self.relays_kv:
            # No local source will refill the outgoing buffer, so the group
            # continues past this rank only if we pass on what we received.
            assert self.main_kv_rows is not None
            assert self.indexer_kv_rows is not None
            num_tokens = main_rows.shape[0]
            self.main_kv_rows[:num_tokens].copy_(main_rows)
            self.indexer_kv_rows[:num_tokens].copy_(indexer_rows)

    def _accept_remote_candidates(
        self, intermediate_tensors: IntermediateTensors
    ) -> None:
        """Take over the shortlist published on an earlier rank.

        Unconditional on every rank that carries it, forwarder or not: a rank
        with no index source of its own still has to hand the shortlist to the
        next one, and copying into the local buffer is what makes the send at
        the end of :meth:`forward` pick it up.
        """
        rows = intermediate_tensors.tensors.get("candidates")
        if rows is None or self.candidate_idxs_buffer is None:
            return
        num_tokens = rows.shape[0]
        self.candidate_idxs_buffer[:num_tokens].copy_(rows)

    def _engram_layers(self):
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            engram = getattr(layer, "engram", None)
            if engram is not None:
                yield engram

    def _prepare_engram(
        self, input_ids: torch.Tensor | None, positions: torch.Tensor
    ) -> None:
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
                input_ids.new_full(
                    (max(_num_reqs(md), 1), self.engram_hash.lookback_depth), -1
                ),
                input_ids.new_zeros(
                    (max(_num_reqs(md), 1), self.engram_hash.lookback_depth),
                    dtype=torch.bool,
                ),
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
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        """Embed -> local layers -> leftover FFN post -> collapse or PP send.

        Returns ``(hidden_states, aux)`` instead of a bare tensor when a DSpark
        draft asked for aux layers. Aux captured on an earlier PP rank is packed
        into the intermediate tensors (``EagleModelMixin``) and concatenated on
        the last rank, which is where the drafter runs. PP=1 leaves the packed
        keys empty and returns only this rank's captures.
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
            # Before any layer runs: the stranded group's caches have to hold
            # this step's rows by the time the first consumer reads them.
            self._accept_remote_kv(intermediate_tensors, positions)
            self._accept_remote_candidates(intermediate_tensors)

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
            tensors = {"hidden_states": hidden_states, "pre_mix": pre_mix}
            num_tokens = positions.shape[0]
            if self.pp_plan.transfers_kv:
                assert self.main_kv_rows is not None
                assert self.indexer_kv_rows is not None
                tensors["main_kv"] = self.main_kv_rows[:num_tokens]
                tensors["indexer_kv"] = self.indexer_kv_rows[:num_tokens]
            if self.pp_plan.transfers_topk:
                # No designated sender: every index source rewrites this
                # buffer, so at the end of the rank it already holds the
                # latest selection -- this rank's, or the one it took over.
                tensors["topk"] = self.topk_indices_buffer[:num_tokens]
            if self.pp_plan.transfers_candidates:
                # Same story as the top-k: whichever rank last wrote the
                # shortlist left it in this buffer, so the send needs no
                # designated sender.
                assert self.candidate_idxs_buffer is not None
                tensors["candidates"] = self.candidate_idxs_buffer[:num_tokens]
            tensors.update(self.pack_local_aux_hidden_states(aux_hidden_states))
            return IntermediateTensors(tensors)
        hidden_states = mhc.hc_pre(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        aux_hidden_states = (
            self.collect_remote_aux_hidden_states(intermediate_tensors)
            + aux_hidden_states
        )
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_engram_weights(self) -> None:
        """Load the engram tensors that bypass the vLLM weight path."""
        for layer_id in self.engram_layer_ids:
            layer = self.layers[layer_id]
            if (
                isinstance(layer, PPMissingLayer)
                or getattr(layer, "engram", None) is None
            ):
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
            item for item in weights if not _ENGRAM_SKIP.search(item[0])
        )
        if getattr(self.model, "engram_layer_ids", None):
            self.model.load_engram_weights()
        self.model._weights_ready = True
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

    @staticmethod
    def get_model_state_cls():
        """Community V4.1 state, for SWA bounded replay.

        ``CacheConfig.swa_bounded_replay`` keeps the sliding-window KV out of
        prefix caching and rebuilds it after a prefix hit by recomputing the
        hit's last window. The scheduler tells each request where that replay
        starts; this state is what carries it into the batch -- it pads the
        replayed tokens' slots in the prefix-cacheable groups so their cached KV
        is left alone, and hands the starts to the sliding-window metadata
        builder (``IluDeepseekSparseSWAMetadataBuilder`` subclasses the
        community one, which is what the state dispatches on).
        """
        from vllm.models.deepseek_v41.nvidia.model_state import (
            DeepseekV41ModelState,
        )

        return DeepseekV41ModelState

    @property
    def token_lookback_depth(self) -> int:
        """Zero: this model needs no runner-gathered lookback window.

        Community V4.1 has the runner gather the ids preceding each chunk start
        for the engram n-gram hash. Here ``IluEngram`` keeps its own paged slot
        cache of token ids and reads the window from that (see
        ``engram.lookback_paged``), so there is nothing for the runner to
        gather. Only the state's bounded-replay half is wanted.
        """
        return 0

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        # Sorts the ids and fills the PP slot cache the runner's
        # ``reserve_aux_intermediate_tensor_slots`` reads.
        self.model._set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return default_aux_hidden_state_layers(self.config)


def register_model() -> None:
    from vllm import ModelRegistry

    # Two names, one class. ``DeepseekV41ForCausalLM`` is what the released
    # raw-FP8 checkpoint declares; the Iluvatar-quantized conversions declare
    # ``IluDeepseekV41ForCausalLM``, following the V4 convention where the
    # ``Ilu`` prefix marks a checkpoint quantized for this platform. The two
    # differ only in ``quantization_config`` -- compressed-tensors
    # ``iluvatar-mixed`` versus raw FP8 -- which already selects the quant
    # method per layer, and in module names they agree.
    for arch in ("DeepseekV41ForCausalLM", "IluDeepseekV41ForCausalLM"):
        ModelRegistry.register_model(
            arch,
            "vllm_iluvatar.models.deepseek_v41.model:IluDeepseekV41ForCausalLM",
        )
