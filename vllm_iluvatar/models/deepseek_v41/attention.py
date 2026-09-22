"""DeepSeek-V4.1-Flash attention: projections, caches, sparse MLA."""
from __future__ import annotations

import re

import ixformer.inference.functions as ixfops
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_iluvatar.models.deepseek_v41.common.ops import (
    indexer_k_norm_rope_store,
    rope_quant_insert,
)
from vllm_iluvatar.models.deepseek_v41.compressor import (
    IluDeepseekV41Compressor,
    V41CompressKVCache,
)
from vllm_iluvatar.models.deepseek_v41.sparse_mla import (
    BLOCK_SIZE,
    V41AttentionMetadata,
    V41CompressorBackend,
    V41WindowKVCache,
    decode_sparse_mla,
    gather_group_rows,
    gather_window_rows,
)
from vllm_iluvatar.models.deepseek_v41.rope import apply_rope, get_shared_cos_sin

FP8_BLOCK_SIZE = 32
FP4_BLOCK_SIZE = 32

# Peak bytes of dense score rows held at once; the token axis is split to fit.
_MMA_INDEX_BUDGET = 192 << 20
# Below this token count the fused SIMT kernel wins on the part that is not FLOPs:
# dense scoring needs a host-side chunk plan and a few small H2D copies per call,
# and decode is already CPU-bound here. See ``_mma_index_applies``.
_MMA_INDEX_MIN_TOKENS = 512
# ixinfer v3 on ivcore30 requires topk % 64 == 0.
_IXINFER_TOPK_MULTIPLE = 64
# ...and reaches its tiled MMA path only when the head count is exactly one of
# its five template tiers. Any other count -- including multiples of 16 such as
# 48 -- falls back to a scalar dot product. Measured at 8192 tokens: 12.2 ms at
# h=16 against 29.5 at h=8 and 58.1 at h=48.
_IXINFER_HEAD_TIERS = (16, 32, 64, 128, 256)


def _source_layer_prefix(prefix: str, layer_id: int) -> str:
    """Rewrite ``prefix`` so it names ``layer_id`` instead of its own layer.

    Cross-layer state is never handed between layers. A borrower resolves the
    module that owns what it needs out of ``static_forward_context``, keyed by
    the source layer's prefix, which it can build from construction-time
    knowledge alone. Nothing has to be published at forward time, so a layer
    cannot read a value some earlier step left behind, and a source that is not
    on this rank fails at load instead of silently going missing.
    """
    out, n = re.subn(r"\.layers\.\d+\.", f".layers.{layer_id}.", prefix)
    assert n == 1, f"no layer index in prefix {prefix!r}"
    return out


def _seq_spans(md: V41AttentionMetadata) -> list[tuple[int, int, int, int]]:
    qsl = md.query_start_loc_cpu.tolist()
    slens = md.seq_lens_cpu.tolist()
    spans = []
    for i in range(md.num_reqs):
        start, end = qsl[i], qsl[i + 1]
        if end > start:
            spans.append((i, start, end, slens[i]))
    return spans


def prefill_attn(
    q: torch.Tensor,
    window_kv: torch.Tensor,
    compress_cache: torch.Tensor | None,
    md: V41AttentionMetadata | None,
    spans: list[tuple[int, int, int, int]],
    ratio: int,
    group_idxs: torch.Tensor | None,
    group_lens: torch.Tensor | None,
    positions: torch.Tensor,
    window_size: int,
    attn_sink: torch.Tensor,
    scale: float,
    window_cache: torch.Tensor | None = None,
    window_md: V41AttentionMetadata | None = None,
) -> torch.Tensor:
    dev = q.device
    n_win_max = min(window_size, max((sl for *_, sl in spans), default=0))
    plane_parts: list[torch.Tensor] = []
    base = 0
    win_bases: list[int] = []
    grp_bases: list[int] = []
    n_groups: list[int] = []

    cflat = None if compress_cache is None else compress_cache
    wflat = None if window_cache is None else window_cache
    win_origins: list[int] = []
    for req, t0, t1, seq_len in spans:
        n_g = seq_len // ratio if cflat is not None else 0
        if n_g:
            plane_parts.append(gather_group_rows(cflat, md, req, 0, n_g, ratio))
        grp_bases.append(base)
        base += n_g
        n_groups.append(n_g)

        start_pos = seq_len - (t1 - t0)
        # First prefill uses the current activation as the window plane.
        # Gathering through block_table is not equivalent (slot/block mismatch).
        if wflat is not None and window_md is not None and start_pos > 0:
            win_lo = max(0, start_pos + 1 - window_size)
            hist = start_pos - win_lo
            parts = []
            if hist:
                parts.append(
                    gather_window_rows(wflat, window_md, req, win_lo, hist)
                )
            parts.append(window_kv[t0:t1])
            plane_parts.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=0))
            win_origins.append(win_lo)
            win_bases.append(base)
            base += hist + (t1 - t0)
        else:
            plane_parts.append(window_kv[t0:t1])
            win_origins.append(start_pos)
            win_bases.append(base)
            base += t1 - t0

    plane = torch.cat(plane_parts, dim=0) if plane_parts else window_kv[:0]

    topk_w = 0 if group_idxs is None else group_idxs.size(-1)
    width = n_win_max + topk_w
    pad = -width % _IXINFER_TOPK_MULTIPLE
    idxs = torch.full((q.size(0), width + pad), -1, dtype=torch.int32, device=dev)
    lens = torch.zeros(q.size(0), dtype=torch.int32, device=dev)

    for (_req, t0, t1, _seq_len), wb, gb, n_g, wlo in zip(
        spans, win_bases, grp_bases, n_groups, win_origins
    ):
        has_groups = bool(n_g) and group_idxs is not None
        ixfops.ds4_build_v41_index_plane(
            idxs,
            lens,
            positions,
            group_idxs if has_groups else None,
            group_lens if has_groups else None,
            t0,
            t1 - t0,
            wb,
            gb,
            wlo,
            window_size,
        )

    # Pad up to the nearest tier so the kernel takes its MMA path. Off-tier head
    # counts hit a scalar fallback that is 2.4x slower *in absolute time* than
    # the padded call despite doing half the work. Heads are independent, so the
    # padded lanes are computed and dropped; they cost nothing in accuracy either
    # -- the MMA path lands closer to an fp32 reference (2.6e-3 relative) than
    # the scalar path it replaces (5.2e-3). TP8 gives 64/8 = 8 heads and lands on
    # the slow side; TP4 gives 16 and pays nothing here.
    heads = q.size(1)
    tier = next((t for t in _IXINFER_HEAD_TIERS if t >= heads), heads)
    pad_h = tier - heads
    if pad_h:
        q_in = q.new_zeros((q.size(0), heads + pad_h, q.size(2)))
        q_in[:, :heads] = q
        sink_in = attn_sink.new_zeros(heads + pad_h)
        sink_in[:heads] = attn_sink
    else:
        q_in, sink_in = q, attn_sink

    o = ixfops.ds4_flash_mla_sparse_fwd(
        q_in,
        plane.unsqueeze(1),
        idxs.unsqueeze(1).contiguous(),
        scale,
        d_v=q.size(-1),
        attn_sink=sink_in,
        topk_length=lens,
    )
    # Dropping the padded lanes leaves a strided view, and the RoPE that consumes
    # this output requires contiguous memory.
    return o[:, :heads].contiguous() if pad_h else o


def _mma_index_applies(
    md: V41AttentionMetadata, num_tokens: int, candidate_span: int
) -> bool:
    """Can this batch be scored densely without changing what gets selected?

    Two conditions. The candidate machinery has to be vacuous: a restricted layer
    selects among the blocks the source layer published, so when that listIluDeepseekV41IndexerBackend
    already spans every block a token can see, the restriction removes nothing --
    the source can skip publishing and the restricted layers can score the full
    range with identical results. ``candidate_topk_blocks * candidate_block_size``
    is that span. It is measured in the compressed domain of the *source* layer,
    and the answer has to hold for every index layer at once (it is what tells
    the source whether to publish), so the comparison is against the token-domain
    sequence length -- the form that is conservative for every ratio.

    And the batch has to be one this path can be captured around. Dense scoring
    plans its chunks on the host and uploads the per-chunk seqlens, so a graph
    would freeze one batch's lengths and silently replay them for every later
    one -- the fused kernel derives its chunk count from run-time lengths on the
    device precisely so it can replay at any length. ``num_decodes`` does not
    catch this: V4.1's builder keeps ``reorder_batch_threshold`` at 1 without
    ``supports_spec_as_decode``, so a DSpark verify step arrives here counted as
    a prefill, and it is exactly what gets captured. Hence the explicit capture
    check, plus a token floor -- below it the host-side plan costs more than the
    FLOPs it saves, and decode is CPU-bound here to begin with.
    """
    if md.num_decodes or not md.num_reqs:
        return False
    if num_tokens < _MMA_INDEX_MIN_TOKENS:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    return int(md.seq_lens_cpu[: md.num_reqs].max()) <= candidate_span


def _short_context_bypass_applies(
    md: V41AttentionMetadata, ratio: int, topk: int
) -> bool:
    """Whether the indexer cannot change the selection, so it can be skipped.
    """
    if ratio <= 0 or not md.num_reqs:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    return int(md.seq_lens_cpu[: md.num_reqs].max()) // ratio <= topk


def _mma_index_chunks(
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_reqs: int,
    rows: int,
) -> list[tuple[int, int, list[int], list[int], list[int]]]:
    """Split the token axis into pieces that fit the dense-score budget.

    Any contiguous sub-span of a request is a valid batch entry for the scoring
    kernel: the only thing its causal mask reads is ``kv_len - q_len``, so giving
    it the position just past the sub-span's last token as ``kv_len`` reproduces
    that span's absolute positions exactly. That makes a request splittable at
    any token, which is what lets one budget serve any context length.

    Returns ``(token_start, token_end, cu_seqlens_q, cu_seqlens_kv, req_ids)``
    per chunk. ``req_ids`` is one entry per span rather than per request, because
    a request split across chunks contributes a span -- and a block-table row --
    to each of them.
    """
    chunks: list[tuple[int, int, list[int], list[int], list[int]]] = []
    cu_q, cu_kv, reqs = [0], [0], []
    start = 0
    for r in range(num_reqs):
        lo, hi = int(query_start_loc_cpu[r]), int(query_start_loc_cpu[r + 1])
        # Absolute position of this request's first query token.
        base = int(seq_lens_cpu[r]) - (hi - lo)
        cursor = lo
        while cursor < hi:
            if cu_q[-1] == rows:
                chunks.append((start, start + cu_q[-1], cu_q, cu_kv, reqs))
                start += cu_q[-1]
                cu_q, cu_kv, reqs = [0], [0], []
            take = min(rows - cu_q[-1], hi - cursor)
            cu_q.append(cu_q[-1] + take)
            cu_kv.append(cu_kv[-1] + base + (cursor - lo) + take)
            reqs.append(r)
            cursor += take
    if cu_q[-1]:
        chunks.append((start, start + cu_q[-1], cu_q, cu_kv, reqs))
    return chunks


class V41IndexKeyCache(V41CompressKVCache):
    """Indexer keys: same page geometry as compressed KV, ``index_head_dim`` rows."""


class IluDeepseekV41Indexer(nn.Module):
    """Index-query projections, the K cache for key owners, and the selection.

    Key ownership is ``layer_id in kv_source_layer_ids``: index keys come from the
    compressor's pre-RoPE latent, so only those layers own ``wk`` / ``k_norm`` and a
    cache to write them to. Everyone else scores against the cache a source
    published.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        compress_ratio: int,
        quant_config=None,
        prefix: str = "",
        *,
        owns_k: bool,
        key_cache_prefix: str,
        static_forward_context: dict,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.owns_k = owns_k
        # Index keys come from a kv-source layer's compressor latent, so only
        # those layers own a cache to write them to; the rest score against the
        # nearest one, looked up here rather than published into shared state.
        self.key_cache_prefix = key_cache_prefix
        self._static_forward_context = static_forward_context
        # Bound by the model once the shared selection buffers exist.
        self.topk_idxs_buf: torch.Tensor | None = None
        self.topk_lens_buf: torch.Tensor | None = None
        self.candidate_idxs_buf: torch.Tensor | None = None
        self.candidate_lens_buf: torch.Tensor | None = None
        self.compress_ratio = compress_ratio
        self.is_candidate_source = layer_id == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_id
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.cos_sin: torch.Tensor | None = None

        # Dense scoring needs a page the mqa_logits paged loader accepts, which is
        # a multiple of 16 rows. A compressed layer's page holds
        # ``BLOCK_SIZE / compress_ratio`` rows, so ratio 2 gives 8 and only the
        # ratio-1 layers qualify -- which are also the expensive ones, since they
        # see twice the groups.
        def page_ok(ratio: int) -> bool:
            page = BLOCK_SIZE // ratio if ratio else 0
            return page >= 16 and page % 16 == 0

        self.mma_page_ok = page_ok(compress_ratio)
        # Taking this path means the candidate list is never published, which is
        # only safe when every layer that would consume it also takes the path.
        # Otherwise a restricted layer falls back to the fused kernel and finds no
        # mask to read.
        self.mma_chain_ok = all(
            page_ok(config.compress_ratios[i])
            for i in config.index_source_layer_ids
            if i >= config.candidate_source_layer_id and config.compress_ratios[i]
        )
        downstream = [
            config.compress_ratios[i]
            for i in config.index_source_layer_ids
            if i >= config.candidate_source_layer_id and config.compress_ratios[i]
        ]
        self.bypass_ratio = (
            min([compress_ratio] + downstream)
            if self.is_candidate_source
            else compress_ratio
        )

        # Replicated, not sharded over heads: the scoring kernel shards candidate
        # blocks over the TP group, so every rank needs every head anyway and a
        # sharded projection would only gather them straight back -- 64 MiB of q
        # per index-source layer at 8k tokens, against 7/8 of a
        # [T, 1280] x [1280, 4096] GEMM to compute them everywhere.
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank, config.index_n_heads * config.index_head_dim,
            bias=False, quant_config=quant_config, prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size, config.index_n_heads, bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16, prefix=f"{prefix}.weights_proj",
        )
        if self.owns_k:
            self.wk = ReplicatedLinear(
                config.head_dim, config.index_head_dim, bias=False,
                params_dtype=torch.bfloat16, prefix=f"{prefix}.wk",
            )
            self.k_norm = RMSNorm(config.index_head_dim, config.rms_norm_eps)
            self.key_cache = V41IndexKeyCache(
                head_size=config.index_head_dim,
                dtype=torch.get_default_dtype(),
                compress_ratio=compress_ratio,
                block_size=BLOCK_SIZE,
                prefix=key_cache_prefix,
            )
            assert self.key_cache.prefix == key_cache_prefix
        elif key_cache_prefix not in static_forward_context:
            raise NotImplementedError(
                f"layer {layer_id} scores against index keys owned by "
                f"{key_cache_prefix!r}, which is not on this rank"
            )

    def _index_keys(self) -> tuple[torch.Tensor, V41AttentionMetadata]:
        """The paged index-key cache this layer scores against, and its metadata."""
        cache = self._static_forward_context[self.key_cache_prefix]
        return cache.kv_cache, get_forward_context().attn_metadata[
            self.key_cache_prefix
        ]

    def publish_keys(self, latent: torch.Tensor) -> None:
        md: V41AttentionMetadata = get_forward_context().attn_metadata[
            self.key_cache_prefix
        ]
        indexer_k_norm_rope_store(
            self.wk(latent)[0],
            md.positions,
            self.cos_sin,
            self.k_norm.weight,
            self.k_norm.variance_epsilon,
            self.key_cache.kv_cache,
            md.slot_mapping,
            self.compress_ratio,
            True,
            block_size=md.block_size,
            block_table=md.block_table,
            token_to_req_indices=md.token_to_req_indices,
        )

    def score_and_select(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.cos_sin is not None
        index_k, md = self._index_keys()

        if _short_context_bypass_applies(md, self.bypass_ratio, self.index_topk):
            n = positions.size(0)
            idxs = self.topk_idxs_buf[:n]
            lens = self.topk_lens_buf[:n]
            ixfops.v41_fill_short_context_topk(
                idxs, lens, positions, self.compress_ratio
            )
            return idxs, lens

        q = self.wq_b(qr)[0].unflatten(-1, (self.n_heads, self.index_head_dim))

        weights = self.weights_proj(x)[0]
        ixfops.v41_indexer_q_rope_qdq(
            q,
            positions,
            self.cos_sin,
            FP4_BLOCK_SIZE,
            "fp4_pow2",
            weights,
            weights,
            self.softmax_scale * self.n_heads**-0.5,
        )

        return self._select_paged(q, weights, positions, index_k, md)

    def _select_paged(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
        index_k: torch.Tensor,
        md: V41AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score and select by walking the paged index cache.

        Nothing sequence-wide may be materialised here. A dense
        ``[num_tokens, n_heads, max_groups]`` score is seq_len**2 on a
        compress_ratio=1 layer and wants 17 GiB at 32k tokens; the kernel carries a
        running top-k per token instead, so the widest allocation is
        ``[num_tokens, index_topk]``.

        That also dictates how the work is split. A rank needs the *complete*
        per-head sum for any group it scores, and the ``[num_tokens, seq_len]``
        score is exactly what must not exist to be all-reduced, so q and the
        weights are replicated -- see the projections -- and the kernel shards
        *candidate blocks* round-robin over the TP group. Each rank's local top-k
        contains every group of its own blocks that could reach the global top-k,
        so merging the compact results is exact rather than approximate.

        ``max_visible_groups`` must come from the CPU seq_lens: it is a launch
        hint only, and reading ``seq_lens.max()`` off the device would sync during
        CUDA graph capture.

        The selection leaves in preallocated buffers the model owns, written in
        place: they are what the layers between index sources read, and at 8192
        tokens the top-k alone would otherwise be a 16 MiB allocation per source
        layer per step.
        """
        ratio = self.compress_ratio
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        num_tokens = q.size(0)
        dev = q.device
        if (
            self.mma_page_ok
            and self.mma_chain_ok
            and _mma_index_applies(
                md, num_tokens, self.candidate_topk_blocks * self.candidate_block_size
            )
        ):
            return self._select_mma(q, weights, positions, md, index_k, num_tokens)
        if self.uses_candidates:
            cand_in_idxs = self.candidate_idxs_buf[:num_tokens]
            cand_in_lens = self.candidate_lens_buf[:num_tokens]
        else:
            cand_in_idxs = torch.empty((num_tokens, 0), dtype=torch.int32, device=dev)
            cand_in_lens = torch.zeros(num_tokens, dtype=torch.int32, device=dev)

        # Only the candidate source emits blocks; everyone else still has to pass
        # a buffer, so give it the minimum width the binding accepts.
        cand_capacity = self.candidate_topk_blocks if self.is_candidate_source else 1
        topk_idxs = torch.empty((num_tokens, self.index_topk), dtype=torch.int32, device=dev)
        topk_scores = torch.empty((num_tokens, self.index_topk), dtype=torch.float32, device=dev)
        topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=dev)
        cand_idxs = torch.empty((num_tokens, cand_capacity), dtype=torch.int32, device=dev)
        cand_scores = torch.empty((num_tokens, cand_capacity), dtype=torch.float32, device=dev)
        cand_lens = torch.empty(num_tokens, dtype=torch.int32, device=dev)

        max_seq = int(md.seq_lens_cpu[: md.num_reqs].max().item()) if md.num_reqs else 0
        ixfops.ds4_v41_paged_index_select_out(
            q.contiguous(),
            index_k,
            weights.contiguous(),
            md.token_to_req_indices,
            md.seq_lens,
            md.block_table,
            positions.to(torch.int64),
            cand_in_idxs,
            cand_in_lens,
            topk_idxs,
            topk_scores,
            topk_lens,
            cand_idxs,
            cand_scores,
            cand_lens,
            ratio,
            self.index_topk,
            cand_capacity,
            self.candidate_block_size,
            int(self.uses_candidates),
            self.is_candidate_source,
            tp_rank,
            tp_size,
            max_seq // ratio,
        )

        def merge(
            idxs: torch.Tensor,
            scores: torch.Tensor,
            lens: torch.Tensor,
            capacity: int,
            dst_idxs: torch.Tensor,
            dst_lens: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out_idxs = dst_idxs[:num_tokens]
            out_lens = dst_lens[:num_tokens]
            if tp_size == 1:
                out_idxs.copy_(idxs)
                out_lens.copy_(lens)
                return out_idxs, out_lens
            out_scores = torch.empty((num_tokens, capacity), dtype=torch.float32, device=dev)
            ixfops.ds4_v41_merge_topk_out(
                tensor_model_parallel_all_gather(idxs, dim=1),
                tensor_model_parallel_all_gather(scores, dim=1),
                tensor_model_parallel_all_gather(lens.unsqueeze(1), dim=1),
                out_idxs,
                out_scores,
                out_lens,
                capacity,
            )
            return out_idxs, out_lens

        if self.is_candidate_source:
            merge(
                cand_idxs, cand_scores, cand_lens, self.candidate_topk_blocks,
                self.candidate_idxs_buf, self.candidate_lens_buf,
            )
        return merge(
            topk_idxs, topk_scores, topk_lens, self.index_topk,
            self.topk_idxs_buf, self.topk_lens_buf,
        )

    def _select_mma(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
        md: V41AttentionMetadata,
        index_k: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score on the MMA units against a dense row, then radix top-k.

        The fused kernel scores at 1.5 TFLOPS against this device's 11.2 TFLOPS
        fp32 FMA ceiling, and the shape is the reason: it holds q in shared
        memory and broadcasts it, so every FMA needs a companion shared-memory
        read and half the issue slots go to loads. ``mqa_logits`` computes the
        same ReLU-weighted head sum out of a tensor-core GEMM at 133 TFLOPS.

        What the fused kernel buys with that rate is a footprint -- it keeps a
        running top-k per token and never materialises a score row, which at 128k
        would be seq_len**2. Here the row exists but the token axis is split to a
        byte budget, so the peak allocation is a constant rather than a function
        of context.

        Scoring every group on every rank also retires the cross-rank merge. The
        ranks already all-gather q, so each one holds all 32 heads and can select
        independently; ``ds4_v41_merge_topk_out`` (2.6 ms a call at 8k tokens) and
        its three all-gathers disappear with it.
        """
        dev = q.device
        ratio = self.compress_ratio
        block = md.block_size
        seq_cpu, qsl_cpu = md.seq_lens_cpu, md.query_start_loc_cpu
        max_seq = int(seq_cpu[: md.num_reqs].max())

        # The stock causal mask is `col <= i + (kv_len - q_len)`: slope 1 in the
        # token domain. A compressed layer needs slope 1/ratio, which no pair of
        # seqlen arrays can express, so the mask is left a superset and the exact
        # bound is applied by `visible` in the selection below. The widened mask
        # reaches pages the compressed block table does not list, so pad it with
        # one it does -- those columns score against the wrong key and are cut
        # before anything reads them.
        pages = (max_seq + block - 1) // block
        bt = md.block_table
        if bt.size(1) < pages:
            bt = torch.cat(
                [bt, bt[:, :1].expand(bt.size(0), pages - bt.size(1))], dim=1
            ).contiguous()
        width = pages * block

        # Compressed-domain count of groups this token may see. A prefill token's
        # position is always inside its own sequence, so the fused kernel's second
        # clamp against seq_lens is redundant here.
        visible = (positions.to(torch.int32) + 1) // ratio
        topk = self.index_topk
        idxs = self.topk_idxs_buf[:num_tokens]
        zeros = torch.zeros(num_tokens, dtype=torch.int32, device=dev)

        rows = max(1, min(num_tokens, _MMA_INDEX_BUDGET // (width * 2)))
        chunks = _mma_index_chunks(qsl_cpu, seq_cpu, md.num_reqs, rows)
        scratch = torch.empty((rows, width), dtype=torch.bfloat16, device=dev)
        for t0, t1, cu_q, cu_kv, reqs in chunks:
            n = t1 - t0
            # One block-table row per span, not per request: the scoring kernel
            # indexes it by its own batch index.
            rows_bt = bt[torch.tensor(reqs, dtype=torch.int32, device=dev)]
            ixfops.dsa_indexer_mqa_logits_with_blocks_bf16(
                q[t0:t1].contiguous(),
                torch.tensor(cu_q, dtype=torch.int32, device=dev),
                torch.tensor(cu_kv, dtype=torch.int32, device=dev),
                index_k,
                rows_bt,
                weights[t0:t1].contiguous(),
                scratch[:n],
                max(cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1)),
                max(cu_kv[i + 1] - cu_kv[i] for i in range(len(cu_kv) - 1)),
                width,
            )
            ixfops.dsa_update_topk_indices(
                scratch[:n], zeros[:n], visible[t0:t1], topk, idxs[t0:t1]
            )
        return idxs, torch.clamp(visible, max=topk, out=self.topk_lens_buf[:num_tokens])


class IluDeepseekV41Attention(nn.Module, AttentionLayerBase):
    """Q/KV/O projections, vLLM-owned caches, and the sparse MLA forward.

    ``compress_ratio > 0`` does not mean this layer compresses its own KV: only
    ``kv_source_layer_ids`` do; the rest read that cache.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        dtype = vllm_config.model_config.dtype
        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)

        self.layer_id = layer_id
        self.n_heads = config.num_attention_heads
        self.n_local_heads = config.num_attention_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratios[layer_id]
        self.softmax_scale = self.head_dim**-0.5
        self.max_seq_len = vllm_config.model_config.max_model_len
        self.eps = config.rms_norm_eps

        is_backbone = layer_id < config.num_hidden_layers
        self.is_kv_source = is_backbone and layer_id in config.kv_source_layer_ids
        self.is_index_source = is_backbone and layer_id in config.index_source_layer_ids
        self.compressor = None
        self.indexer = None
        # Top-k selections cross layers through buffers the model preallocates
        # and every layer writes in place. The V4 layer plumbing already passes
        # the indices one down; ``bind_index_buffers`` fills in its companions.
        self.topk_idxs_buf = topk_indices_buffer
        self.topk_lens_buf: torch.Tensor | None = None

        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32), requires_grad=False
        )
        self.attn_sink.weight_loader = self._load_attn_sink

        quant_config = vllm_config.quant_config
        # ``wq_a`` and ``wkv`` both read the layer's normed hidden state, so as two
        # linears they walk the same activation twice -- and under fp8 they also
        # quantize it twice. Merged (V4's ``fused_wqa_wkv``) that becomes one pass
        # and, at decode, one launch instead of two on each side.
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            config.hidden_size, [self.q_lora_rank, self.head_dim], bias=False,
            quant_config=quant_config, prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # replicated, like the two linears it replaces
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.wq_b",
        )
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False, quant_config=quant_config, prefix=f"{prefix}.wo_a",
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank, config.hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.wo_b",
        )

        if self.compress_ratio:
            original_seq_len = config.original_seq_len
            rope_theta = config.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, config.rope_theta
        self._freqs_key = (
            self.rope_head_dim, self.max_seq_len, original_seq_len, rope_theta,
            config.rope_factor, config.beta_fast, config.beta_slow,
        )
        self._cos_sin: torch.Tensor | None = None

        self.window_cache = V41WindowKVCache(
            head_size=self.head_dim, dtype=dtype, window_size=self.window_size,
            block_size=BLOCK_SIZE, prefix=f"{prefix}.window_kv",
        )
        # ``compress_ratio > 0`` on a non-source layer means it reads a cache some
        # earlier layer owns. Resolve whose here, while the layer list is still
        # construction-time knowledge, and keep only the prefix: registering the
        # same cache module under two layers would put it in the module tree twice.
        sfc = vllm_config.compilation_config.static_forward_context
        self._static_forward_context = sfc
        if prefix in sfc:
            raise ValueError(f"Duplicate layer name: {prefix}")
        sfc[prefix] = self

        # A kv-source layer owns the compressed-KV cache itself: the spec and the
        # handle live on the attention, so a borrower resolves the source layer
        # rather than a cache module hanging off it.
        self.kv_dtype = dtype
        self.kv_cache = torch.tensor([])
        self._compress_cache_prefix = ""
        if self.is_kv_source:
            if BLOCK_SIZE % self.compress_ratio:
                raise ValueError(
                    f"block_size {BLOCK_SIZE} must be a multiple of compress_ratio "
                    f"{self.compress_ratio}, or a page would hold a fractional row "
                    f"(layer {prefix})"
                )
            self._compress_cache_prefix = prefix
            self.compressor = IluDeepseekV41Compressor(
                config, self.compress_ratio, prefix=f"{prefix}.compressor",
                k_cache_prefix=self._compress_cache_prefix,
            )
        elif self.compress_ratio:
            self._compress_cache_prefix = self._source_prefix(
                prefix, config.kv_source_layer_ids, ""
            )
        if self.is_index_source:
            # Index keys are written from a compressor latent, so their cache
            # lives on the layers that are both a kv source and an index source.
            key_owners = sorted(
                set(config.kv_source_layer_ids) & set(config.index_source_layer_ids)
            )
            self.indexer = IluDeepseekV41Indexer(
                config, self.layer_id, self.compress_ratio,
                quant_config=vllm_config.quant_config, prefix=f"{prefix}.indexer",
                owns_k=layer_id in key_owners,
                key_cache_prefix=self._source_prefix(
                    prefix, key_owners, "indexer.index_k"
                ),
                static_forward_context=sfc,
            )

    def _source_prefix(
        self, prefix: str, source_layer_ids, suffix: str
    ) -> str:
        """Prefix of the nearest source layer at or below this one.

        The caches are published forward, so the source is the largest source
        index that does not exceed this layer's. It is always constructed before
        this layer, which is what makes the lookup resolvable here.
        """
        sources = [i for i in sorted(source_layer_ids) if i <= self.layer_id]
        assert sources, (
            f"layer {self.layer_id} reads {suffix} but no source layer precedes it"
        )
        src = _source_layer_prefix(prefix, sources[-1])
        if suffix:
            src += f".{suffix}"
        # This layer being its own source means the cache is built a few lines
        # below, so there is nothing to look up yet.
        if sources[-1] != self.layer_id and src not in self._static_forward_context:
            raise NotImplementedError(
                f"layer {self.layer_id} reads {src!r}, which is not on this rank"
            )
        return src

    def bind_index_buffers(
        self,
        topk_lens: torch.Tensor,
        candidate_idxs: torch.Tensor | None,
        candidate_lens: torch.Tensor | None,
    ) -> None:
        """Hand this layer the shared selection buffers the model preallocated.

        Called once after construction, before any forward. The indices buffer
        already arrived through the V4 layer plumbing; these are the companions
        our kernels need, which take an explicit length instead of reading it off
        a padded row.
        """
        self.topk_lens_buf = topk_lens
        if self.indexer is not None:
            self.indexer.topk_idxs_buf = self.topk_idxs_buf
            self.indexer.topk_lens_buf = topk_lens
            self.indexer.candidate_idxs_buf = candidate_idxs
            self.indexer.candidate_lens_buf = candidate_lens

    def _compressed_kv(self) -> tuple[torch.Tensor, V41AttentionMetadata]:
        """The compressed KV cache this layer attends over, and its metadata."""
        cache = self._static_forward_context[self._compress_cache_prefix]
        return cache.kv_cache, get_forward_context().attn_metadata[
            self._compress_cache_prefix
        ]

    @property
    def cos_sin(self) -> torch.Tensor:
        if self._cos_sin is None:
            self._cos_sin = get_shared_cos_sin(self._freqs_key, self.attn_sink.device)
        return self._cos_sin

    def _load_attn_sink(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        rank = get_tensor_model_parallel_rank()
        shard = loaded.narrow(0, rank * self.n_local_heads, self.n_local_heads)
        param.data.copy_(shard.to(param.dtype))

    def _o_proj(self, o: torch.Tensor) -> torch.Tensor:
        """Grouped output projection: ``wo_a`` is block-diagonal over groups.

        V4 runs this as ``ds4_inv_rope_fp8_quant`` + ``fp8_per_channel_einsum`` over an
        fp8 ``wo_a``, which fits V4.1's geometry exactly (G=1, D=1024, R=4096 at TP8) and
        was measured here: decode moved 47.6 -> 46.3..47.6 ms, i.e. inside run-to-run
        noise, because the einsum only saves 4 MiB of weight traffic per layer. GSM8K over
        two runs came out 96.59% / 96.97% against 97.27% bf16 -- 0.9 sigma, so not proven,
        but consistently on the low side. The likely cause is that
        ``ds4_inv_rope_fp8_quant`` emits one scale per (group, token), i.e. 4096 values
        spanning 8 heads share a scale, where the other dense layers quantize RMSNorm
        output that is uniformly scaled by construction. No measurable speedup is not worth
        that, so this stays bf16.
        """
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        z = torch.einsum("bsgd,grd->bsgr", o.to(wo_a.dtype), wo_a)
        return self.wo_b(z.flatten(2))[0]

    def _write_compressed(self, latent: torch.Tensor, md: V41AttentionMetadata) -> None:
        rope_quant_insert(
            latent,
            md.positions,
            self.cos_sin,
            self.kv_cache,
            md.slot_mapping,
            self.compress_ratio,
            block_size=md.block_size,
            block_table=md.block_table,
            token_to_req_indices=md.token_to_req_indices,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Only kv-source layers own one: the rest read the source's through the
        # forward context, and cr == 0 layers are pure sliding-window attention.
        # The window cache stays a module of its own.
        if not self.is_kv_source:
            return None
        return MLAAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.kv_dtype,
            compress_ratio=self.compress_ratio,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return V41CompressorBackend

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        md_by_layer = get_forward_context().attn_metadata
        profiling = not isinstance(md_by_layer, dict)
        md: V41AttentionMetadata | None = (
            None if profiling else md_by_layer[self.window_cache.prefix]
        )
        spans = [] if md is None else _seq_spans(md)
        x = hidden_states
        num_tokens = x.size(0)
        pos32 = positions[:num_tokens].to(torch.int32)

        if self.indexer is not None and self.indexer.cos_sin is None:
            self.indexer.cos_sin = self.cos_sin

        qr_kv = self.fused_wqa_wkv(x)[0]
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)

        qr, kv = ixfops.rms_norm_qk(
            qr, kv, self.q_norm.weight, self.kv_norm.weight, self.eps
        )

        q = self.wq_b(qr)[0].unflatten(-1, (self.n_local_heads, self.head_dim))

        if profiling:
            o = q.new_zeros(q.shape)
            return self._o_proj(
                o.view(1, num_tokens, self.n_local_groups, -1)
            ).view(num_tokens, -1)

        # One launch for what was four: RoPE on q, then RoPE, fp8 qdq and the
        # window-cache insert on kv. The kv half never round-trips to memory
        # between the three steps, which is the point.
        ixfops.v41_qkv_rope_qdq_insert(
            q,
            self.window_cache.kv_cache,
            kv,
            md.slot_mapping,
            pos32,
            self.cos_sin,
            FP8_BLOCK_SIZE,
            "fp8_pow2",
        )

        if self.compress_ratio:
            if self.is_kv_source:
                latent = self.compressor(x)
                if latent is not None:
                    if self.is_index_source:
                        self.indexer.publish_keys(latent)
                    self._write_compressed(
                        latent,
                        get_forward_context().attn_metadata[
                            self._compress_cache_prefix
                        ],
                    )
            if self.is_index_source:
                group_idxs, group_lens = self.indexer.score_and_select(x, qr, pos32)
            else:
                # Reuse the selection of the nearest index source above.
                group_idxs = self.topk_idxs_buf[:num_tokens]
                group_lens = self.topk_lens_buf[:num_tokens]
        else:
            group_idxs = group_lens = None

        o = self._attend(q, kv, md, spans, group_idxs, group_lens, pos32)
        apply_rope(o.unsqueeze(0), pos32, self.cos_sin, inverse=True)
        return self._o_proj(
            o.view(1, num_tokens, self.n_local_groups, -1)
        ).view(num_tokens, -1)

    def _attend(
        self, q, window_kv, md: V41AttentionMetadata, spans, group_idxs, group_lens, positions
    ) -> torch.Tensor:
        nd = md.num_decode_tokens
        assert sum(t1 - t0 for _r, t0, t1, _sl in spans[: md.num_decodes]) == nd, (
            f"decode span boundary disagrees with num_decode_tokens: {nd} expected, "
            f"spans={[(r, t0, t1) for r, t0, t1, _ in spans]}, num_decodes={md.num_decodes}"
        )
        cache, cache_md = self._compressed_kv() if self.compress_ratio else (None, None)
        outs = []

        if nd:
            outs.append(
                decode_sparse_mla(
                    q[:nd].contiguous(),
                    self.window_cache.kv_cache,
                    md,
                    positions[:nd],
                    self.window_size,
                    self.attn_sink,
                    self.softmax_scale,
                    compress_ratio=self.compress_ratio,
                    compress_cache=cache,
                    compress_md=cache_md,
                    group_idxs=None if group_idxs is None else group_idxs[:nd],
                    group_lens=None if group_lens is None else group_lens[:nd],
                )
            )

        if md.num_prefills:
            pre_spans = [
                (r, t0 - nd, t1 - nd, sl) for r, t0, t1, sl in spans[md.num_decodes :]
            ]
            outs.append(
                prefill_attn(
                    q[nd:].contiguous(),
                    window_kv[nd:],
                    cache,
                    cache_md,
                    pre_spans,
                    self.compress_ratio or 1,
                    None if group_idxs is None else group_idxs[nd:],
                    None if group_lens is None else group_lens[nd:],
                    positions[nd:],
                    self.window_size,
                    self.attn_sink,
                    self.softmax_scale,
                    window_cache=self.window_cache.kv_cache,
                    window_md=md,
                )
            )

        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
