"""DeepSeek-V4.1-Flash attention: projections, caches, sparse MLA."""
from __future__ import annotations


import ixformer.inference.functions as ixfops
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import extract_layer_index

from vllm_iluvatar.models.deepseek_v41.compressor import IluDeepseekV41Compressor
from vllm_iluvatar.models.deepseek_v41.decode_wkv import (
    decode_attn_wkv,
    decode_wkv_enabled,
    prefill_as_decode_enabled,
)
from vllm_iluvatar.models.deepseek_v41.indexer import (
    IluDeepseekV41Indexer,
    select_candidate_blocks,
)
from vllm_iluvatar.models.deepseek_v41.paged_caches import (
    BLOCK_SIZE,
    V41AttentionMetadata,
    V41CompressKVCache,
    V41IndexKeyCache,
    V41WindowKVCache,
    compress_rope_qdq_insert,
    gather_group_rows,
    gather_window_rows,
    write_cache,
)
from vllm_iluvatar.models.deepseek_v41.rope import apply_rope, get_shared_cos_sin

FP8_BLOCK_SIZE = 32
FP4_BLOCK_SIZE = 32
# Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.
_COMPRESS_KV_FP4_BLOCK = 16
# ixinfer v3 on ivcore30 requires topk % 64 == 0.
_IXINFER_TOPK_MULTIPLE = 64


_STEP_N = 0

class PerSeqRuntime:

    def __init__(self) -> None:
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.topk_lens: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None
        # Borrowers cannot look metadata up: the key is the owning layer's prefix.
        self.compress_kv_md: V41AttentionMetadata | None = None
        self.index_k_md: V41AttentionMetadata | None = None


paged_runtime = PerSeqRuntime()


def _seq_spans(md: V41AttentionMetadata) -> list[tuple[int, int, int, int]]:
    qsl = md.query_start_loc_cpu.tolist()
    slens = md.seq_lens_cpu.tolist()
    spans = []
    for i in range(md.num_reqs):
        start, end = qsl[i], qsl[i + 1]
        if end > start:
            spans.append((i, start, end, slens[i]))
    return spans


def visible_groups(positions: torch.Tensor, ratio: int) -> torch.Tensor:
    return ((positions + 1) // ratio).to(torch.int32)


def index_scores(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    md: V41AttentionMetadata,
    spans: list[tuple[int, int, int, int]],
    ratio: int,
    n_groups: list[int],
) -> torch.Tensor:
    max_groups = max(n_groups) if n_groups else 0
    mid = q.new_zeros(q.size(0), q.size(1), max(max_groups, 1))
    if max_groups == 0:
        return mid

    for (req, t0, t1, _seq_len), n_g in zip(spans, n_groups):
        if n_g == 0:
            continue
        k_req = gather_group_rows(k_cache, md, req, 0, n_g, ratio)
        mid[t0:t1, :, :n_g] = torch.einsum("shd,td->sht", q[t0:t1], k_req)
    return mid


def topk_groups(
    index_score: torch.Tensor, visible: torch.Tensor, index_topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = index_score.size(0)
    topk = min(index_topk, index_score.size(-1))
    buf = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=index_score.device)
    ixfops.dsa_update_topk_indices(
        index_score.contiguous(), torch.zeros_like(visible), visible, topk, buf
    )
    return IluDeepseekV41Attention._compact_idxs(buf)


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

    o = ixfops.ds4_flash_mla_sparse_fwd(
        q,
        plane.unsqueeze(1),
        idxs.unsqueeze(1).contiguous(),
        scale,
        d_v=q.size(-1),
        attn_sink=attn_sink,
        topk_length=lens,
    )
    return o


class V41PagedIndexer(IluDeepseekV41Indexer):
    def __init__(
        self,
        config,
        layer_id: int,
        compress_ratio: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config, layer_id, compress_ratio, quant_config=quant_config, prefix=prefix
        )
        self.layer_id = layer_id
        if self.owns_k:
            self.key_cache = V41IndexKeyCache(
                head_size=config.index_head_dim,
                dtype=torch.get_default_dtype(),
                compress_ratio=compress_ratio,
                block_size=BLOCK_SIZE,
                prefix=f"{prefix}.index_k",
            )
        else:
            self.key_cache = None

    def publish_keys(
        self,
        latent: torch.Tensor,
        groups: list[tuple[int, range]],
        shared: PerSeqRuntime,
    ) -> None:
        md: V41AttentionMetadata = get_forward_context().attn_metadata[
            self.key_cache.prefix
        ]
        k = self.k_norm(self.wk(latent)[0])
        compress_rope_qdq_insert(
            self.key_cache.kv_cache, k, md, self.compress_ratio,
            FP4_BLOCK_SIZE, "fp4_pow2", self.cos_sin,
        )
        shared.index_k = self.key_cache.kv_cache
        shared.index_k_md = md

    def score_and_select(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        spans: list[tuple[int, int, int, int]],
        shared: PerSeqRuntime,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.cos_sin is not None
        assert shared.index_k is not None, "index keys were never published by a kv_source layer"
        ratio = self.compress_ratio

        q = self.wq_b(qr)[0].unflatten(-1, (self.n_local_heads, self.index_head_dim))
        apply_rope(q.unsqueeze(0), positions.to(torch.int32), self.cos_sin)
        ixfops.ds4_act_qdq(q, FP4_BLOCK_SIZE, "fp4_pow2")

        visible = visible_groups(positions, ratio)
        n_groups = [seq_len // ratio for _r, _a, _b, seq_len in spans]
        assert shared.index_k_md is not None, "index-key metadata was never published"
        mid = index_scores(q, shared.index_k, shared.index_k_md, spans, ratio, n_groups)

        weights = self.weights_proj(x)[0] * (self.softmax_scale * self.n_heads**-0.5)
        index_score = ixfops.ds4_index_score_reduce(
            mid.contiguous(), weights.contiguous(), visible
        )
        # ds4_index_score_reduce already writes -inf for s >= visible_lens[t],
        # so masking against ``visible`` again here only rescans the tensor --
        # [num_tokens, seq_len] fp32, 268 MiB at 8192 tokens on the candidate
        # source layer.
        #
        # Replicated index heads already give every rank the complete per-head
        # sum, so there is nothing to combine.
        #
        # This has to name the tensor-parallel group: the bare
        # ``torch.distributed.all_reduce`` this replaces used the default world
        # group, which happens to equal the TP group only when
        # pipeline_parallel_size == 1. Under PP the world group spans stages that
        # never reach this layer, so the collective hung in ixccl's bootstrap
        # waiting for ranks that would never join.
        if not self.heads_are_replicated and get_tensor_model_parallel_world_size() > 1:
            index_score = tensor_model_parallel_all_reduce(index_score)

        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(
                index_score.unsqueeze(0),
                visible.unsqueeze(-1),
                self.candidate_topk_blocks,
                self.candidate_block_size,
            ).squeeze(0)
        elif self.uses_candidates:
            assert shared.candidates is not None, "candidate source never published a mask"
            index_score.masked_fill_(~shared.candidates, -torch.inf)

        return topk_groups(index_score, visible, self.index_topk)


class IluDeepseekV41Attention(nn.Module):
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
        del topk_indices_buffer
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

        is_backbone = layer_id < config.num_hidden_layers
        self.is_kv_source = is_backbone and layer_id in config.kv_source_layer_ids
        self.is_index_source = is_backbone and layer_id in config.index_source_layer_ids
        self.compressor = None
        self.indexer = None
        self.shared = paged_runtime

        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32), requires_grad=False
        )
        self.attn_sink.weight_loader = self._load_attn_sink

        quant_config = vllm_config.quant_config
        self.wq_a = ReplicatedLinear(
            config.hidden_size, self.q_lora_rank, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.wq_a",
        )
        self.q_norm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.wq_b",
        )
        self.wkv = ReplicatedLinear(
            config.hidden_size, self.head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.wkv",
        )
        self.kv_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
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
        if self.is_kv_source:
            self.compress_cache = V41CompressKVCache(
                head_size=self.head_dim, dtype=dtype,
                compress_ratio=self.compress_ratio,
                block_size=BLOCK_SIZE, prefix=f"{prefix}.compress_kv",
            )
            self.compressor = IluDeepseekV41Compressor(
                config, self.compress_ratio, prefix=f"{prefix}.compressor"
            )
        else:
            self.compress_cache = None
        if self.is_index_source:
            self.indexer = V41PagedIndexer(
                config, self.layer_id, self.compress_ratio,
                quant_config=vllm_config.quant_config, prefix=f"{prefix}.indexer",
            )

    @property
    def cos_sin(self) -> torch.Tensor:
        if self._cos_sin is None:
            self._cos_sin = get_shared_cos_sin(self._freqs_key, self.attn_sink.device)
        return self._cos_sin

    def _load_attn_sink(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        rank = get_tensor_model_parallel_rank()
        shard = loaded.narrow(0, rank * self.n_local_heads, self.n_local_heads)
        param.data.copy_(shard.to(param.dtype))

    @staticmethod
    def _compact_idxs(idxs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense prefix of valid entries plus per-row lengths.

        ``ds4_flash_mla_sparse_fwd`` does not skip ``-1`` inside ``[0, length)``.
        """
        valid = idxs >= 0
        lens = valid.sum(-1, dtype=torch.int32)
        width = idxs.size(-1)
        dst = torch.where(valid, valid.cumsum(-1) - 1, width)
        out = idxs.new_full(idxs.shape[:-1] + (width + 1,), -1)
        out.scatter_(-1, dst, idxs)
        return out[..., :width].contiguous(), lens

    def _o_proj(self, o: torch.Tensor) -> torch.Tensor:
        """Grouped output projection: ``wo_a`` is block-diagonal over groups."""
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        z = torch.einsum("bsgd,grd->bsgr", o.to(wo_a.dtype), wo_a)
        return self.wo_b(z.flatten(2))[0]

    def _write_compressed(
        self, latent: torch.Tensor, groups: list[tuple[int, range]], md: V41AttentionMetadata
    ) -> None:
        compress_rope_qdq_insert(
            self.compress_cache.kv_cache, latent, md, self.compress_ratio,
            _COMPRESS_KV_FP4_BLOCK, "fp4_e4m3", self.cos_sin,
        )
        self.shared.compress_kv = self.compress_cache.kv_cache
        self.shared.compress_kv_md = md

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

        qr = self.q_norm(self.wq_a(x)[0])
        q = self.wq_b(qr)[0].unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rope(q.unsqueeze(0), pos32, self.cos_sin)

        kv = self.kv_norm(self.wkv(x)[0])
        apply_rope(kv.unsqueeze(0), pos32, self.cos_sin)
        ixfops.ds4_act_qdq(kv, FP8_BLOCK_SIZE, "fp8_pow2")

        if profiling:
            o = q.new_zeros(q.shape)
            return self._o_proj(
                o.view(1, num_tokens, self.n_local_groups, -1)
            ).view(num_tokens, -1)

        write_cache(self.window_cache.kv_cache, kv, md.slot_mapping)

        if self.compress_ratio:
            if self.is_kv_source:
                latent = self.compressor(x, spans)
                groups = [
                    (
                        req,
                        range(
                            *IluDeepseekV41Compressor.groups_completed(
                                seq_len - (t1 - t0), seq_len, self.compress_ratio
                            )
                        ),
                    )
                    for req, t0, t1, seq_len in spans
                ]
                groups = [(r, g) for r, g in groups if len(g)]
                if latent is not None:
                    if self.is_index_source:
                        self.indexer.publish_keys(latent, groups, self.shared)
                    self._write_compressed(
                        latent,
                        groups,
                        get_forward_context().attn_metadata[self.compress_cache.prefix],
                    )
                md_by = get_forward_context().attn_metadata
                self.shared.compress_kv = self.compress_cache.kv_cache
                self.shared.compress_kv_md = md_by[self.compress_cache.prefix]
                if self.is_index_source and self.indexer.key_cache is not None:
                    self.shared.index_k = self.indexer.key_cache.kv_cache
                    self.shared.index_k_md = md_by[self.indexer.key_cache.prefix]
            if self.is_index_source:
                self.shared.topk_idxs, self.shared.topk_lens = (
                    self.indexer.score_and_select(x, qr, pos32, spans, self.shared)
                )
            group_idxs, group_lens = self.shared.topk_idxs, self.shared.topk_lens
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
        cache = self.shared.compress_kv if self.compress_ratio else None
        cache_md = self.shared.compress_kv_md if self.compress_ratio else None
        outs = []

        if decode_wkv_enabled() and prefill_as_decode_enabled():
            # ``kv`` was already written to the window cache above, so there is
            # no plane to build and no decode / prefill split to make.
            return decode_attn_wkv(
                q.contiguous(),
                self.window_cache.kv_cache,
                md,
                positions,
                self.window_size,
                self.attn_sink,
                self.softmax_scale,
                compress_ratio=self.compress_ratio,
                compress_cache=cache,
                compress_md=cache_md,
                group_idxs=group_idxs,
                group_lens=group_lens,
            )

        if nd:
            dec_spans = spans[: md.num_decodes]
            if decode_wkv_enabled():
                outs.append(
                    decode_attn_wkv(
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
            else:
                outs.append(
                    prefill_attn(
                        q[:nd].contiguous(),
                        window_kv[:nd],
                        cache,
                        cache_md,
                        dec_spans,
                        self.compress_ratio or 1,
                        None if group_idxs is None else group_idxs[:nd],
                        None if group_lens is None else group_lens[:nd],
                        positions[:nd],
                        self.window_size,
                        self.attn_sink,
                        self.softmax_scale,
                        window_cache=self.window_cache.kv_cache,
                        window_md=md,
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
