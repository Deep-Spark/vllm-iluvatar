# Source: vllm_iluvatar/v1/attention/backends/ilu_gdn_attn.py at 47fb804; paired ixaite metadata and state.
# SPDX-License-Identifier: Apache-2.0
"""Iluvatar GDN attention backend metadata.
Corresponds to: vllm/v1/attention/backends/gdn_attn.py
Baseline: main_6e448d0
Why: Iluvatar GDN kernels need CPU query starts and an output reordering index
     for mixed speculative/non-speculative batches.
How: Reuse the official ``GDN_ATTN`` enum member via platform backend
     registration and extend only the metadata/builder surface consumed by the
     Iluvatar Qwen GDN OOT layer.
Scope: enabled only when ``VLLM_ILU_GDN_ATTN_BACKEND`` is true.
Remove: when community GDN metadata exposes the same ordering information.
"""

from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec


def _n_chunk_max_from_cu(query_start_loc_cpu: torch.Tensor | None) -> int:
    """Host-side ceil(max_seq_len / 16). No device D2H."""
    if query_start_loc_cpu is None or query_start_loc_cpu.numel() < 2:
        return 1
    lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    if lens.numel() == 0:
        return 1
    max_len = int(lens.max().item())
    return max(1, (max_len + 15) // 16)


def _pack_int32_1d(
    src: torch.Tensor | None, buf: torch.Tensor | None
) -> torch.Tensor | None:
    """Once-per-step int32 packed 1D for ixformer ``check_tensor_int_contiguous``.

    Eager: ``query_start_loc[:B+1]`` is already packed; ``block_table[:, 0]``
    may have stride ``max_blocks``. Copy the latter into the builder's
    persistent graph buffer (same storage FULL decode will capture). Do not
    realloc per layer.

    FULL graph: the pad block already wrote a contiguous ``[:capture_bs]``
    view of that buffer. Leave it; shrinking to live ``B`` would break replay.
    """
    if src is None:
        return None
    if src.dim() != 1:
        src = src.reshape(-1)
    if src.dtype == torch.int32 and src.stride(0) == 1:
        return src
    n = int(src.numel())
    if buf is not None and n <= buf.numel():
        dst = buf[:n]
        dst.copy_(src, non_blocking=True)
        return dst
    return _dense_slot_ids(src.to(dtype=torch.int32))


def _dense_slot_ids(index: torch.Tensor) -> torch.Tensor:
    # Speculative mamba block tables are (num_spec+1) wide, so [:, 0] carries
    # that row stride. FlyDSL layout-dynamic memrefs need a stride-1 axis.
    # A single-request slice still reports is_contiguous(); clone/contiguous
    # keep the non-unit stride, so allocate a dense buffer and copy.
    if 1 in index.stride():
        return index
    dense = torch.empty(index.shape, dtype=index.dtype, device=index.device)
    if index.numel() > 0:
        dense.copy_(index)
    return dense


class IluGDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["IluGDNAttentionMetadataBuilder"]:
        return IluGDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class IluGDNAttentionMetadata(GDNAttentionMetadata):
    non_spec_query_start_loc_cpu: torch.Tensor | None = None
    output_index: torch.Tensor | None = None  # shape: [total_tokens,]
    spec_conv_state_indices: torch.Tensor | None = None
    spec_token_starts: torch.Tensor | None = None
    non_spec_token_starts: torch.Tensor | None = None
    n_chunk_max: int = 1
    n_chunk_total: int = 1
    # Dense first-column view used by causal_conv1d_update. Keep it separate
    # from the 2D state table so every GDN layer reuses the same packed vector.
    spec_conv_state_indices_tensor: torch.Tensor | None = None
    state_block_table: torch.Tensor | None = None
    initial_state_idx: torch.Tensor | None = None
    block_idx_first_scheduled_token: torch.Tensor | None = None
    block_idx_last_scheduled_token: torch.Tensor | None = None
    output_state_indices: torch.Tensor | None = None
    num_computed_tokens: torch.Tensor | None = None
    num_computed_tokens_cpu: torch.Tensor | None = None
    cache_block_size: int = 0


class IluGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.spec_conv_state_indices = torch.empty(
            self.decode_cudagraph_max_bs, dtype=torch.int32, device=device
        )
        self.apc_all = vllm_config.cache_config.mamba_cache_mode == "all"
        if self.apc_all and self.use_spec_decode:
            raise NotImplementedError(
                "GDN all prefix caching does not support speculative decoding"
            )
        # Persistent vectors are replay inputs: addresses stay fixed across steps.
        self.apc_vectors = torch.empty(
            (5, self.decode_cudagraph_max_bs), dtype=torch.int32, device=device
        )
        self.spec_conv_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> IluGDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        spec_sequence_masks_cpu: torch.Tensor | None = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = async_tensor_h2d(
                    spec_sequence_masks_cpu, device=query_start_loc.device
                )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            # >>> ILU MODIFICATION >>>
            # With a speculative config the mamba block table is num_spec + 1
            # wide, so this column slice carries that row stride. Downstream
            # consumers want a dense vector (view(-1, 1) reshapes, and the
            # FlyDSL kernels need a stride-1 axis), so densify once per step
            # rather than per layer.
            non_spec_state_indices_tensor = _pack_int32_1d(
                block_table_tensor[:, 0],
                getattr(self, "non_spec_state_indices_tensor", None),
            )
            # <<< ILU MODIFICATION <<<
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
            # >>> ILU MODIFICATION >>>
            output_index = None
            # <<< ILU MODIFICATION <<<
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                # >>> ILU MODIFICATION >>>
                output_index = None
                # <<< ILU MODIFICATION <<<
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ].contiguous()
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                # >>> ILU MODIFICATION >>>
                index_range = torch.arange(
                    index.numel(), dtype=torch.int32, device=index.device
                )
                output_index = torch.empty_like(index_range)
                output_index.index_copy_(
                    0, non_spec_token_indx, index_range[:num_non_spec_tokens]
                )
                output_index.index_copy_(
                    0, spec_token_indx, index_range[num_non_spec_tokens:]
                )
                non_spec_token_indx = non_spec_token_indx.to(torch.int32)
                spec_token_indx = spec_token_indx.to(torch.int32)
                # <<< ILU MODIFICATION <<<

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ].contiguous()
                # >>> ILU MODIFICATION >>>
                # Same column-slice stride as the non-spec-only path above.
                non_spec_state_indices_tensor = _pack_int32_1d(
                    block_table_tensor[~spec_sequence_masks_cpu, 0],
                    getattr(self, "non_spec_state_indices_tensor", None),
                )
                # <<< ILU MODIFICATION <<<

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        spec_conv_state_indices_tensor = (
            None
            if spec_state_indices_tensor is None
            else _dense_slot_ids(spec_state_indices_tensor[:, 0])
        )

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        n_chunk_total = 1

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
            assert non_spec_query_start_loc_cpu is not None
            query_lens_cpu = (
                non_spec_query_start_loc_cpu[1:] - non_spec_query_start_loc_cpu[:-1]
            )
            chunk_counts_cpu = torch.div(
                query_lens_cpu + 15, 16, rounding_mode="floor"
            ).to(torch.int32)
            chunk_offsets_cpu = torch.zeros(
                chunk_counts_cpu.numel() + 1, dtype=torch.int32
            )
            torch.cumsum(chunk_counts_cpu, dim=0, out=chunk_offsets_cpu[1:])
            n_chunk_total = max(1, int(chunk_offsets_cpu[-1].item()))
            chunk_offsets = async_tensor_h2d(
                chunk_offsets_cpu, device=query_start_loc.device
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
            has_initial_state = has_initial_state.to(torch.int32)
        else:
            has_initial_state = None

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_sequence_masks is not None
            assert spec_conv_state_indices_tensor is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_conv_state_indices_tensor[:num_spec_decodes].copy_(
                spec_conv_state_indices_tensor, non_blocking=True
            )
            spec_conv_state_indices_tensor = self.spec_conv_state_indices_tensor[
                :batch_size
            ]
            spec_conv_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        # G7: pack after graph padding so capture views stay fixed-shape.
        idx_buf = getattr(self, "non_spec_state_indices_tensor", None)
        cu_buf = getattr(self, "non_spec_query_start_loc", None)
        non_spec_state_indices_tensor = _pack_int32_1d(
            non_spec_state_indices_tensor, idx_buf
        )
        non_spec_query_start_loc = _pack_int32_1d(non_spec_query_start_loc, cu_buf)
        spec_conv_state_indices = None
        if spec_state_indices_tensor is not None:
            spec_conv_state_indices = _pack_int32_1d(
                spec_state_indices_tensor[:, 0],
                getattr(self, "spec_conv_state_indices", None),
            )
        spec_token_starts = None
        non_spec_token_starts = None
        if spec_sequence_masks_cpu is not None and (
            num_prefills > 0 or num_decodes > 0
        ):
            starts_cpu = query_start_loc_cpu[:-1]
            spec_token_starts = async_tensor_h2d(
                starts_cpu[spec_sequence_masks_cpu].to(torch.int32),
                device=query_start_loc.device,
            )
            non_spec_token_starts = async_tensor_h2d(
                starts_cpu[~spec_sequence_masks_cpu].to(torch.int32),
                device=query_start_loc.device,
            )

        attn_metadata = IluGDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            non_spec_query_start_loc_cpu=non_spec_query_start_loc_cpu,
            output_index=output_index,
            spec_conv_state_indices=spec_conv_state_indices,
            spec_token_starts=spec_token_starts,
            non_spec_token_starts=non_spec_token_starts,
            n_chunk_max=_n_chunk_max_from_cu(non_spec_query_start_loc_cpu),
            n_chunk_total=n_chunk_total,
            spec_conv_state_indices_tensor=spec_conv_state_indices_tensor,
        )
        if self.apc_all:
            # Column arithmetic uses the actual hybrid Mamba block size.
            # Complete source blocks remain immutable; partial blocks belong to
            # one request and can be updated until the next boundary.
            block_size = self.kv_cache_spec.block_size
            computed = context_lens_tensor.to(torch.int32)
            initial = ((computed + block_size - 1) // block_size - 1).clamp(min=0)
            first = computed // block_size
            last = ((m.seq_lens + block_size - 1) // block_size - 1).clamp(min=0)
            source = block_table_tensor.gather(1, initial.long().unsqueeze(1)).squeeze(
                1
            )
            target = block_table_tensor.gather(1, last.long().unsqueeze(1)).squeeze(1)
            if num_prefills == 0 and self.use_full_cuda_graph:
                for dst, value in zip(
                    self.apc_vectors, (initial, first, last, computed, target)
                ):
                    dst[:batch_size].copy_(value)
                initial, first, last, computed, target = self.apc_vectors[
                    :, :batch_size
                ]
                self.non_spec_state_indices_tensor[:batch_size].copy_(source)
                source = self.non_spec_state_indices_tensor[:batch_size]
            attn_metadata.non_spec_state_indices_tensor = source
            attn_metadata.state_block_table = block_table_tensor
            attn_metadata.initial_state_idx = initial
            attn_metadata.block_idx_first_scheduled_token = first
            attn_metadata.block_idx_last_scheduled_token = last
            attn_metadata.output_state_indices = target
            attn_metadata.num_computed_tokens = computed
            attn_metadata.num_computed_tokens_cpu = (
                m.num_computed_tokens_cpu if num_prefills > 0 else None
            )
            attn_metadata.cache_block_size = block_size
        return attn_metadata
