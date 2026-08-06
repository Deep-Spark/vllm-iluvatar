# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import ixformer.inference.functions as ixf_ops
import ixformer.inference.functions as ixfops
import numpy as np
import torch
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadata,
    get_mla_dims,
)
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_compressed_slot_mapping,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger("vllm." + __name__)

# For FP8 sparse attention we have two implementations:
# 1. Mixed batch mode: use the FP8 decode kernel for both prefill and decode this is
#    done by treating all tokens as single batch.
# 2. Separate prefill and decode mode: use the BF16 prefill kernel for prefill
#    (upconverting the FP8 cache to BF16 then calling the prefill kernel) and using
#    the FP8 decode kernel for decode.
# Currently we use #1 when the number of heads per rank is low (i.e. TP) since the BF16
# prefill kernel requires padding the number of heads to 128 while the decode does not
# so when the per ranke head count is below MIN_HEADS_FOR_BF16_PREFILL we use the mixed
# batch mode (#2).
MIN_HEADS_FOR_BF16_PREFILL = 32

"""
NOTE: FlashMLA Sparse uses an fp8 cache with the following format

In the "FP8 with scale" format, each token's KV cache is 656 Bytes,
structured as:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512
    `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values.
    The first `float32` is the scale for the first 128 `float8_e4m3` values,
    the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
    part is not quantized for accuracy.
"""


class IluFlashMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["IluFlashMLASparseMetadataBuilder"]:
        return IluFlashMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["IluFlashMLASparseImpl"]:
        return IluFlashMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [9, 10]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)


@dataclass
class IluFlashMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    # Decode/prefill split of the (already reordered) batch. Consumed by the
    # MoE decode dispatch (``_is_only_decode``) to pick the GEMV path.
    num_decodes: int
    num_prefills: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor | None
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048
    # DeepSeek-V4: compress_ratio carried per-group (4 / 128). Defaults to 1 so the
    # legacy V3.2 path (no compressor) keeps the same metadata shape.
    compress_ratio: int = 1

    # Pre-computed C128A topk metadata (only populated when compress_ratio == 128).
    # Decode rows: paged global slot ids + per-token valid lengths.
    c128a_global_decode_topk_indices: torch.Tensor | None = None
    c128a_decode_topk_lens: torch.Tensor | None = None
    # Prefill rows: local indices [0, n-1, -1, ...] consumed by
    # ds4_combine_topk_swa_indices.
    c128a_prefill_topk_indices: torch.Tensor | None = None

    # fp8-only (dormant for W4A8, stays None): merged all-token global-paged
    # C128A top-k, i.e. decode-global rows concatenated with globalized prefill
    # rows over ``[0:num_actual_tokens]``. Populated once per step by
    # ``IluDeepseekV4Fp8FlashMLASparseMetadataBuilder`` (方案 A) so the fp8
    # ``forward_mqa`` reads the ready-to-use extra top-k directly instead of
    # re-globalizing/merging per C128 layer. Both are views of fixed-address
    # builder buffers for CUDA-graph capture/replay.
    c128a_global_topk_indices: torch.Tensor | None = None  # [num_actual, W] int32
    c128a_topk_lens: torch.Tensor | None = None  # [num_actual] int32


def get_prefill_workspace_size(max_model_len: int):
    # NOTE(Lucas): 5 is a magic number for controlling the prefill buffer size.
    # May be tuned later.
    # Memory usage: 5 * max_model_len * 576 * 2 bytes
    #   Example: DeepSeek-V3.2 with max_model_len=163840 ->
    #            5 * 163840 * 576 * 2 = ~900 MB
    # This fits nicely below the typical MoE workspace size of >2GB so this is "free"
    return max_model_len * 5


_C128A_TOPK_ALIGNMENT = 128


class IluFlashMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[IluFlashMLASparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.device = device

        # Treat requests with query length <= 1 as decodes to match the
        # DeepGEMM indexer constraint (fp8_paged_mqa_logits only supports next_n <= 2)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # Shape: [max_num_seqs], all elements = topk_tokens (constant for full-CG)
        self.topk_tokens_tensor = torch.full(
            (max_num_seqs,), self.topk_tokens, device=device, dtype=torch.int32
        )
        # Shape: [max_num_seqs], all elements = max_model_len
        self.max_model_len_tensor = torch.full(
            (max_num_seqs,),
            self.model_config.max_model_len,
            device=device,
            dtype=torch.int32,
        )
        # this is ignored by `flash_mla_with_kvcache` if indices not None
        self.dummy_block_table = torch.empty(
            (max_num_seqs, 1), dtype=torch.int32, device=self.device
        )

        self.req_id_per_token_buffer = torch.empty(
            (max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

        # DeepSeek-V4 detection: hf_config carries compress_ratios for V4.
        hf_config = vllm_config.model_config.hf_config
        self.is_deepseek_v4 = (
            hasattr(hf_config, "compress_ratios")
            and len(getattr(hf_config, "compress_ratios", []) or []) > 0
        )
        self.compress_ratio = 1
        if self.is_deepseek_v4 and hasattr(self.kv_cache_spec, "compress_ratio"):
            self.compress_ratio = self.kv_cache_spec.compress_ratio

        # Pre-allocate compressed slot mapping buffer for CUDA graph address
        # stability when this group needs compression (compress_ratio > 1).
        if self.compress_ratio > 1:
            self.compressed_slot_mapping_buffer = torch.empty(
                max_num_batched_tokens,
                dtype=torch.int64,
                device=self.device,
            )

        # Pre-allocate C128A topk buffers (only this group needs them).
        if self.compress_ratio == 128:
            # Pad to B_TOPK alignment (128 covers h_q in {64, 128}). Padded slots
            # stay -1 and decode_lens caps them via topk_length so the extra
            # columns are no-op.
            c128a_max_compressed = cdiv(
                self.model_config.max_model_len, self.compress_ratio
            )
            c128a_max_compressed = (
                cdiv(c128a_max_compressed, _C128A_TOPK_ALIGNMENT)
                * _C128A_TOPK_ALIGNMENT
            )
            self.c128a_max_compressed = c128a_max_compressed
            self.c128a_global_decode_buffer = torch.empty(
                (max_num_batched_tokens, c128a_max_compressed),
                dtype=torch.int32,
                device=self.device,
            )
            self.c128a_decode_lens_buffer = torch.empty(
                max_num_batched_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            self.c128a_prefill_buffer = torch.empty(
                (max_num_batched_tokens, c128a_max_compressed),
                dtype=torch.int32,
                device=self.device,
            )

    def _build_c128a_metadata(
        self,
        cm: CommonAttentionMetadata,
        req_id_per_token: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
        num_prefill_tokens: int,
    ) -> dict[str, torch.Tensor | None]:
        """Pre-compute C128A topk metadata (decode + prefill in a single launch)."""
        num_total = num_decode_tokens + num_prefill_tokens
        if num_total == 0:
            return {}

        assert cm.positions is not None, (
            "positions is required for C128A metadata build"
        )
        # ixformer requires positions / slot_mapping / token_to_req to share
        # the same leading dimension (= num_total). CUDA-graph capture may
        # allocate longer tensors; always slice to the active token count.
        assert cm.slot_mapping.shape[0] >= num_total, (
            f"C128A slot_mapping length {cm.slot_mapping.shape[0]} < "
            f"num_total={num_total}"
        )
        assert req_id_per_token.shape[0] >= num_total, (
            f"C128A req_id_per_token length {req_id_per_token.shape[0]} < "
            f"num_total={num_total}"
        )

        # Compressed-domain block size used by the kernel for paged lookup.
        # ``self.kv_cache_spec.block_size`` is the per-group kernel-aligned block
        # size (V4 main MLA: 256), so the compressed page size is 256/128 = 2.
        block_size = self.kv_cache_spec.block_size // self.compress_ratio

        positions_int32 = cm.positions[:num_total]
        if positions_int32.dtype != torch.int32:
            positions_int32 = positions_int32.to(torch.int32)

        global_decode, decode_lens, prefill_local = (
            ixfops.ds4_build_c128a_topk_metadata(
                positions_int32,
                self.compress_ratio,
                num_decode_tokens,
                req_id_per_token[:num_total],
                cm.block_table_tensor[:num_decodes],
                block_size,
                cm.slot_mapping[:num_total],
                self.c128a_global_decode_buffer,
                self.c128a_decode_lens_buffer,
                self.c128a_prefill_buffer,
                max_compressed_tokens=self.c128a_max_compressed,
            )
        )

        result: dict[str, torch.Tensor | None] = {}
        if num_decode_tokens > 0:
            result["c128a_global_decode_topk_indices"] = global_decode.view(
                num_decode_tokens, 1, -1
            )
            result["c128a_decode_topk_lens"] = decode_lens
        if num_prefill_tokens > 0:
            result["c128a_prefill_topk_indices"] = prefill_local
        return result

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> IluFlashMLASparseMetadata:
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens
        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]

        # V4 main MLA: convert raw token slot_mapping to compressed-domain so
        # ``DeepseekCompressor._fused_kernel`` writes to the right cache slot.
        slot_mapping = cm.slot_mapping
        if self.compress_ratio > 1:
            slot_mapping = get_compressed_slot_mapping(
                cm.num_actual_tokens,
                cm.query_start_loc,
                cm.seq_lens,
                cm.block_table_tensor.clamp(min=0),
                int(self.kv_cache_spec.storage_block_size),
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )

        # Match SWA's decode split (no `require_uniform=True`) so the per-token
        # output rows line up with `q` in `_forward_decode` / `_forward_prefill`.
        (
            num_decodes,
            num_prefills,
            num_decode_tokens,
            num_prefill_tokens,
        ) = split_decodes_and_prefills(
            cm,
            decode_threshold=self.reorder_batch_threshold or 1,
        )

        c128a_fields: dict[str, torch.Tensor | None] = {}
        if self.is_deepseek_v4 and self.compress_ratio == 128:
            c128a_fields = self._build_c128a_metadata(
                cm,
                req_id_per_token,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
                num_prefill_tokens=num_prefill_tokens,
            )

        metadata = IluFlashMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,
            query_start_loc_cpu=cm.query_start_loc_cpu,
            seq_lens=cm.seq_lens,
            slot_mapping=slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            compress_ratio=self.compress_ratio,
            **c128a_fields,
        )

        return metadata


class IluFlashMLASparseImpl(SparseMLACommonImpl[IluFlashMLASparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim = mla_args["qk_rope_head_dim"]
        self.qk_head_dim = mla_args["qk_head_dim"]
        self.v_head_dim = mla_args["v_head_dim"]
        self.kv_b_proj = mla_args["kv_b_proj"]
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.softmax_scale = scale
        # The indexer carries the shared buffer for normal layers; skip-topk
        # layers reuse the explicitly-passed buffer from the previous layer.
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )
        # Prefill BF16 kernel requires 64 on Hopper, 128 on Blackwell
        self.prefill_padding = (
            128 if current_platform.is_device_capability_family(100) else 64
        )

        vllm_config = get_current_vllm_config()
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        q_concat_shape = (max_tokens, num_heads, head_size)

        (self.q_concat_buffer,) = current_workspace_manager().get_simultaneous(
            (q_concat_shape, torch.bfloat16),
        )
        self.use_int8_mla = envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == "INT8"

    def _k_up_proj(self, q_nope: torch.Tensor) -> torch.Tensor:
        # q_nope: (B, N, P), self.W_UK_T: (N, P, L)
        q_t = q_nope.transpose(0, 1)  # (N, B, P)
        ql = torch.bmm(q_t, self.W_UK_T)  # (N, B, L)
        return ql.transpose(0, 1)  # (B, N, L)

    def _v_up_proj(
        self, x: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (B, N, L), self.W_UV: (N, L, V)
        x_t = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        result = torch.bmm(x_t, self.W_UV)  # (N, B, V)
        return result.transpose(0, 1)  # (B, N, V)

    def forward_full(
        self,
        layer: "MLAAttention",
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        output: torch.Tensor,
        positions: torch.Tensor | None = None,
        kv_cache_scale: list[torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Iluvatar-optimized forward_impl aligned with v0.17.0.

        Key differences from the default path:
        - Rotary embedding is applied inside (not in the wrapper)
        - KV cache write is done inside (not in forward())
        - Prefill uses ixf_ops.mla_rope for fused rope
        - Decode uses ixf_ops.vllm_paged_attention_mla_fused
        """
        if attn_metadata is None:
            _ = torch.empty(
                (
                    layer.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )
            output = torch.empty(
                output.shape[0],
                self.v_head_dim * self.num_heads,
                device=q.device,
                dtype=q.dtype,
            )
            return output

        if self.dcp_world_size == -1:
            self.dcp_world_size = get_dcp_group().world_size

        num_actual_toks = attn_metadata.num_actual_tokens
        # Inputs and outputs may be padded for CUDA graphs
        k_pe = k_pe.unsqueeze(1)
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]
        positions = positions[:num_actual_toks, ...]
        num_mqa_tokens = q.size(0)
        if kv_cache_scale:
            kv_cache_scale = kv_cache_scale[0]
        if num_mqa_tokens > 0:
            q = q[:num_mqa_tokens]
            q_nope, q_pe = q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            q_pe, k_pe = layer.rotary_emb(positions[:num_actual_toks], q_pe, k_pe)

            q_nope = self._k_up_proj(q_nope)
            q_nope = q_nope.view(-1, self.num_heads, self.kv_lora_rank)
            q = torch.cat([q_nope, q_pe], dim=-1)
            if kv_cache.numel() > 0:
                if self.use_int8_mla:
                    # block_num, block_size, 2 = kv_cache_scale.shape
                    nope_scale_cache, rope_scale_cache = kv_cache_scale.unbind(2)
                    nope_scale_cache = nope_scale_cache.unsqueeze(-1)
                    rope_scale_cache = rope_scale_cache.unsqueeze(-1)
                    nope_i8, nope_scale = ixf_ops.scaled_int8_quant_for_attn(
                        k_c_normed.unsqueeze(1), 2
                    )
                    rope_i8, rope_scale = ixf_ops.scaled_int8_quant_for_attn(k_pe, 2)
                    kv_i8 = torch.cat([nope_i8, rope_i8], dim=-1).squeeze(1)
                    kv_cache_scale = (nope_scale_cache, rope_scale_cache)
                    torch.ops._C.concat_and_cache_dsa_int8(
                        kv_i8,
                        nope_scale,
                        rope_scale,
                        kv_cache,
                        nope_scale_cache,
                        rope_scale_cache,
                        attn_metadata.slot_mapping.flatten(),
                    )
                else:
                    ops.concat_and_cache_mla(
                        k_c_normed,
                        k_pe,
                        kv_cache,
                        attn_metadata.slot_mapping.flatten(),
                        kv_cache_dtype=self.kv_cache_dtype,
                        scale=layer._k_scale,
                    )
            attn_out = self.forward_mqa(
                q, kv_cache, attn_metadata, self, kv_cache_scale
            )
            output = torch.empty(
                output.shape[0],
                self.num_heads,
                self.v_head_dim,
                device=q.device,
                dtype=q.dtype,
            )
            output[:num_actual_toks] = self._v_up_proj(attn_out)

            return output.view(output.shape[0], self.v_head_dim * self.num_heads)

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: IluFlashMLASparseMetadata,
    ) -> torch.Tensor:
        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill).
        topk_indices = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        return self._bf16_flash_mla_kernel(q, kv_c_and_k_pe_cache, topk_indices)

    def flash_mla_sparse_prefill(
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        sm_scale: float,
        d_v: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sparse attention prefill kernel

        Args:
        - q: [s_q, h_q, d_qk], bfloat16
        - kv: [s_kv, h_kv, d_qk], bfloat16
        - indices: [s_q, h_kv, topk], int32.
            Invalid indices should be set to -1 or numbers >= s_kv
        - sm_scale: float
        - d_v: The dimension of value vectors. Can only be 512

        Returns:
        - (output, max_logits, lse)
            About the definition of output,
            max_logits and lse, please refer to README.md
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, 2-based log-sum-exp
        """
        results = ops.sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)
        return results

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # NOTE(Chen): kernel requires num_local_head to be a multiple of
        # 64 on hopper and 128 on blackwell
        if self.num_heads % self.prefill_padding != 0:
            assert self.prefill_padding % self.num_heads == 0
            logger.warning_once(
                f"Padding num_heads from {self.num_heads} to "
                f"{self.prefill_padding} for BF16 sparse prefill kernel"
            )
            q_padded = q.new_empty((q.shape[0], self.prefill_padding, q.shape[2]))
            q_padded[:, : self.num_heads, :] = q
            q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = self.flash_mla_sparse_prefill(
            q, kv_c_and_k_pe_cache, topk_indices, self.softmax_scale
        )
        output = output[:, : self.num_heads, :]
        return output

    def _forward_int8_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: IluFlashMLASparseMetadata,
        kv_cache_scale: torch.Tensor,
    ) -> torch.Tensor:
        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill).
        topk_indices = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        return self._int8_flash_mla_kernel(
            q, kv_c_and_k_pe_cache, topk_indices, kv_cache_scale
        )

    def _int8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kv_cache_scale: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        topk_indices = topk_indices.view(num_tokens, 1, -1)
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.reshape(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        nope_scale_cache, rope_scale_cache = kv_cache_scale
        nope_scale_cache = nope_scale_cache.reshape(-1, 1)
        rope_scale_cache = rope_scale_cache.reshape(-1, 1)
        i8_q, q_scale = ixf_ops.scaled_int8_quant_for_attn(
            q,
            2,
            transpose_scale=False,
        )
        return ixf_ops.sparse_flash_attn_int8(
            i8_q,
            kv_c_and_k_pe_cache,
            kv_c_and_k_pe_cache[..., :512],
            q_scale,
            nope_scale_cache,
            rope_scale_cache,
            topk_indices,
            self.softmax_scale,
            output_dtype=q.dtype,
        )

    def setup_projections(
        self,
        layer: "MLAAttention",
        act_dtype: torch.dtype,
    ) -> bool:
        return False

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: IluFlashMLASparseMetadata,
        layer: AttentionLayer,
        kv_cache_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        # Concatenate q if it's a tuple (ql_nope, q_pe)
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)

        num_actual_toks = q.shape[0]

        # Get topk indices
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        if self.use_int8_mla:
            attn_out = self._forward_int8_kv(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata, kv_cache_scale
            )
        else:
            attn_out = self._forward_bf16_kv(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )

        return attn_out


# ---------------------------------------------------------------------------
# DeepSeek-V4 (ilu W4A8) FlashMLA sparse backend
# ---------------------------------------------------------------------------
# V4 与 V3.2 在 FlashMLA sparse 路径上共享 metadata/builder/impl，区别仅在：
#   - backend 名称（用于 attention layer 选择）
#   - 我们禁用 fp8_ds_mla（W4A8 模型 KV cache 走 bf16）
#   - kernel block size 与 KV cache spec 对齐
# 通过派生在文件末尾追加，原 V3.2 路径完全不受影响。


class IluDeepseekV4FlashMLASparseBackend(IluFlashMLASparseBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",
    ]

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            return (num_blocks, block_size, 584)
        return (num_blocks, block_size, head_size)
