# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention for Iluvatar platform."""

import copy
from dataclasses import dataclass
from typing import ClassVar, List, Union

import ixformer.inference.functions as ixf_ops
import numpy as np
import torch
import vllm.envs as envs
from ixformer.contrib.vllm_flash_attn import merge_attn_states  # type: ignore[import]
from vllm import _custom_ops as ops  # type: ignore[import]
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
    get_layers_from_vllm_config,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_sinks,
    flash_attn_varlen_func,
    flash_attn_varlen_int8_func,
    flash_attn_with_kvcache,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
    reshape_and_cache_flash,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)
from enum import Enum


class QueryLenSupport(Enum):
    """Defines the level of query length support for an attention backend's
    decode pipeline.

    - SINGLE_ONLY: Decode pipeline only supports single-token queries
                   (query_len=1)
    - UNIFORM: Decode pipeline supports uniform multi-token queries
               (all requests must have same query_len > 1)
    - VARLEN: Decode pipeline supports variable-length queries
              (mixed query lengths in same batch)
    """

    SINGLE_ONLY = "single_only"
    UNIFORM = "uniform"
    VARLEN = "varlen"


class IluFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return list(range(16, 257, 16))

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "ILU_FLASH_ATTN"

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["IluFlashAttentionImpl"]:
        return IluFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["IluFlashAttentionMetadataBuilder"]:
        return IluFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        if envs.VLLM_ATTN_OPT_LEVEL == 1:
            return (3, num_blocks, num_kv_heads, block_size, head_size)
        return (2, num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 2, 3, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    SUPPORTED_HEAD_SIZES = (64, 96, 128, 160, 192, 256, 512)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in cls.SUPPORTED_HEAD_SIZES

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return False
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        return None


@dataclass
class IluFlashAttentionPrefillMetadata:
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    key_start_loc: torch.Tensor
    max_query_len: int


@dataclass
class IluFlashAttentionDecodeMetadata:
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    max_decode_seq_len: int
    use_graph: bool
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None


@dataclass
class IluFlashAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    key_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None
    cu_prefix_kv_lens: torch.Tensor | None
    cu_suffix_kv_lens: torch.Tensor | None

    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None
    dcp_cu_seqlens_k: torch.Tensor | None = None

    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    prefill: IluFlashAttentionPrefillMetadata | None = None
    decode: IluFlashAttentionDecodeMetadata | None = None

    causal: bool = True


def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model."""
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, IluFlashAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class IluFlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[IluFlashAttentionMetadata]
):
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = True
    reorder_batch_threshold: ClassVar[int] = 1

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.attention_config = vllm_config.attention_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0
        self.aot_schedule = False

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size
        # Align decode/prefill split threshold with speculative decode query length
        # when backend supports treating spec requests as decode.
        supports_spec_decode = self.query_len_support != QueryLenSupport.SINGLE_ONLY
        self._init_reorder_batch_threshold(
            self.reorder_batch_threshold, supports_spec_decode
        )

        if self.dcp_world_size > 1:
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            self._dcp_context_kv_lens = torch.zeros(
                max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )
            self._dcp_cu_seqlens_k = torch.zeros(
                max_num_reqs + 1,
                dtype=torch.int32,
                device=self.device,
            )

        self.aot_sliding_window: tuple[int, int] | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> IluFlashAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        key_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=self.device)
        torch.cumsum(seq_lens[:num_reqs], dim=0, out=key_start_loc[1:])

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=(self.query_len_support != QueryLenSupport.VARLEN),
            )
        )
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_actual_tokens

        max_num_splits = 0

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None
        dcp_cu_seqlens_k = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None
        cu_prefix_kv_lens = None
        cu_suffix_kv_lens = None
        scheduler_metadata = None

        if self.dcp_world_size > 1:
            query_kv_lens = query_start_loc[1:] - query_start_loc[:-1]
            local_context_kv_lens = get_dcp_local_seq_lens(
                seq_lens - query_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            self._dcp_context_kv_lens[:num_reqs] = local_context_kv_lens
            self._dcp_context_kv_lens[num_reqs:] = 0
            dcp_context_kv_lens = self._dcp_context_kv_lens[:num_reqs]

            self._dcp_cu_seqlens_k[0] = 0
            torch.cumsum(
                dcp_context_kv_lens,
                dim=0,
                dtype=torch.int32,
                out=self._dcp_cu_seqlens_k[1 : num_reqs + 1],
            )
            dcp_cu_seqlens_k = self._dcp_cu_seqlens_k[: num_reqs + 1]

            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            max_dcp_context_kv_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens],
                dtype=torch.int32,
                device=self.device,
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            cu_prefix_kv_lens = torch.tensor(
                [0, common_prefix_len],
                dtype=torch.int32,
                device=self.device,
            )
            cu_suffix_kv_lens = torch.zeros(
                num_reqs + 1, dtype=torch.int32, device=self.device
            )
            torch.cumsum(suffix_kv_lens, dim=0, out=cu_suffix_kv_lens[1:])

        prefill_metadata = None
        decode_metadata = None

        if num_prefills > 0:
            prefill_metadata = IluFlashAttentionPrefillMetadata(
                block_table=block_table_tensor[num_decodes:],
                query_start_loc=(query_start_loc[num_decodes:] - num_decode_tokens),
                key_start_loc=(
                    key_start_loc[num_decodes:] - key_start_loc[num_decodes]
                ),
                max_query_len=max_query_len,
            )

        if num_decodes > 0:
            decode_query_start_loc = query_start_loc[: num_decodes + 1]
            decode_query_lens = decode_query_start_loc[1:] - decode_query_start_loc[:-1]
            decode_seq_lens = seq_lens[:num_decodes]
            decode_cu_seqlens_q = torch.arange(
                0,
                num_decodes + 1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            decode_cu_seqlens_k = torch.zeros(
                num_decodes + 1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            decode_cu_seqlens_k[1:] = decode_seq_lens.cumsum(0)
            decode_metadata = IluFlashAttentionDecodeMetadata(
                block_table=block_table_tensor[:num_decodes],
                query_start_loc=decode_query_start_loc,
                seq_lens=decode_seq_lens,
                max_query_len=decode_query_lens.max().item(),
                max_decode_seq_len=int(decode_seq_lens.max().item()),
                use_graph=(
                    self.use_full_cuda_graph
                    and self.max_cudagraph_size is not None
                    and num_actual_tokens <= self.max_cudagraph_size
                ),
                cu_seqlens_q=decode_cu_seqlens_q,
                cu_seqlens_k=decode_cu_seqlens_k,
            )

        attn_metadata = IluFlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            key_start_loc=key_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            dcp_cu_seqlens_k=dcp_cu_seqlens_k,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            cu_prefix_kv_lens=cu_prefix_kv_lens,
            cu_suffix_kv_lens=cu_suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            prefill=prefill_metadata,
            decode=decode_metadata,
            causal=causal,
        )
        return attn_metadata

    def update_block_table(
        self,
        metadata: IluFlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> IluFlashAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        if new_metadata.decode is not None:
            new_decode = copy.copy(new_metadata.decode)
            new_decode.block_table = blk_table[: new_metadata.num_decodes]
            new_metadata.decode = new_decode
        if new_metadata.prefill is not None:
            new_prefill = copy.copy(new_metadata.prefill)
            new_prefill.block_table = blk_table[new_metadata.num_decodes :]
            new_metadata.prefill = new_prefill
        return new_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class IluFlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
        )
        logger.info_once(
            "Using IluFlashAttention (ixformer backend), FA version %s",
            self.vllm_flash_attn_version,
            scope="local",
        )
        self.batch_invariant_enabled = envs.VLLM_BATCH_INVARIANT

        self.sinks = sinks
        if self.sinks is not None:
            assert (
                flash_attn_supports_sinks()
            ), "Sinks are only supported in FlashAttention 3"
            assert self.sinks.shape[0] == num_heads

        self.supports_quant_query_input = True
        self.supports_per_head_quant_scales = (
            self.vllm_flash_attn_version >= 3
            if self.vllm_flash_attn_version is not None
            else False
        )
        assert envs.VLLM_ATTN_OPT_LEVEL in [
            0,
            1,
            2,
        ], "VLLM_ATTN_OPT_LEVEL only support [0 for non-quant, 1 for I8Q_I8K_F16V, 2 for I8Q_I8K_I8V] now! but got {}".format(
            envs.VLLM_ATTN_OPT_LEVEL
        )
        """
            quant_type = 0
                attention:f16 qkv
                cache:f16 kv cache
            quant_type = 2
                attention:int8q int8k int8v
                cache:
                    int8 k cache && fp32 k cache scale
                    int8 v cache && fp32 v cache scale(load from file, dont update)
            quant_type = 1
                attention:int8q int8k fp16v
                cache:
                    int8 k cache && fp32 k cache scale
                    fp16 v cache
        """
        self.quant_type = int(envs.VLLM_ATTN_OPT_LEVEL)

        vllm_config = get_current_vllm_config_or_none()
        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        self.dcp_combine = dcp_a2a_lse_reduce if dcp_a2a else cp_lse_ag_out_rs

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: IluFlashAttentionMetadata,
        output: torch.Tensor,
        kv_cache_scale: Union[torch.Tensor, List[torch.Tensor]] | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention (ixformer backend).

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.
            kv_cache_scale = [num_blocks, num_kv_heads, block_size] + [num_kv_heads, head_size]
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported")

        if attn_metadata is None:
            return output.fill_(0)

        softmax_scale: float = self.scale
        window_size = self.sliding_window
        alibi_slopes: torch.Tensor = self.alibi_slopes
        logits_soft_cap: float = self.logits_soft_cap

        attn_type = self.attn_type
        num_actual_tokens = attn_metadata.num_actual_tokens

        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_only = has_decode and not has_prefill
        if self.quant_type == 0:
            key_cache, value_cache = kv_cache.unbind(0)
        elif self.quant_type == 2:
            i8_key_cache, i8_value_cache = kv_cache.unbind(0)
            num_blocks, num_kv_heads, block_size, head_size = i8_key_cache.shape

            key_scale_cache, value_scale_cache = kv_cache_scale
            assert (
                key_scale_cache.shape == (num_blocks, num_kv_heads, block_size)
                and key_scale_cache.dtype == torch.float32
            ), f"key_scale_cache.shape {key_scale_cache.shape} != (num_blocks, num_kv_heads, block_size) or key_scale_cache.dtype {key_scale_cache.dtype} != torch.float32"
            assert (
                value_scale_cache.shape == (num_kv_heads, head_size)
                and value_scale_cache.dtype == torch.float32
            ), f"value_scale_cache.shape {value_scale_cache.shape} != (num_kv_heads, head_size) or value_scale_cache.dtype {value_scale_cache.dtype} != torch.float32"
            value_cache_info = (i8_value_cache, value_scale_cache)

        elif self.quant_type == 1:
            # key_cache 是 int8，value_cache 是 fp16
            i8_key_cache = kv_cache[0]
            num_blocks, num_kv_heads, block_size, head_size = i8_key_cache.shape
            value_cache = (
                kv_cache[1:]
                .view(query.dtype)
                .reshape(num_blocks, num_kv_heads, block_size, head_size)
            )
            key_scale_cache = kv_cache_scale
            value_cache_info = (value_cache, None)

        if is_quantized_kv_cache(self.kv_cache_dtype):
            dtype = IluFlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            prefill_output = output[num_decode_tokens:]
            decode_output = output[:num_decode_tokens]

            if self.quant_type == 2:
                if decode_only:
                    int8_query, query_scale = ixf_ops.scaled_int8_quant_for_attn(
                        query, 2, transpose_scale=False
                    )
                    i8_key, key_scale = ixf_ops.scaled_int8_quant_for_attn(
                        key, 2, transpose_scale=False
                    )
                    i8_value, _value_scale = ixf_ops.scaled_int8_quant_for_attn(
                        value, 0, transpose_scale=False, scale=value_cache_info[1]
                    )
                else:
                    int8_query, query_scale = ixf_ops.scaled_int8_quant_for_attn(
                        query, 2, transpose_scale=True
                    )
                    i8_key, key_scale = ixf_ops.scaled_int8_quant_for_attn(
                        key, 2, transpose_scale=False
                    )
                    i8_value, _value_scale = ixf_ops.scaled_int8_quant_for_attn(
                        value, 0, transpose_scale=False, scale=value_cache_info[1]
                    )
            elif self.quant_type == 1:
                """
                origin key cache
                    num_blocks, num_kv_heads, block_size, head_size  f16
                reformat key cache
                    key_cache_i8 : num_blocks, num_kv_heads, block_size, head_size  int8
                    key_scale_cache : num_blocks, num_kv_heads, block_size  fp32
                """

                if decode_only:
                    int8_query, query_scale = ixf_ops.scaled_int8_quant_for_attn(
                        query, 2, transpose_scale=False
                    )
                    i8_key, key_scale = ixf_ops.scaled_int8_quant_for_attn(
                        key, 2, transpose_scale=False
                    )
                else:
                    int8_query, query_scale = ixf_ops.scaled_int8_quant_for_attn(
                        query, 2, transpose_scale=True
                    )
                    i8_key, key_scale = ixf_ops.scaled_int8_quant_for_attn(
                        key, 2, transpose_scale=False
                    )
            else:
                if (
                    layer.quant_manager is not None
                    and layer.quant_manager.check_enable()
                ):
                    i8_value, value_scale = ixf_ops.scaled_int8_quant_for_attn(
                        value, 0, transpose_scale=False
                    )
                    layer.quant_manager.update_data(value_scale)

            # key and value may be None in the case of cross attention. They are
            # calculated once based on the output from the encoder and then cached
            # in KV cache.
            if (
                self.kv_sharing_target_layer_name is None
                and key is not None
                and value is not None
            ):
                # Reshape the input keys and values and store them in the cache.
                # Skip this if sharing KV cache with an earlier attention layer.
                # NOTE(woosuk): Here, key and value are padded while slot_mapping is
                # not padded. However, we don't need to do key[:num_actual_tokens]
                # and value[:num_actual_tokens] because the reshape_and_cache_flash
                # op uses the slot_mapping's shape to determine the number of
                # actual tokens.
                if self.quant_type == 2:
                    if has_prefill:
                        ixf_ops.reshape_and_cache_flash_int8(
                            key=i8_key,
                            value=i8_value,
                            k_scale=key_scale,
                            key_cache=i8_key_cache,
                            value_cache=value_cache_info[0],
                            key_scale_cache=key_scale_cache,
                            slot_mapping=attn_metadata.slot_mapping,
                            kv_cache_dtype="",
                        )
                elif self.quant_type == 1:
                    if has_prefill:
                        ixf_ops.reshape_and_cache_flash_mix(
                            key=i8_key,
                            value=value,
                            k_scale=key_scale,
                            key_cache=i8_key_cache,
                            value_cache=value_cache_info[0],
                            key_scale_cache=key_scale_cache,
                            slot_mapping=attn_metadata.slot_mapping,
                            kv_cache_dtype="",
                        )

                else:
                    ops.reshape_and_cache_flash(
                        key,
                        value,
                        key_cache,
                        value_cache,
                        attn_metadata.slot_mapping,
                        self.kv_cache_dtype,
                        layer._k_scale,
                        layer._v_scale,
                    )

            if has_prefill and attn_metadata.prefill is not None:
                prefill_q = query[num_decode_tokens:num_actual_tokens]
                prefill_k = key[num_decode_tokens:num_actual_tokens]
                prefill_v = value[num_decode_tokens:num_actual_tokens]
                prefill_out_slice = prefill_output[: prefill_q.shape[0]]
                sliding_window_size = (
                    list(self.sliding_window)
                    if self.sliding_window is not None
                    else (-1, -1)
                )
                if self.dcp_world_size > 1:
                    if self.quant_type > 0:
                        raise NotImplementedError(
                            "DCP prefill is not supported with int8 attention yet"
                        )
                    self._forward_prefill_with_dcp(
                        prefill_q,
                        prefill_k,
                        prefill_v,
                        prefill_out_slice,
                        attn_metadata,
                    )
                elif self.quant_type > 0:
                    flash_attn_varlen_int8_func(
                        q=int8_query[num_decode_tokens:],
                        k=i8_key_cache,
                        v=value_cache_info[0],
                        q_scale=query_scale[:, num_decode_tokens:],
                        k_scale=key_scale_cache,
                        v_scale=value_cache_info[1],
                        cu_seqlens_q=attn_metadata.prefill.query_start_loc,
                        cu_seqlens_k=attn_metadata.prefill.key_start_loc,
                        max_seqlen_q=attn_metadata.prefill.max_query_len,
                        max_seqlen_k=attn_metadata.max_seq_len,
                        softmax_scale=softmax_scale,
                        causal=attn_metadata.causal,
                        window_size=sliding_window_size,
                        alibi_slopes=alibi_slopes,
                        softcap=logits_soft_cap,
                        out=prefill_out_slice,
                        block_table=attn_metadata.prefill.block_table,
                        output_dtype=query.dtype,
                    )
                else:
                    flash_attn_varlen_func(
                        q=prefill_q,
                        k=key_cache,
                        v=value_cache,
                        out=prefill_out_slice,
                        cu_seqlens_q=attn_metadata.prefill.query_start_loc,
                        cu_seqlens_k=attn_metadata.prefill.key_start_loc,
                        max_seqlen_q=attn_metadata.prefill.max_query_len,
                        max_seqlen_k=attn_metadata.max_seq_len,
                        softmax_scale=softmax_scale,
                        causal=attn_metadata.causal,
                        alibi_slopes=alibi_slopes,
                        sinks=self.sinks,
                        window_size=sliding_window_size,
                        block_table=attn_metadata.prefill.block_table,
                        softcap=logits_soft_cap,
                    )

            if has_decode and attn_metadata.decode is not None:
                if self.dcp_world_size > 1:
                    self._forward_with_dcp(
                        query[:num_decode_tokens],
                        key[:num_decode_tokens],
                        value[:num_decode_tokens],
                        key_cache,
                        value_cache,
                        decode_output,
                        attn_metadata,
                    )
                elif self.quant_type in [1, 2]:
                    decode_q = query[:num_decode_tokens]
                    para_dict = dict(
                        output=decode_output,
                        query=int8_query[:num_decode_tokens],
                        key_cache=i8_key_cache,
                        query_scale=(
                            query_scale[:num_decode_tokens]
                            if decode_only
                            else query_scale[:, :num_decode_tokens].t().contiguous()
                        ),
                        key_scale_cache=key_scale_cache,
                        num_kv_heads=self.num_kv_heads,
                        scale=softmax_scale,
                        block_tables=attn_metadata.decode.block_table,
                        context_lens=attn_metadata.decode.seq_lens,
                        block_size=i8_key_cache.shape[-2],
                        softcap=logits_soft_cap,
                        alibi_slopes=alibi_slopes,
                        causal=True,
                        window_left=window_size[0],
                        window_right=window_size[1],
                        use_cuda_graph=(
                            attn_metadata.decode.use_graph if decode_only else False
                        ),
                        max_context_len=attn_metadata.decode.max_decode_seq_len,
                    )

                    if self.quant_type == 2:
                        para_dict.update(
                            dict(
                                value_cache=value_cache_info[0],
                                value_scale_cache=value_cache_info[1],
                            )
                        )
                        # for kv + k_scale write fusion
                        if decode_only:
                            para_dict.update(
                                dict(
                                    save_key=i8_key[:num_decode_tokens],
                                    save_value=i8_value[:num_decode_tokens],
                                    save_key_scale=key_scale[:num_decode_tokens],
                                )
                            )
                        ixf_ops.vllm_paged_attention_int8(**para_dict)
                    elif self.quant_type == 1:
                        para_dict.update(
                            dict(
                                value_cache=value_cache,
                            )
                        )
                        if decode_only:
                            para_dict.update(
                                dict(
                                    save_key=i8_key[:num_decode_tokens],
                                    save_value=value[:num_decode_tokens].contiguous(),
                                    save_key_scale=key_scale[:num_decode_tokens],
                                )
                            )
                        ixf_ops.vllm_paged_attention_mix(**para_dict)
                else:
                    decode_q = query[:num_decode_tokens]
                    if self.head_size >= 512:
                        sliding_window_size = (
                            list(self.sliding_window)
                            if self.sliding_window is not None
                            else (-1, -1)
                        )
                        flash_attn_varlen_func(
                            q=decode_q,
                            k=key_cache,
                            v=value_cache,
                            out=decode_output,
                            cu_seqlens_q=attn_metadata.decode.cu_seqlens_q,
                            cu_seqlens_k=attn_metadata.decode.cu_seqlens_k,
                            max_seqlen_q=1,
                            max_seqlen_k=attn_metadata.decode.max_decode_seq_len,
                            softmax_scale=softmax_scale,
                            causal=True,
                            alibi_slopes=alibi_slopes,
                            window_size=sliding_window_size,
                            block_table=attn_metadata.decode.block_table,
                            softcap=logits_soft_cap,
                        )
                    else:
                        flash_attn_with_kvcache(
                            q=decode_q.unsqueeze(1),
                            k_cache=key_cache,
                            v_cache=value_cache,
                            block_table=attn_metadata.decode.block_table,
                            cache_seqlens=attn_metadata.decode.seq_lens,
                            max_query_len=attn_metadata.decode.max_query_len,
                            cu_query_lens=attn_metadata.decode.query_start_loc,
                            softmax_scale=softmax_scale,
                            causal=True,
                            window_size=window_size,
                            alibi_slopes=alibi_slopes,
                            softcap=logits_soft_cap,
                            sinks=self.sinks,
                            out=decode_output.unsqueeze(1),
                            use_cuda_graph=attn_metadata.decode.use_graph,
                            max_context_len=attn_metadata.decode.max_decode_seq_len,
                        )

            return output

        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            cu_prefix_kv_lens=attn_metadata.cu_prefix_kv_lens,
            cu_suffix_kv_lens=attn_metadata.cu_suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            s_aux=self.sinks,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ):
            return

        key_cache, value_cache = kv_cache.unbind(0)

        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def _forward_dcp_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        block_table: torch.Tensor,
        dcp_cu_seqlens_k: torch.Tensor,
        max_dcp_context_kv_len: int,
        query_causal: bool,
    ) -> None:
        """DCP two-phase attention: context from paged KV + new Q/K/V, then merge."""
        query = query.contiguous()
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )

        # dcp_cu_seqlens_k is prepared in metadata build() so this forward path
        # only reads it. Writing cu_seqlens_k inside a captured CUDA graph and
        # then consuming it in flash_attn_varlen_func can hang on replay.
        fa_kwargs: dict = dict(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            out=None,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=dcp_cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_dcp_context_kv_len,
            softmax_scale=self.scale,
            sinks=self.sinks,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
        )
        context_attn_out, context_lse = flash_attn_varlen_func(**fa_kwargs)

        context_attn_out_cor, context_lse_cor = self.dcp_combine(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous().to(torch.float32)

        query_fa_kwargs: dict = dict(
            q=query,
            k=key,
            v=value,
            out=None,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            sinks=self.sinks,
            causal=query_causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
        )
        query_attn_out, query_lse = flash_attn_varlen_func(**query_fa_kwargs)
        query_lse = query_lse.to(torch.float32)
        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        merge_attn_states(
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
            output,
        )

    def _forward_prefill_dcp_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        query_causal: bool,
    ) -> None:
        query = query.contiguous()
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        query_fa_kwargs: dict = dict(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            sinks=self.sinks,
            causal=query_causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            return_softmax_lse=False,
        )
        flash_attn_varlen_func(**query_fa_kwargs)

    def _forward_prefill_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: IluFlashAttentionMetadata,
    ) -> None:
        assert attn_metadata.prefill is not None
        prefill_meta = attn_metadata.prefill
        self._forward_prefill_dcp_attention(
            query,
            key,
            value,
            output,
            cu_seqlens_q=prefill_meta.query_start_loc,
            max_seqlen_q=prefill_meta.max_query_len,
            query_causal=attn_metadata.causal,
        )

    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: IluFlashAttentionMetadata,
    ) -> None:
        assert attn_metadata.decode is not None
        decode_meta = attn_metadata.decode
        num_decodes = attn_metadata.num_decodes
        assert attn_metadata.dcp_cu_seqlens_k is not None
        assert attn_metadata.max_dcp_context_kv_len is not None
        self._forward_dcp_attention(
            query,
            key,
            value,
            key_cache,
            value_cache,
            output,
            cu_seqlens_q=decode_meta.cu_seqlens_q,
            max_seqlen_q=decode_meta.max_query_len,
            block_table=decode_meta.block_table,
            dcp_cu_seqlens_k=attn_metadata.dcp_cu_seqlens_k[: num_decodes + 1],
            max_dcp_context_kv_len=attn_metadata.max_dcp_context_kv_len,
            query_causal=True,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: IluFlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache."""
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
    dcp_world_size: int,
) -> bool:
    if common_prefix_len < 256:
        return False
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    if dcp_world_size > 1:
        return False

    num_queries_per_kv = num_query_heads // num_kv_heads
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        return True

    num_tokens = num_reqs
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    cu_prefix_kv_lens: torch.Tensor,
    cu_suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    assert sliding_window == (
        -1,
        -1,
    ), "Cascade attention does not support sliding window."

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-2]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0

    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        cu_seqlens_k=cu_prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=list(sliding_window),
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
    )

    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=list(sliding_window),
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
    )

    merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse, output)
