"""DeepSeek-V4.1 FlashMLA sparse backend, metadata, and metadata builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.mla.sparse_swa import (
    _LAYER_TYPE_C1A,
    _LAYER_TYPE_C2A,
    _LAYER_TYPE_SWAONLY,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    is_quantized_main_spec,
    main_kv_format,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa import (
    IluDeepseekSparseSWABackend,
    IluDeepseekSparseSWAMetadataBuilder,
)

# v4.1 per-layer compress ratios: 0 = pure sliding window, 1 = full-length
# compressed cache, 2 = ratio-2 compressed cache. Ratio-1 and ratio-2 layers
# both attend over indexer topk indices into a shared compressed cache but
# differ in compressed page block size (block_size // ratio), so each needs its
# own FlashMLA tile-scheduler plan.
_V41_LAYER_TYPES: dict[int, str] = {
    0: _LAYER_TYPE_SWAONLY,
    1: _LAYER_TYPE_C1A,
    2: _LAYER_TYPE_C2A,
}


def deepseek_v41_layer_type(compress_ratio: int) -> str:
    layer_type = _V41_LAYER_TYPES.get(compress_ratio)
    if layer_type is None:
        raise ValueError(
            f"Unsupported DeepSeek V4.1 compress_ratio={compress_ratio}; "
            "expected 0, 1, or 2."
        )
    return layer_type


def _spec_compress_ratio(kv_cache_spec: AttentionSpec) -> int:
    """Read the compressed-cache ratio from this tree's MLA spec.

    Named ``compress_ratio`` up to 0.27.1, ``tokens_per_state`` from 0.30.1.
    """
    return int(getattr(kv_cache_spec, "tokens_per_state", 1))


def _spec_compressed_block_size(
    kv_cache_spec: AttentionSpec, compress_ratio: int
) -> int:
    """Compressed-domain page size (official ``num_states``).

    Up to 0.27.1 this was the ``storage_block_size`` property. In 0.30.1 that
    name became an optional override field defaulting to None, and the
    computed value moved to ``num_states``.
    """
    if hasattr(kv_cache_spec, "num_states"):
        return int(kv_cache_spec.num_states)
    return int(kv_cache_spec.block_size // compress_ratio)


class IluDeepseekV41SparseSWAMetadataBuilder(IluDeepseekSparseSWAMetadataBuilder):
    """SWA metadata builder base for v4.1.

    The shared builder classifies decode layer types by the v4.0 ratios
    (1 = SWA-only, 4, 128). v4.1 uses 0 / 1 / 2, so recompute the set of
    tile-scheduler plans from the v4.1 topology.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        compress_ratios = getattr(
            self.vllm_config.model_config.hf_config, "compress_ratios", None
        ) or [0]
        self._layer_types = {
            deepseek_v41_layer_type(int(ratio)) for ratio in compress_ratios
        }


class IluDeepseekV4SparseMLABackend(AttentionBackend):
    """DeepSeek-V4.1 sparse-MLA backend base.

    Subclasses ``AttentionBackend`` directly (not the V3.2
    ``FlashMLASparseBackend``): DeepSeek-V4.1 runs its own attention layer
    (``IluDeepseekV41Attention``), so it does not reuse the V3.2 builder or
    impl, and only needs to declare its own metadata builder, KV-cache
    layout, and the sparse-MLA capability flags.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        fmt = main_kv_format()
        if fmt is not None:
            rows = fmt.rows_per_page(512)
            return [rows, rows * 2]
        return [64]

    @staticmethod
    def get_builder_cls() -> type[IluDeepseekV4SparseMLAMetadataBuilder]:
        return IluDeepseekV4SparseMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[Any]:
        raise NotImplementedError(
            "IluDeepseekV4SparseMLABackend has no separate impl class; "
            "DeepSeek-V4.1 attention runs through IluDeepseekV41Attention."
        )

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # DeepSeek V4 layout: 448 NoPE + 64 RoPE = 512.
        return [512]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


@dataclass
class IluDeepseekV4FlashMLAMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int
    topk_tokens: int
    compressed_kv_format: Any = None


class IluDeepseekV4SparseMLAMetadataBuilder(
    AttentionMetadataBuilder[IluDeepseekV4FlashMLAMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        # Classify single-token queries (plus num_speculative_tokens via
        # supports_spec_as_decode=True) as decodes; longer queries go to prefill.
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.topk_tokens = self.model_config.hf_config.index_topk

        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.req_id_per_token_buffer = torch.empty(
            (max_num_batched_tokens,), dtype=torch.int32, device=device
        )

        self.compress_ratio = _spec_compress_ratio(self.kv_cache_spec)
        self.compressed_kv_format = (
            main_kv_format() if is_quantized_main_spec(kv_cache_spec) else None
        )
        if self.compressed_kv_format is not None:
            expected_rows = self.compressed_kv_format.rows_per_page(512)
            if (
                _spec_compressed_block_size(kv_cache_spec, self.compress_ratio)
                != expected_rows
            ):
                raise ValueError(
                    "Compressed KV metadata cannot split a physical payload page"
                )
        # Only kv-source layers own a compressed cache, and v4.1 compresses at
        # ratio 1 or 2.
        if self.compress_ratio not in (1, 2):
            raise ValueError(
                "DeepSeek V4.1 compressed-KV caches use compress_ratio 1 or 2; "
                f"got {self.compress_ratio}."
            )

        # Pre-allocate compressed slot mapping buffer for CUDA graph address
        # stability when compress_ratio > 1.
        if self.compress_ratio > 1:
            self.compressed_slot_mapping_buffer = torch.empty(
                max_num_batched_tokens, dtype=torch.int64, device=device
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> IluDeepseekV4FlashMLAMetadata:
        cm = common_attn_metadata
        req_id_per_token = cm.token_to_req_indices(self.req_id_per_token_buffer)

        slot_mapping = cm.slot_mapping
        if self.compress_ratio > 1:
            slot_mapping = get_compressed_slot_mapping(
                cm.num_actual_tokens,
                cm.slot_mapping,
                cm.query_start_loc,
                cm.seq_lens,
                (
                    cm.block_table_tensor
                    if self.compressed_kv_format is not None
                    else cm.block_table_tensor.clamp_(min=0)
                ),
                int(self.kv_cache_spec.num_states),
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )

        return IluDeepseekV4FlashMLAMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            compressed_kv_format=self.compressed_kv_format,
        )


class IluDeepseekV4FlashMLAMetadataBuilder(IluDeepseekV4SparseMLAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class IluDeepseekV4FlashMLABackend(IluDeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "ILU_FLASHMLA_SPARSE_DSV41"

    @staticmethod
    def get_builder_cls() -> type[IluDeepseekV4FlashMLAMetadataBuilder]:
        return IluDeepseekV4FlashMLAMetadataBuilder


class IluDeepseekSparseSWAFlashMLAMetadataBuilder(
    IluDeepseekV41SparseSWAMetadataBuilder
):
    """SWA metadata for the FlashMLA decode path, which allows varlen decode."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class IluDeepseekSparseSWAFlashMLABackend(IluDeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[IluDeepseekSparseSWAFlashMLAMetadataBuilder]:
        return IluDeepseekSparseSWAFlashMLAMetadataBuilder
