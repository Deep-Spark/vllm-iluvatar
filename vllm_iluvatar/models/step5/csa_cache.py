"""Allocator-owned Step5 CSA caches and metadata for vLLM 0.31."""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheLayout,
    MLAAttentionSpec,
)


@dataclass
class Step5CSAMetadata:
    query_start: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int


class Step5CSAMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.kv_cache_spec, (CircularBufferSpec, MLAAttentionSpec)):
            raise TypeError("Step5 CSA expects a ring or compressed full cache")
        if (
            isinstance(self.kv_cache_spec, MLAAttentionSpec)
            and self.kv_cache_spec.tokens_per_state != 8
        ):
            raise ValueError("Step5 CSA summaries compress exactly 8 tokens")

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Step5CSAMetadata:
        del common_prefix_len, fast_build
        return Step5CSAMetadata(
            common_attn_metadata.query_start_loc,
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.slot_mapping,
            self.kernel_block_size or self.kv_cache_spec.block_size,
        )


class Step5CSABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "Step5CSABackend"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("CSA cache layers have no attention implementation")

    @staticmethod
    def get_builder_cls():
        return Step5CSAMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [MultipleOf(8)]

    @classmethod
    def get_supported_head_sizes(cls):
        return [256, 512]

    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.LBHNC,)


class _Step5Cache(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        context = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate CSA layer name: {prefix}")
        context[prefix] = self

    def get_attn_backend(self):
        return Step5CSABackend


class Step5CSARing(_Step5Cache):
    def __init__(self, prefix: str, speculative_tokens: int = 0):
        super().__init__(prefix)
        if speculative_tokens < 0:
            raise ValueError("speculative_tokens must be nonnegative")
        self.minimum_capacity = max(11, 8 + speculative_tokens)
        self.requested_capacity = max(16, 1 << (self.minimum_capacity - 1).bit_length())

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        return CircularBufferSpec(
            block_size=self.requested_capacity,
            num_kv_heads=1,
            head_size=512,
            head_size_v=0,
            dtype=torch.bfloat16,
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        cache = kv_cache.squeeze(1)
        if (
            cache.ndim != 3
            or cache.shape[1] < self.minimum_capacity
            or cache.shape[2] != 512
            or (cache.dtype != torch.bfloat16)
            or (cache.stride(2) != 1)
            or (cache.stride(1) != 512)
            or (cache.stride(0) < cache.shape[1] * 512)
        ):
            raise ValueError("Invalid resolved Step5 CSA ring layout")
        self.kv_cache = cache


class Step5CSASummary(_Step5Cache):
    def __init__(self, prefix: str, cache_config):
        super().__init__(prefix)
        self.cache_config = cache_config

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        block_size = self.cache_config.block_size
        if block_size % 8:
            raise ValueError("Step5 CSA cache pages must be aligned to 8 tokens")
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.uint8,
            tokens_per_state=8,
            alignment=512,
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        cache = kv_cache.squeeze(1)
        if (
            cache.ndim != 3
            or cache.shape[2] != 256
            or cache.dtype != torch.uint8
            or (not cache.is_contiguous())
        ):
            raise ValueError("Invalid resolved Step5 CSA summary layout")
        self.kv_cache = cache


class Step5CSA(nn.Module):
    """Publish summaries using two caches owned by the native KV manager.

    The sparse indexer consumes this allocator-owned summary cache and enforces
    the causal historical bound. The explicit dense S1 mode does not use CSA.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        speculative = vllm_config.speculative_config
        count = speculative.num_speculative_tokens if speculative is not None else 0
        self.ring = Step5CSARing(f"{prefix}.ring", count)
        self.summary = Step5CSASummary(f"{prefix}.summary", vllm_config.cache_config)

    def forward(
        self, keys: torch.Tensor, logits: torch.Tensor, positions: torch.Tensor
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return
        ring_meta = metadata[self.ring.prefix]
        summary_meta = metadata[self.summary.prefix]
        if ring_meta.block_size != self.ring.kv_cache.shape[1]:
            raise ValueError("CSA ring metadata differs from resolved capacity")
        if summary_meta.block_size != self.summary.kv_cache.shape[1] * 8:
            raise ValueError("CSA summary metadata differs from resolved page size")
        from ixformer.inference.functions.models.step5.cache.fp8.csa_cache_update import (
            csa_cache_update,
        )

        csa_cache_update(
            keys,
            logits,
            positions,
            ring_meta.query_start,
            ring_meta.block_table[:, 0],
            summary_meta.slot_mapping,
            self.ring.kv_cache,
            self.summary.kv_cache.view(-1, 256),
            slot_scale=8,
            backend="ixformer_native",
        )
