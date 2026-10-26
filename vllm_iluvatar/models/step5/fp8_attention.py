"""Step5 sparse FP8 cache metadata and model-owned K/V scales."""

import json
import math
from dataclasses import dataclass
from typing import ClassVar

from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder

from vllm_iluvatar import envs
from vllm_iluvatar.v1.attention.backends.ilu_flash_attn import IluFlashAttentionBackend


def sparse_fp8_enabled():
    return envs.VLLM_ILU_STEP5_SPARSE_FP8


def sparse_fp8_cache_config(cache_config):
    from dataclasses import replace

    if cache_config is None or cache_config.cache_dtype != "auto":
        raise ValueError("Step5 sparse-only FP8 requires --kv-cache-dtype auto")
    return replace(cache_config, cache_dtype="fp8")


def configure_sparse_fp8_attention(attention, ring_capacity):
    """Keep 16-token sparse KV pages through hybrid allocator unification."""
    from types import MethodType

    attention.attn_backend = Step5SparseFP8Backend
    attention._ilu_step5_external_fp8_scales = True
    attention._step5_fp8_page_budget = max(
        16 * attention.num_kv_heads * 384 * 2, ring_capacity * 512 * 2
    )
    attention.get_kv_cache_spec = MethodType(sparse_fp8_cache_spec, attention)


def sparse_fp8_cache_spec(attention, vllm_config):
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

    if vllm_config.cache_config.block_size != 16:
        raise ValueError("Step5 sparse FP8 requires --block-size 16")
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=attention.num_kv_heads,
        head_size=192,
        head_size_v=192,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
        page_size_padded=attention._step5_fp8_page_budget,
    )


def load_scales(prefix):
    """Resolve fixed K/V scales once when constructing a sparse attention layer.

    Without a scale file, use unit scales for both cache writes and attention
    reads. An explicit file overrides that default and must contain valid,
    independent K/V scales for every requested layer (including MTP layers).
    This does not change the attention softmax scale or checkpoint weights.
    """
    path = envs.VLLM_ILU_STEP5_FP8_KV_SCALE_PATH
    if not path:
        return 1.0, 1.0
    with open(path, encoding="utf-8") as stream:
        table = json.load(stream)
    if not isinstance(table, dict):
        raise ValueError(f"Step5 K/V scale table must be a JSON object: {path}")
    entry = table.get(prefix)
    if not isinstance(entry, dict) or not {"k", "v"} <= entry.keys():
        raise ValueError(f"Missing independent K/V scales for {prefix} in {path}")
    result = []
    for role in ("k", "v"):
        value = entry[role]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (not math.isfinite(value))
            or (not 1e-12 <= value <= 3.402823e38)
        ):
            raise ValueError(f"Invalid Step5 {role} scale for {prefix}: {value!r}")
        result.append(float(value))
    return tuple(result)


@dataclass
class Step5SparseFP8Metadata:
    query_start_loc: object
    slot_mapping: object
    block_table: object
    max_seq_len: int
    max_query_len: int
    num_actual_tokens: int


class Step5SparseFP8MetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        meta = common_attn_metadata
        table = meta.block_table_tensor
        columns = min(table.shape[1], max(1, (meta.max_seq_len + 15) // 16))
        return Step5SparseFP8Metadata(
            meta.query_start_loc,
            meta.slot_mapping,
            table[:, :columns],
            meta.max_seq_len,
            meta.max_query_len,
            meta.num_actual_tokens,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        meta = self.build(0, common_attn_metadata)
        meta.block_table = common_attn_metadata.block_table_tensor
        meta.max_seq_len = self.vllm_config.model_config.max_model_len
        return meta


class Step5SparseFP8Backend(IluFlashAttentionBackend):
    @staticmethod
    def get_name():
        return "STEP5_SPARSE_FP8"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [16]

    @staticmethod
    def get_builder_cls():
        return Step5SparseFP8MetadataBuilder
