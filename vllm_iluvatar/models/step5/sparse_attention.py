"""Step5 sparse chain using native 0.31 caches and ixformer operators."""

import torch
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_iluvatar import envs

from .csa_cache import Step5CSA


def indexer_parallelism(tp_size: int, tp_rank: int, providers: int = 4):
    """Provider-shard Q/W; ranks sharing a KV head share its indexer heads."""
    if (
        any(type(value) is not int for value in (tp_size, tp_rank, providers))
        or tp_size < 1
        or (not 0 <= tp_rank < tp_size)
        or (providers != 4)
        or (providers % tp_size and tp_size % providers)
    ):
        raise ValueError("Unsupported Step5 indexer TP/provider geometry")
    size = min(tp_size, providers)
    return {"tp_size": size, "tp_rank": tp_rank // (tp_size // size)}


def validate_sparse_geometry(heads: int, groups: int, head_dim: int) -> None:
    """Fail before allocating caches for unsupported native per-rank GQA."""
    if (
        any(type(value) is not int for value in (heads, groups, head_dim))
        or head_dim != 192
        or groups not in (1, 2, 4)
        or (heads <= 0)
        or heads % groups
        or (heads // groups not in (8, 16))
    ):
        raise ValueError(
            f"Step5 native sparse GQA requires per-rank D192, G=1/2/4 and H/G=8/16; got H={heads}, G={groups}, D={head_dim}"
        )


class Step5SparseAttention(nn.Module):
    """No request/page ownership outside the vLLM KV manager.

    Projection parameters remain on Step5Attention with their checkpoint
    names. This module owns only derived RoPE and allocator-registered caches.
    Tensor allocations follow the normal vLLM Graph memory-pool lifecycle;
    every mathematical operation and cache write uses a native operator.
    """

    def __init__(self, vllm_config, config, prefix, rope_parameters):
        super().__init__()
        sparse = config.sparse_config
        required = {
            "proxy_dim": 256,
            "sparse_indexer_rope_dim": 32,
            "sparse_indexer_num_heads": 16,
            "sparse_indexer_num_k_heads": 1,
            "num_provider_groups": 4,
            "topk": 512,
            "region_block_size": 8,
        }
        for name, expected in required.items():
            if sparse.get(name, expected) != expected:
                raise ValueError(f"Unsupported Step5 sparse {name}")
        if not sparse.get("sparse_indexer_use_rope", True):
            raise ValueError("Step5 native indexer requires RoPE")
        self.fp8_backend = envs.VLLM_ILU_STEP5_FP8_ATTENTION_BACKEND
        self.layer_name = f"{prefix}.sparse_attention"
        self.attention_layer_name = f"{prefix}.attn"
        context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in context:
            raise ValueError(f"Duplicate Step5 sparse layer name: {self.layer_name}")
        context[self.layer_name] = self
        self.eps = config.rms_norm_eps
        self.max_model_len = vllm_config.model_config.max_model_len
        parameters = dict(rope_parameters)
        parameters["partial_rotary_factor"] = 32 / 256
        parameters["rope_dim"] = 32
        self.rotary = get_rope(
            head_size=256,
            max_position=config.max_position_embeddings,
            is_neox_style=True,
            rope_parameters=parameters,
            dtype=torch.bfloat16,
        )
        self.csa = Step5CSA(vllm_config, f"{prefix}.csa")
        from .fp8_attention import load_scales

        sk, sv = load_scales(self.attention_layer_name)
        device = self.rotary.cos_sin_cache.device
        self.register_buffer(
            "fp8_k_scale",
            torch.tensor(sk, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "fp8_v_scale",
            torch.tensor(sv, dtype=torch.float32, device=device),
            persistent=False,
        )

    def forward(
        self,
        query,
        key,
        value,
        index_q,
        index_k,
        index_z,
        weights,
        positions,
        q_weight,
        k_weight,
        k_bias,
    ):
        output = torch.empty(query.shape, dtype=torch.bfloat16, device=query.device)
        torch.ops.vllm.ilu_step5_sparse_attention(
            query,
            key,
            value,
            index_q,
            index_k,
            index_z,
            weights,
            positions,
            q_weight,
            k_weight,
            k_bias,
            output,
            self.layer_name,
        )
        return output

    def attention_impl(
        self,
        query,
        key,
        value,
        index_q,
        index_k,
        index_z,
        weights,
        positions,
        q_weight,
        k_weight,
        k_bias,
        output,
    ) -> None:
        """Run the metadata-dependent chain behind the opaque-op boundary."""
        from ixformer.inference.functions.models.step5.attention.fp8.sparse_gqa_fp8 import (
            SparseGQAWorkspace,
            sparse_gqa_fp8,
            sparse_gqa_fp8_workspace_size,
        )
        from ixformer.inference.functions.models.step5.indexer.float.top512 import (
            top512,
            top512_workspace_size,
        )
        from ixformer.inference.functions.models.step5.indexer.fp8.indexer_norm_rope import (
            indexer_norm_rope,
        )
        from ixformer.inference.functions.models.step5.indexer.fp8.indexer_scores import (
            indexer_scores,
        )

        tokens, heads, _ = query.shape
        groups = key.shape[1]
        context = get_forward_context()
        attention = context.no_compile_layers[self.attention_layer_name]
        metadata = context.attn_metadata
        meta = None if metadata is None else metadata[self.attention_layer_name]
        bound = self.max_model_len if meta is None else meta.max_seq_len
        width = max(2, (bound + 15) // 16 * 2)
        q_out = torch.empty(
            (tokens, groups * 4, 256), dtype=torch.uint8, device=query.device
        )
        k_out = torch.empty((tokens, 256), dtype=torch.bfloat16, device=query.device)
        z_out = torch.empty_like(k_out)
        scores = torch.empty(
            (tokens, groups, width), dtype=torch.bfloat16, device=query.device
        )
        starts = torch.empty(tokens * groups, dtype=torch.int32, device=query.device)
        ends = torch.empty_like(starts)
        selected = torch.empty(
            (tokens, groups, 512), dtype=torch.int32, device=query.device
        )
        scratch = torch.empty(
            top512_workspace_size(tokens, groups, width),
            dtype=torch.int32,
            device=query.device,
        )
        regions = torch.empty(
            (tokens, groups, 513), dtype=torch.int32, device=query.device
        )
        counts = torch.empty((tokens, groups), dtype=torch.int32, device=query.device)
        columns = (
            (self.max_model_len + 15) // 16
            if meta is None
            else meta.block_table.shape[1]
        )
        sizes = sparse_gqa_fp8_workspace_size(
            tokens, heads, groups, 513, columns, backend=self.fp8_backend
        )
        fp8_workspace = SparseGQAWorkspace(
            **{
                name: torch.empty(
                    shape, dtype=getattr(torch, dtype), device=query.device
                )
                for name, (shape, dtype) in sizes.items()
            }
        )
        if meta is None:
            return
        dtype = torch.float8_e4m3fn
        cache = attention.kv_cache
        if cache.dtype == torch.uint8:
            cache = cache.view(torch.float8_e4m3fn)
        if cache.dtype != dtype or attention.kv_cache_dtype not in ("fp8", "fp8_e4m3"):
            raise ValueError(f"Step5 sparse GQA cache must have dtype {dtype}")
        if cache.ndim != 4 or cache.shape[1] != groups or cache.shape[-1] != 384:
            raise ValueError("Step5 sparse GQA expects packed HND [pages,G,N,384]")
        key_cache, value_cache = (cache[..., :192], cache[..., 192:])
        indexer_norm_rope(
            index_q.view(tokens, groups * 4, 256),
            index_k,
            index_z,
            q_weight,
            k_weight,
            k_bias,
            self.rotary.cos_sin_cache,
            positions,
            meta.query_start_loc,
            q_out,
            k_out,
            z_out,
            self.eps,
            backend="ixformer_native",
        )
        self.csa(k_out, z_out, positions)
        summary = self.csa.summary.kv_cache
        summary_meta = metadata[self.csa.summary.prefix]
        indexer_scores(
            q_out.view(tokens, groups, 4, 256),
            weights.view(tokens, groups, 4),
            summary,
            summary_meta.block_table,
            meta.query_start_loc,
            positions,
            scores,
            backend="ixformer_native",
        )
        top512(
            scores,
            positions,
            meta.query_start_loc,
            summary_meta.block_table,
            summary.shape[1],
            summary.shape[0],
            starts,
            ends,
            selected,
            scratch,
            regions,
            counts,
            backend="ixformer_native",
        )
        sparse_gqa_fp8(
            query,
            key,
            value,
            key_cache,
            value_cache,
            regions,
            counts,
            meta.query_start_loc,
            positions,
            meta.block_table,
            meta.slot_mapping,
            self.fp8_k_scale,
            self.fp8_v_scale,
            output,
            fp8_workspace,
            softmax_scale=192 ** (-0.5),
            backend=self.fp8_backend,
        )


def ilu_step5_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Write output; allocator-owned KV/CSA caches stay in the forward context."""
    layer = get_forward_context().no_compile_layers[layer_name]
    layer.attention_impl(
        query,
        key,
        value,
        index_q,
        index_k,
        index_z,
        weights,
        positions,
        q_weight,
        k_weight,
        k_bias,
        output,
    )


def ilu_step5_sparse_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Tracing only: output is supplied and no runtime metadata may be read."""
    return None


direct_register_custom_op(
    op_name="ilu_step5_sparse_attention",
    op_func=ilu_step5_sparse_attention,
    mutates_args=["output"],
    fake_impl=ilu_step5_sparse_attention_fake,
)
