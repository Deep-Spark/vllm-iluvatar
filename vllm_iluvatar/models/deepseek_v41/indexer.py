"""V4.1 indexer weights. Scoring lives on ``V41PagedIndexer``."""
from __future__ import annotations

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Keep the top-scoring compressed blocks, always including the newest one."""
    width = logits.size(-1)
    tail = -width % block_size
    scores = logits if tail == 0 else torch.nn.functional.pad(
        logits, (0, tail), value=-torch.inf
    )
    # amax allocates, so the masked_fill below is free to run in place.
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores.masked_fill_(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )

    if topk_blocks >= num_blocks:
        # k == num_blocks makes the selection an identity: topk hands back every
        # column exactly once, so the scatter below would just write each
        # column's own `> -inf` back into place. Skipping it is bit-exact and
        # drops a full sort of num_blocks per row. With the shipped config
        # (topk_blocks=2048, block_size=8, compress_ratio=1 on the candidate
        # source layer) this is the live path for any context <= 16384 tokens.
        keep = scores > -torch.inf
    else:
        top = scores.topk(topk_blocks, dim=-1)
        keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
            -1, top.indices, top.values > -torch.inf
        )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class IluDeepseekV41Indexer(nn.Module):
    """Index-query projections and, for key owners, ``wk`` / ``k_norm``.

    Key ownership is ``layer_id in kv_source_layer_ids``: index keys come from
    the compressor's pre-RoPE latent, so only those layers write the K cache.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        compress_ratio: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()

        self.owns_k = layer_id in config.kv_source_layer_ids
        self.compress_ratio = compress_ratio
        self.is_candidate_source = layer_id == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_id
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.n_heads = config.index_n_heads
        self.n_local_heads = config.index_n_heads // tp_size
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.cos_sin: torch.Tensor | None = None

        self.wq_b = ColumnParallelLinear(
            config.q_lora_rank, config.index_n_heads * config.index_head_dim,
            bias=False, quant_config=quant_config, prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ColumnParallelLinear(
            config.hidden_size, config.index_n_heads, bias=False,
            params_dtype=torch.bfloat16, prefix=f"{prefix}.weights_proj",
        )
        if self.owns_k:
            self.wk = ReplicatedLinear(
                config.head_dim, config.index_head_dim, bias=False,
                params_dtype=torch.bfloat16, prefix=f"{prefix}.wk",
            )
            self.k_norm = RMSNorm(config.index_head_dim, config.rms_norm_eps)
