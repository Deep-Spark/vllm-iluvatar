"""Optional CQ prefill PP transport: symmetric INT8 blocks of 512 elements."""

from math import prod

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

from vllm_iluvatar.ilu_util.arch import is_ivcore30

_BLOCK_SIZE = 512


def is_pure_prefill(attn_metadata: object, num_tokens: int) -> bool:
    if not isinstance(attn_metadata, dict):
        return False
    metadata = next(
        (
            value
            for value in attn_metadata.values()
            if isinstance(value, DeepseekSparseSWAMetadata)
        ),
        None,
    )
    return (
        metadata is not None
        and metadata.num_decode_tokens == 0
        and metadata.num_prefill_tokens == num_tokens
    )


def maybe_encode_hidden_states(tensors: dict, senders: str, pp_group) -> None:
    """Replace the PP payload only when the native per-token codec can encode it."""
    x = tensors["hidden_states"]
    if senders != "all" and str(pp_group.rank_in_group) not in (
        rank.strip() for rank in senders.split(",")
    ):
        return
    if (
        pp_group.use_cpu_custom_send_recv
        or get_tensor_model_parallel_world_size() != 1
        or not is_ivcore30()
        or not x.is_cuda
        or x.ndim < 2
        or x.shape[0] <= 128
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        return
    row = prod(x.shape[1:])
    if (
        row == 0
        or row % _BLOCK_SIZE
        or x.numel() // _BLOCK_SIZE > 2**31 - 1
        or x.data_ptr() % 16
        or not is_pure_prefill(get_forward_context().attn_metadata, x.shape[0])
        or not hasattr(torch.ops._ixformer_C, "v41_pp_quant_int8")
        or not hasattr(torch.ops._ixformer_C, "v41_pp_dequant_int8")
    ):
        return
    # Token stays the leading dimension for IntermediateTensors slicing.
    out = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(
        (x.shape[0], row // _BLOCK_SIZE), device=x.device, dtype=torch.float32
    )
    torch.ops._ixformer_C.v41_pp_quant_int8(x, out, scales)
    tensors["hidden_states"] = out
    tensors["prefill_pp_scale"] = scales


def decode(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.bfloat16)
    torch.ops._ixformer_C.v41_pp_dequant_int8(x, scales, out)
    return out
