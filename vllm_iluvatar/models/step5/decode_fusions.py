"""Dispatch small-op fusions only for verified pure-decode metadata."""

import torch
from vllm.forward_context import get_forward_context, is_forward_context_available


def decode_actual_tokens(attention, rows):
    if not 0 < rows <= 32 or not is_forward_context_available():
        return None
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return None
    if attention is None:
        meta = next(
            (
                m
                for m in metadata.values()
                if getattr(m, "num_prefills", 1) == 0
                and getattr(m, "num_decodes", 0) > 0
            ),
            None,
        )
    else:
        meta = metadata.get(attention.layer_name)
    if (
        meta is None
        or getattr(meta, "num_prefills", 1) != 0
        or getattr(meta, "num_decodes", 0) == 0
    ):
        return None
    starts = meta.query_start_loc
    if starts.dtype != torch.int32 or not starts.is_cuda or starts.ndim != 1:
        return None
    # Read the current CUDA count in the kernel, not a capture-time Python int.
    # vLLM owns this buffer and refreshes query_start_loc on every replay.
    return starts[-1:]


def accepts_prequant(layer):
    supports = getattr(layer.quant_method, "supports_prequantized_input", None)
    return callable(supports) and supports(layer)


def can_fuse_gate(layer, x, actual):
    return (
        actual is not None
        and 6 < x.shape[0] <= 32
        and x.dtype == torch.bfloat16
        and x.shape[-1] in (1536, 3072, 6144)
        and layer.input_is_parallel
        and accepts_prequant(layer)
    )


def can_share_norm_quant(attention, x, actual):
    return (
        actual is not None
        and 6 < x.shape[0] <= 32
        and x.shape[-1] == 4096
        and x.dtype in (torch.bfloat16, torch.float32)
        and attention.use_head_wise_attn_gate
        and accepts_prequant(attention.qkv_proj)
        and accepts_prequant(attention.g_proj)
    )
