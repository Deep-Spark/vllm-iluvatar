"""Dispatch Step5 small-op fusions for decode and prefill.

Decode / CUDA-graph / MTP must use the live ``query_start_loc[-1:]`` scalar
(vLLM refreshes it on every replay). Prefill may allocate a fresh int32[1]
row count outside capture. Dense row masks keep one address for their lifetime.
Missing graph-safe state skips fusion and logs once so FULL_DECODE_ONLY graphs
are not silently recorded unfused.
"""

from __future__ import annotations

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)

# Attention widths that gate_fp8 templates cover (heads * 192).
_GATE_FP8_WIDTHS = (1536, 3072, 6144, 12288)

_warned_capture_without_actual = False
_warned_row_mask_capacity = False


def dense_row_mask_capacity():
    """Reserve dense row masks for prefill and every decode/MTP graph size."""
    config = get_current_vllm_config_or_none()
    if config is None:
        return 32
    speculative = config.speculative_config
    drafts = 0 if speculative is None else (speculative.num_speculative_tokens or 0)
    budget = config.scheduler_config.max_num_seqs * (1 + drafts)
    capture_sizes = config.compilation_config.cudagraph_capture_sizes or []
    return max(
        1, budget, config.scheduler_config.max_num_batched_tokens, *capture_sizes
    )


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _decode_only_meta(meta) -> bool | None:
    """True/False when meta exposes prefill/decode counts; None if omitted.

    Step5 sparse-FP8 / CSA builders only ship ``query_start_loc`` (no
    ``num_prefills``). Treating missing counts as ``num_prefills=1`` falsely
    rejected decode CUDA-graph capture and fell through to a host→device copy.
    """
    has_prefills = hasattr(meta, "num_prefills")
    has_decodes = hasattr(meta, "num_decodes")
    if not has_prefills and not has_decodes:
        return None
    return getattr(meta, "num_prefills", 1) == 0 and getattr(meta, "num_decodes", 0) > 0


def _query_start_loc_tail(meta):
    starts = getattr(meta, "query_start_loc", None)
    if (
        starts is None
        or starts.dtype != torch.int32
        or not starts.is_cuda
        or starts.ndim != 1
        or starts.numel() < 1
    ):
        return None
    # Read the current CUDA count in the kernel, not a capture-time Python int.
    # vLLM owns this buffer and refreshes query_start_loc on every replay.
    return starts[-1:]


def decode_actual_tokens(attention, rows):
    # All fusion kernels support prefill-sized row counts. An arbitrary decode
    # cap would silently record unfused graphs when max_seqs * (1 + MTP) > 32.
    if rows <= 0 or not is_forward_context_available():
        return None
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return None
    if attention is None:
        meta = next(
            (
                m
                for m in metadata.values()
                if _decode_only_meta(m) is not False
                and _query_start_loc_tail(m) is not None
            ),
            None,
        )
    else:
        meta = metadata.get(attention.layer_name)
    if meta is None or _decode_only_meta(meta) is False:
        return None
    return _query_start_loc_tail(meta)


def fusion_actual_tokens(attention, rows, device=None):
    """Return int32[1] valid-row count for fusion kernels, or None.

    Prefers ``query_start_loc[-1:]`` when metadata allows (CUDA-graph–safe view).
    Otherwise allocates a fresh int32[1]=``rows`` so concurrent prefills cannot
    race a shared mutable counter. Host→device copies are illegal under CUDA
    graph capture, so the fallback is skipped while capturing (with a warning).
    """
    decode = decode_actual_tokens(attention, rows)
    if decode is not None:
        return decode
    if rows <= 0:
        return None
    if _capturing():
        global _warned_capture_without_actual
        if not _warned_capture_without_actual:
            _warned_capture_without_actual = True
            logger.warning(
                "Step5 fusion skipped during CUDA graph capture: no "
                "query_start_loc[-1:] for rows=%s (sparse-FP8/MTP decode must "
                "expose int32 CUDA query_start_loc). Graph will record the "
                "unfused path for this size.",
                rows,
            )
        return None
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    # Device fill avoids the torch.tensor(list, device=cuda) host copy.
    actual = torch.empty(1, device=device, dtype=torch.int32)
    actual.fill_(int(rows))
    return actual


def accepts_prequant(layer):
    supports = getattr(layer.quant_method, "supports_prequantized_input", None)
    return callable(supports) and supports(layer)


def can_fuse_gate(layer, x, actual):
    return (
        actual is not None
        and x.shape[0] > 0
        and x.dtype == torch.bfloat16
        and x.shape[-1] in _GATE_FP8_WIDTHS
        and layer.input_is_parallel
        and accepts_prequant(layer)
    )


def can_share_norm_quant(attention, x, actual):
    return (
        actual is not None
        and x.shape[0] > 0
        and x.shape[-1] == 4096
        and x.dtype in (torch.bfloat16, torch.float32)
        and attention.use_head_wise_attn_gate
        and accepts_prequant(attention.qkv_proj)
        and accepts_prequant(attention.g_proj)
    )


def can_fuse_swiglu_fp8(down_proj, gate_up, row_mask):
    """Gate SwiGLU+FP8 fusion for decode (incl. MTP T=4) and prefill sizes."""
    return (
        row_mask is not None
        and gate_up.shape[0] > 0
        and gate_up.dtype == torch.bfloat16
        and gate_up.dim() == 2
        and gate_up.shape[1] % 16 == 0
        and down_proj.input_is_parallel
        and accepts_prequant(down_proj)
        and row_mask.dtype == torch.int32
        and row_mask.shape == (gate_up.shape[0],)
        and row_mask.device == gate_up.device
    )


def can_fuse_qk_norm_rope(qkv, rope_cache, actual, *, head_dim: int, rotary_dim: int):
    """Gate qk_norm_rope: needs head_dim=192 and even rotary_dim (0 disables rope)."""
    return (
        actual is not None
        and qkv.shape[0] > 0
        and qkv.dtype == torch.bfloat16
        and rope_cache.dtype == torch.bfloat16
        and head_dim == 192
        and rotary_dim >= 0
        and rotary_dim % 2 == 0
    )


def dense_valid_row_mask(buffer: torch.Tensor, rows: int, device: torch.device):
    """Return a view of the fixed dense mask, or skip fusion if it cannot fit.

    Zeros mark every row valid. Never replace this buffer, including outside
    capture: an existing CUDA graph may still reference its original address.
    An unexpected size or device uses the unfused path without relocating it.
    """
    if rows <= 0:
        return buffer, None
    if buffer.device != device or buffer.numel() < rows:
        global _warned_row_mask_capacity
        if not _warned_row_mask_capacity:
            _warned_row_mask_capacity = True
            logger.warning(
                "Step5 swiglu fusion skipped: fixed dense row_mask has %s "
                "rows on %s, need %s on %s. The original buffer is retained "
                "for existing CUDA graphs.",
                buffer.numel(),
                buffer.device,
                rows,
                device,
            )
        return buffer, None
    return buffer, buffer[:rows]
