# SPDX-License-Identifier: Apache-2.0
"""Decode-only metadata helpers for Iluvatar MoE paths.

The decode GEMV paths must avoid ``expert_sizes_gpu.cpu()`` during CUDA graph
capture. These helpers keep the attention-metadata shape checks shared by W16A16
and quantized MoE implementations without importing their kernel modules.

Two distinct questions are answered by two distinct helpers:
``_is_only_decode`` decides batch semantics (is this a pure decode batch?) from
the metadata counters alone, while ``_ensure_cuda_graph_safe_dispatch`` decides
capture safety (may the D2H-dependent general/EP path run right now?). Keeping
them separate avoids forcing non-decode batches onto the decode GEMV path just
because some CUDA graph happens to be capturing.
"""

from __future__ import annotations

import torch


def _metadata_values(attn_metadata) -> list[object]:
    if attn_metadata is None:
        return []
    if isinstance(attn_metadata, dict):
        return list(attn_metadata.values())
    return [attn_metadata]


def _decode_counts(value) -> tuple[int, int] | None:
    """Return the effective ``(num_prefills, num_decodes)`` of one metadata.

    Speculative decode: hybrid backends keep ``num_decodes`` and
    ``num_spec_decodes`` mutually exclusive, so a pure spec-decode batch
    reports ``num_prefills == num_decodes == 0`` and carries the real request
    count in ``num_spec_decodes`` (see vllm ``GDNAttentionMetadata``); treat
    those as decode. Qwen3.5 is a GDN + full-attention hybrid, so its linear
    layers report exactly that under MTP.

    An entry reporting no work at all (and no spec decode) carries no batch
    semantics, so it abstains by returning ``None`` and lets the remaining
    metadata decide. Counters are only read as ``int``; coercing a Tensor here
    would force an implicit D2H sync that is illegal during graph capture.
    """
    num_prefills = getattr(value, "num_prefills", None)
    num_decodes = getattr(value, "num_decodes", None)
    if not (isinstance(num_prefills, int) and isinstance(num_decodes, int)):
        return None
    if num_prefills == 0 and num_decodes == 0:
        num_spec_decodes = getattr(value, "num_spec_decodes", 0)
        if isinstance(num_spec_decodes, int) and num_spec_decodes > 0:
            return 0, num_spec_decodes
        return None
    return num_prefills, num_decodes


def _describe_metadata(attn_metadata) -> str:
    """Compact metadata summary used only in the capture-safety error message.

    Aggregates identical ``(type, counts)`` entries because a model emits one
    metadata object per layer, and formats plain ints only so the diagnostic
    never reads device memory.
    """
    if attn_metadata is None:
        return "attn_metadata=None"

    counts_by_signature: dict[str, int] = {}
    for value in _metadata_values(attn_metadata):
        counts = _decode_counts(value)
        if counts is None:
            signature = f"{value.__class__.__name__}(counts=unavailable)"
        else:
            signature = (
                f"{value.__class__.__name__}"
                f"(num_prefills={counts[0]}, num_decodes={counts[1]})"
            )
        counts_by_signature[signature] = counts_by_signature.get(signature, 0) + 1
    if not counts_by_signature:
        return "<empty>"
    return "; ".join(f"{sig} x{n}" for sig, n in counts_by_signature.items())


def _is_only_decode(attn_metadata) -> bool:
    """Decide whether to take the CG-safe decode GEMV path.

    Driven purely by the batch's decode/prefill counters. The CUDA graph
    capture state is deliberately not consulted: "currently capturing" does not
    make a batch pure decode, and assuming it does would silently route prefill
    or mixed batches into the decode GEMV path under any graph mode other than
    FULL_DECODE_ONLY. ``_ensure_cuda_graph_safe_dispatch`` answers the capture
    safety question instead.

    A hybrid model contributes one metadata object per layer and its backends
    may describe the same batch differently, so every metadata that reports
    batch semantics has to agree before the decode path is taken.

    Metadata is recognised by the counter fields it exposes rather than by a
    backend class-name allowlist: dispatch depends on what a batch contains,
    not on which class describes it, and a name list silently goes stale
    whenever a backend is added or renamed.
    """
    counts: list[tuple[int, int]] = []
    for value in _metadata_values(attn_metadata):
        value_counts = _decode_counts(value)
        if value_counts is not None:
            counts.append(value_counts)

    # Fail closed: a batch is only pure decode when at least one metadata
    # reports what it contains and every reporting metadata agrees. Letting the
    # first reporting entry decide would allow a decode-reporting backend to
    # mask a prefill-reporting one in a hybrid model.
    if not counts:
        return False
    return all(
        num_prefills == 0 and num_decodes > 0
        for num_prefills, num_decodes in counts
    )


def _ensure_cuda_graph_safe_dispatch(only_decode: bool, attn_metadata) -> None:
    """Fail loudly instead of running the D2H-dependent path inside a graph.

    The general/EP MoE path copies ``expert_sizes_gpu`` to CPU, which is not
    valid during CUDA graph capture. If we are capturing a batch that was not
    classified as pure decode, raise instead of silently producing negative
    dims / absurd allocations. The observed metadata is reported so a
    misclassification is debuggable.
    """
    try:
        is_capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
    except (RuntimeError, AssertionError):
        is_capturing = False

    if is_capturing and not only_decode:
        raise RuntimeError(
            "Iluvatar MoE general/EP execution is not CUDA-graph safe: it "
            "copies expert_sizes_gpu to CPU. The current batch was not "
            "classified as pure decode; capture a decode-only graph or run "
            "this batch eagerly. Observed metadata: "
            + _describe_metadata(attn_metadata)
        )
