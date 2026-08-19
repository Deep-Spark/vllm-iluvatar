# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for sparse MLA Triton warmup.

Corresponds to:
    vllm/model_executor/warmup/sparse_mla_triton_warmup.py
    vllm/v1/attention/backends/mla/sparse_swa.py

Target   : ``vllm.v1.attention.backends.mla.sparse_swa``.
Baseline : main_6e448d0
Why      : community warmup imports ``_compute_prefill_metadata_kernel`` from
           ``sparse_swa`` and launches it with Triton kernel syntax. The
           Iluvatar ``sparse_swa`` fork replaces that Triton kernel with
           ``ixfops.ds4_compute_swa_prefill_gather_lens`` and removes the
           original symbol, so warmup logs an ImportError and skips.
How      : when the symbol is missing, inject a tiny kernel-like object whose
           ``kernel[(grid)](...)`` launch delegates to the ixformer equivalent.
Scope    : sparse MLA warmup only; runtime metadata build already calls ixformer
           directly from the forked ``sparse_swa`` implementation.
Remove   : when ``sparse_swa`` is restored to community or warmup learns the
           ixformer path natively.
"""

from __future__ import annotations

from collections.abc import Callable

from vllm_iluvatar.ilu_patch import general_patch as p

_KERNEL_NAME = "_compute_prefill_metadata_kernel"


class _IluPrefillMetadataWarmupKernel:
    def __getitem__(self, _grid) -> Callable[..., None]:
        def launch(
            prefill_gather_lens,
            seq_lens,
            query_start_loc,
            num_prefills,
            num_decodes,
            window_size,
            *,
            BLOCK_SIZE,
        ) -> None:
            del BLOCK_SIZE
            import ixformer.inference.functions as ixfops

            ixfops.ds4_compute_swa_prefill_gather_lens(
                prefill_gather_lens=prefill_gather_lens,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                num_prefills=num_prefills,
                num_decodes=num_decodes,
                window_size=window_size,
            )

        return launch


def _apply() -> None:
    try:
        from vllm.v1.attention.backends.mla import sparse_swa
    except ImportError:
        return

    if hasattr(sparse_swa, _KERNEL_NAME):
        return

    setattr(sparse_swa, _KERNEL_NAME, _IluPrefillMetadataWarmupKernel())


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="sparse_swa prefill metadata warmup kernel shim",
    )
