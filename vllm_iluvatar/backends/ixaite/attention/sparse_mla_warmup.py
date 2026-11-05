# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable


# Keep the kernel launch ABI while binding only the ixaite implementation.
class _IxaitePrefillMetadataWarmupKernel:
    # Warmup callers expect Triton-style indexing.
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
            from vllm_iluvatar.backends.ixaite import ops as ilu_ops

            ilu_ops.ds4_compute_swa_prefill_gather_lens(
                prefill_gather_lens=prefill_gather_lens,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                num_prefills=num_prefills,
                num_decodes=num_decodes,
                window_size=window_size,
            )

        return launch
