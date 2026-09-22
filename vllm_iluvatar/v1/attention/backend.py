# SPDX-License-Identifier: Apache-2.0
"""Iluvatar override for ``vllm.v1.attention.backend.CommonAttentionMetadata.token_to_req_indices``.

Corresponds to: vllm/v1/attention/backend.py
Target   : CommonAttentionMetadata.token_to_req_indices
Baseline : main_a67969c5a0 (pre commit 13e221f8308)
Why      : Community commit 13e221f8308 replaced the ``torch.repeat_interleave``
           per-token→request mapping with a Triton binary-search kernel
           (``_token_request_mapping_kernel``). That kernel is a pure device
           kernel whose Python int args (``num_mapped``, ``num_reqs``,
           ``num_output_tokens``) get baked into the CUDA graph at capture
           time. Under FULL_DECODE_ONLY cudagraph + dspark adaptive
           verification the live ``num_mapped`` (= ``query_start_loc_cpu[-1]``)
           varies per step, so the replayed (frozen) kernel draws the
           real/padding boundary at the wrong place and corrupts the
           token→request mapping; the compressor / SWA ixformer ops then write
           KV/score into the wrong request slots and decode output garbles.
           The pre-commit ``repeat_interleave`` path does a CPU sync
           (``int(query_start_loc_cpu[-1])``) plus a dynamic-output torch op,
           which CUDA graph cannot capture, so it runs on the host each step
           with live data and is correct under cudagraph.
How      : Keep the pre-13e221f8308 ``repeat_interleave`` implementation.
Scope    : every V1 attention backend on Iluvatar that consumes
           ``common_attn_metadata.token_to_req_indices``.
Remove   : when the CoreX Triton backend handles ``_token_request_mapping_kernel``
           correctly under cudagraph, or community reverts the regression.
"""

from __future__ import annotations

import torch


def token_to_req_indices(self, buffer: torch.Tensor) -> torch.Tensor:
    """Build or reuse the per-token request index mapping."""
    num_tokens = self.num_actual_tokens
    if self._token_to_req_indices_cache is not None:
        assert self._token_to_req_indices_cache.device == buffer.device
        assert self._token_to_req_indices_cache.dtype == torch.int32
        assert self._token_to_req_indices_cache.shape[0] >= num_tokens
        return self._token_to_req_indices_cache[:num_tokens]

    # >>> ILU MODIFICATION >>>
    # Built from the device query_start_loc: adaptive verification decides
    # the per-request draft split on device, so the CPU copy carries the
    # right total but not the right per-request boundaries. Padding
    # requests have a query length of zero and drop out of the repeat.
    # Community commit 13e221f8308 replaced this with a Triton binary-search
    # kernel whose int args are frozen into the CUDA graph; under
    # FULL_DECODE_ONLY + dspark the frozen ``num_mapped`` diverges from the
    # live value and mis-maps tokens. Keep the pre-commit repeat_interleave
    # path, which CUDA graph cannot capture (CPU sync + dynamic output), so
    # it runs on the host each step with live data.
    num_mapped_tokens = int(self.query_start_loc_cpu[-1])
    query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
    assert buffer.shape[0] >= max(num_mapped_tokens, num_tokens)
    token_to_req_indices = torch.repeat_interleave(
        torch.arange(query_lens.shape[0], dtype=torch.int32, device=buffer.device),
        query_lens,
        output_size=num_mapped_tokens,
    )
    buffer[:num_mapped_tokens].copy_(token_to_req_indices)
    if num_mapped_tokens < num_tokens:
        buffer[num_mapped_tokens:num_tokens].zero_()
    self._token_to_req_indices_cache = buffer[: max(num_mapped_tokens, num_tokens)]
    # <<< ILU MODIFICATION <<<
    return self._token_to_req_indices_cache[:num_tokens]
