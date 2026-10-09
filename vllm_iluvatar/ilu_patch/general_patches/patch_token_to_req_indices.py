# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar ``token_to_req_indices`` override.

Corresponds to: vllm/v1/attention/backend.py
Target   : vllm.v1.attention.backend.CommonAttentionMetadata.token_to_req_indices
Baseline : v0.31.0 (db9527a468) (post commit 13e221f8308)
Why      : Community commit 13e221f8308 ("[Perf] Fuse DSV4.1 input metadata
           preparation with Triton") replaced the ``torch.repeat_interleave``
           per-token→request mapping with a Triton binary-search kernel
           (``_token_request_mapping_kernel``). That kernel is a pure device
           kernel whose Python int args (``num_mapped``, ``num_reqs``,
           ``num_output_tokens``) get baked into the CUDA graph at capture
           time. Under FULL_DECODE_ONLY cudagraph + dspark adaptive
           verification the live ``num_mapped`` (= ``query_start_loc_cpu[-1]``)
           varies per step, so the replayed (frozen) kernel draws the
           real/padding boundary at the wrong place and corrupts the
           token→request mapping. The compressor / SWA ixformer ops then write
           KV/score into the wrong request slots and decode output garbles.
           The pre-commit ``repeat_interleave`` path does a CPU sync
           (``int(query_start_loc_cpu[-1])``) plus a dynamic-output torch op,
           which CUDA graph cannot capture, so it runs on the host each step
           with live data and is correct under cudagraph.
How      : Fingerprint the live community ``token_to_req_indices`` source
           against the baseline snapshot below, then rebind the method to the
           pre-commit ``repeat_interleave`` implementation in
           ``vllm_iluvatar.v1.attention.backend``.
Scope    : every V1 attention backend on Iluvatar that consumes
           ``common_attn_metadata.token_to_req_indices``.
Remove   : when the CoreX Triton backend handles ``_token_request_mapping_kernel``
           correctly under cudagraph, or community reverts the regression.
Fingerprint: the live community ``CommonAttentionMetadata.token_to_req_indices``
           source is checked against ``TOKEN_TO_REQ_INDICES`` before rebinding;
           gated by VLLM_ILU_PATCH_CHECK_FINGERPRINT.
"""

from __future__ import annotations

import inspect

from vllm_iluvatar.ilu_patch import general_patch as p

# Verbatim community-baseline source of
# ``CommonAttentionMetadata.token_to_req_indices`` (post 13e221f8308), used as
# the drift anchor. Keep in sync with vllm/v1/attention/backend.py.
TOKEN_TO_REQ_INDICES = '''\
    def token_to_req_indices(self, buffer: torch.Tensor) -> torch.Tensor:
        """Build or reuse the per-token request index mapping."""
        num_tokens = self.num_actual_tokens
        if self._token_to_req_indices_cache is not None:
            assert self._token_to_req_indices_cache.device == buffer.device
            assert self._token_to_req_indices_cache.dtype == torch.int32
            assert self._token_to_req_indices_cache.shape[0] >= num_tokens
            return self._token_to_req_indices_cache[:num_tokens]

        # Built from the device query_start_loc: adaptive verification decides the
        # per-request draft split on device, so the CPU copy carries the right total
        # but not the right per-request boundaries. Padding requests have a query
        # length of zero and are skipped by the device boundary search.
        num_mapped_tokens = int(self.query_start_loc_cpu[-1])
        from vllm.v1.attention.ops.metadata import _token_request_mapping_kernel

        num_output_tokens = max(num_mapped_tokens, num_tokens)
        assert buffer.shape[0] >= num_output_tokens
        _token_request_mapping_kernel[((num_output_tokens + 255) // 256,)](
            self.query_start_loc,
            buffer,
            self.query_start_loc.shape[0] - 1,
            num_mapped_tokens,
            num_output_tokens,
            num_warps=4,
        )
        self._token_to_req_indices_cache = buffer[:num_output_tokens]
        return self._token_to_req_indices_cache[:num_tokens]
'''


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="token_to_req_indices repeat_interleave path",
    )


def _apply() -> None:
    from vllm.v1.attention.backend import CommonAttentionMetadata

    import vllm_iluvatar.v1.attention.backend as impl

    p.check_fingerprint(
        "vllm.v1.attention.backend.CommonAttentionMetadata.token_to_req_indices",
        inspect.getsource(CommonAttentionMetadata.token_to_req_indices),
        TOKEN_TO_REQ_INDICES,
        resync_hint=(
            "Refresh TOKEN_TO_REQ_INDICES to the new community source, then "
            "review whether the pre-13e221f8308 repeat_interleave override "
            "in vllm_iluvatar/v1/attention/backend.py still applies."
        ),
    )
    p.check_signatures(
        CommonAttentionMetadata,
        impl,
        ["token_to_req_indices"],
        strict=True,
    )

    # [ILU override] CommonAttentionMetadata.token_to_req_indices — delta: keep
    # the pre-13e221f8308 repeat_interleave path instead of the Triton
    # binary-search kernel, which mis-maps tokens under FULL_DECODE_ONLY
    # cudagraph + dspark adaptive verification on CoreX.
    CommonAttentionMetadata.token_to_req_indices = impl.token_to_req_indices
