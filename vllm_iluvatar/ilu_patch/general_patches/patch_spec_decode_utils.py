# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar spec-decode utility overrides.

Corresponds to: vllm/v1/spec_decode/utils.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

Target   : vllm.v1.spec_decode.utils.eagle_prepare_next_token_padded_kernel
Baseline : main_dcfebf93f4e
Why      : Iluvatar Triton can infer ``tl.sum(bool_mask)`` as ``int1``, causing
           a phi-node dtype mismatch with the discarded-request branch.
How      : rebind the module-level Triton kernel and propagate to from-import
           consumers such as ``llm_base_proposer``.
Scope    : EAGLE speculative decoding on Iluvatar.
Remove   : when Iluvatar Triton promotes bool reductions compatibly with
           upstream, or upstream pins the dtype explicitly.
Fingerprint: the community kernel source is hashed at register-time against
           the baseline below; gated by VLLM_ILU_PATCH_CHECK_FINGERPRINT.
"""

from __future__ import annotations

import inspect

from vllm_iluvatar.ilu_patch import general_patch as p

_OVERRIDES = ["eagle_prepare_next_token_padded_kernel"]

# Verbatim community-baseline source of
# vllm.v1.spec_decode.utils.eagle_prepare_next_token_padded_kernel,
# used as the drift anchor by the fingerprint guard.
#
# Baseline: main_dcfebf93f4e
_EAGLE_PREPARE_NEXT_TOKEN_PADDED_KERNEL_BASELINE = """\
@triton.jit
def eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,  # [num_reqs, num_sampled_tokens_per_req]
    discard_request_mask_ptr,  # [num_reqs]
    backup_next_token_ids_ptr,  # [num_reqs]
    next_token_ids_ptr,  # [num_reqs] (output)
    valid_sampled_tokens_count_ptr,  # [num_reqs] (output)
    vocab_size,  # tl.int32
    num_sampled_tokens_per_req,  # tl.int32 (num_spec_tokens + 1)
    num_reqs,  # tl.int32
    stride_sampled_token_ids,  # tl.int32 (stride for dim 0)
    BLOCK_SIZE_TOKENS: tl.constexpr,  # Power-of-2 >= num_sampled_tokens_per_req
):
    \"\"\"
    Fused kernel for Eagle prepare_next_token_ids_padded. This kernel computes the
    number of valid (1 + accepted) tokens for each request, and the corresponding
    "next" token id to sample from during speculative decoding. This is the
    "last accepted token" from the sampled tokens, or the backup token if no
    tokens were accepted or if the request is marked as discarded.
    \"\"\"
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # Check if this request is discarded.
    is_discarded = tl.load(discard_request_mask_ptr + req_idx)

    if is_discarded:
        backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
        valid_count = tl.full((), 0, dtype=tl.uint32)
        tl.store(next_token_ids_ptr + req_idx, backup_token)
        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
    else:
        # Count the number of valid tokens among the sampled tokens.
        token_offs = tl.arange(0, BLOCK_SIZE_TOKENS)
        token_mask = token_offs < num_sampled_tokens_per_req

        row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
        token_ids = tl.load(row_ptr + token_offs, mask=token_mask, other=-1)

        # Rejected tokens are -1, valid tokens are in [0, vocab_size)
        is_valid_mask = (token_ids != -1) & (token_ids < vocab_size) & token_mask
        valid_count = tl.sum(is_valid_mask)

        if valid_count > 0:
            # Guaranteed to be well-defined since
            # valid_count > 0 implies is_valid_mask is not empty
            last_valid_index = tl.max(tl.where(is_valid_mask, token_offs, -1))

            # Select the token at that index, using a sum trick since
            # we don't want to load again to access token_ids[last_valid_index].
            last_valid_token = tl.sum(
                tl.where(token_offs == last_valid_index, token_ids, 0)
            )
            tl.store(next_token_ids_ptr + req_idx, last_valid_token)
        else:
            # No valid tokens found, use backup token
            backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
            tl.store(next_token_ids_ptr + req_idx, backup_token)

        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
"""


def register() -> None:
    """Register Iluvatar spec-decode utility overrides."""
    p.register_once(
        __name__,
        _apply,
        detail="EAGLE next-token padded kernel int32 valid-count override",
    )


def _source_for_fingerprint(symbol: object) -> str:
    """Return Python source for a function or a Triton JITFunction wrapper."""
    return inspect.getsource(getattr(symbol, "fn", symbol))


def _apply() -> None:
    import vllm.v1.spec_decode.utils as _m

    import vllm_iluvatar.v1.spec_decode.utils as impl

    p.check_fingerprint(
        "vllm.v1.spec_decode.utils.eagle_prepare_next_token_padded_kernel",
        _source_for_fingerprint(_m.eagle_prepare_next_token_padded_kernel),
        _EAGLE_PREPARE_NEXT_TOKEN_PADDED_KERNEL_BASELINE,
        resync_hint=(
            "Re-sync eagle_prepare_next_token_padded_kernel in "
            "vllm_iluvatar/v1/spec_decode/utils.py and the baseline in "
            "ilu_patch/general_patches/patch_spec_decode_utils.py to the new "
            "community source."
        ),
    )

    p.check_signatures(_m, impl, _OVERRIDES, strict=True)
    for name in _OVERRIDES:
        # [ILU override] eagle_prepare_next_token_padded_kernel - delta: force
        # valid_count to int32 for Iluvatar Triton codegen.
        p.rebind_module_func(_m, name, getattr(impl, name))
