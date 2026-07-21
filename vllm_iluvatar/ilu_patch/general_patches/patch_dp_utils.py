# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar DP batch coordination overrides.

Corresponds to: vllm/v1/worker/dp_utils.py
                vllm/forward_context.py

Target   : vllm.v1.worker.dp_utils.coordinate_batch_across_dp
           vllm.forward_context.DPMetadata.make
           vllm.forward_context.set_forward_context
Baseline : plugin/vllm 0.24.0 runtime source
Why      : hybrid DeepSeek-V4 EP may switch decode to LL only when every DP
           rank is decode-only; otherwise collective paths can diverge.
How      : coordinate_batch_across_dp is a full-function copy with optional
           all-ranks decode-only synchronization. The DPMetadata methods are
           lightweight wrappers that attach the synchronized bit while keeping
           existing call contracts.
Scope    : DP+EP MoE forward on Iluvatar.
Remove   : when upstream DPMetadata carries this field or exposes a hook.
Fingerprint: coordinate_batch_across_dp community source is hashed at
           register-time against the baseline snapshot below; gated by
           VLLM_ILU_PATCH_CHECK_FINGERPRINT (default on) so startup fails fast
           if upstream changed the body.
"""

from __future__ import annotations

import inspect
import types
from contextlib import contextmanager
from functools import wraps

from vllm_iluvatar.ilu_patch import general_patch as p

# Verbatim community-baseline source of
# vllm.v1.worker.dp_utils.coordinate_batch_across_dp, used as the drift anchor.
# Baseline: plugin/vllm 0.24.0 runtime source.
COORDINATE_BATCH_ACROSS_DP = '''\
def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    allow_microbatching: bool,
    parallel_config: ParallelConfig,
    num_tokens_padded: int | None = None,
    uniform_decode: bool | None = None,
    cudagraph_mode: int = 0,
) -> tuple[bool, torch.Tensor | None, int]:
    """
    Coordinates amongst all DP ranks to determine if and how the full batch
    should be split into microbatches.

    Args:
        num_tokens_unpadded: Number of tokens without accounting for padding
        allow_microbatching: If microbatching should be attempted
        parallel_config: The parallel config
        num_tokens_padded: Number of tokens including any non-DP padding (CUDA graphs,
            TP, etc)
        uniform_decode: Only used if allow_microbatching is True. True if the batch
            only contains single token decodes
        cudagraph_mode: The cudagraph mode for this rank (0=NONE, 1=PIECEWISE, 2=FULL).
            DP padding is enabled when synced cudagraph mode across ranks is not NONE.

    Returns: tuple[
        ubatch_slices: if this is set then all DP ranks have agreed to
        microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        padded up to the max value across all DP ranks when cudagraph is enabled.
        synced_cudagraph_mode: The synchronized cudagraph mode (min across ranks)
    ]

    """
    if parallel_config.data_parallel_size == 1:
        # Early exit.
        return False, None, cudagraph_mode

    # If the caller has explicitly enabled microbatching.
    should_attempt_ubatching = False
    if allow_microbatching:
        # Check preconditions for microbatching
        assert uniform_decode is not None
        should_attempt_ubatching = check_ubatch_thresholds(
            parallel_config,
            num_tokens_unpadded,
            uniform_decode=uniform_decode,
        )

    if num_tokens_padded is None:
        num_tokens_padded = num_tokens_unpadded

    (should_ubatch, num_tokens_after_padding, synced_cudagraph_mode) = (
        _synchronize_dp_ranks(
            num_tokens_unpadded,
            num_tokens_padded,
            should_attempt_ubatching,
            cudagraph_mode,
            parallel_config,
        )
    )

    return (should_ubatch, num_tokens_after_padding, synced_cudagraph_mode)
'''


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="DP coordination decode-only metadata",
    )


def _apply() -> None:
    import vllm.forward_context as fc_mod
    import vllm.v1.worker.dp_utils as dp_mod

    import vllm_iluvatar.v1.worker.dp_utils as impl

    p.check_fingerprint(
        "vllm.v1.worker.dp_utils.coordinate_batch_across_dp",
        inspect.getsource(dp_mod.coordinate_batch_across_dp),
        COORDINATE_BATCH_ACROSS_DP,
        resync_hint=(
            "Re-sync vllm_iluvatar/v1/worker/dp_utils.py and the "
            "COORDINATE_BATCH_ACROSS_DP baseline in "
            "ilu_patch/general_patches/patch_dp_utils.py to the new community "
            "source of coordinate_batch_across_dp."
        ),
    )
    p.check_signatures(
        dp_mod,
        impl,
        ["coordinate_batch_across_dp"],
        strict=True,
    )

    p.rebind_module_func(
        dp_mod,
        "coordinate_batch_across_dp",
        impl.coordinate_batch_across_dp,
    )

    orig_make = fc_mod.DPMetadata.make

    @staticmethod
    @wraps(orig_make)
    def make_with_decode_only(*args, all_ranks_decode_only=None, **kwargs):
        metadata = orig_make(*args, **kwargs)
        value = (
            impl.get_last_all_ranks_decode_only()
            if all_ranks_decode_only is None
            else bool(all_ranks_decode_only)
        )
        metadata.all_ranks_decode_only = value
        return metadata

    fc_mod.DPMetadata.make = make_with_decode_only

    orig_set_forward_context = fc_mod.set_forward_context

    @contextmanager
    @wraps(orig_set_forward_context)
    def set_forward_context_with_decode_only(
        *args, all_ranks_decode_only=None, **kwargs
    ):
        with orig_set_forward_context(*args, **kwargs):
            ctx = fc_mod.get_forward_context()
            if ctx.dp_metadata is not None:
                value = (
                    impl.get_last_all_ranks_decode_only()
                    if all_ranks_decode_only is None
                    else bool(all_ranks_decode_only)
                )
                ctx.dp_metadata.all_ranks_decode_only = value
            yield

    p.check_signatures(
        types.SimpleNamespace(set_forward_context=orig_set_forward_context),
        types.SimpleNamespace(set_forward_context=set_forward_context_with_decode_only),
        ["set_forward_context"],
        strict=True,
    )
    p.rebind_module_func(
        fc_mod,
        "set_forward_context",
        set_forward_context_with_decode_only,
    )
