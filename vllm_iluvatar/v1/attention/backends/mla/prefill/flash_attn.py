# SPDX-License-Identifier: Apache-2.0
"""Iluvatar mirror of vLLM MLA FlashAttention prefill backend.

Corresponds to: vllm/v1/attention/backends/mla/prefill/flash_attn.py
Target        : FlashAttnPrefillBackend._flash_attn_varlen_diff_headdims
Baseline      : main_61ac3680210
Why           : the paired Iluvatar runtime ``flash_attn_varlen_func`` rejects
                the community ``output_scale`` kwarg, while ILU MLA still needs
                the V-dim padding behavior to stay outside community vLLM.
How           : override the official MLA prefill backend enum with a mirrored
                subclass and keep the delta in this upstream-shaped file.
Scope         : Iluvatar ``MLAPrefillBackendEnum.FLASH_ATTN`` only.
Remove        : when the paired runtime accepts community ``out/output_scale``
                kwargs and upstream keeps V unpadding compatible with ILU MLA.
"""

from __future__ import annotations

import inspect

import torch
import vllm.envs as envs
from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FA4_MLA_PREFILL_KERNEL,
)
from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend as CommunityFlashAttnPrefillBackend,
)

from vllm_iluvatar.ilu_patch import general_patch as p

# Verbatim community-baseline source of
# FlashAttnPrefillBackend._flash_attn_varlen_diff_headdims (4-space method
# indent), used as the drift anchor by the fingerprint guard.
#
# Baseline: main_b35ff5f7739e
_FLASH_ATTN_VARLEN_DIFF_HEADDIMS_BASELINE = """\
    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool = False,
        softmax_scale: float | None = None,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        maybe_padded_v = v
        if self.requires_v_padding:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        if self._is_vllm_fa:
            kwargs["return_softmax_lse"] = return_softmax_lse
            kwargs["out"] = out
            kwargs["output_scale"] = output_scale
        else:
            # ROCm leverages the upstream flash_attn, which takes a parameter
            # called "return_attn_probs" instead of return_softmax_lse
            kwargs["return_attn_probs"] = return_softmax_lse
            assert out is None and output_scale is None
        if envs.VLLM_BATCH_INVARIANT:
            kwargs["num_splits"] = 1

        attn_out = FA4_MLA_PREFILL_KERNEL(
            q=q,
            k=k,
            v=maybe_padded_v,
            runtime_kernel=self.flash_attn_varlen_func,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        # Unpack the output if there are multiple results
        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # Remain consistent with old `flash_attn_varlen_func` where there
        # is only one output tensor if `return_softmax_lse` is False.
        if return_softmax_lse:
            return attn_out, lse
        return attn_out
"""

_CHECKED_FLASH_ATTN_PREFILL_FINGERPRINT = False


def _check_flash_attn_prefill_fingerprint() -> None:
    global _CHECKED_FLASH_ATTN_PREFILL_FINGERPRINT
    if _CHECKED_FLASH_ATTN_PREFILL_FINGERPRINT:
        return
    community_method = CommunityFlashAttnPrefillBackend._flash_attn_varlen_diff_headdims
    p.check_fingerprint(
        "FlashAttnPrefillBackend._flash_attn_varlen_diff_headdims",
        inspect.getsource(community_method),
        _FLASH_ATTN_VARLEN_DIFF_HEADDIMS_BASELINE,
        resync_hint=(
            "Re-sync vllm_iluvatar/v1/attention/backends/mla/prefill/"
            "flash_attn.py with the paired community "
            "FlashAttnPrefillBackend._flash_attn_varlen_diff_headdims baseline."
        ),
    )
    _CHECKED_FLASH_ATTN_PREFILL_FINGERPRINT = True


class FlashAttnPrefillBackend(CommunityFlashAttnPrefillBackend):
    """Iluvatar FLASH_ATTN MLA prefill backend."""

    def __init__(self, *args, **kwargs) -> None:
        _check_flash_attn_prefill_fingerprint()
        super().__init__(*args, **kwargs)

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool = False,
        softmax_scale: float | None = None,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        maybe_padded_v = v
        if self.requires_v_padding:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        if self._is_vllm_fa:
            kwargs["return_softmax_lse"] = return_softmax_lse
            # >>> ILU MODIFICATION >>>
            # The paired ixformer/vllm-fa runtime rejects community out/output_scale
            # kwargs, even when their values are None.
            # <<< ILU MODIFICATION <<<
        else:
            kwargs["return_attn_probs"] = return_softmax_lse
            assert out is None and output_scale is None
        if envs.VLLM_BATCH_INVARIANT:
            kwargs["num_splits"] = 1

        attn_out = FA4_MLA_PREFILL_KERNEL(
            q=q,
            k=k,
            v=maybe_padded_v,
            runtime_kernel=self.flash_attn_varlen_func,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]
        if self.requires_v_padding:
            # >>> ILU MODIFICATION >>>
            # Keep ILU MLA output shape at v_head_dim after padding V to q_head_dim.
            attn_out = attn_out[..., : v.shape[-1]].contiguous()
            # <<< ILU MODIFICATION <<<
        if return_softmax_lse:
            return attn_out, lse
        return attn_out
