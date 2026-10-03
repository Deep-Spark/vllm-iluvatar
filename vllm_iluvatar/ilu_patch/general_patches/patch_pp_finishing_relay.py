# SPDX-License-Identifier: Apache-2.0
"""Leave requests that finish this step out of the PP sampled-token relay.

Corresponds to:
    vllm/v1/worker/gpu/pp_utils.py :: compute_need_sampled_mask
    vllm/v1/worker/gpu/model_runner.py :: GPUModelRunner.prepare_inputs
Baseline : v0.30.1
Why/How  : see vllm_iluvatar/v1/worker/gpu/pp_utils.py
Scope    : V2 model runner, PP>1, no speculative decoding;
           VLLM_ILU_PP_SKIP_FINISHING_RELAY=0 disables it.
Remove   : when upstream restores the finishing-request exclusion.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="PP sampled-token relay skips requests finishing this step",
    )


def _apply() -> None:
    import vllm.envs as envs

    if not envs.VLLM_ILU_PP_SKIP_FINISHING_RELAY:
        return

    from vllm.v1.worker.gpu import pp_utils
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_iluvatar.v1.worker.gpu import pp_utils as impl

    pp_utils.compute_need_sampled_mask = impl.compute_need_sampled_mask

    if not getattr(
        GPUModelRunner.prepare_inputs, "_ilu_pp_finishing_relay_patched", False
    ):
        prep = impl.make_prepare_inputs_max_seq_len(GPUModelRunner.prepare_inputs)
        prep._ilu_pp_finishing_relay_patched = True  # type: ignore[attr-defined]
        GPUModelRunner.prepare_inputs = prep  # type: ignore[method-assign]
