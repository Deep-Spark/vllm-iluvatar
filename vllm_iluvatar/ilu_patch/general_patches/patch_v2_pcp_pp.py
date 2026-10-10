# SPDX-License-Identifier: Apache-2.0
"""Wire eager MRV2 PCP + PP request-state handoff on Iluvatar."""

from types import SimpleNamespace

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(__name__, _apply, detail="V2 eager PCP + PP global request handoff")


def _apply() -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

    from vllm_iluvatar.v1.worker.gpu.model_runner import make_pcp_pp_sample_tokens
    from vllm_iluvatar.v1.worker.gpu.pcp_manager import make_validate_config

    validate_config = make_validate_config(PCPManager.validate_config)
    sample_tokens = make_pcp_pp_sample_tokens(GPUModelRunner.sample_tokens)
    p.check_signatures(
        PCPManager,
        SimpleNamespace(validate_config=validate_config),
        ["validate_config"],
    )
    p.check_signatures(
        GPUModelRunner, SimpleNamespace(sample_tokens=sample_tokens), ["sample_tokens"]
    )
    PCPManager.validate_config = staticmethod(validate_config)
    GPUModelRunner.sample_tokens = sample_tokens
