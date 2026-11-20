# SPDX-License-Identifier: Apache-2.0
"""Wire GLM RecoverSSM validation before VllmConfig instances are constructed."""

from types import SimpleNamespace

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(__name__, _apply, detail="GLM KDA RecoverSSM configuration")


def _apply() -> None:
    import vllm.config.vllm as module
    from pydantic.dataclasses import rebuild_dataclass

    from vllm_iluvatar.config.recoverssm import make_validate_mamba_cached_kernel

    cls = module.VllmConfig
    name = "validate_mamba_cached_kernel"
    replacement = make_validate_mamba_cached_kernel(getattr(cls, name))
    p.check_signatures(cls, SimpleNamespace(**{name: replacement}), [name], strict=True)
    setattr(cls, name, replacement)
    # Pydantic has already captured the validator callable. Updating only the
    # class method leaves normal constructor validation on the Kimi-only path.
    cls.__pydantic_decorators__.model_validators[name].func = replacement
    rebuild_dataclass(cls, force=True, _types_namespace=vars(module))

    from vllm.v1.worker.gpu import cudagraph_utils
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_iluvatar.v1.worker.gpu.model_runner import (
        IluvatarGPUModelRunner,
        make_recoverssm_initialize_kv_cache,
        make_recoverssm_profiling_teardown,
    )

    for runner_cls in (GPUModelRunner, IluvatarGPUModelRunner):
        method = make_recoverssm_initialize_kv_cache(runner_cls.initialize_kv_cache)
        p.check_signatures(
            runner_cls,
            SimpleNamespace(initialize_kv_cache=method),
            ["initialize_kv_cache"],
            strict=True,
        )
        runner_cls.initialize_kv_cache = method
    p.rebind_module_func(
        cudagraph_utils,
        "_teardown_profiling_state",
        make_recoverssm_profiling_teardown(cudagraph_utils._teardown_profiling_state),
    )
