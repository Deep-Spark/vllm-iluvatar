# SPDX-License-Identifier: Apache-2.0
"""Adapt EngineArgs.create_speculative_config and the V2 init_speculator factory.

vLLM 0.31 (341bf7d4) can validate a custom proposer but its GPU V2 factory
does not construct the Flash DFly proposer. Remove these wrappers once both
configuration and construction support DFly upstream.
"""

import inspect
from functools import wraps

from vllm_iluvatar.ilu_patch import general_patch as p

_CLASS = "vllm_iluvatar.models.hy_v4_flash.speculator.FlashDFlySpeculator"


def register():
    p.register_once(
        __name__, _apply, detail="HY4 Flash DFly configuration and V2 proposer"
    )


def _apply():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.gpu import spec_decode

    original_config = EngineArgs.create_speculative_config
    inspect.signature(original_config).bind(None, None, None)

    @wraps(original_config)
    def config(self, target_model_config, target_parallel_config):
        raw = self.speculative_config
        if raw and raw.get("method") == "dfly":
            if target_model_config.hf_text_config.model_type != "hy_v4_flash":
                raise ValueError("DFly requires a HY4 Flash checkpoint")
            self.speculative_config = {**raw, "method": "custom_class", "model": _CLASS}
        try:
            result = original_config(self, target_model_config, target_parallel_config)
        finally:
            self.speculative_config = raw
        if result and result.model == _CLASS and result.method == "custom_class":
            result.method = "dfly"
        return result

    original_init = spec_decode.init_speculator
    inspect.signature(original_init).bind(None, None)

    @wraps(original_init)
    def init(vllm_config, device):
        spec = vllm_config.speculative_config
        if spec and spec.method == "dfly" and spec.model == _CLASS:
            from vllm_iluvatar.models.hy_v4_flash.speculator import FlashDFlySpeculator

            return FlashDFlySpeculator(vllm_config, device)
        return original_init(vllm_config, device)

    EngineArgs.create_speculative_config = config
    p.rebind_module_func(spec_decode, "init_speculator", init)
