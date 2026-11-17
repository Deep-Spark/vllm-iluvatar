# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401, I001
"""Deterministic import-time registration for nested quantization packages.

``ilu_register`` discovers only top-level modules. Importing this module loads
all nested config and runtime-method registrars in dependency order.
"""

# Backend methods that register under their own name. The configs below import
# them too, but only as a side effect of their own wiring; list them here so
# the registration does not depend on that.
from .backends.ivcore30.fp4 import fp8_moe as _fp8_moe
from .backends.ivcore30.fp8 import linear as _fp8_linear

# Runtime registry and method implementations must exist before configs start
# resolving ``additional_config.ilu_quant`` groups.
from .ilu_runtime import base_config as _online_base_config
from .ilu_runtime.ivcore30.linear import fp8_to_w8a8 as _fp8_to_w8a8
from .ilu_runtime.ivcore30.linear import native_to_fp8 as _native_to_fp8
from .ilu_runtime.ivcore30.linear import native_to_w8a8 as _native_to_w8a8
from .ilu_runtime.ivcore30.moe import fp16_to_fp4 as _fp16_to_fp4
from .ilu_runtime.ivcore30.moe import fp8_to_fp4 as _fp8_to_fp4
from .ilu_runtime.ivcore30.moe import fp8_to_w4a8_nn as _fp8_to_w4a8_nn
from .ilu_runtime.ivcore30.moe import mxfp4_to_w4a8_tn as _mxfp4_to_w4a8_tn

# Checkpoint quantization configs.
from .configs import awq as _awq
from .configs import fp8 as _fp8
from .configs import gptq as _gptq
from .configs import mxfp8 as _mxfp8
from .configs.deepseek import v4_fp8 as _deepseek_v4_fp8
from .configs.deepseek import v41_fp8 as _deepseek_v41_fp8
from .configs.modelopt import mxfp8 as _modelopt_mxfp8
from .configs.modelopt import nvfp4 as _modelopt_nvfp4

__all__: list[str] = []
