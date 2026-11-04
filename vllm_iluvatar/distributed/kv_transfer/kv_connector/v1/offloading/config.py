# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py

Target   : build_offloading_config
Baseline : v0.31.0 (341bf7d4d124)
Why      : Propagate non-uniform PP rank geometry into native offload configuration.
How      : Wrappers delegate unchanged paths to the community implementation.
Scope    : DeepSeek V4.1 paired / non-uniform PP adapters.
Remove   : When community provides equivalent geometry and storage ownership.
"""

from __future__ import annotations

from vllm.logger import init_logger

from vllm_iluvatar.models.deepseek_v41.offloading import (
    _GEOMETRY_ATTR,
    _RANK_SAFE_REQUIRED_ATTR,
    _needs_rank_safe_geometry,
)

logger = init_logger(__name__)

_ORIGINAL_BUILD_CONFIG = None


# [ILU override] _ilu_build_offloading_config — delta: adapt paired pages / rank-local PP geometry.
def _ilu_build_offloading_config(vllm_config, kv_cache_config):
    assert _ORIGINAL_BUILD_CONFIG is not None
    needs_geometry = _needs_rank_safe_geometry(vllm_config)
    geometry = getattr(kv_cache_config, _GEOMETRY_ATTR, None)
    if needs_geometry and geometry is None:
        raise RuntimeError("DSV4.1 PP CPU offload geometry was not propagated")
    config = _ORIGINAL_BUILD_CONFIG(vllm_config, kv_cache_config)
    if geometry is not None:
        object.__setattr__(config, _GEOMETRY_ATTR, tuple(geometry))
    if needs_geometry:
        object.__setattr__(config, _RANK_SAFE_REQUIRED_ATTR, True)
    return config
