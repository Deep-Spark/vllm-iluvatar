"""Wire paired indexer scale budgeting without changing upstream allocation."""

from functools import wraps


def register() -> None:
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.engine import core

    from vllm_iluvatar.models.deepseek_v41.indexer_kv import (
        reserve_indexer_scale_memory,
    )

    original = kv_cache_utils.get_kv_cache_configs
    if getattr(original, "_ilu_indexer_scales", False):
        return

    @wraps(original)
    def paired(vllm_config, kv_cache_specs, available_memory):
        return reserve_indexer_scale_memory(
            original, vllm_config, kv_cache_specs, available_memory
        )

    paired._ilu_indexer_scales = True
    kv_cache_utils.get_kv_cache_configs = paired
    core.get_kv_cache_configs = paired
