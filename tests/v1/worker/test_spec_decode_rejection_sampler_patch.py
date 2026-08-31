# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the V2 rejection-sampler chunk-size patch."""

from __future__ import annotations

import sys
from types import ModuleType

from vllm_iluvatar import envs as ilu_envs
from vllm_iluvatar.ilu_patch.general_patches import (
    patch_spec_decode_rejection_sampler,
)

_DEFAULT_MAX_CHUNK_BYTES = 128 * 1024 * 1024
_ENV_MAX_CHUNK_BYTES = "VLLM_ILU_REJECTION_SAMPLER_MAX_CHUNK_BYTES"


def test_rejection_sampler_chunk_env_is_registered(monkeypatch):
    import vllm.envs as vllm_envs

    monkeypatch.setenv(_ENV_MAX_CHUNK_BYTES, "4096")
    ilu_envs.apply()
    vllm_envs.disable_envs_cache()

    assert vllm_envs.VLLM_ILU_REJECTION_SAMPLER_MAX_CHUNK_BYTES == 4096


def test_rejection_sampler_chunk_env_invalid_value_uses_default(monkeypatch):
    import vllm.envs as vllm_envs

    monkeypatch.setenv(_ENV_MAX_CHUNK_BYTES, "not-an-int")
    ilu_envs.apply()
    vllm_envs.disable_envs_cache()

    assert (
        vllm_envs.VLLM_ILU_REJECTION_SAMPLER_MAX_CHUNK_BYTES == _DEFAULT_MAX_CHUNK_BYTES
    )


def test_rejection_sampler_chunk_patch_reads_registered_env(monkeypatch):
    import vllm.envs as vllm_envs

    community = ModuleType("vllm.v1.worker.gpu.spec_decode.rejection_sampler")
    community.MAX_CHUNK_BYTES = 2**30
    monkeypatch.setitem(sys.modules, community.__name__, community)
    monkeypatch.setenv(_ENV_MAX_CHUNK_BYTES, "8192")
    ilu_envs.apply()
    vllm_envs.disable_envs_cache()

    patch_spec_decode_rejection_sampler._apply()

    assert community.MAX_CHUNK_BYTES == 8192
