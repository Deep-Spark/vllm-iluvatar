# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import vllm_iluvatar.models.deepseek_v4.compressor as ilu_compressor


def _config(
    *,
    cache_dtype: str = "bfloat16",
    staging: bool | None = None,
    role: str = "kv_producer",
    connector: str = "IluNixlConnector",
):
    transfer_config = None
    if staging is not None:
        extra_config = {"enable_fp8_ds_mla_wire_staging": staging}
        transfer_config = SimpleNamespace(
            kv_role=role,
            kv_connector=connector,
            get_from_extra_config=lambda key, default: extra_config.get(key, default),
        )
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        kv_transfer_config=transfer_config,
    )


def _state(*, block_size: int, state_dim: int = 1024):
    return SimpleNamespace(
        block_size=block_size,
        state_dim=state_dim,
        dtype="dtype",
        sliding_window=state_dim,
    )


@pytest.fixture(autouse=True)
def _capture_spec(monkeypatch):
    monkeypatch.setattr(
        ilu_compressor,
        "SlidingWindowMLASpec",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )


@pytest.mark.parametrize(
    ("block_size", "staging", "expected"),
    [
        (4, None, 8),
        (8, None, 16),
        (4, False, 8),
        (8, False, 16),
        (4, True, 4),
        (8, True, 8),
    ],
)
def test_bf16_state_block_size_follows_wire_staging(
    block_size,
    staging,
    expected,
):
    spec = ilu_compressor._ilu_get_kv_cache_spec(
        _state(block_size=block_size),
        _config(staging=staging),
    )

    assert spec.block_size == expected


@pytest.mark.parametrize(
    ("role", "connector"),
    [
        ("kv_consumer", "IluNixlConnector"),
        ("kv_producer", "NixlConnector"),
    ],
)
def test_bf16_native_state_blocks_require_supported_producer(role, connector):
    spec = ilu_compressor._ilu_get_kv_cache_spec(
        _state(block_size=4),
        _config(staging=True, role=role, connector=connector),
    )

    assert spec.block_size == 8


@pytest.mark.parametrize(("state_dim", "expected"), [(2048, 2), (1024, 4)])
def test_fp8_cache_without_staging_preserves_bucket_aligned_block_size(
    state_dim,
    expected,
):
    spec = ilu_compressor._ilu_get_kv_cache_spec(
        _state(block_size=8, state_dim=state_dim),
        _config(cache_dtype="fp8_ds_mla", staging=False),
    )

    assert spec.block_size == expected


@pytest.mark.parametrize("block_size", [4, 8])
def test_wire_staging_overrides_fp8_bucket_formula(block_size):
    spec = ilu_compressor._ilu_get_kv_cache_spec(
        _state(block_size=block_size, state_dim=2048),
        _config(cache_dtype="fp8_ds_mla", staging=True),
    )

    assert spec.block_size == block_size
