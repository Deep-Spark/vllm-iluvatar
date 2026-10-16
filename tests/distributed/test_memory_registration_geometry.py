"""Scheduler-block registrations of shared layer-outer and block-outer views."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.handshake_validation import (
    validate_descriptor_geometry,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.memory_registration import (
    MemoryRegistrationPlanner,
    TensorRegistration,
)


@pytest.mark.parametrize("block_outer", [False, True])
def test_bf16_multiple_owner_views_share_one_registered_allocation(block_outer):
    pages, rows, width, layers = 3, 4, 16, 2
    page_bytes = rows * width * 2
    owner_stride = page_bytes if block_outer else page_bytes * pages
    block_stride = page_bytes * layers if block_outer else page_bytes
    size = page_bytes * pages * layers
    config = SimpleNamespace(
        kv_cache_tensors=[
            SimpleNamespace(
                offset=0,
                layer_stride=owner_stride,
                block_stride=block_stride,
                size=size,
                layers=["a", "b"],
            )
        ]
    )
    planner = MemoryRegistrationPlanner(config)
    raw = torch.empty(size, dtype=torch.uint8)
    spec = MLAAttentionSpec(
        block_size=rows, num_kv_heads=1, head_size=width, dtype=torch.bfloat16
    )
    registrations = []
    for index, name in enumerate(("a", "b")):
        cache = torch.as_strided(
            raw.view(torch.bfloat16),
            (pages, rows, width),
            (block_stride // 2, width, 1),
            index * owner_stride // 2,
        )
        geometry = planner.describe(
            layer_name=name, tensor=cache, spec=spec, num_blocks=pages
        )
        assert geometry.registration_base == raw.data_ptr()
        assert geometry.registration_nbytes == size
        assert geometry.payload_segments == ((index * owner_stride, page_bytes),)
        assert geometry.descriptor_span <= size
        registrations.append(geometry)
    validate_descriptor_geometry(
        local_ptr=[0, 0],
        remote_ptr=[0, 0],
        local_desc=[r.payload_segments for r in registrations],
        remote_desc=[r.payload_segments for r in registrations],
        local_block_len=[block_stride] * 2,
        remote_block_len=[block_stride] * 2,
        layer_desc=[spec] * 2,
    )


def test_overlay_alias_has_identical_address_geometry():
    raw = torch.empty(192, dtype=torch.uint8)
    config = SimpleNamespace(
        kv_cache_tensors=[
            SimpleNamespace(
                offset=0, layer_stride=64, block_stride=64, size=192, layers=[name]
            )
            for name in ("a", "alias")
        ]
    )
    spec = MLAAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=16, dtype=torch.bfloat16
    )
    cache = raw.view(torch.bfloat16).view(3, 2, 16)
    planner = MemoryRegistrationPlanner(config)
    assert planner.describe(
        layer_name="a", tensor=cache, spec=spec, num_blocks=3
    ) == planner.describe(layer_name="alias", tensor=cache, spec=spec, num_blocks=3)


@pytest.mark.parametrize("offset,length", [(96, 40), (256, 64)])
def test_global_offsets_still_reject_cross_page_or_allocation_overrun(offset, length):
    registration = TensorRegistration(
        registration_base=1 << 40,
        registration_nbytes=384,
        block_stride=64,
        payload_segments=((offset, length),),
        num_blocks=3,
        kernel_blocks=1,
        device_id=0,
    )
    with pytest.raises(ValueError, match="stride|span"):
        registration.validate("layer")


def test_tensor_stride_disagreement_is_rejected():
    raw = torch.empty(384, dtype=torch.uint8)
    cache = raw.view(torch.bfloat16).view(6, 2, 16)[::2]
    planner = MemoryRegistrationPlanner(SimpleNamespace(kv_cache_tensors=[]))
    spec = MLAAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=16, dtype=torch.bfloat16
    )
    with pytest.raises(ValueError, match="allocation size|stride disagreement"):
        planner.describe(layer_name="layer", tensor=cache, spec=spec, num_blocks=3)


def test_descriptor_geometry_retains_cross_block_bounds_check():
    with pytest.raises(ValueError, match="exceeds block stride"):
        validate_descriptor_geometry(
            local_ptr=[0],
            remote_ptr=[0],
            local_desc=[[(3 * 64 + 32, 40)]],
            remote_desc=[[(0, 40)]],
            local_block_len=[64],
            remote_block_len=[64],
            layer_desc=[None],
        )
