# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for DSv4.1 -> FP8 DS MLA wire staging helpers."""

import torch

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.fp8_staging import (
    _dequant_grouped_main_rows,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common.dsv41 import (
    validate_wire_format_tags,
)


def test_dequant_i8g64_matches_scale_broadcast():
    rows = 3
    payload = (
        torch.arange(rows * 512, dtype=torch.int8).view(rows, 512).view(torch.uint8)
    )
    scales = torch.full((rows, 8), 0.5, dtype=torch.float16)
    out = _dequant_grouped_main_rows(payload, scales, format_name="i8g64")
    assert out.dtype == torch.bfloat16
    assert out.shape == (rows, 512)
    expected = payload.view(torch.int8).to(torch.float32) * 0.5
    assert torch.allclose(out.float(), expected, atol=2e-2, rtol=2e-2)


def test_dequant_i4g32_unpacks_nibbles():
    # Pack int4 codes [-2, 3] repeating into bytes.
    even = torch.full((2, 256), (-2) & 0xF, dtype=torch.uint8)
    odd = torch.full((2, 256), 3 & 0xF, dtype=torch.uint8)
    payload = even | (odd << 4)
    scales = torch.ones((2, 16), dtype=torch.float16)
    out = _dequant_grouped_main_rows(payload, scales, format_name="i4g32")
    assert out.shape == (2, 512)
    assert torch.equal(out[:, 0::2], torch.full_like(out[:, 0::2], -2))
    assert torch.equal(out[:, 1::2], torch.full_like(out[:, 1::2], 3))


def test_wire_staging_handshake_overrides_ignore_compute_codecs():
    # P compute is i8g64/split, but wire staging advertises none/legacy.
    assert (
        validate_wire_format_tags(
            None,
            remote_engine_id="d0",
            remote_main_kv_format="none",
            remote_indexer_kv_layout="legacy",
            local_main_kv_format="none",
            local_indexer_kv_layout="legacy",
        )
        is False
    )
