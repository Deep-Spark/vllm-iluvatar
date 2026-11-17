# SPDX-License-Identifier: Apache-2.0
"""Routing contracts at the small-batch MHC boundary; native math tests live in IxFormer."""

from unittest.mock import Mock

import pytest
import torch

from vllm_iluvatar.models.glm5next import mhc


@pytest.mark.parametrize("m", [4, 12, 15, 16, 17, 24, 36, 48, 127, 128, 119, 120, 121])
@pytest.mark.parametrize("enabled", [False, True])
def test_sqrsum_handoff_kept_outside_splitk(monkeypatch, m, enabled):
    monkeypatch.setattr(mhc.envs, "VLLM_ILU_GLM5NEXT_MHC_DECODE_SPLITK", enabled)
    residual = torch.empty(m, 4, 4096, dtype=torch.bfloat16)
    sqrsum = torch.empty(m, dtype=torch.float32)
    expected = (object(), object(), object())
    split = Mock(return_value=expected)
    reuse = Mock(return_value=(*expected, None))
    monkeypatch.setattr(mhc.ixfops, "ds4_mhc_pre_v2", split)
    monkeypatch.setattr(mhc.ixf_mhc, "mhc_pre_with_sqrsum", reuse)
    actual = mhc.mhc_pre(residual, None, None, None, 1e-6, 1e-6, 2.0, 20, sqrsum=sqrsum)
    assert actual == expected
    if enabled and 16 <= m <= 120:
        split.assert_called_once()
        reuse.assert_not_called()
    else:
        reuse.assert_called_once()
        assert reuse.call_args.args[4] is sqrsum
        split.assert_not_called()


@pytest.mark.parametrize("m", [15, 16, 120, 121])
def test_splitk_never_skips_required_tp_allreduce(monkeypatch, m):
    monkeypatch.setattr(mhc.envs, "VLLM_ILU_GLM5NEXT_MHC_DECODE_SPLITK", True)
    residual = torch.empty(m, 4, 4096, dtype=torch.bfloat16)
    reduced = torch.empty_like(residual)
    sqrsum = torch.empty(m, dtype=torch.float32)
    group = object()
    reduce = Mock(return_value=(reduced, sqrsum))
    post = Mock(side_effect=AssertionError("required allreduce was skipped"))
    pre = Mock(return_value=(object(), object(), object()))
    monkeypatch.setattr(mhc, "allreduce_mhc_post_with_sqrsum", reduce)
    monkeypatch.setattr(mhc, "mhc_post", post)
    monkeypatch.setattr(mhc, "mhc_pre", pre)
    result = mhc.mhc_fused_post_pre(
        None,
        residual,
        None,
        None,
        None,
        None,
        None,
        1e-6,
        1e-6,
        2.0,
        20,
        reduce_x=True,
        tp_group=group,
    )
    reduce.assert_called_once()
    assert reduce.call_args.kwargs["group"] is group
    assert pre.call_args.args[0] is reduced
    assert pre.call_args.kwargs["sqrsum"] is sqrsum
    assert result[0] is reduced
    post.assert_not_called()
