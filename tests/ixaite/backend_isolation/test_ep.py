# SPDX-License-Identifier: Apache-2.0
"""Exercise the private naive EP path without entering the vendor transport."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls
from vllm_iluvatar.backends.ixaite.moe import ep


# The gather-to-reduce lifecycle must use the selected communicator before slicing.
def test_naive_ep_reduces_through_ixaite_communicator(monkeypatch):
    comm = SimpleNamespace(all_reduce=Mock(side_effect=lambda x: x.mul_(2)))
    group = SimpleNamespace(
        rank_in_group=1,
        device_communicator=comm,
        all_gatherv=lambda xs, **kwargs: [torch.cat([x, x]) for x in xs],
    )
    metadata = SimpleNamespace(
        sp_local_sizes=lambda _: nullcontext(),
        get_chunk_sizes_across_dp_rank=lambda: [2, 2],
    )
    monkeypatch.setattr(ep, "get_ep_group", lambda: group)
    monkeypatch.setattr(ep, "get_dp_group", lambda: group)
    monkeypatch.setattr(
        ep, "get_forward_context", lambda: SimpleNamespace(dp_metadata=metadata)
    )
    runner = SimpleNamespace(
        moe_config=SimpleNamespace(
            pcp_size=1, sp_size=1, hidden_dim=3, is_sequence_parallel=False
        ),
        _quant_method=SimpleNamespace(
            skip_forward_padding=True, topk_indices_dtype=torch.int64
        ),
        shared_experts=None,
        gate=lambda x: (x, None),
        router=SimpleNamespace(select_experts=lambda **kwargs: (None, None)),
        routed_experts=SimpleNamespace(
            _ensure_moe_quant_config_init=lambda: None,
            forward_modular=lambda x, **kwargs: x.clone(),
        ),
        apply_routed_input_transform=lambda x: (x, None),
        apply_routed_output_transform=lambda x: x,
        _fused_output_is_reduced=False,
        _maybe_add_zero_expert_output=lambda x: x,
    )
    layer = ep.IluFusedMoEEP()
    layer.experts = runner
    x = torch.arange(6.0).reshape(2, 3)
    with forbid_ixformer_calls():
        actual = layer.forward_dp_ep_naive(x)
    comm.all_reduce.assert_called_once()
    assert comm.all_reduce.call_args.args[0].shape == (4, 3)
    torch.testing.assert_close(actual, x * 2, rtol=0, atol=0)
