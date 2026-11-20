# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_iluvatar.models.glm5next import model as glm


def post_mix(x, residual, post, comb):
    return (comb @ residual + post * x.unsqueeze(1)).to(x.dtype)


class BoundaryLayer(nn.Module):
    """Deterministic stand-in for a layer producing the deferred mHC tuple."""

    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, positions, x, residual, post, comb):
        if post is not None:
            x = post_mix(x, residual, post, comb)
        elif self.idx == 0:
            x = x.unsqueeze(1).expand(-1, 4, -1)
        assert x.ndim == 3 and x.shape[1] == 4
        y = x.mean(1) + self.idx + 1
        if self.idx == 2:
            return y, None, None, None
        post = y.new_full((len(y), 4, 1), 0.5)
        comb = torch.eye(4, dtype=y.dtype).expand(len(y), 4, 4)
        return y, x, post, comb


def stage(rank, compact, capture):
    model = glm.IluGlm5NextModel.__new__(glm.IluGlm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc_num_residual_streams=4, hidden_size=16)
    model._send_completed_mhc_residual = compact
    model.is_sequence_parallel = False
    model.start_layer, model.end_layer = rank, rank + 1
    model._active_layers = [BoundaryLayer(rank)]
    model.aux_hidden_state_layers = (1, 2, 3) if capture else ()
    model.norm = nn.Identity()
    return model


@pytest.mark.parametrize("tokens", [1, 17, 128])
@pytest.mark.parametrize("capture", [False, True])
def test_pp_mhc_boundary(tokens, capture):
    torch.manual_seed(14)
    x = torch.randn(tokens, 16)
    positions = torch.arange(tokens)

    def run(compact):
        incoming = None
        for rank in range(3):
            model = stage(rank, compact, capture)
            group = SimpleNamespace(is_first_rank=rank == 0, is_last_rank=rank == 2)
            with patch.object(glm, "get_pp_group", return_value=group):
                out = model._forward_impl(
                    None, positions, incoming, inputs_embeds=x if rank == 0 else None
                )
            if rank == 2:
                return out
            incoming = model.make_empty_intermediate_tensors(tokens, x.dtype, x.device)
            assert incoming.tensors.keys() == out.tensors.keys()
            if compact:
                assert list(out.tensors) == ["hidden_states"]
                assert out["hidden_states"].shape == (tokens, 4, 16)
            else:
                assert set(out.tensors) == {"hidden_states", "residual", "post", "comb"}
            for key, tensor in out.items():
                incoming[key].copy_(tensor)
        raise AssertionError("last PP stage did not return")

    with patch.object(glm.mhc, "mhc_post", side_effect=post_mix):
        expected, actual = run(False), run(True)
    if capture:
        expected, expected_aux = expected
        actual, actual_aux = actual
        assert len(expected_aux) == len(actual_aux) == 2
        for left, right in zip(expected_aux, actual_aux):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
