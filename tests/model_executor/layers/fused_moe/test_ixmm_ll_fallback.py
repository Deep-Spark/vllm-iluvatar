# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_iluvatar.model_executor.layers.fused_moe import ixmm_ll_decode as decode


@pytest.mark.parametrize("rank", [0, 1])
def test_missing_interface_on_one_rank_falls_back_before_shmem_init(monkeypatch, rank):
    group = SimpleNamespace(rank_in_group=rank, world_size=2, cpu_group=object())
    monkeypatch.setenv("VLLM_ILU_V41_IXMM_LL_DECODE", "1")
    monkeypatch.setenv("VLLM_ILU_IXMM_LL_LIBRARY", "routed.so")
    runtime = Mock(uid_size=64, sizes=[(128, 256)])
    reason = "undefined symbol: ixmm_shmem_uid_size"
    loader = Mock(
        return_value=runtime, side_effect=AttributeError(reason) if rank else None
    )
    monkeypatch.setattr(decode, "IxmmLL", loader)
    votes = [(True, None, (runtime.uid_size, runtime.sizes)), (True, reason, None)]

    def gather(output, vote, *, group):
        assert vote == votes[rank]
        output[:] = votes

    monkeypatch.setattr(decode.dist, "all_gather_object", gather)
    warning = Mock()
    monkeypatch.setattr(decode.logger, "warning", warning)
    layer = SimpleNamespace(w13_weight=SimpleNamespace(device="cuda"))
    for _ in range(2):
        decode.prepare_layer(layer, 18, group)
        assert not hasattr(layer, "_ilu_ixmm_ll")
    loader.assert_called_once_with("routed.so")
    runtime.initialize.assert_not_called()
    assert warning.call_count == int(rank == 0)


def test_missing_optional_library(tmp_path):
    with pytest.raises(OSError):
        decode.IxmmLL(str(tmp_path / "missing.so"))
