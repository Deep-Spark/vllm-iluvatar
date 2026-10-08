"""Differential tests against the exact upstream DP agreement protocol."""
import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode as Mode
from vllm.v1.worker.gpu import dp_utils as upstream
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor as Desc
from vllm_iluvatar.v1.worker.gpu import dp_utils, merged_dp_sync as merged
from vllm_iluvatar.v1.worker.dp_utils import get_last_all_ranks_decode_only

@pytest.mark.parametrize('scenario', ['decode', 'mixed_vote', 'mixed_query', 'eager', 'empty', 'skip', 'varlen'])
@pytest.mark.parametrize('rank', [0, 1])
def test_agreement_matches_upstream(monkeypatch, scenario, rank):
    # Columns represent two different ranks; preserve metadata agreement even
    # when decode votes differ, one rank requests eager, or all ranks are idle.
    data = torch.tensor([[6,12],[Mode.FULL.value,Mode.FULL.value],[6,6],[-1,-1],[0,0],[1,2],[1,1]], dtype=torch.int32)
    if scenario == 'mixed_vote': data[6,1] = 0
    if scenario == 'mixed_query': data[2,1] = 3
    if scenario == 'eager': data[1,1] = Mode.NONE.value
    if scenario == 'empty': data[0] = 0
    if scenario == 'varlen': data[3] = torch.tensor([6,8])
    calls = []
    def reduce(tensor, group):
        calls.append(tuple(tensor.shape))
        assert torch.equal(tensor[:,rank], data[:len(tensor),rank])
        tensor.copy_(data[:len(tensor)])
    original = dp_utils._ORIGINAL_SYNC or upstream.sync_cudagraph_and_dp_padding
    monkeypatch.setattr(merged.dist, 'all_reduce', reduce)
    monkeypatch.setattr(merged, 'get_dp_group', lambda: SimpleNamespace(cpu_group='test'))
    monkeypatch.setattr(upstream, 'get_dp_group', lambda: SimpleNamespace(cpu_group='test'))
    for module in (merged, upstream):
        monkeypatch.setattr(module, 'should_skip_dp_coordination', lambda: scenario == 'skip')
    def dispatch(nreq, ntok, uniform, **kwargs):
        return Desc(cg_mode=Mode.FULL if uniform else Mode.NONE, num_tokens=ntok, num_reqs=nreq, uniform_token_count=uniform)
    manager = SimpleNamespace(dispatch=dispatch)
    args = dict(cudagraph_manager=manager, desired_batch_desc=Desc(cg_mode=Mode(int(data[1,rank])),num_tokens=int(data[0,rank]),num_reqs=int(data[5,rank])),num_tokens=int(data[0,rank]),num_reqs=int(data[5,rank]),uniform_token_count=int(data[2,rank]),dp_size=2,dp_rank=rank,max_query_len=None if data[3,rank]==-1 else int(data[3,rank]),uniform_decode=bool(data[6,rank]))
    before = original(**args)
    after = merged.sync_cudagraph_and_dp_padding(**args)
    assert before[0] == after[0]
    if before[1] is None:
        assert after[1] is None
    else:
        for name in before[1].__dataclass_fields__:
            a,b = getattr(before[1],name),getattr(after[1],name)
            assert torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b
    expected = bool(data[6,rank]) if scenario=='skip' else bool(torch.all(data[6]==1))
    assert get_last_all_ranks_decode_only() == expected
    assert calls == ([] if scenario=='skip' else [(6,2),(7,2)])

def test_signature():
    assert inspect.signature(merged.sync_cudagraph_and_dp_padding) == inspect.signature(upstream.sync_cudagraph_and_dp_padding)
