import pytest
import torch

from vllm_iluvatar._custom_ops import vocab_parallel_embedding


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layout", ["vector", "matrix", "strided"])
def test_shard_lookup(id_dtype, layout):
    ids = torch.tensor([99, 100, 169, 170, 999, 1000, 1012, 1013], dtype=id_dtype)
    weight = torch.randn(96, 192, dtype=torch.bfloat16)
    expected = torch.zeros(ids.numel(), weight.shape[1], dtype=weight.dtype)
    expected[1] = weight[0]
    expected[2] = weight[69]
    expected[5] = weight[76]
    expected[6] = weight[88]
    if layout == "matrix":
        ids = ids.reshape(2, 4)
        expected = expected.reshape(2, 4, -1)
    ids, weight = ids.cuda(), weight.cuda()
    if layout == "strided":
        storage = torch.empty(
            weight.shape[0],
            weight.shape[1] * 2,
            device=weight.device,
            dtype=weight.dtype,
        )
        storage[:, ::2].copy_(weight)
        weight = storage[:, ::2]
    actual = vocab_parallel_embedding(ids, weight, 100, 170, 6, 1000, 1013)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
