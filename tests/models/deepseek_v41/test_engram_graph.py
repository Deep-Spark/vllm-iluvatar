import ixformer.inference.functions as shared_ops
import torch
from ixformer.inference.backend_registry import registry
from ixformer.inference.functions import deepseekv41 as ops
from safetensors.torch import save_file

from vllm_iluvatar.models.deepseek_v41.engram import _map_locked_rows


def _hash_case():
    device = "cuda"
    input_ids = torch.tensor([1, 7, 3, -4, 99], dtype=torch.int64, device=device)
    token_map = torch.tensor(
        [10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.int32, device=device
    )
    positions = torch.tensor([2, 3, 4, 0, 1], dtype=torch.int64, device=device)
    token_to_req = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=device)
    query_start = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    block_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device=device)
    slot_mapping = torch.tensor([6, 7, 8, 12, -1], dtype=torch.int64, device=device)
    id_cache = torch.full((20,), -1, dtype=torch.int32, device=device)
    id_cache[4:6] = torch.tensor([31, 32], dtype=torch.int32, device=device)
    multipliers = torch.tensor(
        [[3, 5, 7, 9], [11, 13, 15, 17]], dtype=torch.int64, device=device
    )
    primes = torch.tensor(
        [
            [101, 103, 107, 109, 113, 127],
            [131, 137, 139, 149, 151, 157],
        ],
        dtype=torch.int64,
        device=device,
    )
    offsets = torch.tensor(
        [[0, 101, 204, 311, 420, 533], [0, 131, 268, 407, 556, 707]],
        dtype=torch.int64,
        device=device,
    )
    hash_out = torch.empty((5, 2, 6), dtype=torch.int32, device=device)
    return [
        input_ids,
        token_map,
        positions,
        token_to_req,
        query_start,
        block_table,
        slot_mapping,
        id_cache,
        multipliers,
        primes,
        offsets,
        hash_out,
        2,
        -1,
        4,
    ]


def test_readonly_mmap_uva_lookup_graph(tmp_path):
    # Above the ixformer in-place register threshold: tests the real mmap path.
    rows, dim = 8192, 256
    weight = torch.ones(rows, dim, dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.full((rows, dim // 32), 127, dtype=torch.uint8)
    path = str(tmp_path / "table.safetensors")
    save_file({"weight": weight, "scale": scales}, path)
    host_w, region_w = _map_locked_rows(path, "weight", 0, rows)
    host_s, region_s = _map_locked_rows(path, "scale", 0, rows)
    dev_w = shared_ops.get_cuda_view_from_cpu_tensor(host_w)
    dev_s = shared_ops.get_cuda_view_from_cpu_tensor(host_s)
    ids = torch.tensor(
        [[0, rows - 1, -1], [3, 4, rows]], dtype=torch.int32, device="cuda"
    )
    out = torch.zeros(2, 3, dim, dtype=torch.bfloat16, device="cuda")

    def run():
        ops.engram_lookup_out(ids, dev_w, dev_s, out, 0, rows, 0, 3)

    run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    g.replay()
    torch.cuda.synchronize()
    assert torch.all(out[:, :2] == 1) and torch.all(out[:, 2] == 0)
    ids.fill_(-1)
    g.replay()
    torch.cuda.synchronize()
    assert torch.all(out == 0)


def test_hash_graph_cross_chunk_and_padding():
    args = _hash_case()
    expected = [x.clone() if isinstance(x, torch.Tensor) else x for x in args]
    registry.get_reference_impl("engram_hash_out")(*expected)
    ops.engram_hash_out(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.engram_hash_out(*args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(args[11], expected[11])
    args[0][0] = 3
    registry.get_reference_impl("engram_hash_out")(
        *[args[i] if i != 11 else expected[11] for i in range(len(args))]
    )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(args[11], expected[11])
