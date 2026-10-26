"""Storage rejection and real shared-page/UVA/side-stream graph regression."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_iluvatar.models.deepseek_v41 import engram as mod
from vllm_iluvatar.models.deepseek_v41 import engram_storage as storage


def _config(**settings):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                engram_layer_ids=[1, 14],
                engram_num_embeddings=[100, 200],
                engram_head_dim=256,
                engram_max_ngram_size=4,
                engram_n_heads=8,
            )
        ),
        parallel_config=SimpleNamespace(data_parallel_size=4, tensor_parallel_size=2),
        engram_config=SimpleNamespace(
            cpu_offload=True,
            use_thp=False,
            dp_shared_memory=settings.get("dp_shared_memory", True),
            embedding_across_dp=settings.get("embedding_across_dp", False),
        ),
    )


@pytest.mark.parametrize(
    "setting", ["cpu_offload", "use_thp", "private", "legacy", "shm"]
)
def test_reject_storage_before_allocation(setting, monkeypatch):
    config = _config()
    monkeypatch.delenv("VLLM_PLE_CPU_OFFLOAD", raising=False)
    monkeypatch.delenv("VLLM_ILU_V41_DISABLE_ENGRAM", raising=False)
    if setting == "cpu_offload":
        config.engram_config.cpu_offload = False
    elif setting == "use_thp":
        config.engram_config.use_thp = True
    elif setting == "private":
        config.engram_config.dp_shared_memory = False
    elif setting == "legacy":
        monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "0")
    else:
        monkeypatch.setattr(
            storage.shutil, "disk_usage", lambda p: SimpleNamespace(total=1)
        )
    with pytest.raises(ValueError):
        storage.validate_engram_config(config)


def test_across_dp_and_disabled_engram(monkeypatch):
    storage.validate_engram_config(
        _config(dp_shared_memory=False, embedding_across_dp=True)
    )
    monkeypatch.setenv("VLLM_ILU_V41_DISABLE_ENGRAM", "1")
    storage.validate_engram_config(_config(dp_shared_memory=False))


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_shared_topology_accepts_tp4_dp4_and_tp8(tp):
    config = _config(embedding_across_dp=True)
    config.parallel_config.tensor_parallel_size = tp
    storage.validate_engram_config(config)  # shm takes precedence over across-DP


def test_across_dp_empty_heads_rejected():
    config = _config(dp_shared_memory=False, embedding_across_dp=True)
    config.parallel_config.tensor_parallel_size = 4
    with pytest.raises(AssertionError, match="Engram ranks without hash heads"):
        storage.validate_engram_config(config)


@pytest.mark.parametrize(
    ("tp", "shared"), [(1, True), (2, True), (4, True), (8, True), (2, False)]
)
def test_head_ownership_and_collective_selection(tp, shared, monkeypatch):
    config = _config(dp_shared_memory=shared, embedding_across_dp=True)
    config.parallel_config.tensor_parallel_size = tp
    dp_rank = 3
    group = SimpleNamespace(world_size=4, rank_in_group=dp_rank)
    monkeypatch.setattr(mod, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(mod, "get_engram_dp_group", lambda: group)
    monkeypatch.setattr(mod, "get_dp_group", lambda: group)
    monkeypatch.setattr(mod, "check_shared_group", lambda group: None)
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: tp - 1)
    monkeypatch.setattr(mod, "_max_batched_tokens", lambda: 2)
    heads = tuple(range(1, 25))
    embedding = mod.EngramEmbedding(sum(heads), 256, heads)
    shards = tp if shared else tp * 4
    width = (24 + shards - 1) // shards
    rank = tp - 1 if shared else (tp - 1) * 4 + dp_rank
    assert embedding.local_cols == width
    assert embedding.head_start == rank * width
    assert embedding.start_idx == sum(heads[: rank * width])
    assert embedding.end_idx == sum(heads[: (rank + 1) * width])
    assert (embedding._dp_group is None) == shared
    assert embedding._lookup_rows.shape[0] == (2 if shared else 8)


@pytest.mark.parametrize("across_dp", [False, True])
def test_dummy_forward_without_global_config(across_dp):
    from vllm_iluvatar.models.deepseek_v41.model import IluDeepseekV41Model

    calls = []
    engram = SimpleNamespace(
        embed=SimpleNamespace(_dp_group=object() if across_dp else None),
        prepare_embeddings=lambda hashes: calls.append(hashes.clone()),
    )
    model = SimpleNamespace(
        engram_hash=None,
        engram_layout=object(),
        _engram_dummy=torch.zeros(2, 24, dtype=torch.int32),
        _engram_layers=lambda: iter([engram]),
    )
    IluDeepseekV41Model._prepare_engram(model, torch.zeros(2), torch.arange(2))
    assert len(calls) == int(across_dp)
    if across_dp:
        assert torch.all(calls[0] == -1)
        assert engram._rows is None


def _shared_worker(rank, rendezvous, checkpoint):
    import torch.distributed as dist
    from ixformer.inference.functions import deepseekv41 as formal_ops

    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)

    class Group:
        world_size = 2
        rank_in_group = rank
        cpu_group = dist.group.WORLD

        def broadcast_object(self, value):
            objects = [value]
            dist.broadcast_object_list(objects, src=0)
            return objects[0]

    group = Group()
    storage.check_shared_group(group)
    host, device, mapping = storage.map_shared_checkpoint_rows(
        checkpoint,
        "weight",
        3,
        8195,
        group,
        mod._safetensors_entry,
    )
    scales, dev_scales, scale_mapping = storage.map_shared_checkpoint_rows(
        checkpoint,
        "scale",
        3,
        8195,
        group,
        mod._safetensors_entry,
    )
    # Changing one process's mapping must change the peer's bytes, not a copy.
    if rank == 0:
        host[0].fill_(2)
    dist.barrier()
    assert torch.all(host[0].float() == 2)
    ids = torch.tensor([[3, 4, -1], [8194, 8195, 3]], device="cuda", dtype=torch.int32)
    staged = torch.zeros(2, 3, 256, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(staged)
    stream = torch.cuda.Stream()
    done = torch.cuda.Event()

    def run():
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            formal_ops.engram_lookup_out(
                ids, device, dev_scales, staged, 3, 8195, 0, 3
            )
            done.record(stream)
        # Represents preceding decoder work on the main stream.
        output.zero_()
        torch.cuda.current_stream().wait_event(done)
        output.copy_(staged)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(output[0, 0] == 2)
    assert torch.all(output[0, 1] == 1)
    assert torch.all(output[0, 2] == 0)
    assert torch.all(output[1, 0] == 1)
    assert torch.all(output[1, 1] == 0)
    ids.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(output == 0)
    # Inspect our mapping, allowing another service to initialize its own tables.
    with open("/proc/self/maps") as maps:
        for line in maps:
            lo, hi = (int(addr, 16) for addr in line.split()[0].split("-"))
            if lo <= host.data_ptr() < hi:
                assert "/dev/shm/ilu_engram_" in line and "(deleted)" in line
                break
        else:
            raise AssertionError("shared Engram mapping not found")
    dist.destroy_process_group()


def test_shared_pages_uva_prefetch_graph(tmp_path):
    rows, dim = 8195, 256
    path = str(tmp_path / "table.safetensors")
    save_file(
        {
            "weight": torch.ones(rows, dim, dtype=torch.float32).to(
                torch.float8_e4m3fn
            ),
            "scale": torch.full((rows, dim // 32), 127, dtype=torch.uint8),
        },
        path,
    )
    torch.multiprocessing.spawn(
        _shared_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", path),
        nprocs=2,
    )
