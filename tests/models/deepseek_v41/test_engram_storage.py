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


@pytest.mark.parametrize("setting", ["cpu_offload", "use_thp", "private", "legacy"])
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
    with pytest.raises(ValueError):
        storage.validate_engram_config(config)


def test_shared_checkpoint_mmap_does_not_require_dev_shm_capacity(monkeypatch):
    config = _config(dp_shared_memory=True)
    monkeypatch.delenv("VLLM_PLE_CPU_OFFLOAD", raising=False)
    monkeypatch.delenv("VLLM_ILU_V41_DISABLE_ENGRAM", raising=False)
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
        _mark_batch_image_tokens=lambda input_ids: None,
    )
    IluDeepseekV41Model._prepare_engram(model, torch.zeros(2), torch.arange(2))
    assert len(calls) == int(across_dp)
    if across_dp:
        assert torch.all(calls[0] == -1)
        assert engram._rows is None


def _shared_worker(rank, rendezvous, checkpoint):
    import os

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
    # Updating the underlying checkpoint file must be visible through both
    # processes' shared page-cache mappings without copying into /dev/shm.
    if rank == 0:
        offset, _, _ = mod._safetensors_entry(checkpoint, "weight")
        with open(checkpoint, "r+b") as f:
            f.seek(offset + 3 * 256)
            f.write(
                torch.full((256,), 2.0, dtype=torch.float32)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            f.flush()
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
            formal_ops.engram_lookup_out(ids, device, dev_scales, staged, 3, 8195, 0, 3)
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
    # Inspect our mapping: must be a read-only shared mapping of the checkpoint
    # file itself (r--s), with zero /dev/shm tmpfs copies.
    real_checkpoint = os.path.realpath(checkpoint)
    with open("/proc/self/maps") as maps:
        for line in maps:
            lo, hi = (int(addr, 16) for addr in line.split()[0].split("-"))
            if lo <= host.data_ptr() < hi:
                perms = line.split()[1]
                assert perms == "r--s"
                assert real_checkpoint in line
                assert "/dev/shm" not in line
                break
        else:
            raise AssertionError("shared Engram mapping not found")
    dist.destroy_process_group()


@pytest.mark.parametrize("retain_view", [False, True])
def test_map_shared_checkpoint_rows_maps_file_directly_without_shm(
    tmp_path, monkeypatch, retain_view
):
    import mmap
    import os

    import ixformer.inference.functions as ixf_ops

    rows, dim = 64, 256
    path = str(tmp_path / "table.safetensors")
    save_file(
        {
            "weight": torch.arange(rows * dim, dtype=torch.float32)
            .reshape(rows, dim)
            .to(torch.float8_e4m3fn),
        },
        path,
    )
    registered = []
    unregistered = []

    class FakeCudart:
        def cudaHostRegister(self, ptr, size, flags):
            registered.append((ptr, size, flags))
            return SimpleNamespace(value=0)

        def cudaHostUnregister(self, ptr):
            unregistered.append(ptr)
            return SimpleNamespace(value=0)

    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    monkeypatch.setattr(ixf_ops, "get_cuda_view_from_cpu_tensor", lambda t: t)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, obj, group=None: out.__setitem__(0, obj),
    )
    group = SimpleNamespace(world_size=1, rank_in_group=0, cpu_group=None)

    host, view, mapping = storage.map_shared_checkpoint_rows(
        path, "weight", 3, 19, group, mod._safetensors_entry
    )
    assert host.shape == (16, dim)
    assert view is host
    assert len(registered) == 1
    reg_base, reg_size, reg_flags = registered[0]
    assert reg_flags == 2
    assert reg_base % mmap.PAGESIZE == 0
    assert reg_size % mmap.PAGESIZE == 0
    assert reg_base <= host.data_ptr()
    assert reg_base + reg_size >= host.data_ptr() + host.nbytes

    real_checkpoint = os.path.realpath(path)
    with open("/proc/self/maps") as maps:
        for line in maps:
            lo, hi = (int(addr, 16) for addr in line.split()[0].split("-"))
            if lo <= host.data_ptr() < hi:
                assert line.split()[1] == "r--s"
                assert real_checkpoint in line
                assert "/dev/shm" not in line
                break
        else:
            raise AssertionError("shared Engram mapping not found")

    retained = host.view(-1) if retain_view else None
    del host, view
    import gc

    gc.collect()
    if retain_view:
        assert not mapping.closed
        assert not unregistered
        assert retained.numel() == 16 * dim
        del retained
        gc.collect()
    assert unregistered == [reg_base]
    assert mapping.closed


def test_map_shared_checkpoint_rows_cleans_up_on_register_view_and_peer_errors(
    tmp_path, monkeypatch
):
    import mmap

    import ixformer.inference.functions as ixf_ops

    rows, dim = 32, 256
    path = str(tmp_path / "table.safetensors")
    save_file(
        {
            "weight": torch.ones(rows, dim, dtype=torch.float32).to(
                torch.float8_e4m3fn
            ),
        },
        path,
    )

    created_mmaps: list[mmap.mmap] = []
    real_mmap = mmap.mmap

    def tracking_mmap(*args, **kwargs):
        m = real_mmap(*args, **kwargs)
        created_mmaps.append(m)
        return m

    monkeypatch.setattr(mmap, "mmap", tracking_mmap)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)

    # Case 1: cudaHostRegister fails -> sticky error cleared, mmap closed.
    cleared_errors = []
    unregistered = []

    class FailRegisterCudart:
        def cudaHostRegister(self, ptr, size, flags):
            return SimpleNamespace(value=11)

        def cudaHostUnregister(self, ptr):
            unregistered.append(ptr)
            return SimpleNamespace(value=0)

        def cudaGetLastError(self):
            cleared_errors.append(True)
            return 11

    monkeypatch.setattr(torch.cuda, "cudart", lambda: FailRegisterCudart())
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, obj, group=None: out.__setitem__(0, obj),
    )
    group = SimpleNamespace(world_size=1, rank_in_group=0, cpu_group=None)

    with pytest.raises(RuntimeError, match="Engram cudaHostRegister failed"):
        storage.map_shared_checkpoint_rows(
            path, "weight", 0, rows, group, mod._safetensors_entry
        )
    assert cleared_errors == [True]
    assert unregistered == []
    assert len(created_mmaps) == 1 and created_mmaps[-1].closed

    # Case 2: cudaHostRegister succeeds, get_cuda_view_from_cpu_tensor fails
    # -> cudaHostUnregister called and mmap closed immediately.
    registered = []

    class OkCudart:
        def cudaHostRegister(self, ptr, size, flags):
            registered.append(ptr)
            return SimpleNamespace(value=0)

        def cudaHostUnregister(self, ptr):
            unregistered.append(ptr)
            return SimpleNamespace(value=0)

    monkeypatch.setattr(torch.cuda, "cudart", lambda: OkCudart())

    def fail_view(_t):
        raise RuntimeError("UVA view failed")

    monkeypatch.setattr(ixf_ops, "get_cuda_view_from_cpu_tensor", fail_view)
    with pytest.raises(RuntimeError, match="UVA view failed"):
        storage.map_shared_checkpoint_rows(
            path, "weight", 0, rows, group, mod._safetensors_entry
        )
    assert len(registered) == 1
    assert unregistered == registered
    assert len(created_mmaps) == 2 and created_mmaps[-1].closed

    # Case 3: Local rank succeeds, peer rank fails in all_gather_object
    # -> local rank unregisters and closes mmap before raising.
    registered.clear()
    unregistered.clear()
    monkeypatch.setattr(ixf_ops, "get_cuda_view_from_cpu_tensor", lambda t: t)
    group2 = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=None)

    def peer_failure_all_gather(out, obj, group=None):
        out[0] = obj
        out[1] = "RuntimeError: peer failed"

    monkeypatch.setattr(torch.distributed, "all_gather_object", peer_failure_all_gather)
    with pytest.raises(RuntimeError, match="peer failed"):
        storage.map_shared_checkpoint_rows(
            path, "weight", 0, rows, group2, mod._safetensors_entry
        )
    assert len(registered) == 1
    assert unregistered == registered
    assert len(created_mmaps) == 3 and created_mmaps[-1].closed


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


def test_failed_mapping_waits_for_storage_retained_by_callback(tmp_path, monkeypatch):
    import gc
    import mmap

    import ixformer.inference.functions as ixf_ops

    path = str(tmp_path / "small.safetensors")
    save_file({"weight": torch.ones(8, 256, dtype=torch.uint8)}, path)
    held, mappings, events = [], [], []
    real_mmap = mmap.mmap

    def tracked_mmap(*args, **kwargs):
        mapping = real_mmap(*args, **kwargs)
        mappings.append(mapping)
        return mapping

    class FakeCudart:
        def cudaHostRegister(self, *args):
            events.append("register")
            return SimpleNamespace(value=0)

        def cudaHostUnregister(self, *args):
            events.append("unregister")
            return SimpleNamespace(value=0)

    def fail_view(tensor):
        held.append(tensor._base)
        raise RuntimeError("retained UVA callback failure")

    monkeypatch.setattr(mmap, "mmap", tracked_mmap)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    monkeypatch.setattr(ixf_ops, "get_cuda_view_from_cpu_tensor", fail_view)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, obj, group=None: out.__setitem__(0, obj),
    )
    with pytest.raises(RuntimeError, match="retained UVA callback failure"):
        storage.map_shared_checkpoint_rows(
            path,
            "weight",
            0,
            8,
            SimpleNamespace(world_size=1, cpu_group=None),
            mod._safetensors_entry,
        )
    assert held[0] is not None and held[0].numel() == 2048
    assert not mappings[0].closed
    assert events == ["register"]
    assert held[0].eq(1).all()
    held.clear()
    gc.collect()
    assert mappings[0].closed
    assert events == ["register", "unregister"]
