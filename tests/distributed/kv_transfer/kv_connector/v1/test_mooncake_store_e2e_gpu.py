# SPDX-License-Identifier: Apache-2.0
"""End-to-end CoreX GPU + real `mooncake_master` 2P integration tests.

Automatically skipped when CoreX/CUDA GPUs or `mooncake_master` are unavailable.
Covers:
  1. Multi-process (`spawn`) P1 -> P2 GPU KV offload & bit-exact restore for both
     standard FullAttention and DeepSeek-V4.1-Flash (paired i4g32 + FP16 scale +
     split INT8 indexer + FP32 scale).
  2. Live `rdma` / `tcp` `standalone-store` (`global_segment_size="0"`) P2 pull
     directly from P1's DRAM segment over Mooncake transfer engine.
"""

from __future__ import annotations

import fcntl
import hashlib
import multiprocessing as mp
import os
import shutil
import socket
import struct
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.outputs import KVConnectorOutput

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
    IluMooncakeStoreConnector,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.mooncake_store.layout import (
    _resolve_dsv41_quant_format,
)
from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    PAYLOAD_PAGE_BYTES,
    CompressedKVState,
    plan_scale_layout,
)
from vllm_iluvatar.models.deepseek_v41.indexer_kv import (
    SPLIT_INDEXER_FORMAT,
    IndexerScaleState,
    plan_indexer_scales,
)


def _find_mooncake_master() -> str | None:
    try:
        import mooncake  # type: ignore[import-not-found]

        candidate = Path(mooncake.__file__).resolve().parent / "mooncake_master"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    except Exception:
        pass
    return shutil.which("mooncake_master")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _netdev_ipv4(ifname: str) -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            packed = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", ifname[:15].encode()))
            return socket.inet_ntoa(packed[20:24])
    except OSError:
        return None


def _pick_rdma_device_and_ip() -> tuple[str, str] | None:
    ib_dir = Path("/sys/class/infiniband")
    if not ib_dir.is_dir():
        return None
    candidates = sorted((p.name for p in ib_dir.iterdir() if p.name.startswith("mlx5_")), key=lambda n: (n != "mlx5_2", n))
    for hca in candidates:
        state_file = ib_dir / hca / "ports" / "1" / "state"
        net_dir = ib_dir / hca / "device" / "net"
        if state_file.is_file() and "ACTIVE" in state_file.read_text() and net_dir.is_dir():
            for netdev in sorted(p.name for p in net_dir.iterdir()):
                if ip_addr := _netdev_ipv4(netdev):
                    return hca, ip_addr
    return None


@contextmanager
def _run_mooncake_master() -> Iterator[tuple[int, int]]:
    master_bin = _find_mooncake_master()
    assert master_bin is not None
    rpc_port, http_port, metrics_port = _free_port(), _free_port(), _free_port()
    proc = subprocess.Popen(
        [
            master_bin,
            "--enable_http_metadata_server=true",
            "--http_metadata_server_host=0.0.0.0",
            f"--http_metadata_server_port={http_port}",
            f"--port={rpc_port}",
            f"--metrics_port={metrics_port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", rpc_port), timeout=0.2):
                    with socket.create_connection(("127.0.0.1", metrics_port), timeout=0.2):
                        break
            except OSError:
                time.sleep(0.05)
        else:
            out = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"mooncake_master failed to start on ports {rpc_port}/{http_port}:\n{out}")
        time.sleep(0.1)
        yield rpc_port, http_port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        if proc.stdout is not None:
            proc.stdout.close()


def _make_hashes(tokens: list[int], hash_block_size: int) -> list[BlockHash]:
    hashes: list[BlockHash] = []
    acc = b""
    for start in range(0, len(tokens) // hash_block_size * hash_block_size, hash_block_size):
        acc = hashlib.sha256(acc + repr(tokens[start : start + hash_block_size]).encode()).digest()[:16]
        hashes.append(BlockHash(acc))
    return hashes


def _sched_out(
    new_reqs: list[NewRequestData] | None = None,
    num_scheduled_tokens: dict[str, int] | None = None,
) -> SchedulerOutput:
    scheduled = num_scheduled_tokens or {}
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs or [],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        ),
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _new_req_data(req_id: str, tokens: list[int], block_ids: tuple[list[int], ...]) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=tokens,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=block_ids,
        num_computed_tokens=0,
        lora_request=None,
    )


def _mock_req(req_id: str, tokens: list[int], hashes: list[BlockHash]) -> Any:
    return SimpleNamespace(
        request_id=req_id,
        num_tokens=len(tokens),
        prompt_token_ids=tokens,
        all_token_ids=tokens,
        block_hashes=hashes,
        num_computed_tokens=0,
    )


def _make_gpu_vllm_cfg(
    engine_id: str,
    dev_idx: int,
    *,
    model: str,
    use_mla: bool,
    num_kv_heads: int,
    block_size: int,
    prefix_match_unit: int | None,
    fwd_ctx: dict[str, Any],
    extra_config: dict[str, Any],
) -> Any:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=model, use_mla=use_mla, get_total_num_kv_heads=lambda: num_kv_heads
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        parallel_config=SimpleNamespace(
            rank=0,
            local_rank=dev_idx,
            world_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_index=0,
        ),
        cache_config=SimpleNamespace(
            block_size=block_size,
            prefix_match_unit=prefix_match_unit,
            cache_dtype="auto",
            prefix_caching_hash_algo="sha256",
            enable_prefix_caching=True,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        compilation_config=SimpleNamespace(static_forward_context=fwd_ctx),
        speculative_config=None,
        kv_events_config=None,
        kv_transfer_config=SimpleNamespace(
            kv_connector="IluMooncakeStoreConnector",
            kv_role="kv_both",
            engine_id=engine_id,
            is_kv_producer=True,
            kv_connector_extra_config=extra_config,
        ),
    )


def _build_dsv41_gpu_node(
    engine_id: str,
    dev_idx: int,
    extra_config: dict[str, Any],
    *,
    fill: bool,
) -> tuple[Any, Any, dict[str, torch.Tensor], dict[str, Any], list[Any]]:
    os.environ["VLLM_ILU_DS41_COMPRESSED_KV_FORMAT"] = "i4g32"
    os.environ["VLLM_ILU_DS41_INDEXER_KV_LAYOUT"] = "split"
    os.environ["VLLM_ILU_DS4_INDEXER_KV_DTYPE"] = "int8"

    torch.cuda.set_device(dev_idx)
    device = torch.device("cuda", dev_idx)
    fmt = _resolve_dsv41_quant_format("i4g32")
    rows, payload_b = fmt.rows_per_page(512), fmt.payload_bytes(512)
    block_size, num_blocks = rows * 2, 8

    swa_spec = SlidingWindowMLASpec(
        bounded_replay=True, block_size=64, sliding_window=128, num_kv_heads=1, head_size=512, head_size_v=0, dtype=torch.bfloat16
    )
    main_spec = MLAAttentionSpec(
        block_size=block_size, tokens_per_state=2, num_kv_heads=1, head_size=512, dtype=torch.uint8, state_content_bytes=payload_b, cache_dtype_str="i4g32", storage_block_size=block_size
    )
    idx_spec = MLAAttentionSpec(
        block_size=1024, tokens_per_state=2, num_kv_heads=1, head_size=128, dtype=torch.uint8, state_content_bytes=128, cache_dtype_str=SPLIT_INDEXER_FORMAT
    )

    m_layers = ["model.layers.2.attn", "model.layers.8.attn"]
    i_layers = ["model.layers.2.attn.indexer.k_cache", "model.layers.8.attn.indexer.k_cache"]
    idx_page = idx_spec.page_size_bytes
    stride = 2 * PAYLOAD_PAGE_BYTES + 2 * idx_page
    total = num_blocks * stride
    raw = (
        torch.randint(1, 250, (total,), dtype=torch.uint8, device=device)
        if fill
        else torch.zeros(total, dtype=torch.uint8, device=device)
    )
    kv_cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=total, layers=m_layers, layer_stride=PAYLOAD_PAGE_BYTES, block_stride=stride, offset=0),
            KVCacheTensor(size=total, layers=i_layers, layer_stride=idx_page, block_stride=stride, offset=2 * PAYLOAD_PAGE_BYTES),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.0.swa"], swa_spec),
            KVCacheGroupSpec(m_layers, main_spec),
            KVCacheGroupSpec(i_layers, idx_spec),
        ],
    )
    kv_cfg.kv_cache_layout = KVCacheLayout.LBHNC
    fwd_ctx: dict[str, Any] = {
        name: SimpleNamespace(compressed_kv_format=fmt, kv_cache_scales=None) for name in m_layers
    }
    for name in i_layers:
        fwd_ctx[name] = SimpleNamespace(kv_cache_scales=None)
    m_state = CompressedKVState(plan_scale_layout(kv_cfg, fmt), fmt, device, fwd_ctx)
    i_state = IndexerScaleState(plan_indexer_scales(kv_cfg), device, fwd_ctx)
    if fill:
        m_state.scale_storage.uniform_(-3.0, 3.0)
        i_state.storage.uniform_(-5.0, 5.0)
    torch.cuda.synchronize(device)

    kv_caches = {
        name: torch.as_strided(raw, (num_blocks, rows, payload_b), (stride, payload_b, 1), pos * PAYLOAD_PAGE_BYTES)
        for pos, name in enumerate(m_layers)
    }
    for pos, name in enumerate(i_layers):
        kv_caches[name] = torch.as_strided(
            raw, (num_blocks, idx_spec.num_states, 128), (stride, 128, 1), 2 * PAYLOAD_PAGE_BYTES + pos * idx_page
        )

    vllm_cfg = _make_gpu_vllm_cfg(
        engine_id,
        dev_idx,
        model="/models/DeepSeek-V4.1-Flash",
        use_mla=True,
        num_kv_heads=1,
        block_size=64,
        prefix_match_unit=64,
        fwd_ctx=fwd_ctx,
        extra_config=extra_config,
    )
    worker = IluMooncakeStoreConnector(vllm_cfg, KVConnectorRole.WORKER, kv_cfg)
    worker.register_kv_caches(kv_caches)
    sched = IluMooncakeStoreConnector(vllm_cfg, KVConnectorRole.SCHEDULER, kv_cfg)
    return sched, worker, kv_caches, fwd_ctx, [m_state, i_state]


def _std_worker_proc(
    role: str,
    dev_idx: int,
    extra_config: dict[str, Any],
    rpc_base: str,
    pipe_q: Any,
    done_ev: Any,
) -> None:
    import vllm.envs

    vllm.envs.VLLM_RPC_BASE_PATH = rpc_base
    torch.cuda.set_device(dev_idx)
    device = torch.device("cuda", dev_idx)
    num_blocks, block_size = 8, 16
    spec = FullAttentionSpec(block_size=block_size, num_kv_heads=4, head_size=64, dtype=torch.float16)
    layers = ["model.layers.0.attn", "model.layers.1.attn"]
    page = spec.page_size_bytes
    stride = len(layers) * page
    total = num_blocks * stride
    kv_cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=total, layers=layers, layer_stride=page, block_stride=stride, offset=0)
        ],
        kv_cache_groups=[KVCacheGroupSpec(layers, spec)],
    )
    kv_cfg.kv_cache_layout = KVCacheLayout.LBHNC
    vllm_cfg = _make_gpu_vllm_cfg(
        f"e2e-std-{role}",
        dev_idx,
        model="/models/Qwen3-32B" if role == "p1" else "/other/mount/Qwen3-32B",
        use_mla=False,
        num_kv_heads=4,
        block_size=block_size,
        prefix_match_unit=None,
        fwd_ctx={},
        extra_config=extra_config,
    )
    backing = torch.zeros(total // 2, dtype=torch.float16, device=device)
    kv_caches = {
        layer: torch.as_strided(
            backing, (num_blocks, 4, block_size, 64), (stride // 2, block_size * 64, 64, 1), idx * (page // 2)
        )
        for idx, layer in enumerate(layers)
    }
    worker = IluMooncakeStoreConnector(vllm_cfg, KVConnectorRole.WORKER, kv_cfg)
    sched = IluMooncakeStoreConnector(vllm_cfg, KVConnectorRole.SCHEDULER, kv_cfg)
    try:
        worker.register_kv_caches(kv_caches)
        tokens_p1 = list(range(1000, 1064))
        if role == "p1":
            req = _mock_req("e2e-std-1", tokens_p1, _make_hashes(tokens_p1, block_size))
            assert sched.get_num_new_matched_tokens(req, 0) == (0, False)
            sched.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda group_ids=None: ([1, 2, 3, 4],)), 0)
            expected_cpu: dict[str, list[torch.Tensor]] = {}
            for l_idx, (lname, t) in enumerate(kv_caches.items()):
                expected_cpu[lname] = []
                for b in (1, 2, 3, 4):
                    pat = torch.arange(t[b].numel(), dtype=torch.float32, device=device).view_as(t[b]) * 0.01 + float(l_idx * 10 + b)
                    t[b].copy_(pat.to(torch.float16))
                    expected_cpu[lname].append(t[b].cpu())
            torch.cuda.synchronize(device)
            meta = sched.build_connector_meta(
                _sched_out([_new_req_data("e2e-std-1", tokens_p1, ([1, 2, 3, 4],))], {"e2e-std-1": 64})
            )
            worker.bind_connector_metadata(meta)
            worker.wait_for_save()
            assert worker.connector_worker and worker.connector_worker.kv_send_thread
            worker.connector_worker.kv_send_thread.request_queue.join()
            sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=worker.build_connector_worker_meta()))
            worker.clear_connector_metadata()
            pipe_q.put(expected_cpu)
            done_ev.wait(timeout=30.0)
        else:
            expected_cpu = pipe_q.get(timeout=20.0)
            tokens_p2 = tokens_p1 + list(range(2000, 2016))
            req = _mock_req("e2e-std-2", tokens_p2, _make_hashes(tokens_p2, block_size))
            assert sched.get_num_new_matched_tokens(req, 0) == (64, True)
            dst_blocks = [7, 5, 6, 3]
            sched.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda group_ids=None: (dst_blocks,)), 64)
            meta = sched.build_connector_meta(_sched_out())
            worker.bind_connector_metadata(meta)
            worker.start_load_kv(SimpleNamespace())  # type: ignore[arg-type]
            assert worker.connector_worker
            worker.connector_worker.recv_request_queue.join()
            res = worker.get_transfer_results(set())
            assert res.finished_recving == {"e2e-std-2"} and not res.failed_recving
            torch.cuda.synchronize(device)
            for lname, exp in expected_cpu.items():
                for idx, b in enumerate(dst_blocks):
                    assert torch.equal(kv_caches[lname][b].cpu(), exp[idx])
            done_ev.set()
    finally:
        sched.shutdown()
        worker.shutdown()


def _dsv41_worker_proc(
    role: str,
    dev_idx: int,
    extra_config: dict[str, Any],
    rpc_base: str,
    pipe_q: Any,
    done_ev: Any,
) -> None:
    import vllm.envs

    vllm.envs.VLLM_RPC_BASE_PATH = rpc_base
    sched, worker, kv, ctx, states = _build_dsv41_gpu_node(f"e2e-ds-{role}", dev_idx, extra_config, fill=(role == "p1"))
    try:
        tokens = list(range(1088))
        hashes = _make_hashes(tokens, 64)
        if role == "p1":
            req = _mock_req("ds-p1", tokens, hashes)
            assert sched.get_num_new_matched_tokens(req, 0) == (0, False)
            sched.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda group_ids=None: ([1, 2, 3], [1, 2])), 0)
            meta = sched.build_connector_meta(
                _sched_out([_new_req_data("ds-p1", tokens, ([1, 2, 3], [1, 2, 3], [1, 2]))], {"ds-p1": len(tokens)})
            )
            worker.bind_connector_metadata(meta)
            worker.wait_for_save()
            assert worker.connector_worker and worker.connector_worker.kv_send_thread
            worker.connector_worker.kv_send_thread.request_queue.join()
            sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=worker.build_connector_worker_meta()))
            worker.clear_connector_metadata()
            snap = {
                k: [kv[k][b].cpu() for b in (1, 2)]
                for k in ("model.layers.2.attn", "model.layers.8.attn", "model.layers.2.attn.indexer.k_cache", "model.layers.8.attn.indexer.k_cache")
            }
            snap_scales = {
                k: [ctx[k].kv_cache_scales[b].cpu() for b in (1, 2)]
                for k in snap
            }
            pipe_q.put((snap, snap_scales))
            done_ev.wait(timeout=30.0)
        else:
            snap, snap_scales = pipe_q.get(timeout=20.0)
            req = _mock_req("ds-p2", tokens, hashes)
            matched, is_async = sched.get_num_new_matched_tokens(req, 0)
            assert matched == 1024 and is_async is True
            sched.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda group_ids=None: ([5, 6, 7], [5, 7])), matched)
            worker.bind_connector_metadata(sched.build_connector_meta(_sched_out()))
            worker.start_load_kv(SimpleNamespace())  # type: ignore[arg-type]
            assert worker.connector_worker
            worker.connector_worker.recv_request_queue.join()
            res = worker.get_transfer_results(set())
            assert res.finished_recving == {"ds-p2"} and not res.failed_recving
            torch.cuda.synchronize(torch.device("cuda", dev_idx))
            for k in ("model.layers.2.attn", "model.layers.8.attn"):
                for idx, b in enumerate((5, 6)):
                    assert torch.equal(kv[k][b].cpu(), snap[k][idx])
                    assert torch.equal(ctx[k].kv_cache_scales[b].cpu(), snap_scales[k][idx])
            for k in ("model.layers.2.attn.indexer.k_cache", "model.layers.8.attn.indexer.k_cache"):
                assert torch.equal(kv[k][5].cpu(), snap[k][0])
                assert torch.equal(ctx[k].kv_cache_scales[5].cpu(), snap_scales[k][0])
            done_ev.set()
    finally:
        sched.shutdown()
        worker.shutdown()
        for st in states:
            st.release()


@pytest.mark.skipif(
    not torch.cuda.is_available() or _find_mooncake_master() is None,
    reason="Requires CoreX/CUDA GPU and installed mooncake_master binary",
)
def test_real_mooncake_master_multiprocess_gpu_2p_standard_and_dsv41(tmp_path) -> None:
    rdma_pick = _pick_rdma_device_and_ip()
    proto, rdma_dev, host = (
        ("rdma", rdma_pick[0], rdma_pick[1]) if rdma_pick else ("tcp", "", "127.0.0.1")
    )
    dev_p1 = 0
    dev_p2 = 1 if torch.cuda.device_count() >= 2 else 0
    ctx = mp.get_context("spawn")

    with _run_mooncake_master() as (rpc_port, http_port):
        extra = {
            "local_hostname": host,
            "master_server_address": f"{host}:{rpc_port}",
            "metadata_server": f"http://{host}:{http_port}/metadata",
            "protocol": proto,
            "device_name": rdma_dev,
            "cpu_bytes_to_use": "64MiB",
        }
        for target in (_std_worker_proc, _dsv41_worker_proc):
            q = ctx.Queue()
            done = ctx.Event()
            p1 = ctx.Process(target=target, args=("p1", dev_p1, extra, str(tmp_path), q, done))
            p2 = ctx.Process(target=target, args=("p2", dev_p2, extra, str(tmp_path), q, done))
            p1.start()
            p2.start()
            p2.join(timeout=45.0)
            p1.join(timeout=45.0)
            assert p2.exitcode == 0 and p1.exitcode == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or _find_mooncake_master() is None,
    reason="Requires CoreX/CUDA GPU and installed mooncake_master binary",
)
@pytest.mark.parametrize("requested_proto", ["rdma", "tcp"])
def test_real_mooncake_master_gpu_2p_dsv41_standalone_store_proto(
    tmp_path, monkeypatch, requested_proto: str
) -> None:
    if requested_proto == "rdma":
        rdma_pick = _pick_rdma_device_and_ip()
        if rdma_pick is None:
            pytest.skip("No active IPv4 RDMA HCA detected on host")
        rdma_dev, host = rdma_pick
        proto = "rdma"
    else:
        proto, rdma_dev, host = "tcp", "", "127.0.0.1"

    monkeypatch.setattr("vllm.envs.VLLM_RPC_BASE_PATH", str(tmp_path), raising=False)
    dev_p1 = 0
    dev_p2 = 1 if torch.cuda.device_count() >= 2 else 0

    with _run_mooncake_master() as (rpc_port, http_port):
        base_extra = {
            "local_hostname": host,
            "metadata_server": f"http://{host}:{http_port}/metadata",
            "master_server_address": f"{host}:{rpc_port}",
            "protocol": proto,
            "device_name": rdma_dev,
        }
        p1_sched, p1_worker, p1_kv, p1_ctx, p1_st = _build_dsv41_gpu_node(
            "live-p1", dev_p1, {**base_extra, "mode": "embedded", "global_segment_size": "256MB"}, fill=True
        )
        # P2 mounts 0B local segment to force cross-client Direct GPU RDMA/TCP pull from P1
        p2_sched, p2_worker, p2_kv, p2_ctx, p2_st = _build_dsv41_gpu_node(
            "live-p2", dev_p2, {**base_extra, "mode": "standalone-store", "global_segment_size": "0"}, fill=False
        )
        try:
            tokens = list(range(1088))
            hashes = _make_hashes(tokens, 64)
            req1 = _mock_req("gpu-req-1", tokens, hashes)
            assert p1_sched.get_num_new_matched_tokens(req1, 0) == (0, False)
            p1_sched.update_state_after_alloc(req1, SimpleNamespace(get_block_ids=lambda group_ids=None: ([1, 2, 3], [1, 2])), 0)
            meta1 = p1_sched.build_connector_meta(
                _sched_out([_new_req_data("gpu-req-1", tokens, ([1, 2, 3], [1, 2, 3], [1, 2]))], {"gpu-req-1": len(tokens)})
            )
            torch.cuda.set_device(dev_p1)
            p1_worker.bind_connector_metadata(meta1)
            p1_worker.wait_for_save()
            assert p1_worker.connector_worker and p1_worker.connector_worker.kv_send_thread
            p1_worker.connector_worker.kv_send_thread.request_queue.join()
            p1_sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=p1_worker.build_connector_worker_meta()))
            p1_worker.clear_connector_metadata()

            req2 = _mock_req("gpu-req-2", tokens, hashes)
            matched, is_async = p2_sched.get_num_new_matched_tokens(req2, 0)
            assert matched == 1024 and is_async is True
            p2_sched.update_state_after_alloc(req2, SimpleNamespace(get_block_ids=lambda group_ids=None: ([5, 6, 7], [5, 7])), matched)
            torch.cuda.set_device(dev_p2)
            p2_worker.bind_connector_metadata(p2_sched.build_connector_meta(_sched_out()))
            p2_worker.start_load_kv(SimpleNamespace())  # type: ignore[arg-type]
            assert p2_worker.connector_worker
            p2_worker.connector_worker.recv_request_queue.join()
            xfer = p2_worker.get_transfer_results(set())
            assert "gpu-req-2" in xfer.finished_recving and not xfer.failed_recving
            p2_worker.clear_connector_metadata()
            torch.cuda.synchronize(dev_p2)

            for lname in ("model.layers.2.attn", "model.layers.8.attn"):
                for src_blk, dst_blk in ((1, 5), (2, 6)):
                    assert torch.equal(p2_kv[lname][dst_blk].cpu(), p1_kv[lname][src_blk].cpu())
                    assert torch.equal(p2_ctx[lname].kv_cache_scales[dst_blk].cpu(), p1_ctx[lname].kv_cache_scales[src_blk].cpu())
            for iname in ("model.layers.2.attn.indexer.k_cache", "model.layers.8.attn.indexer.k_cache"):
                assert torch.equal(p2_kv[iname][5].cpu(), p1_kv[iname][1].cpu())
                assert torch.equal(p2_ctx[iname].kv_cache_scales[5].cpu(), p1_ctx[iname].kv_cache_scales[1].cpu())
        finally:
            p2_sched.shutdown()
            p1_sched.shutdown()
            p2_worker.shutdown()
            p1_worker.shutdown()
            for st in (*p2_st, *p1_st):
                st.release()
