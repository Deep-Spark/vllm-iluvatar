"""Bounded real V2 profiler/capture/cleanup; no model or service startup."""

import gc
import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from ixformer.inference.functions.deepseek.quantized_kv_cache import (
    dequantize_paged_kv,
    quantize_paged_kv,
)
from ixformer.inference.functions.deepseek.quantized_kv_format import I4G32, I8G64
from test_compressed_kv_profiling import ProfilingRunner, assert_clean
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, override_forward_context
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.worker.gpu.cudagraph_utils import _teardown_profiling_state
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    WORKSPACE_BYTES,
    CompressedKVState,
    layer_specs,
    plan_scale_layout,
)
from vllm_iluvatar.v1.worker.gpu.compressed_kv_profiling import compressed_kv_profiling

pytestmark = pytest.mark.skipif(
    os.environ.get("KV_COMPRESSED_KV_GPU_TESTS") != "1",
    reason="requires an explicitly leased GPU and KV_COMPRESSED_KV_GPU_TESTS=1",
)


class PairModel(torch.nn.Module):
    """Replace model mathematics with real caller-owned encode/read operations."""

    def __init__(self, runner, fmt):
        super().__init__()
        self.runner, self.fmt = runner, fmt
        self.rows = torch.full((2, 512), 0.5, dtype=torch.bfloat16, device="cuda")
        self.slots = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        self.output = torch.empty_like(self.rows)
        self.witness = torch.empty((), dtype=torch.uint8, device="cuda")

    def forward(self):
        main = self.runner.compilation_config.static_forward_context[
            "model.layers.2.attn"
        ]
        pool = main.compressed_kv_workspace
        with pool.forward_scope() as workspace:
            quantize_paged_kv(
                self.rows,
                self.slots,
                main.kv_cache,
                main.kv_cache_scales,
                format=self.fmt,
            )
            pool.main_cache_written()
            dequantize_paged_kv(
                main.kv_cache,
                main.kv_cache_scales,
                self.slots,
                self.output,
                format=self.fmt,
            )
            workspace[:1].copy_(self.rows[:1, :1].flatten().view(torch.uint8)[:1])
            self.witness.copy_(workspace[0])
        return self.output


class PairCaptureManager:
    """Isolate backend metadata; actual CUDAGraphWrapper owns real graphs."""

    def __init__(self, runner):
        self.runner = runner
        self.pool = None
        self.use_breakable_cg = False
        self._capture_descs = {CUDAGraphMode.FULL: [BatchDescriptor(2)]}

    def needs_capture(self):
        return True

    def capture(self, *args, **kwargs):
        runner = self.runner
        runner.inspect_cache()
        runner.capture_ledger.append(memory())
        for _ in range(3):
            runner.model()
        if runner.failure == "capture":
            raise RuntimeError("injected capture failure")
        context = SimpleNamespace(
            batch_descriptor=BatchDescriptor(2),
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
        )
        runner.graph_wrapper.graph_pool = self.pool
        before = torch.cuda.mem_get_info()[0]
        with override_forward_context(context):
            runner.graph_wrapper()
            for value in (0.25, 1.0, 0.5):
                runner.model.rows.fill_(value)
                main = runner.compilation_config.static_forward_context[
                    "model.layers.2.attn"
                ]
                with main.compressed_kv_workspace.acquire():
                    runner.graph_wrapper()
                torch.cuda.synchronize()
                assert torch.equal(runner.model.output, runner.model.rows)
        self._capture_mem_samples.append(max(before - torch.cuda.mem_get_info()[0], 0))
        assert runner.graph_wrapper.concrete_cudagraph_entries


class GraphProfilingRunner(ProfilingRunner):
    """Use original V2 graph-memory profiler and capture_model implementations."""

    profile_cudagraph_memory = GPUModelRunner.profile_cudagraph_memory
    capture_model = GPUModelRunner.capture_model

    def __init__(self, fmt):
        super().__init__(fmt, "cuda")
        self.speculator = None
        self.pcp_manager = None
        self.input_buffers = None
        self.intermediate_tensors = None
        self.block_tables = None
        self.use_aux_hidden_state_outputs = False
        self.adaptive_verification = None
        self.kv_connector = SimpleNamespace(reset_capture_state=lambda: None)
        self.capture_ledger = []
        self.model = PairModel(self, fmt)
        self.graph_wrapper = CUDAGraphWrapper(
            self.model, self.vllm_config, CUDAGraphMode.FULL
        )

    def maybe_setup_dummy_loras(self, _):
        return nullcontext()

    def initialize_kv_cache(self, *args, **kwargs):
        super().initialize_kv_cache(*args, **kwargs)
        specs = layer_specs(self.kv_cache_config.kv_cache_groups)
        self.attn_groups = [
            [
                SimpleNamespace(
                    kv_cache_group_id=index,
                    kv_cache_spec=specs[name],
                    layer_names=[name],
                )
                for name in group.layer_names
            ]
            for index, group in enumerate(self.kv_cache_config.kv_cache_groups)
        ]
        self.cudagraph_manager = PairCaptureManager(self)


def memory():
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "free": free,
        "total": total,
    }


def assert_released(runner, baseline):
    assert_clean(runner)
    assert not runner.graph_wrapper.concrete_cudagraph_entries
    gc.collect()
    torch.cuda.synchronize()
    now = memory()
    assert now["allocated"] == baseline["allocated"], (baseline, now)
    assert now["reserved"] <= baseline["reserved"], (baseline, now)
    assert now["free"] >= 1024**3
    for receipt in runner.receipts:
        assert all(receipt[key]() is None for key in ("payload", "scale", "workspace"))
    return now


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_actual_v2_profile_capture_cleanup_and_formal_rebind(fmt, record_property):
    # Includes the one D128 arena, tiny paired pages, input/output and a
    # generous graph/allocator allowance. The outer driver checks headroom
    # before importing the tested GPU fixture as well.
    assert torch.cuda.mem_get_info()[0] >= 320 * 1024**2 + 1024**3
    gc.collect()
    runner = GraphProfilingRunner(fmt)
    torch.cuda.empty_cache()
    baseline = memory()
    torch.cuda.reset_peak_memory_stats()
    original_pool = type(current_platform)._global_graph_pool
    original_methods = (
        runner.initialize_kv_cache.__func__,
        runner.profile_cudagraph_memory.__func__,
    )
    original_zeroer = runner.kv_block_zeroer = object()
    profile_ledgers = []
    for _ in range(2):
        with compressed_kv_profiling(
            runner,
            fmt,
            workspace_bytes=WORKSPACE_BYTES - 4202496,
            prefill_producer=False,
        ):
            estimate = runner.profile_cudagraph_memory()
        assert 0 <= estimate < WORKSPACE_BYTES // 2
        assert runner.kv_block_zeroer is original_zeroer
        assert type(current_platform)._global_graph_pool is original_pool
        assert runner.initialize_kv_cache.__func__ is original_methods[0]
        assert runner.profile_cudagraph_memory.__func__ is original_methods[1]
        profile_ledgers.append(assert_released(runner, baseline))

    # A later formal cache uses a new state and different actual V2 geometry.
    runner.cache_config.num_gpu_blocks_override = 7
    cache = get_kv_cache_config_from_groups(
        runner.vllm_config, get_kv_cache_groups(runner.vllm_config, runner.specs), 0
    )
    runner.cache_config.num_gpu_blocks_override = None
    state = CompressedKVState(
        plan_scale_layout(cache, fmt),
        fmt,
        "cuda",
        runner.compilation_config.static_forward_context,
        workspace_bytes=WORKSPACE_BYTES - 4202496,
    )
    try:
        runner.initialize_kv_cache(cache, is_profiling=True)
        runner._init_kv_zero_meta()
        state.bind_zeroer(runner.kv_block_zeroer)
        main = runner.compilation_config.static_forward_context["model.layers.2.attn"]
        assert main.kv_cache.shape[0] == main.kv_cache_scales.shape[0] == 7
        with state.workspace.acquire():
            runner.model()
        torch.cuda.synchronize()
        assert torch.equal(runner.model.output, runner.model.rows)
    finally:
        _teardown_profiling_state(runner)
        runner.kv_block_zeroer = original_zeroer
        state.release()
        gc.collect()
        torch.cuda.empty_cache()
    final = assert_released(runner, baseline)
    record_property(
        "profile_allocated_reserved_free",
        {
            "baseline": baseline,
            "capture": runner.capture_ledger,
            "after_each": profile_ledgers,
            "final": final,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
        },
    )
    assert torch.cuda.max_memory_allocated() - baseline["allocated"] <= 320 * 1024**2
