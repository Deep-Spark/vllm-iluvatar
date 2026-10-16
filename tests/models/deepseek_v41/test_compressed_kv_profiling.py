"""Real V2 profiling grouping/allocation/binding/teardown without model weights.

The Meta fixture isolates accelerator cleanup and zeroer addresses only. GPU
tests reuse the same fixture with real CUDA storage, metadata and graph capture.
"""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from ixformer.inference.functions.deepseek.quantized_kv_format import I4G32, I8G64
from test_compressed_kv_v2_planning import config, specs_for_rank
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.attn_utils import init_kv_cache
from vllm.v1.worker.gpu.cudagraph_utils import (
    _init_minimal_kv_cache_for_profiling,
    _teardown_profiling_state,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.utils import KVBlockZeroer

from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    PREFILL_WORKSPACE_BYTES,
    WORKSPACE_BYTES,
    CompressedKVState,
    plan_scale_layout,
)
from vllm_iluvatar.v1.worker.gpu.compressed_kv_profiling import compressed_kv_profiling


class ProfilingRunner:
    """Isolate model/backend construction; keep V2 cache and teardown methods."""

    def __init__(self, fmt, device="meta"):
        cfg = config()
        self.device = torch.device(device)
        self.specs, context = specs_for_rank(cfg, fmt, 2, 3)
        for layer in context.values():
            if getattr(layer, "compressed_kv_format", None) is not None:
                layer.kv_cache_scales = None
                layer.compressed_kv_workspace = None
        cfg.model_config.hf_config = SimpleNamespace(model_type="deepseek_v41")
        cfg.compilation_config = SimpleNamespace(
            max_cudagraph_capture_size=2,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            static_forward_context=context,
        )
        self.vllm_config = cfg
        self.cache_config = cfg.cache_config
        self.compilation_config = cfg.compilation_config
        self.max_num_reqs = 2
        self.model_state = SimpleNamespace(supports_mm_inputs=False)
        self.model = torch.nn.Module()
        self.lora_config = None
        self.kv_caches = []
        self.attn_groups = []
        self.kv_block_zeroer = None
        self.cudagraph_manager = None
        self.receipts = []
        self.failure = None

    def get_kv_cache_spec(self):
        return self.specs

    def maybe_remove_all_loras(self, _):
        pass

    def initialize_kv_cache(
        self, cache, is_profiling=False, kv_cache_allocation_context=None
    ):
        assert is_profiling
        self.kv_cache_config = cache
        self.kernel_block_sizes = [
            group.kv_cache_spec.block_size for group in cache.kv_cache_groups
        ]
        self.attn_groups = [
            [
                SimpleNamespace(
                    kv_cache_group_id=index,
                    kv_cache_spec=group.kv_cache_spec,
                    layer_names=group.layer_names,
                )
            ]
            for index, group in enumerate(cache.kv_cache_groups)
        ]
        self.kv_caches = list(
            init_kv_cache(
                self.compilation_config.static_forward_context,
                cache,
                self.device,
                None,
                self.vllm_config,
                kv_cache_allocation_context,
            ).values()
        )
        if self.failure == "initialize":
            raise RuntimeError("injected initialization failure after real binding")

    def _init_kv_zero_meta(self):
        if self.device.type == "meta":
            # Meta addresses are all zero and cannot form a real zeroer table.
            # Keep the actual State metadata-reservation path; address/zeroing
            # qualification belongs to the CUDA fixture.
            self.kv_block_zeroer = KVBlockZeroer(
                self.device, (), [], {}, self.kv_cache_config.num_blocks
            )
        else:
            GPUModelRunner._init_kv_zero_meta(self)

    def profile_cudagraph_memory(self):
        _init_minimal_kv_cache_for_profiling(self)
        self.inspect_cache()
        if self.failure == "capture":
            raise RuntimeError("injected capture failure")
        _teardown_profiling_state(self)
        return 12345

    def inspect_cache(self):
        main = self.compilation_config.static_forward_context["model.layers.2.attn"]
        pool = getattr(main, "compressed_kv_workspace", None)
        scale = getattr(main, "kv_cache_scales", None)
        self.receipts.append(
            {
                "blocks": self.kv_cache_config.num_blocks,
                "payload": weakref.ref(main.kv_cache),
                "scale": None if scale is None else weakref.ref(scale),
                "workspace": None if pool is None else weakref.ref(pool.storage),
                "arena_bytes": None if pool is None else pool.storage.numel(),
                "yield_bytes": None if pool is None else pool.tensor.numel(),
                "producer": None if pool is None else pool.prefill_producer,
            }
        )
        if pool is not None:
            assert pool._metadata_reserved
            assert any(cache is scale for cache in self.kv_caches)
            with pool.forward_scope():
                assert pool._forward_depth == 1


@pytest.fixture
def meta_accelerator_cleanup(monkeypatch):
    # Only isolate device operations unavailable on Meta. Keep the actual
    # upstream teardown and every cache/layer/method operation under test.
    calls = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: calls.append("empty"))
    return calls


def assert_clean(runner):
    assert not runner.kv_caches and not runner.attn_groups
    assert not hasattr(runner, "kv_cache_config")
    assert runner.cudagraph_manager is None
    assert runner.cache_config.num_gpu_blocks is None
    assert runner.cache_config.num_gpu_blocks_override is None
    for layer in runner.compilation_config.static_forward_context.values():
        assert layer.kv_cache.numel() == 0
        assert getattr(layer, "kv_cache_scales", None) is None
        assert getattr(layer, "compressed_kv_workspace", None) is None


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("producer", [False, True])
def test_actual_minimal_profile_binds_paired_state_then_restores_runner(
    fmt, producer, meta_accelerator_cleanup
):
    runner = ProfilingRunner(fmt)
    original_methods = (
        runner.initialize_kv_cache.__func__,
        runner.profile_cudagraph_memory.__func__,
    )
    previous_zeroer = runner.kv_block_zeroer = object()
    arena = (PREFILL_WORKSPACE_BYTES if producer else WORKSPACE_BYTES) - 4202496
    with compressed_kv_profiling(
        runner, fmt, workspace_bytes=arena, prefill_producer=producer
    ):
        assert runner.profile_cudagraph_memory() == 12345
        assert_clean(runner)
        assert runner.kv_block_zeroer is previous_zeroer
        assert runner.profile_cudagraph_memory() == 12345
    assert runner.kv_block_zeroer is previous_zeroer
    assert runner.initialize_kv_cache.__func__ is original_methods[0]
    assert runner.profile_cudagraph_memory.__func__ is original_methods[1]
    assert "initialize_kv_cache" not in vars(runner)
    assert "profile_cudagraph_memory" not in vars(runner)
    assert runner.vllm_config.model_config.max_model_len == 1048576
    for receipt in runner.receipts:
        assert receipt["blocks"] == 2
        assert receipt["arena_bytes"] == arena and receipt["yield_bytes"] < arena
        assert receipt["producer"] is producer
        assert all(receipt[key]() is None for key in ("payload", "scale", "workspace"))


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("failure", ["initialize", "capture"])
def test_actual_profiling_exception_clears_partial_payload_and_sidecar(
    fmt, failure, meta_accelerator_cleanup
):
    runner = ProfilingRunner(fmt)
    runner.failure = failure
    with compressed_kv_profiling(
        runner, fmt, workspace_bytes=WORKSPACE_BYTES - 4202496, prefill_producer=False
    ):
        with pytest.raises(RuntimeError, match="injected"):
            runner.profile_cudagraph_memory()
        assert_clean(runner)
        assert runner.kv_block_zeroer is None
        runner.failure = None
        assert runner.profile_cudagraph_memory() == 12345
    assert_clean(runner)


def test_instance_overrides_absent_zeroer_and_nested_profile_restore_exactly(
    meta_accelerator_cleanup,
):
    runner = ProfilingRunner(I8G64)
    initialize = runner.initialize_kv_cache
    profile = runner.profile_cudagraph_memory
    runner.initialize_kv_cache, runner.profile_cudagraph_memory = initialize, profile
    del runner.kv_block_zeroer
    with compressed_kv_profiling(
        runner, I8G64, workspace_bytes=1024**2, prefill_producer=False
    ):
        with (
            pytest.raises(RuntimeError, match="already active"),
            compressed_kv_profiling(
                runner, I8G64, workspace_bytes=1024**2, prefill_producer=False
            ),
        ):
            pytest.fail("nested profile context entered")
        runner.profile_cudagraph_memory()
    assert runner.initialize_kv_cache is initialize
    assert runner.profile_cudagraph_memory is profile
    assert "kv_block_zeroer" not in vars(runner)


def test_unwrapped_quantized_profile_exposes_original_scale_binding_gap(
    meta_accelerator_cleanup,
):
    runner = ProfilingRunner(I8G64)
    with pytest.raises(ValueError, match="scale view is not paired"):
        runner.profile_cudagraph_memory()
    _teardown_profiling_state(runner)


def test_none_original_minimal_profile_needs_no_sidecar(meta_accelerator_cleanup):
    runner = ProfilingRunner(None)
    assert runner.profile_cudagraph_memory() == 12345
    assert_clean(runner)
    assert runner.receipts[0]["scale"] is None


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_formal_geometry_after_profiling_has_only_new_bindings(
    fmt, meta_accelerator_cleanup
):
    runner = ProfilingRunner(fmt)
    with compressed_kv_profiling(
        runner, fmt, workspace_bytes=WORKSPACE_BYTES - 4202496, prefill_producer=False
    ):
        runner.profile_cudagraph_memory()
    runner.cache_config.num_gpu_blocks_override = 7
    # Reuse the real helper only to generate a different valid V2 config;
    # the real formal State/allocator/binder are then invoked without wrapper.
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )

    cache = get_kv_cache_config_from_groups(
        runner.vllm_config, get_kv_cache_groups(runner.vllm_config, runner.specs), 0
    )
    runner.cache_config.num_gpu_blocks_override = None
    state = CompressedKVState(
        plan_scale_layout(cache, fmt),
        fmt,
        "meta",
        runner.compilation_config.static_forward_context,
        workspace_bytes=WORKSPACE_BYTES - 4202496,
    )
    try:
        runner.initialize_kv_cache(cache, is_profiling=True)
        main = runner.compilation_config.static_forward_context["model.layers.2.attn"]
        assert main.kv_cache.shape[0] == 7
        assert main.kv_cache_scales.shape[0] == 7
        assert main.compressed_kv_workspace is state.workspace
        gc.collect()
        assert all(
            runner.receipts[0][key]() is None
            for key in ("payload", "scale", "workspace")
        )
    finally:
        _teardown_profiling_state(runner)
        state.release()


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("producer", [False, True])
def test_worker_profiles_before_budget_and_deducts_fixed_storage_once(
    fmt, producer, meta_accelerator_cleanup, monkeypatch
):
    from vllm.v1.worker.gpu_worker import Worker

    from vllm_iluvatar.models.deepseek_v41 import compressed_kv
    from vllm_iluvatar.v1.worker.gpu_worker import IluWorker

    runner = ProfilingRunner(fmt)
    cfg = runner.vllm_config
    cfg.use_v2_model_runner = True
    cfg.scheduler_config.max_num_batched_tokens = 2048
    cfg.model_config.hf_text_config = SimpleNamespace(
        model_type="deepseek_v41", index_topk=512
    )
    cfg.kv_transfer_config = SimpleNamespace(
        kv_role="kv_producer" if producer else "kv_consumer"
    )
    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", fmt.name)
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", "", raising=False)
    monkeypatch.setattr(envs, "VLLM_ILU_MLA_ATTN_QUANT_TYPE", "", raising=False)
    worker = IluWorker.__new__(IluWorker)
    worker.vllm_config, worker.cache_config = cfg, cfg.cache_config
    worker.model_runner, worker.device = runner, runner.device
    events = []
    state_cls = compressed_kv.CompressedKVState

    def state(*args, **kwargs):
        assert events[-1] == "ordinary profile snapshot complete"
        events.append("temporary fixed allocation before capture")
        return state_cls(*args, **kwargs)

    monkeypatch.setattr(compressed_kv, "CompressedKVState", state)
    total = 2 * 1024**3

    def base_profile(self):
        assert not hasattr(self, "_compressed_kv_budget")
        main = cfg.compilation_config.static_forward_context["model.layers.2.attn"]
        assert main.kv_cache_scales is None
        assert main.compressed_kv_workspace is None
        events.append("ordinary profile snapshot complete")
        self.cudagraph_memory_estimate = runner.profile_cudagraph_memory()
        assert_clean(runner)
        assert main.compressed_kv_workspace is None
        events.append("original available ledger returned")
        return total

    monkeypatch.setattr(Worker, "determine_available_memory", base_profile)
    payload = worker.determine_available_memory()
    budget = worker._compressed_kv_budget
    expected_workspace = PREFILL_WORKSPACE_BYTES if producer else WORKSPACE_BYTES
    page_cost = 65536 + fmt.scale_page_bytes(512)
    assert payload == (total - expected_workspace) // page_cost * 65536
    assert budget.total_bytes == total
    assert budget.workspace_bytes == expected_workspace
    assert budget.retained_mapper_bytes == 4202496
    assert budget.arena_bytes == expected_workspace - 4202496
    assert runner.receipts[0]["arena_bytes"] == budget.arena_bytes
    assert worker.cudagraph_memory_estimate == 12345
    assert events == [
        "ordinary profile snapshot complete",
        "temporary fixed allocation before capture",
        "original available ledger returned",
    ]
    assert not hasattr(worker, "_compressed_kv_state")


def test_no_graph_profile_call_allocates_no_temporary_state(monkeypatch):
    from vllm_iluvatar.models.deepseek_v41 import compressed_kv

    runner = ProfilingRunner(I8G64)
    original = runner.profile_cudagraph_memory.__func__
    monkeypatch.setattr(
        compressed_kv,
        "CompressedKVState",
        lambda *args, **kwargs: pytest.fail("profiler was never called"),
    )
    with compressed_kv_profiling(
        runner, I8G64, workspace_bytes=WORKSPACE_BYTES, prefill_producer=False
    ):
        # Manual KV memory and eager paths return without graph profiling.
        pass
    assert runner.profile_cudagraph_memory.__func__ is original
    assert runner.kv_block_zeroer is None


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_upstream_teardown_then_exception_preserves_original_error(
    fmt, meta_accelerator_cleanup
):
    runner = ProfilingRunner(fmt)
    error = RuntimeError("original capture failure after teardown")

    def original_profile():
        try:
            _init_minimal_kv_cache_for_profiling(runner)
            runner.inspect_cache()
            raise error
        finally:
            _teardown_profiling_state(runner)

    runner.profile_cudagraph_memory = original_profile
    with compressed_kv_profiling(
        runner, fmt, workspace_bytes=WORKSPACE_BYTES, prefill_producer=False
    ):
        with pytest.raises(RuntimeError) as raised:
            runner.profile_cudagraph_memory()
        assert raised.value is error
    assert meta_accelerator_cleanup.count("sync") == 2
    assert_clean(runner)
    assert runner.profile_cudagraph_memory is original_profile
    assert runner.kv_block_zeroer is None


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_zeroer_binding_failure_releases_arena_and_layer_views(
    fmt, meta_accelerator_cleanup, monkeypatch
):
    runner = ProfilingRunner(fmt)
    original = CompressedKVState.bind_zeroer
    references = []

    def bind_then_fail(state, zeroer):
        original(state, zeroer)
        references.extend(
            (weakref.ref(state.workspace.storage), weakref.ref(state.scale_storage))
        )
        raise RuntimeError("injected zeroer binding failure")

    monkeypatch.setattr(CompressedKVState, "bind_zeroer", bind_then_fail)
    with (
        compressed_kv_profiling(
            runner, fmt, workspace_bytes=WORKSPACE_BYTES, prefill_producer=False
        ),
        pytest.raises(RuntimeError, match="zeroer binding failure"),
    ):
        runner.profile_cudagraph_memory()
    gc.collect()
    assert all(reference() is None for reference in references)
    assert_clean(runner)
    assert runner.kv_block_zeroer is None


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_state_constructor_partial_binding_failure_is_atomic(
    fmt, meta_accelerator_cleanup, monkeypatch
):
    runner = ProfilingRunner(fmt)
    runner.specs, context = specs_for_rank(runner.vllm_config, fmt, 2, 9)
    runner.compilation_config.static_forward_context = context
    original = torch.as_strided
    calls = []

    def view_then_fail(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            assert context["model.layers.2.attn"].kv_cache_scales is not None
            raise RuntimeError(
                "injected state construction failure after first binding"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "as_strided", view_then_fail)
    with (
        compressed_kv_profiling(
            runner, fmt, workspace_bytes=WORKSPACE_BYTES, prefill_producer=False
        ),
        pytest.raises(RuntimeError, match="state construction failure"),
    ):
        runner.profile_cudagraph_memory()
    assert calls == [1, 1]
    assert runner.kv_block_zeroer is None
    assert not runner.kv_caches
    assert not hasattr(runner, "kv_cache_config")
    assert runner.cache_config.num_gpu_blocks_override is None
    for layer in context.values():
        assert getattr(layer, "kv_cache_scales", None) is None
        assert getattr(layer, "compressed_kv_workspace", None) is None


@pytest.mark.parametrize("secondary", ["teardown", "release"])
def test_cleanup_errors_are_reported_without_masking_original_failure(
    secondary, meta_accelerator_cleanup, monkeypatch, caplog
):
    import vllm.v1.worker.gpu.cudagraph_utils as utils

    runner = ProfilingRunner(I8G64)
    runner.failure = "capture"
    if secondary == "teardown":

        def cleanup_then_fail(value):
            _teardown_profiling_state(value)
            raise RuntimeError("secondary teardown failure")

        monkeypatch.setattr(utils, "_teardown_profiling_state", cleanup_then_fail)
    else:
        release = CompressedKVState.release

        def release_then_fail(state):
            release(state)
            raise RuntimeError("secondary release failure")

        monkeypatch.setattr(CompressedKVState, "release", release_then_fail)
    with (
        compressed_kv_profiling(
            runner, I8G64, workspace_bytes=WORKSPACE_BYTES, prefill_producer=False
        ),
        pytest.raises(RuntimeError, match="injected capture failure") as raised,
    ):
        runner.profile_cudagraph_memory()
    assert any("secondary" in note for note in raised.value.__notes__)
    assert "secondary" in caplog.text
    assert_clean(runner)
    assert runner.kv_block_zeroer is None


def test_recursive_original_profile_is_rejected_without_runner_mutation():
    runner = ProfilingRunner(I8G64)
    runner.profile_cudagraph_memory = lambda: runner.profile_cudagraph_memory()
    original = runner.profile_cudagraph_memory
    with (
        compressed_kv_profiling(
            runner, I8G64, workspace_bytes=WORKSPACE_BYTES, prefill_producer=False
        ),
        pytest.raises(RuntimeError, match="already active"),
    ):
        runner.profile_cudagraph_memory()
    assert runner.profile_cudagraph_memory is original
    assert "initialize_kv_cache" not in vars(runner)
    assert runner.kv_block_zeroer is None
