# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the IxaiteWorker subclass."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_GPU_WORKER_SRC = (
    Path(__file__).resolve().parents[3]
    / "vllm_iluvatar"
    / "v1"
    / "worker"
    / "gpu_worker.py"
)
_FORBIDDEN_WARMUP_CALLS = frozenset(
    {
        "_dummy_run",
        "_warm_up_ixaite_before_kv",
        "_warm_up_ixaite_gelu",
        "_warm_up_ixaite_video_evs",
        "_warm_up_ixaite_vit_qkv_rope",
        "_warm_up_ixaite_prefill_buckets",
        "_warm_up_ixaite_body_small_m",
        "_warm_up_ixaite_linear_exact_buckets",
        "_warm_up_ixaite_lm_head",
        "_run_ixaite_linear_family_warmup",
        "_clear_ixaite_linear_staging",
        "_warm_up_ixaite_decode_network",
        "_warm_up_ixaite_gdn_chunk_buckets",
        "_run_ixaite_gdn_chunk_warmup",
        "_warm_up_ixaite_gdn_packed_decode",
        "_run_ixaite_gdn_packed_decode_warmup",
        "_warm_up_ixaite_gdn_conv",
        "_run_ixaite_gdn_conv_fn_warmup",
        "_run_ixaite_gdn_conv_update_warmup",
        "_warm_up_ixaite_gdn_post_conv_prep",
        "_run_ixaite_gdn_post_conv_prep_warmup",
        "_release_ixaite_warmup_scratch",
    }
)


@pytest.fixture(autouse=True)
def _stub_gpu_worker(monkeypatch):
    """Stub vllm.v1.worker.gpu_worker.Worker so IxaiteWorker can be imported."""
    # Keep the actual compile-mode enum available behind the fake root package.
    from vllm.config import compilation  # noqa: F401

    def make_package(name):
        package = ModuleType(name)
        package.__path__ = []
        return package

    vllm_mod = make_package("vllm")
    vllm_v1_mod = make_package("vllm.v1")
    worker_package = make_package("vllm.v1.worker")

    logger_mod = ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    )
    vllm_mod.logger = logger_mod

    kv_mod = ModuleType("vllm.v1.kv_cache_interface")
    kv_mod.KVCacheConfig = type("KVCacheConfig", (), {})
    vllm_v1_mod.kv_cache_interface = kv_mod

    gpu_worker_mod = ModuleType("vllm.v1.worker.gpu_worker")

    class _FakeWorker:
        def initialize_from_config(self, kv_cache_config):
            if hasattr(self, "events"):
                self.events.append(("community", kv_cache_config))

        def determine_available_memory(self):
            return "community-profile"

        def compile_or_warm_up_model(self):
            return "community-warmup"

        def execute_dummy_batch(self) -> None:
            raise NotImplementedError

        def shutdown(self) -> None:
            self.model_runner.shutdown()
            if getattr(self, "fail_parent_shutdown", False):
                raise RuntimeError("parent teardown failed")
            self.community_shutdown_complete = True

    gpu_worker_mod.Worker = _FakeWorker

    gpu_model_runner_mod = ModuleType("vllm.v1.worker.gpu_model_runner")
    gpu_model_runner_mod.GPUModelRunner = type("GPUModelRunner", (), {})

    warmup_mod = make_package("vllm.model_executor.warmup")
    kernel_warmup = ModuleType("vllm.model_executor.warmup.kernel_warmup")
    block_warmup = ModuleType("vllm.model_executor.warmup.v1_block_table_warmup")
    kernel_warmup.warm_v1_block_table_kernels = lambda runner: "kernel warmup"
    block_warmup.warm_v1_block_table_kernels = lambda runner: "block warmup"
    warmup_mod.kernel_warmup = kernel_warmup
    warmup_mod.v1_block_table_warmup = block_warmup

    vllm_mod.v1 = vllm_v1_mod
    vllm_v1_mod.worker = worker_package
    worker_package.gpu_worker = gpu_worker_mod
    worker_package.gpu_model_runner = gpu_model_runner_mod

    modules = {
        "vllm": vllm_mod,
        "vllm.logger": logger_mod,
        "vllm.v1": vllm_v1_mod,
        "vllm.v1.kv_cache_interface": kv_mod,
        "vllm.v1.worker": worker_package,
        "vllm.v1.worker.gpu_model_runner": gpu_model_runner_mod,
        "vllm.v1.worker.gpu_worker": gpu_worker_mod,
        warmup_mod.__name__: warmup_mod,
        kernel_warmup.__name__: kernel_warmup,
        block_warmup.__name__: block_warmup,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    # Invalidate any cached import of the plugin module so the stub is used.
    monkeypatch.delitem(
        sys.modules, "vllm_iluvatar.backends.ixaite.worker", raising=False
    )

    monkeypatch.delitem(
        sys.modules, "vllm_iluvatar.v1.worker.gpu_worker", raising=False
    )

    yield gpu_worker_mod

    # Remove fake-base classes before another test imports the real worker.
    sys.modules.pop("vllm_iluvatar.backends.ixaite.worker", None)
    sys.modules.pop("vllm_iluvatar.v1.worker.gpu_worker", None)


def _make_worker(model_runner):
    """Import IxaiteWorker (against the stub) and construct an instance."""
    from vllm_iluvatar.backends.ixaite.worker import IxaiteWorker

    w = IxaiteWorker.__new__(IxaiteWorker)
    w.model_runner = model_runner
    # Both upstream PCP handling and fork warmup inspect normal worker config.
    w.parallel_config = SimpleNamespace(prefill_context_parallel_size=1)
    w.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type=""))
    # Eager-mode stub: _ixaite_warmup_compile_cap reads vllm_config and
    # returns None for any mode other than VLLM_COMPILE.
    w.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=None, compile_ranges_endpoints=[]),
        # Wider than any stub max_model_len, so the token cap stays on
        # max_num_tokens. _ixaite_reachable_token_cap reads this.
        scheduler_config=SimpleNamespace(max_num_seqs=1024),
    )
    return w


# The baseline worker must profile and bind caches without ixaite lifecycle hooks.
def test_baseline_worker_profiles_without_adjusting_mamba_capacity(monkeypatch):
    from vllm_iluvatar.v1.worker.gpu_worker import IluWorker

    class Runner:
        max_num_reqs = 256

    worker = IluWorker.__new__(IluWorker)
    worker.model_runner = Runner()
    worker.events = []
    worker.parallel_config = SimpleNamespace(prefill_context_parallel_size=1)
    worker.scheduler_config = SimpleNamespace(max_num_seqs=256)
    worker.compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: True)
    )
    monkeypatch.setattr(worker, "_try_replace_model_runner_class", lambda: Runner)
    assert worker.determine_available_memory() == "community-profile"
    config = SimpleNamespace(has_mamba_layers=True, num_blocks=56)
    worker.initialize_from_config(config)
    assert worker.scheduler_config.max_num_seqs == 256
    assert worker.model_runner.max_num_reqs == 256
    assert worker.events == [("community", config)]
    assert not hasattr(worker, "_warm_up_ixaite_before_kv")


def _set_model_type(worker, model_type):
    worker.model_config = SimpleNamespace(
        max_model_len=8192, hf_config=SimpleNamespace(model_type=model_type)
    )


def _set_qwen35_gdn_config(worker, model_type="qwen3_5"):
    worker.model_config = SimpleNamespace(
        dtype="bf16",
        hf_config=SimpleNamespace(
            model_type=model_type,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        ),
    )
    worker.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    worker.device = "cpu"


_W8A8_QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "config_groups": {
        "group_0": {
            "weights": {"num_bits": 8, "type": "int"},
            "input_activations": {"num_bits": 8, "type": "int"},
        }
    },
}


def _set_linear_warmup_config(
    worker,
    *,
    model_type="qwen3_5",
    hidden_size=4096,
    tp_size=1,
    image_limit=0,
    quantization=None,
    quantization_config=None,
    hf_config_hidden_size=None,
    text_model_type=None,
    text_hidden_size=None,
):
    if quantization == "compressed-tensors" and quantization_config is None:
        quantization_config = dict(_W8A8_QUANT_CONFIG)
    hf_hidden = hidden_size if hf_config_hidden_size is None else hf_config_hidden_size
    worker.model_config = SimpleNamespace(
        dtype="bf16",
        quantization=quantization,
        quantization_config=quantization_config,
        hf_config=SimpleNamespace(
            model_type=model_type,
            hidden_size=hf_hidden,
            vision_config=SimpleNamespace() if image_limit > 0 else None,
            quantization_config=quantization_config,
        ),
        hf_text_config=SimpleNamespace(
            model_type=model_type if text_model_type is None else text_model_type,
            hidden_size=hidden_size if text_hidden_size is None else text_hidden_size,
        ),
        multimodal_config=SimpleNamespace(
            get_limit_per_prompt=lambda key, limit=image_limit: (
                limit if key == "image" else 0
            )
        ),
    )
    worker.parallel_config = SimpleNamespace(tensor_parallel_size=tp_size)
    worker.device = "cpu"


def _record_linear_warmup(worker, sizes, *, include_lm_head):
    called_nk: list[tuple[int, int]] = []

    def implementation(x, weight, bias=None):
        called_nk.append((int(weight.shape[0]), int(weight.shape[1])))

    worker._run_ixaite_linear_family_warmup(
        implementation, sizes, include_lm_head=include_lm_head
    )
    return called_nk


def _stub_runner_class_swap(worker, monkeypatch):
    monkeypatch.setattr(
        worker,
        "_try_replace_model_runner_class",
        lambda: type(worker.model_runner),
    )


def test_execute_dummy_batch_passes_profile_seq_lens():
    """IxaiteWorker.execute_dummy_batch must forward profile_seq_lens=1."""
    calls: list[dict] = []

    class FakeModelRunner:
        uniform_decode_query_len = 4

        def _dummy_run(self, num_tokens, **kwargs):
            calls.append({"num_tokens": num_tokens, **kwargs})

    worker = _make_worker(FakeModelRunner())
    worker.execute_dummy_batch()

    assert len(calls) == 1
    assert calls[0] == {
        "num_tokens": 4,
        "uniform_decode": True,
        "profile_seq_lens": 1,
    }


def test_execute_dummy_batch_uses_fallback_query_len():
    """When model_runner lacks uniform_decode_query_len, default to 1."""
    calls: list[dict] = []

    class BareModelRunner:
        def _dummy_run(self, num_tokens, **kwargs):
            calls.append({"num_tokens": num_tokens, **kwargs})

    worker = _make_worker(BareModelRunner())
    worker.execute_dummy_batch()

    assert len(calls) == 1
    assert calls[0] == {
        "num_tokens": 1,
        "uniform_decode": True,
        "profile_seq_lens": 1,
    }


def test_ilu_worker_is_subclass():
    """IxaiteWorker must inherit from the community Worker."""
    from vllm_iluvatar.backends.ixaite.worker import IxaiteWorker

    # The stub Worker is the base in the test environment.
    assert issubclass(IxaiteWorker, sys.modules["vllm.v1.worker.gpu_worker"].Worker)


def test_pcp_profile_bound_covers_mixed_dual_chunk_batches():
    import random

    from vllm_iluvatar.backends.ixaite.worker import _pcp_profile_local_tokens

    rng = random.Random(42)
    for pcp in (2, 8, 16):
        for _ in range(100):
            sizes = [rng.randint(1, 1024) for _ in range(rng.randint(1, 64))]
            prefills = [n != 1 for n in sizes]
            bound = _pcp_profile_local_tokens(sum(sizes), pcp, len(sizes))
            for rank in range(pcp):
                local = 0
                for n, prefill in zip(sizes, prefills):
                    if not prefill:
                        local += n
                        continue
                    chunk = (n + 2 * pcp - 1) // (2 * pcp)
                    local += sum(
                        max(0, min(chunk, n - index * chunk))
                        for index in (rank, 2 * pcp - 1 - rank)
                    )
                assert local <= bound <= sum(sizes)


@pytest.mark.parametrize("raise_inside", [False, True])
def test_pcp_profile_only_changes_profile_calls_and_restores(raise_inside):
    from types import SimpleNamespace

    calls = []

    class Runner:
        def _dummy_run(self, n, **kwargs):
            calls.append((n, kwargs))

    runner = Runner()
    worker = _make_worker(runner)
    worker.use_v2_model_runner = True
    worker.parallel_config = SimpleNamespace(prefill_context_parallel_size=16)
    worker.scheduler_config = SimpleNamespace(max_num_seqs=64)
    worker.vllm_config = SimpleNamespace(speculative_config=None)
    try:
        with worker._pcp_memory_profile():
            runner._dummy_run(10240, is_profile=True, skip_attn=True)
            runner._dummy_run(10240, is_profile=False)
            if raise_inside:
                raise RuntimeError("test cleanup")
    except RuntimeError:
        assert raise_inside
    assert calls == [
        (768, {"is_profile": True, "skip_attn": True}),
        (10240, {"is_profile": False}),
    ]
    assert "_dummy_run" not in vars(runner)
    runner._dummy_run(10240, is_profile=True)
    assert calls[-1] == (10240, {"is_profile": True})


@pytest.mark.parametrize(
    "v2,pcp,spec", [(False, 16, None), (True, 1, None), (True, 16, object())]
)
def test_pcp_profile_does_not_change_other_runner_modes(v2, pcp, spec):
    from types import SimpleNamespace

    class Runner:
        def _dummy_run(self, n, **kwargs):
            return n

    runner = Runner()
    worker = _make_worker(runner)
    worker.use_v2_model_runner = v2
    worker.parallel_config = SimpleNamespace(prefill_context_parallel_size=pcp)
    worker.vllm_config = SimpleNamespace(speculative_config=spec)
    with worker._pcp_memory_profile():
        assert runner._dummy_run(10240, is_profile=True) == 10240
        assert "_dummy_run" not in vars(runner)


def test_non_pcp_warmup_does_not_inspect_glm_layers(monkeypatch):
    from types import SimpleNamespace

    base = sys.modules["vllm.v1.worker.gpu_worker"].Worker
    monkeypatch.setattr(
        base, "compile_or_warm_up_model", lambda self: "base", raising=False
    )

    class Model:
        def modules(self):
            raise AssertionError("non-PCP warmup must not inspect model layers")

    worker = _make_worker(SimpleNamespace(model=Model()))
    worker.parallel_config = SimpleNamespace(prefill_context_parallel_size=1)
    assert worker.compile_or_warm_up_model() == "base"


def test_pcp_warmup_temporarily_disables_glm_calibration(monkeypatch):
    from types import SimpleNamespace

    class GlmLayer:
        _fp8_kv_scale_calibration_enabled = True

    attention = ModuleType("vllm_iluvatar.models.glm_moe_dsa.attention")
    attention.IluGlmMoeDsaSparseMLA = GlmLayer
    monkeypatch.setitem(sys.modules, attention.__name__, attention)

    layer = GlmLayer()
    model = SimpleNamespace(modules=lambda: [layer])
    worker = _make_worker(SimpleNamespace(model=model))
    worker.parallel_config = SimpleNamespace(prefill_context_parallel_size=16)
    base = sys.modules["vllm.v1.worker.gpu_worker"].Worker

    def base_warmup(self):
        assert layer._fp8_kv_scale_calibration_enabled is False
        return "base"

    monkeypatch.setattr(base, "compile_or_warm_up_model", base_warmup, raising=False)
    assert worker.compile_or_warm_up_model() == "base"
    assert layer._fp8_kv_scale_calibration_enabled is True


def test_initialize_from_config_warms_attention_after_community_kv_init(monkeypatch):
    class ModelRunner:
        pass

    worker = _make_worker(ModelRunner())
    worker.events = []
    original_class = worker.model_runner.__class__
    monkeypatch.setattr(
        worker, "_try_replace_model_runner_class", lambda: original_class
    )
    monkeypatch.setattr(
        worker,
        "_warm_up_ixaite_attention_after_kv",
        lambda: worker.events.append(("attention", None)),
    )
    config = object()

    worker.initialize_from_config(config)

    assert worker.events == [("community", config), ("attention", None)]
    assert worker.model_runner.__class__ is original_class


def _mamba_seq_worker(*, max_num_seqs, max_num_reqs, num_blocks, full_graphs, mamba):
    class Runner:
        pass

    runner = Runner()
    runner.max_num_reqs = max_num_reqs
    worker = _make_worker(runner)
    worker.events = []
    worker.scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
    worker.compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: full_graphs)
    )
    worker._try_replace_model_runner_class = lambda: Runner
    worker._warm_up_ixaite_attention_after_kv = lambda: None
    config = SimpleNamespace(has_mamba_layers=mamba, num_blocks=num_blocks)
    return worker, runner, config


def test_initialize_from_config_caps_max_num_seqs_at_mamba_blocks():
    worker, runner, config = _mamba_seq_worker(
        max_num_seqs=256,
        max_num_reqs=256,
        num_blocks=56,
        full_graphs=True,
        mamba=True,
    )

    worker.initialize_from_config(config)

    assert worker.scheduler_config.max_num_seqs == 56
    assert runner.max_num_reqs == 56
    assert worker.events == [("community", config)]


def test_initialize_from_config_leaves_a_lower_seq_cap_unchanged():
    worker, runner, config = _mamba_seq_worker(
        max_num_seqs=256,
        max_num_reqs=40,
        num_blocks=56,
        full_graphs=True,
        mamba=True,
    )

    worker.initialize_from_config(config)

    assert worker.scheduler_config.max_num_seqs == 56
    assert runner.max_num_reqs == 40


@pytest.mark.parametrize(
    ("max_num_seqs", "num_blocks", "full_graphs", "mamba"),
    [
        (56, 56, True, True),
        (32, 56, True, True),
        (256, 56, True, False),
        (256, 56, False, True),
        (256, 0, True, True),
    ],
)
def test_initialize_from_config_keeps_max_num_seqs_when_mamba_cap_does_not_apply(
    max_num_seqs, num_blocks, full_graphs, mamba
):
    worker, runner, config = _mamba_seq_worker(
        max_num_seqs=max_num_seqs,
        max_num_reqs=max_num_seqs,
        num_blocks=num_blocks,
        full_graphs=full_graphs,
        mamba=mamba,
    )

    worker.initialize_from_config(config)

    assert worker.scheduler_config.max_num_seqs == max_num_seqs
    assert runner.max_num_reqs == max_num_seqs


def test_attention_warmup_uses_bound_ixaite_layer_and_real_block_table(monkeypatch):
    class FakeIxaiteImpl:
        pass

    backend = ModuleType("vllm_iluvatar.backends.ixaite.attention.flash_attn")
    backend.IxaiteFlashAttentionImpl = FakeIxaiteImpl
    monkeypatch.setitem(sys.modules, backend.__name__, backend)

    kv_cache = SimpleNamespace(ndim=5, shape=(2, 100, 4, 16, 256))
    implementation = FakeIxaiteImpl()
    layer = SimpleNamespace(impl=implementation, kv_cache=kv_cache)
    block_table = SimpleNamespace(ndim=2, shape=(8, 528))
    block_table_owner = SimpleNamespace(
        get_device_tensor=lambda batch: (
            block_table if batch == 8 else pytest.fail(f"unexpected batch {batch}")
        )
    )
    state_table_owner = SimpleNamespace(
        get_device_tensor=lambda batch: SimpleNamespace(ndim=2, shape=(batch, 1))
    )
    runner = SimpleNamespace(
        max_num_tokens=2048,
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(
                block_tables=[state_table_owner, block_table_owner]
            )
        ),
    )
    worker = _make_worker(runner)
    _set_model_type(worker, "qwen3_5")
    worker.compilation_config = SimpleNamespace(
        static_forward_context={"model.layers.0.self_attn": layer}
    )
    worker.scheduler_config = SimpleNamespace(max_num_seqs=8)
    calls = []
    monkeypatch.setattr(
        worker,
        "_run_ixaite_attention_warmup",
        lambda *args: calls.append(args),
    )

    assert worker._warm_up_ixaite_attention_after_kv() is True
    assert calls == [(implementation, kv_cache, block_table, 2048, 8192, 1, (), 0)]


def test_attention_warmup_forwards_uniform_decode_query_len(monkeypatch):
    class FakeIxaiteImpl:
        pass

    backend = ModuleType("vllm_iluvatar.backends.ixaite.attention.flash_attn")
    backend.IxaiteFlashAttentionImpl = FakeIxaiteImpl
    monkeypatch.setitem(sys.modules, backend.__name__, backend)

    kv_cache = SimpleNamespace(ndim=5, shape=(2, 100, 4, 16, 256))
    implementation = FakeIxaiteImpl()
    layer = SimpleNamespace(impl=implementation, kv_cache=kv_cache)
    block_table = SimpleNamespace(ndim=2, shape=(4, 256))
    runner = SimpleNamespace(
        max_num_tokens=8,
        uniform_decode_query_len=4,
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(
                block_tables=[
                    SimpleNamespace(
                        get_device_tensor=lambda batch: (
                            block_table
                            if batch == 4
                            else pytest.fail(f"unexpected batch {batch}")
                        )
                    )
                ]
            )
        ),
    )
    worker = _make_worker(runner)
    _set_model_type(worker, "qwen3_5")
    worker.compilation_config = SimpleNamespace(
        static_forward_context={"model.layers.0.self_attn": layer},
        cudagraph_capture_sizes=(4, 8, 16),
        max_cudagraph_capture_size=16,
        cudagraph_mode=SimpleNamespace(
            decode_mode=lambda: SimpleNamespace(name="NONE")
        ),
    )
    worker.scheduler_config = SimpleNamespace(max_num_seqs=4)
    calls = []
    monkeypatch.setattr(
        worker,
        "_run_ixaite_attention_warmup",
        lambda *args: calls.append(args),
    )

    assert worker._warm_up_ixaite_attention_after_kv() is True
    assert calls == [(implementation, kv_cache, block_table, 8, 8192, 4, (), 0)]


def test_spec_graph_warmup_batches_follow_capture_sizes():
    from vllm_iluvatar.backends.ixaite.worker import ixaite_spec_graph_warmup_batches

    assert ixaite_spec_graph_warmup_batches(4, 256, [1, 2, 4, 8, 16, 256]) == (
        1,
        2,
        4,
        64,
    )
    assert ixaite_spec_graph_warmup_batches(4, 2, [4, 8, 16]) == (1, 2)
    assert ixaite_spec_graph_warmup_batches(4, 8, None) == (1, 8)
    assert ixaite_spec_graph_warmup_batches(4, 8, ()) == (1, 8)


def test_attention_warmup_forwards_cudagraph_capture_sizes(monkeypatch):
    class FakeIxaiteImpl:
        pass

    backend = ModuleType("vllm_iluvatar.backends.ixaite.attention.flash_attn")
    backend.IxaiteFlashAttentionImpl = FakeIxaiteImpl
    monkeypatch.setitem(sys.modules, backend.__name__, backend)

    kv_cache = SimpleNamespace(ndim=5, shape=(2, 100, 4, 16, 256))
    implementation = FakeIxaiteImpl()
    layer = SimpleNamespace(impl=implementation, kv_cache=kv_cache)
    block_table = SimpleNamespace(ndim=2, shape=(8, 256))
    runner = SimpleNamespace(
        max_num_tokens=2048,
        uniform_decode_query_len=4,
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(
                block_tables=[
                    SimpleNamespace(
                        get_device_tensor=lambda batch: (
                            block_table
                            if batch == 8
                            else pytest.fail(f"unexpected batch {batch}")
                        )
                    )
                ]
            )
        ),
    )
    worker = _make_worker(runner)
    _set_model_type(worker, "qwen3_5")
    worker.compilation_config = SimpleNamespace(
        static_forward_context={"model.layers.0.self_attn": layer},
        cudagraph_capture_sizes=(4, 8, 16, 32),
        max_cudagraph_capture_size=16,
        cudagraph_mode=SimpleNamespace(
            decode_mode=lambda: SimpleNamespace(name="FULL")
        ),
    )
    worker.scheduler_config = SimpleNamespace(max_num_seqs=8)
    calls = []
    monkeypatch.setattr(
        worker,
        "_run_ixaite_attention_warmup",
        lambda *args: calls.append(args),
    )

    assert worker._warm_up_ixaite_attention_after_kv() is True
    assert calls == [
        (
            implementation,
            kv_cache,
            block_table,
            2048,
            8192,
            4,
            (4, 8, 16, 32),
            4,
        )
    ]


def test_attention_warmup_compiles_spec_query_and_graph_keys(monkeypatch):
    import torch

    decode_calls: list[dict] = []

    class Impl:
        num_heads = 12
        num_kv_heads = 2
        scale = 0.0625

        def _flash_attn_op(self, kind, *args):
            if kind == 0:
                return lambda **kwargs: None

            def decode(**kwargs):
                decode_calls.append(
                    {
                        "q_tokens": int(kwargs["q"].shape[0]),
                        "batch": int(kwargs["cache_seqlens"].shape[0]),
                        "max_query_len": kwargs["max_query_len"],
                        "use_cuda_graph": kwargs["use_cuda_graph"],
                        "max_context_len": kwargs["max_context_len"],
                    }
                )

            return decode

    kv_cache = torch.zeros((2, 8, 2, 16, 256), dtype=torch.bfloat16)
    block_table = torch.zeros((4, 32), dtype=torch.int32)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_python_op",
        lambda op_name, func_name: ((lambda *args: None), "test"),
    )
    monkeypatch.setattr(
        "ixaite.ops.attention.planner.iter_qwen_simt_decode_schedule_representatives",
        lambda **kwargs: (((1, 1), 1, 128),),
    )
    monkeypatch.setattr(
        "ixaite.ops.attention.planner.iter_qwen_mma_decode_schedule_representatives",
        lambda **kwargs: (
            () if kwargs["query_len"] == 1 else (((16, 16, 16, 2), 1, 2048),)
        ),
    )

    from vllm_iluvatar.backends.ixaite.worker import IxaiteWorker

    IxaiteWorker._run_ixaite_attention_warmup(
        Impl(),
        kv_cache,
        block_table,
        2048,
        4096,
        4,
        (4, 8, 16),
        4,
    )

    assert decode_calls == [
        {
            "q_tokens": 1,
            "batch": 1,
            "max_query_len": 1,
            "use_cuda_graph": False,
            "max_context_len": 128,
        },
        {
            "q_tokens": 2,
            "batch": 1,
            "max_query_len": 2,
            "use_cuda_graph": False,
            "max_context_len": 128,
        },
        {
            "q_tokens": 2,
            "batch": 1,
            "max_query_len": 2,
            "use_cuda_graph": False,
            "max_context_len": 2048,
        },
        {
            "q_tokens": 3,
            "batch": 1,
            "max_query_len": 3,
            "use_cuda_graph": False,
            "max_context_len": 128,
        },
        {
            "q_tokens": 3,
            "batch": 1,
            "max_query_len": 3,
            "use_cuda_graph": False,
            "max_context_len": 2048,
        },
        {
            "q_tokens": 4,
            "batch": 1,
            "max_query_len": 4,
            "use_cuda_graph": False,
            "max_context_len": 128,
        },
        {
            "q_tokens": 4,
            "batch": 1,
            "max_query_len": 4,
            "use_cuda_graph": False,
            "max_context_len": 2048,
        },
        {
            "q_tokens": 4,
            "batch": 1,
            "max_query_len": 4,
            "use_cuda_graph": True,
            "max_context_len": 512,
        },
        {
            "q_tokens": 8,
            "batch": 2,
            "max_query_len": 4,
            "use_cuda_graph": True,
            "max_context_len": 512,
        },
        {
            "q_tokens": 16,
            "batch": 4,
            "max_query_len": 4,
            "use_cuda_graph": True,
            "max_context_len": 512,
        },
    ]


def test_attention_warmup_skips_non_qwen35_models():
    worker = _make_worker(SimpleNamespace())
    _set_model_type(worker, "llama")

    assert worker._warm_up_ixaite_attention_after_kv() is False


def test_attention_warmup_accepts_qwen3_dense_d128(monkeypatch):
    class FakeIxaiteImpl:
        pass

    backend = ModuleType("vllm_iluvatar.backends.ixaite.attention.flash_attn")
    backend.IxaiteFlashAttentionImpl = FakeIxaiteImpl
    monkeypatch.setitem(sys.modules, backend.__name__, backend)

    kv_cache = SimpleNamespace(ndim=5, shape=(2, 100, 8, 16, 128))
    implementation = FakeIxaiteImpl()
    layer = SimpleNamespace(impl=implementation, kv_cache=kv_cache)
    block_table = SimpleNamespace(ndim=2, shape=(8, 528))
    block_table_owner = SimpleNamespace(
        get_device_tensor=lambda batch: (
            block_table if batch == 8 else pytest.fail(f"unexpected batch {batch}")
        )
    )
    runner = SimpleNamespace(
        max_num_tokens=2048,
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(block_tables=[block_table_owner])
        ),
    )
    worker = _make_worker(runner)
    _set_model_type(worker, "qwen3")
    worker.compilation_config = SimpleNamespace(
        static_forward_context={"model.layers.0.self_attn": layer}
    )
    worker.scheduler_config = SimpleNamespace(max_num_seqs=8)
    calls = []
    monkeypatch.setattr(
        worker,
        "_run_ixaite_attention_warmup",
        lambda *args: calls.append(args),
    )

    assert worker._warm_up_ixaite_attention_after_kv() is True
    assert calls == [(implementation, kv_cache, block_table, 2048, 8192, 1, (), 0)]


def test_gdn_warmup_skips_qwen3_dense():
    worker = _make_worker(SimpleNamespace())
    _set_model_type(worker, "qwen3")
    assert worker._warm_up_ixaite_gdn_chunk_buckets() is False
    assert worker._warm_up_ixaite_gdn_packed_decode() is False
    assert worker._warm_up_ixaite_gdn_conv() is False
    assert worker._warm_up_ixaite_gdn_post_conv_prep() is False


def test_linear_warmup_gates_admit_qwen3_dense(monkeypatch):
    worker = _make_worker(SimpleNamespace(max_num_tokens=2048))
    _set_model_type(worker, "qwen3")
    worker.scheduler_config = SimpleNamespace(max_num_seqs=128)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: op_name == "linear",
    )
    calls = []
    monkeypatch.setattr(
        worker,
        "_run_ixaite_linear_family_warmup",
        lambda *a, **k: calls.append(
            (list(a[1]) if len(a) > 1 else [], k.get("include_lm_head"))
        ),
    )
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_python_op",
        lambda op_name, func_name: (
            (object(), "test") if op_name == "linear" else (None, "skip")
        ),
    )
    assert worker._warm_up_ixaite_body_small_m() is True
    assert worker._warm_up_ixaite_linear_exact_buckets() is True
    assert worker._warm_up_ixaite_lm_head() is True
    assert any(include is False for _, include in calls)
    assert any(include is True for _, include in calls)


def test_shutdown_releases_graph_pins_after_runner_before_parent_finishes(monkeypatch):
    events = []

    class ModelRunner:
        def shutdown(self):
            events.append("runner")

    worker = _make_worker(ModelRunner())
    original_class = worker.model_runner.__class__
    monkeypatch.setattr(
        worker, "_try_replace_model_runner_class", lambda: original_class
    )

    def release_pins():
        events.append("pins")

    monkeypatch.setattr("ixaite.release_cuda_graph_pins", release_pins)
    worker.shutdown()

    assert events == ["runner", "pins"]
    assert worker.community_shutdown_complete is True
    assert worker.model_runner.__class__ is original_class


def test_shutdown_skips_graph_pin_release_when_ixaite_is_not_loaded(monkeypatch):
    events = []

    class ModelRunner:
        def shutdown(self):
            events.append("runner")

    worker = _make_worker(ModelRunner())
    original_class = worker.model_runner.__class__
    monkeypatch.setattr(
        worker, "_try_replace_model_runner_class", lambda: original_class
    )
    monkeypatch.delitem(sys.modules, "ixaite", raising=False)

    worker.shutdown()

    assert events == ["runner"]
    assert worker.community_shutdown_complete is True
    assert worker.model_runner.__class__ is original_class


def test_shutdown_keeps_released_pins_when_later_parent_teardown_fails(monkeypatch):
    events = []

    class ModelRunner:
        def shutdown(self):
            events.append("runner")

    worker = _make_worker(ModelRunner())
    worker.fail_parent_shutdown = True
    original_class = worker.model_runner.__class__
    monkeypatch.setattr(
        worker, "_try_replace_model_runner_class", lambda: original_class
    )
    monkeypatch.setattr("ixaite.release_cuda_graph_pins", lambda: events.append("pins"))

    with pytest.raises(RuntimeError, match="parent teardown failed"):
        worker.shutdown()

    assert events == ["runner", "pins"]
    assert worker.model_runner.__class__ is original_class


def test_memory_probe_covers_ixaite_prefill_crop_buckets(monkeypatch):
    calls: list[dict] = []
    released: list[str] = []

    class FakeModelRunner:
        max_num_tokens = 2048

        def _dummy_run(self, num_tokens, **kwargs):
            calls.append({"num_tokens": num_tokens, **kwargs})

    worker = _make_worker(FakeModelRunner())
    _set_model_type(worker, "qwen3_5")
    _stub_runner_class_swap(worker, monkeypatch)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: op_name == "linear",
    )
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_python_op",
        lambda op_name, func_name: (None, "skip"),
    )
    monkeypatch.setattr(
        worker,
        "_release_ixaite_warmup_scratch",
        lambda: released.append("released"),
    )

    result = worker.determine_available_memory()

    assert result == "community-profile"
    assert released == ["released"]
    assert calls == [
        {
            "num_tokens": size,
            "skip_eplb": True,
            "is_profile": True,
            "remove_lora": False,
        }
        for size in (
            1,
            32,
            64,
            80,
            96,
            128,
            129,
            257,
            513,
            769,
            1025,
            1281,
            1537,
            1793,
        )
    ]


def test_memory_probe_compile_cap_runs_community_profile_first(monkeypatch):
    """Under VLLM_COMPILE the community profile runs first and the ixaite
    warmup buckets are capped at the top compiled-range endpoint."""
    from contextlib import contextmanager

    from vllm.config.compilation import CompilationMode

    calls: list[dict] = []
    events: list[str] = []

    class FakeModelRunner:
        max_num_tokens = 2048

        def _dummy_run(self, num_tokens, **kwargs):
            calls.append({"num_tokens": num_tokens, **kwargs})

    worker = _make_worker(FakeModelRunner())
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            compile_ranges_endpoints=[128],
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1024),
    )
    _set_model_type(worker, "qwen3_5")
    _stub_runner_class_swap(worker, monkeypatch)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: op_name == "linear",
    )
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_python_op",
        lambda op_name, func_name: (None, "skip"),
    )
    monkeypatch.setattr(
        worker,
        "_release_ixaite_warmup_scratch",
        lambda: events.append("released"),
    )

    # PCP accounting must cover the profile in compiled mode too.
    @contextmanager
    def pcp_profile():
        events.append("pcp-enter")
        yield
        events.append("pcp-exit")

    monkeypatch.setattr(worker, "_pcp_memory_profile", pcp_profile)

    def community_profile(self):
        events.append("community")
        return "community-profile"

    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.Worker.determine_available_memory",
        community_profile,
    )

    result = worker.determine_available_memory()

    assert result == "community-profile"
    # Community profile first, then the capped warmup, then scratch release.
    assert events == ["pcp-enter", "community", "pcp-exit", "released"]
    assert calls == [
        {
            "num_tokens": size,
            "skip_eplb": True,
            "is_profile": True,
            "remove_lora": False,
        }
        for size in (1, 32, 64, 80, 96, 128)
    ]


def test_vision_gelu_warmup_covers_mode_and_tail_variants(monkeypatch):
    """Vision warmup must compile both modes with and without tail guards."""
    import torch

    calls: list[tuple[str, int]] = []
    worker = _make_worker(SimpleNamespace(device="cpu"))
    worker.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(model_type="qwen3_5"),
    )
    worker.device = "cpu"
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: op_name == "gelu",
    )

    # Record only the public-op contract; no GPU allocation is needed here.
    def gelu(input, approximate):
        calls.append((approximate, input.numel()))
        return input

    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_torch_op",
        lambda op_name: (gelu, "test") if op_name == "gelu" else (None, "skip"),
    )

    assert worker._warm_up_ixaite_gelu() is True
    assert calls == [
        ("none", 4),
        ("none", 1024),
        ("tanh", 4),
        ("tanh", 1024),
    ]


def test_video_evs_warmup_compiles_short_video_shape(monkeypatch):
    import torch

    calls = []
    worker = _make_worker(SimpleNamespace())
    worker.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen3_5",
            vision_config=SimpleNamespace(out_hidden_size=3584),
        ),
        multimodal_config=SimpleNamespace(
            video_pruning_method="evs",
            video_pruning_rate=0.5,
        ),
    )
    worker._ixaite_torch_dtype = lambda: torch.bfloat16
    worker._ixaite_warmup_device = lambda: "cpu"
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_python_op_enabled",
        lambda op_name, func_name: (
            op_name == "video_evs" and func_name == "compute_retention_mask"
        ),
    )

    def video_evs(video, size, spatial_merge_size, q):
        calls.append((video.shape, size, spatial_merge_size, q))
        return torch.empty(video.shape[0], dtype=torch.bool)

    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_python_op",
        lambda op_name, func_name: (video_evs, "test"),
    )

    assert worker._warm_up_ixaite_video_evs() is True
    assert calls == [((392, 3584), (2, 28, 28), 2, 0.5)]


def test_memory_probe_respects_max_num_tokens(monkeypatch):
    calls: list[int] = []

    class FakeModelRunner:
        max_num_tokens = 512

        def _dummy_run(self, num_tokens, **kwargs):
            calls.append(num_tokens)

    worker = _make_worker(FakeModelRunner())
    _set_model_type(worker, "qwen3_5_text")
    _stub_runner_class_swap(worker, monkeypatch)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: True,
    )
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_python_op",
        lambda op_name, func_name: (None, "skip"),
    )
    monkeypatch.setattr(worker, "_release_ixaite_warmup_scratch", lambda: None)

    worker.determine_available_memory()

    assert calls == [1, 32, 64, 80, 96, 128, 129, 257]


def test_ixaite_prefill_warmup_sizes_cover_every_crop_bucket():
    from vllm_iluvatar.backends.ixaite.worker import ixaite_prefill_warmup_sizes

    # One logical size per padded-M bucket: 129 -> 256, then every 256 step.
    assert ixaite_prefill_warmup_sizes(2048) == [
        129,
        257,
        513,
        769,
        1025,
        1281,
        1537,
        1793,
    ]
    assert ixaite_prefill_warmup_sizes(512) == [129, 257]
    # A budget that ends mid-bucket still warms the launcher covering it.
    assert ixaite_prefill_warmup_sizes(600) == [129, 257, 513]
    # Below the crop path there is nothing to warm; small M is a separate size.
    assert ixaite_prefill_warmup_sizes(128) == []


def test_ixaite_body_small_m_warmup_sizes_cover_exact_m():
    from vllm_iluvatar.backends.ixaite.worker import ixaite_body_small_m_warmup_sizes

    assert ixaite_body_small_m_warmup_sizes(2048) == list(range(2, 17))
    assert ixaite_body_small_m_warmup_sizes(8) == list(range(2, 9))
    assert ixaite_body_small_m_warmup_sizes(1) == []


def test_ixaite_hgemm_single_warmup_sizes_cover_exact_small_tiles():
    from vllm_iluvatar.backends.ixaite.worker import ixaite_hgemm_single_warmup_sizes

    assert ixaite_hgemm_single_warmup_sizes(2048) == [32, 64, 80, 96, 128]
    assert ixaite_hgemm_single_warmup_sizes(128) == [32, 64, 80, 96, 128]
    assert ixaite_hgemm_single_warmup_sizes(96) == [32, 64, 80, 96]
    assert ixaite_hgemm_single_warmup_sizes(16) == []


def test_ixaite_hgemm_exact_warmup_sizes_cover_physical_buckets():
    from vllm_iluvatar.backends.ixaite.worker import ixaite_hgemm_exact_warmup_sizes

    assert ixaite_hgemm_exact_warmup_sizes(2048) == [
        256,
        512,
        768,
        1024,
        1280,
        1536,
        1792,
        2048,
    ]
    assert ixaite_hgemm_exact_warmup_sizes(512) == [256, 512]
    assert ixaite_hgemm_exact_warmup_sizes(255) == []


def test_ixaite_qwen35_vision_warmup_sizes_cover_remainder_and_merger():
    from vllm_iluvatar.backends.ixaite.worker import (
        ixaite_qwen35_vision_linear_warmup_sizes,
        ixaite_qwen35_vision_row_counts,
        ixaite_qwen35_vit_qkv_warmup_sizes,
        ixaite_qwen35_vl_prefill_dummy_sizes,
    )

    rows = ixaite_qwen35_vision_row_counts()
    assert 1088 in rows  # 3136 - 2048 remainder of 896^2 ViT
    assert 784 in rows  # 448^2 ViT and 896 merger
    assert 196 in rows  # 448 merger
    assert 912 in rows  # 784 merged + 128 text
    assert 3136 in rows
    assert ixaite_qwen35_vision_linear_warmup_sizes(2048) == rows
    assert ixaite_qwen35_vl_prefill_dummy_sizes(2048) == [784, 912, 926, 1088]
    assert ixaite_qwen35_vl_prefill_dummy_sizes(800) == [784]
    assert ixaite_qwen35_vit_qkv_warmup_sizes() == [784, 1088, 3136]


def test_ixaite_lm_head_warmup_sizes_cover_small_m_and_hgemm():
    from vllm_iluvatar.backends.ixaite.worker import ixaite_lm_head_warmup_sizes

    assert ixaite_lm_head_warmup_sizes(128) == [
        *range(1, 17),
        17,
        32,
        33,
        64,
        65,
        80,
        81,
        96,
        97,
        128,
    ]
    assert ixaite_lm_head_warmup_sizes(16) == list(range(1, 17))
    assert ixaite_lm_head_warmup_sizes(0) == []


def test_ixaite_gdn_chunk_warmup_sizes_cover_dense_and_varlen():
    from vllm_iluvatar.backends.ixaite.worker import ixaite_gdn_chunk_warmup_sizes

    assert ixaite_gdn_chunk_warmup_sizes(16, 2048) == [256, 257]
    assert ixaite_gdn_chunk_warmup_sizes(16, 256) == [256]
    assert ixaite_gdn_chunk_warmup_sizes(8, 2048) == [512, 513]


def test_memory_probe_covers_ixaite_gdn_chunk_cubins(monkeypatch):
    dummy_calls: list[int] = []
    gdn_calls: list[int] = []
    released: list[str] = []

    class FakeModelRunner:
        max_num_tokens = 2048

        def _dummy_run(self, num_tokens, **kwargs):
            dummy_calls.append(num_tokens)

    worker = _make_worker(FakeModelRunner())
    _set_qwen35_gdn_config(worker)
    _stub_runner_class_swap(worker, monkeypatch)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: False,
    )
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_python_op",
        lambda op_name, func_name: (
            (object(), "test") if op_name == "gdn_chunk" else (None, "skip")
        ),
    )
    monkeypatch.setattr(
        worker,
        "_run_ixaite_gdn_chunk_warmup",
        lambda _impl, size: gdn_calls.append(size),
    )
    monkeypatch.setattr(
        worker,
        "_release_ixaite_warmup_scratch",
        lambda: released.append("released"),
    )

    result = worker.determine_available_memory()

    assert result == "community-profile"
    assert released == ["released"]
    assert dummy_calls == []
    assert gdn_calls == [256, 257]


def test_memory_probe_covers_ixaite_gdn_packed_decode(monkeypatch):
    dummy_calls: list[int] = []
    decode_calls: list[str] = []
    released: list[str] = []

    class FakeModelRunner:
        max_num_tokens = 2048

        def _dummy_run(self, num_tokens, **kwargs):
            dummy_calls.append(num_tokens)

    worker = _make_worker(FakeModelRunner())
    _set_qwen35_gdn_config(worker)
    _stub_runner_class_swap(worker, monkeypatch)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: False,
    )
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_python_op",
        lambda op_name, func_name: (
            (object(), "test") if op_name == "gdn_packed_decode" else (None, "skip")
        ),
    )
    monkeypatch.setattr(
        worker,
        "_run_ixaite_gdn_packed_decode_warmup",
        lambda _impl: decode_calls.append("decode"),
    )
    monkeypatch.setattr(
        worker,
        "_release_ixaite_warmup_scratch",
        lambda: released.append("released"),
    )

    result = worker.determine_available_memory()

    assert result == "community-profile"
    assert released == ["released"]
    assert dummy_calls == []
    assert decode_calls == ["decode"]


def test_memory_probe_covers_ixaite_exact_linear_and_lm_head(monkeypatch):
    dummy_calls: list[int] = []
    linear_warmups: list[tuple[list[int], bool]] = []
    released: list[str] = []

    class FakeModelRunner:
        max_num_tokens = 2048

        def _dummy_run(self, num_tokens, **kwargs):
            dummy_calls.append(num_tokens)

    worker = _make_worker(FakeModelRunner())
    _set_model_type(worker, "qwen3_5")
    worker.scheduler_config = SimpleNamespace(max_num_seqs=128)
    _stub_runner_class_swap(worker, monkeypatch)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: op_name == "linear",
    )
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_python_op",
        lambda op_name, func_name: (
            (object(), "test") if op_name == "linear" else (None, "skip")
        ),
    )
    monkeypatch.setattr(
        worker,
        "_run_ixaite_linear_family_warmup",
        lambda _impl, sizes, include_lm_head: linear_warmups.append(
            (list(sizes), include_lm_head)
        ),
    )
    monkeypatch.setattr(
        worker,
        "_release_ixaite_warmup_scratch",
        lambda: released.append("released"),
    )

    result = worker.determine_available_memory()

    assert result == "community-profile"
    assert released == ["released"]
    assert dummy_calls == [
        1,
        32,
        64,
        80,
        96,
        128,
        129,
        257,
        513,
        769,
        1025,
        1281,
        1537,
        1793,
    ]
    assert linear_warmups == [
        (list(range(2, 17)), False),
        ([256, 512, 768, 1024, 1280, 1536, 1792, 2048], False),
        (
            [
                *range(1, 17),
                17,
                32,
                33,
                64,
                65,
                80,
                81,
                96,
                97,
                128,
            ],
            True,
        ),
    ]


def test_memory_probe_covers_ixaite_vision_remainder_shapes(monkeypatch):
    dummy_calls: list[int] = []
    linear_warmups: list[tuple[list[int], bool]] = []
    released: list[str] = []

    class FakeModelRunner:
        max_num_tokens = 2048

        def _dummy_run(self, num_tokens, **kwargs):
            dummy_calls.append(num_tokens)

    worker = _make_worker(FakeModelRunner())
    _set_model_type(worker, "qwen3_5")
    worker.model_config.hf_config.vision_config = SimpleNamespace()
    worker.model_config.multimodal_config = SimpleNamespace(
        get_limit_per_prompt=lambda modality: 1 if modality == "image" else 0
    )
    worker.scheduler_config = SimpleNamespace(max_num_seqs=128)
    _stub_runner_class_swap(worker, monkeypatch)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: op_name == "linear",
    )
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_python_op",
        lambda op_name, func_name: (
            (object(), "test") if op_name == "linear" else (None, "skip")
        ),
    )
    monkeypatch.setattr(
        worker,
        "_run_ixaite_linear_family_warmup",
        lambda _impl, sizes, include_lm_head: linear_warmups.append(
            (list(sizes), include_lm_head)
        ),
    )
    monkeypatch.setattr(
        worker,
        "_release_ixaite_warmup_scratch",
        lambda: released.append("released"),
    )

    result = worker.determine_available_memory()

    assert result == "community-profile"
    assert released == ["released"]
    assert dummy_calls == [
        1,
        32,
        64,
        80,
        96,
        128,
        129,
        257,
        513,
        769,
        1025,
        1281,
        1537,
        1793,
        784,
        912,
        926,
        1088,
    ]
    assert linear_warmups[0] == (list(range(2, 17)), False)
    assert linear_warmups[1] == (
        [
            196,
            256,
            304,
            324,
            512,
            768,
            784,
            912,
            926,
            1024,
            1088,
            1280,
            1536,
            1792,
            2048,
            3136,
        ],
        False,
    )


def _make_vision_qkv_worker(
    *,
    model_type: str = "qwen3_5",
    tp_size: int = 1,
    image_limit: int = 1,
):
    import torch

    worker = _make_worker(SimpleNamespace(device="cpu"))
    vision = None
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        vision = SimpleNamespace(num_heads=16, hidden_size=1152)
    elif model_type == "qwen2_5_vl":
        vision = SimpleNamespace(num_heads=16, hidden_size=1280)
    worker.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(model_type=model_type, vision_config=vision),
        multimodal_config=SimpleNamespace(
            get_limit_per_prompt=lambda modality: (
                image_limit if modality == "image" else 0
            )
        ),
    )
    worker.parallel_config = SimpleNamespace(tensor_parallel_size=tp_size)
    worker.device = "cpu"
    return worker


# Encoder presence is independent of the model-specific warmup support list.
@pytest.mark.parametrize("model_type,has_config,image_limit,expected", [
    ("qwen3_5", True, 1, True),
    ("qwen2_5_vl", True, 1, True),
    ("pixtral", True, 1, True),
    ("qwen3_5", False, 1, False),
    ("qwen2_5_vl", True, 0, False),
])
def test_vision_encoder_detection(model_type, has_config, image_limit, expected):
    worker = _make_vision_qkv_worker(model_type=model_type, image_limit=image_limit)
    worker.model_config.hf_config.vision_config = SimpleNamespace() if has_config else None
    assert worker._ixaite_has_vision_encoder() is expected


# A different encoder must not receive Qwen's packed NeoX preparation inputs.
def test_vit_qkv_warmup_skips_other_vision_encoders(monkeypatch):
    worker = _make_vision_qkv_worker(model_type="pixtral")
    worker.model_config.hf_config.vision_config = SimpleNamespace()
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_torch_op",
        lambda name: pytest.fail("unsupported encoder must not load Qwen preparation"),
    )
    assert worker._ixaite_has_vision_encoder()
    assert worker._warm_up_ixaite_vit_qkv_rope() is False


# A shared encoder predicate must not admit other models to Qwen3.5's GEMM shape table.
@pytest.mark.parametrize("model_type", ["qwen2_5_vl", "pixtral"])
def test_other_encoders_skip_qwen35_vision_linear_shapes(model_type):
    from ixaite.ops.linear_qwen35_shapes import LinearFamily

    worker = _make_worker(SimpleNamespace())
    _set_linear_warmup_config(worker, model_type=model_type, image_limit=1)
    assert worker._ixaite_has_vision_encoder()
    geometries = worker._ixaite_linear_warmup_geometries(include_lm_head=False)
    vision_families = {
        LinearFamily.VIS_QKV, LinearFamily.VIS_O_PROJ,
        LinearFamily.VIS_FC1, LinearFamily.VIS_FC2,
        LinearFamily.MERGER_FC1, LinearFamily.MERGER_FC2,
    }
    assert geometries and all(geom.family not in vision_families for geom in geometries)


def _stub_vision_op(monkeypatch, op_name, implementation):
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda name: name == op_name,
    )
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_torch_op",
        lambda name: (implementation, "test") if name == op_name else (None, "skip"),
    )


# Full spatial RoPE and explicit padding must warm the same cubin used during inference.
def test_vit_qkv_rope_warmup_compiles_full_rotary_and_padding(monkeypatch):
    import torch

    calls = []
    worker = _make_vision_qkv_worker(model_type="qwen3_5_moe", tp_size=2)

    def vit_qkv_rope(packed, cos, sin, num_heads, head_dim, padded_head_dim=None):
        calls.append((tuple(packed.shape), tuple(cos.shape), cos.dtype, num_heads, head_dim, padded_head_dim))
        return packed, packed, packed

    _stub_vision_op(monkeypatch, "vit_qkv_rope", vit_qkv_rope)

    assert worker._warm_up_ixaite_vit_qkv_rope() is True
    assert calls == [
        ((784, 1, 3 * 8 * 72), (784, 36), torch.bfloat16, 8, 72, 128),
        ((784, 1, 3 * 8 * 72), (784, 36), torch.float32, 8, 72, 128),
    ]


# Planner representatives must warm the same padded shape, native scale and boundaries as serving.
def test_vit_warmup_covers_logical_attention_variants(monkeypatch):
    import sys
    from types import SimpleNamespace

    import torch

    from vllm_iluvatar.backends.ixaite import runtime

    worker = _make_vision_qkv_worker(model_type="qwen3_5", tp_size=2)
    worker.model_config.max_model_len = 1024
    calls = []
    plans = []

    # The GPU operator has separate numerical tests; this records startup's ABI.
    def attention(q, k, v, cuq, cuk, sq, sk, **kwargs):
        calls.append((tuple(q.shape), cuq.tolist(), sq, sk, kwargs))

    _stub_vision_op(monkeypatch, "vit_qkv_rope", lambda *args, **kwargs: (None,) * 3)
    monkeypatch.setattr(runtime, "load_ixaite_python_op", lambda *args: (attention, None))
    # Per-sequence lengths from the shared planner determine the packed boundaries.
    def prefill_warmup_calls(*args, **kwargs):
        plans.append((args, kwargs))
        return [[(257, 257)], [(128, 128), (2, 2), (3, 3)]]

    monkeypatch.setitem(sys.modules, "ixaite.ops.attention.flash_attn_prefill", SimpleNamespace(
        prefill_warmup_calls=prefill_warmup_calls))
    assert worker._warm_up_ixaite_vit_qkv_rope()
    assert plans == [((8, 8, 128, 4096), dict(
        dtype=torch.bfloat16, paged=False, max_seqlen_q=4096, max_seqlen_k=4096,
    ))]
    assert calls == [
        ((257, 8, 128), [0, 257], 257, 257,
         {"softmax_scale": 72**-.5, "causal": False, "valid_head_dim": 72}),
        ((133, 8, 128), [0, 128, 130, 133], 128, 128,
         {"softmax_scale": 72**-.5, "causal": False, "valid_head_dim": 72}),
    ]


def test_vit_qkv_rope_warmup_skips_text_only_moe(monkeypatch):
    worker = _make_vision_qkv_worker(model_type="qwen3_5_moe_text", image_limit=0)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: True,
    )
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_torch_op",
        lambda op_name: (_ for _ in ()).throw(AssertionError("should not load")),
    )

    assert worker._warm_up_ixaite_vit_qkv_rope() is False


# Qwen2.5-VL shares preparation and must not recompile it on its first BF16 D80 image.
def test_vit_qkv_warmup_includes_qwen25(monkeypatch):
    calls = []
    worker = _make_vision_qkv_worker(model_type="qwen2_5_vl")

    # Record the logical and stored widths without allocating an actual GPU.
    def prepare(packed, cos, sin, heads, dim, padded_head_dim=None):
        calls.append((heads, dim, padded_head_dim, cos.shape[-1]))
        return packed, packed, packed

    _stub_vision_op(monkeypatch, "vit_qkv_rope", prepare)
    assert worker._warm_up_ixaite_vit_qkv_rope()
    assert calls == [(16, 80, 128, 40)] * 2


# Actual per-rank heads take precedence over config division when vision uses data parallelism.
def test_vit_geometry_uses_loaded_attention_shard():
    from vllm_iluvatar.backends.ixaite.attention.mm_encoder_attention import IluMMEncoderAttention

    worker = _make_vision_qkv_worker(tp_size=2)
    attention = object.__new__(IluMMEncoderAttention)
    attention.num_heads, attention.head_size = 16, 72
    worker.model_runner = SimpleNamespace(model=SimpleNamespace(modules=lambda: iter([attention])))
    assert worker._ixaite_vision_qkv_geometry() == (16, 72)


def _qwen35_9b_text_nks(tp: int) -> dict:
    from ixaite.ops.linear_qwen35_shapes import LinearFamily, geometry_for

    return {
        family: geometry_for(family, tp)
        for family in (
            LinearFamily.QKV,
            LinearFamily.O_PROJ,
            LinearFamily.GATE_UP,
            LinearFamily.DOWN_PROJ,
            LinearFamily.IN_PROJ_QKVZ,
            LinearFamily.IN_PROJ_BA,
            LinearFamily.LM_HEAD,
        )
    }


def test_linear_family_warmup_includes_padded_vision_mlp():
    """TP1 vision MLP is padded by ixaite and must be JIT-warmed."""
    from ixaite.ops.linear_qwen35_shapes import (
        LinearFamily,
        geometry_for,
        ixformer_gated_nk,
    )

    class FakeModelRunner:
        max_num_tokens = 256

    worker = _make_worker(FakeModelRunner())
    _set_linear_warmup_config(worker, image_limit=1)
    called_nk = _record_linear_warmup(worker, [256], include_lm_head=False)
    called_lm = _record_linear_warmup(worker, [1], include_lm_head=True)

    gated = ixformer_gated_nk()
    vis_fc2 = geometry_for(LinearFamily.VIS_FC2, 1)
    vis_fc2_tp8 = geometry_for(LinearFamily.VIS_FC2, 8)
    vis_qkv = geometry_for(LinearFamily.VIS_QKV, 1)
    merger_fc2 = geometry_for(LinearFamily.MERGER_FC2, 1)
    text = _qwen35_9b_text_nks(1)
    assert vis_fc2.nk not in gated
    assert vis_fc2_tp8.nk in gated
    assert vis_qkv.nk not in gated
    assert vis_fc2.nk in called_nk
    assert not any(nk in gated for nk in called_nk)
    assert vis_qkv.nk in called_nk
    assert merger_fc2.nk in called_nk
    assert text[LinearFamily.QKV].nk in called_nk
    assert text[LinearFamily.IN_PROJ_QKVZ].nk in called_nk
    assert text[LinearFamily.GATE_UP].nk in called_nk
    assert text[LinearFamily.LM_HEAD].nk in called_lm


def test_linear_family_warmup_qwen35_9b_text_keeps_body_skips_vision():
    from ixaite.ops.linear_qwen35_shapes import LinearFamily, geometry_for

    class FakeModelRunner:
        max_num_tokens = 256

    worker = _make_worker(FakeModelRunner())
    _set_linear_warmup_config(worker, image_limit=0)
    called_nk = _record_linear_warmup(worker, [256], include_lm_head=False)
    called_lm = _record_linear_warmup(worker, [1], include_lm_head=True)

    text = _qwen35_9b_text_nks(1)
    vis_qkv = geometry_for(LinearFamily.VIS_QKV, 1)
    vis_fc2 = geometry_for(LinearFamily.VIS_FC2, 1)
    merger_fc2 = geometry_for(LinearFamily.MERGER_FC2, 1)
    assert vis_qkv.nk not in called_nk
    assert vis_fc2.nk not in called_nk
    assert merger_fc2.nk not in called_nk
    for family in (
        LinearFamily.QKV,
        LinearFamily.O_PROJ,
        LinearFamily.GATE_UP,
        LinearFamily.DOWN_PROJ,
        LinearFamily.IN_PROJ_QKVZ,
        LinearFamily.IN_PROJ_BA,
    ):
        assert text[family].nk in called_nk
    assert text[LinearFamily.LM_HEAD].nk in called_lm


def test_linear_family_warmup_qwen35_9b_w8a8_keeps_gdn_and_lm_head():
    from ixaite.ops.linear_qwen35_shapes import LinearFamily

    class FakeModelRunner:
        max_num_tokens = 256

    worker = _make_worker(FakeModelRunner())
    _set_linear_warmup_config(
        worker,
        image_limit=0,
        quantization="compressed-tensors",
    )
    called_nk = _record_linear_warmup(worker, [16], include_lm_head=False)
    called_lm = _record_linear_warmup(worker, [1], include_lm_head=True)

    text = _qwen35_9b_text_nks(1)
    assert text[LinearFamily.IN_PROJ_QKVZ].nk in called_nk
    assert text[LinearFamily.IN_PROJ_BA].nk in called_nk
    assert text[LinearFamily.O_PROJ].nk in called_nk
    assert text[LinearFamily.LM_HEAD].nk in called_lm
    assert text[LinearFamily.QKV].nk not in called_nk
    assert text[LinearFamily.GATE_UP].nk not in called_nk
    assert text[LinearFamily.DOWN_PROJ].nk not in called_nk


def test_linear_family_warmup_kvint8_does_not_drop_9b_qkv():
    from ixaite.ops.linear_qwen35_shapes import LinearFamily

    class FakeModelRunner:
        max_num_tokens = 256

    worker = _make_worker(FakeModelRunner())
    _set_linear_warmup_config(worker, image_limit=0)
    worker.model_config.quantization = None
    called_nk = _record_linear_warmup(worker, [16], include_lm_head=False)
    text = _qwen35_9b_text_nks(1)
    assert text[LinearFamily.QKV].nk in called_nk
    assert text[LinearFamily.GATE_UP].nk in called_nk


def test_linear_family_warmup_qwen3_8b_uses_8b_qkv_not_9b():
    from ixaite.ops.linear_qwen35_shapes import (
        LinearFamily,
        geometry_for,
        qwen3_8b_geometries,
    )

    class FakeModelRunner:
        max_num_tokens = 256

    worker = _make_worker(FakeModelRunner())
    _set_linear_warmup_config(
        worker,
        model_type="qwen3",
        hidden_size=4096,
        tp_size=1,
        image_limit=0,
    )
    called_nk = _record_linear_warmup(worker, [16], include_lm_head=False)
    called_lm = _record_linear_warmup(worker, [1], include_lm_head=True)

    geoms_8b = {geom.family: geom for geom in qwen3_8b_geometries() if geom.tp == 1}
    qwen35 = _qwen35_9b_text_nks(1)
    assert geoms_8b[LinearFamily.QKV].nk in called_nk
    assert geoms_8b[LinearFamily.LM_HEAD].nk in called_lm
    assert geometry_for(LinearFamily.GATE_UP, 1).nk in called_nk
    assert geometry_for(LinearFamily.DOWN_PROJ, 1).nk in called_nk
    assert geometry_for(LinearFamily.O_PROJ, 1).nk in called_nk
    assert qwen35[LinearFamily.QKV].nk not in called_nk
    assert qwen35[LinearFamily.LM_HEAD].nk not in called_lm
    assert qwen35[LinearFamily.IN_PROJ_QKVZ].nk not in called_nk


def test_linear_family_warmup_qwen36_35b_skips_9b():
    from ixaite.ops.linear_qwen35_shapes import (
        QWEN36_35B_K,
        LinearFamily,
        geometry_for,
        qwen36_35b_a3b_geometries,
    )

    class FakeModelRunner:
        max_num_tokens = 256

    worker = _make_worker(FakeModelRunner())
    _set_linear_warmup_config(
        worker,
        model_type="qwen3_5_moe",
        hidden_size=QWEN36_35B_K,
        tp_size=2,
        image_limit=0,
        quantization="compressed-tensors",
    )
    called_nk = _record_linear_warmup(worker, [16], include_lm_head=False)
    called_lm = _record_linear_warmup(worker, [1], include_lm_head=True)

    geoms = {geom.family: geom for geom in qwen36_35b_a3b_geometries() if geom.tp == 2}
    qwen35 = _qwen35_9b_text_nks(2)
    assert geoms[LinearFamily.IN_PROJ_QKVZ].nk in called_nk
    assert geoms[LinearFamily.IN_PROJ_BA].nk in called_nk
    assert geoms[LinearFamily.O_PROJ].nk in called_nk
    assert geoms[LinearFamily.LM_HEAD].nk in called_lm
    assert geoms[LinearFamily.QKV].nk not in called_nk
    assert geoms[LinearFamily.GATE_UP].nk not in called_nk
    assert qwen35[LinearFamily.IN_PROJ_QKVZ].nk not in called_nk
    assert qwen35[LinearFamily.LM_HEAD].nk not in called_lm
    assert geometry_for(LinearFamily.MERGER_FC2, 2).nk not in called_nk


def test_linear_family_warmup_qwen36_27b_skips_9b_and_vision():
    from ixaite.ops.linear_qwen35_shapes import (
        QWEN36_27B_K,
        LinearFamily,
        geometry_for,
        qwen36_27b_geometries,
    )

    class FakeModelRunner:
        max_num_tokens = 256

    worker = _make_worker(FakeModelRunner())
    _set_linear_warmup_config(
        worker,
        hidden_size=QWEN36_27B_K,
        tp_size=2,
        image_limit=0,
        quantization="compressed-tensors",
    )
    called_nk = _record_linear_warmup(worker, [16], include_lm_head=False)
    called_lm = _record_linear_warmup(worker, [1], include_lm_head=True)

    geoms = {geom.family: geom for geom in qwen36_27b_geometries() if geom.tp == 2}
    qwen35_qkv = geometry_for(LinearFamily.QKV, 2)
    qwen35_gate = geometry_for(LinearFamily.GATE_UP, 2)
    qwen35_down = geometry_for(LinearFamily.DOWN_PROJ, 2)
    qwen35_lm = geometry_for(LinearFamily.LM_HEAD, 2)

    assert geoms[LinearFamily.IN_PROJ_QKVZ].nk in called_nk
    assert geoms[LinearFamily.IN_PROJ_BA].nk in called_nk
    assert geoms[LinearFamily.O_PROJ].nk in called_nk
    assert geoms[LinearFamily.LM_HEAD].nk in called_lm
    assert geoms[LinearFamily.QKV].nk not in called_nk
    assert geoms[LinearFamily.GATE_UP].nk not in called_nk
    assert geoms[LinearFamily.DOWN_PROJ].nk not in called_nk
    assert geoms[LinearFamily.MERGER_FC2].nk not in called_nk
    assert qwen35_qkv.nk not in called_nk
    assert qwen35_gate.nk not in called_nk
    assert qwen35_down.nk not in called_nk
    assert qwen35_lm.nk not in called_lm


def test_linear_family_warmup_qwen36_27b_bf16_includes_body():
    from ixaite.ops.linear_qwen35_shapes import (
        QWEN36_27B_K,
        LinearFamily,
        geometry_for,
        qwen36_27b_geometries,
    )

    class FakeModelRunner:
        max_num_tokens = 256

    worker = _make_worker(FakeModelRunner())
    _set_linear_warmup_config(
        worker,
        hidden_size=QWEN36_27B_K,
        tp_size=2,
        image_limit=0,
    )
    called_nk = _record_linear_warmup(worker, [16], include_lm_head=False)

    geoms = {geom.family: geom for geom in qwen36_27b_geometries() if geom.tp == 2}
    qwen35_qkv = geometry_for(LinearFamily.QKV, 2)
    assert geoms[LinearFamily.QKV].nk in called_nk
    assert geoms[LinearFamily.GATE_UP].nk in called_nk
    assert geoms[LinearFamily.DOWN_PROJ].nk in called_nk
    assert geoms[LinearFamily.IN_PROJ_QKVZ].nk in called_nk
    assert geoms[LinearFamily.MERGER_FC2].nk not in called_nk
    assert qwen35_qkv.nk not in called_nk


def test_linear_family_warmup_qwen36_27b_vision_adds_shared_vit():
    from ixaite.ops.linear_qwen35_shapes import (
        QWEN36_27B_K,
        LinearFamily,
        geometry_for,
        qwen36_27b_geometries,
    )

    class FakeModelRunner:
        max_num_tokens = 256

    worker = _make_worker(FakeModelRunner())
    _set_linear_warmup_config(
        worker,
        hidden_size=QWEN36_27B_K,
        tp_size=2,
        image_limit=1,
    )
    called_nk = _record_linear_warmup(worker, [16], include_lm_head=False)

    geoms = {geom.family: geom for geom in qwen36_27b_geometries() if geom.tp == 2}
    vis_qkv = geometry_for(LinearFamily.VIS_QKV, 2)
    vis_fc2 = geometry_for(LinearFamily.VIS_FC2, 2)
    assert vis_qkv.nk in called_nk
    assert vis_fc2.nk in called_nk
    assert geoms[LinearFamily.MERGER_FC2].nk in called_nk
    assert geoms[LinearFamily.IN_PROJ_QKVZ].nk in called_nk
    assert geoms[LinearFamily.QKV].nk in called_nk
    assert geometry_for(LinearFamily.QKV, 2).nk not in called_nk


def test_linear_family_warmup_vl_reads_text_hidden_not_vit():
    from ixaite.ops.linear_qwen35_shapes import (
        QWEN36_27B_K,
        LinearFamily,
        geometry_for,
        qwen36_27b_geometries,
    )

    class FakeModelRunner:
        max_num_tokens = 256

    worker = _make_worker(FakeModelRunner())
    _set_linear_warmup_config(
        worker,
        model_type="qwen3_5",
        hidden_size=QWEN36_27B_K,
        hf_config_hidden_size=1152,
        text_model_type="qwen3_5_text",
        tp_size=2,
        image_limit=1,
    )
    called_nk = _record_linear_warmup(worker, [16], include_lm_head=False)

    geoms = {geom.family: geom for geom in qwen36_27b_geometries() if geom.tp == 2}
    qwen35 = _qwen35_9b_text_nks(2)
    vis_qkv = geometry_for(LinearFamily.VIS_QKV, 2)
    assert geoms[LinearFamily.IN_PROJ_QKVZ].nk in called_nk
    assert geoms[LinearFamily.QKV].nk in called_nk
    assert vis_qkv.nk in called_nk
    assert geoms[LinearFamily.MERGER_FC2].nk in called_nk
    assert qwen35[LinearFamily.IN_PROJ_QKVZ].nk not in called_nk
    assert qwen35[LinearFamily.QKV].nk not in called_nk
    assert geometry_for(LinearFamily.MERGER_FC2, 2).nk not in called_nk


def test_linear_family_warmup_compressed_tensors_without_w8a8_keeps_qkv():
    from ixaite.ops.linear_qwen35_shapes import LinearFamily

    class FakeModelRunner:
        max_num_tokens = 256

    worker = _make_worker(FakeModelRunner())
    _set_linear_warmup_config(
        worker,
        image_limit=0,
        quantization="compressed-tensors",
        quantization_config={"quant_method": "compressed-tensors"},
    )
    called_nk = _record_linear_warmup(worker, [16], include_lm_head=False)
    text = _qwen35_9b_text_nks(1)
    assert text[LinearFamily.QKV].nk in called_nk
    assert text[LinearFamily.GATE_UP].nk in called_nk
    assert text[LinearFamily.IN_PROJ_QKVZ].nk in called_nk


def test_linear_family_warmup_w4a8_compressed_tensors_keeps_qkv():
    from ixaite.ops.linear_qwen35_shapes import LinearFamily

    class FakeModelRunner:
        max_num_tokens = 256

    worker = _make_worker(FakeModelRunner())
    _set_linear_warmup_config(
        worker,
        image_limit=0,
        quantization="compressed-tensors",
        quantization_config={
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "type": "int"},
                    "input_activations": {"num_bits": 8, "type": "int"},
                }
            },
        },
    )
    called_nk = _record_linear_warmup(worker, [16], include_lm_head=False)
    text = _qwen35_9b_text_nks(1)
    assert text[LinearFamily.QKV].nk in called_nk
    assert text[LinearFamily.GATE_UP].nk in called_nk


def test_linear_family_warmup_qwen36_27b_tp1_does_not_fall_back_to_9b():
    from ixaite.ops.linear_qwen35_shapes import QWEN36_27B_K, LinearFamily

    class FakeModelRunner:
        max_num_tokens = 256

    worker = _make_worker(FakeModelRunner())
    _set_linear_warmup_config(
        worker,
        hidden_size=QWEN36_27B_K,
        tp_size=1,
        image_limit=0,
    )
    called_nk = _record_linear_warmup(worker, [16], include_lm_head=False)
    called_lm = _record_linear_warmup(worker, [1], include_lm_head=True)
    qwen35 = _qwen35_9b_text_nks(1)
    assert called_nk == []
    assert called_lm == []
    assert qwen35[LinearFamily.QKV].nk not in called_nk
    assert qwen35[LinearFamily.IN_PROJ_QKVZ].nk not in called_nk
    assert qwen35[LinearFamily.LM_HEAD].nk not in called_lm


def test_memory_probe_covers_ixaite_gdn_conv_and_prep(monkeypatch):
    dummy_calls: list[int] = []
    conv_calls: list[str] = []
    prep_calls: list[str] = []
    released: list[str] = []

    class FakeModelRunner:
        max_num_tokens = 2048

        def _dummy_run(self, num_tokens, **kwargs):
            dummy_calls.append(num_tokens)

    worker = _make_worker(FakeModelRunner())
    _set_qwen35_gdn_config(worker)
    _stub_runner_class_swap(worker, monkeypatch)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: False,
    )

    def _load(op_name, func_name):
        if op_name == "causal_conv1d":
            return object(), "test"
        if op_name == "gdn_post_conv_prep":
            return object(), "test"
        return None, "skip"

    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_python_op",
        _load,
    )
    monkeypatch.setattr(
        worker,
        "_run_ixaite_gdn_conv_fn_warmup",
        lambda _impl: conv_calls.append("fn"),
    )
    monkeypatch.setattr(
        worker,
        "_run_ixaite_gdn_conv_update_warmup",
        lambda _impl: conv_calls.append("update"),
    )
    monkeypatch.setattr(
        worker,
        "_run_ixaite_gdn_post_conv_prep_warmup",
        lambda _impl: prep_calls.append("prep"),
    )
    monkeypatch.setattr(
        worker,
        "_release_ixaite_warmup_scratch",
        lambda: released.append("released"),
    )

    result = worker.determine_available_memory()

    assert result == "community-profile"
    assert released == ["released"]
    assert dummy_calls == []
    assert conv_calls == ["fn", "update"]
    assert prep_calls == ["prep"]


def test_memory_probe_decode_network_when_linear_disabled(monkeypatch):
    dummy_calls: list[int] = []
    released: list[str] = []

    class FakeModelRunner:
        max_num_tokens = 2048

        def _dummy_run(self, num_tokens, **kwargs):
            dummy_calls.append(num_tokens)

    worker = _make_worker(FakeModelRunner())
    _set_model_type(worker, "qwen3_5")
    _stub_runner_class_swap(worker, monkeypatch)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: op_name == "silu_and_mul",
    )
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.load_ixaite_python_op",
        lambda op_name, func_name: (None, "skip"),
    )
    monkeypatch.setattr(
        worker,
        "_release_ixaite_warmup_scratch",
        lambda: released.append("released"),
    )

    result = worker.determine_available_memory()

    assert result == "community-profile"
    assert released == ["released"]
    assert dummy_calls == [1]


def test_compile_or_warm_up_model_does_not_run_crop_dummy(monkeypatch):
    class FakeModelRunner:
        max_num_tokens = 2048

        def _dummy_run(self, num_tokens, **kwargs):
            raise AssertionError("crop dummy must not run after KV reservation")

    worker = _make_worker(FakeModelRunner())
    _set_model_type(worker, "qwen3_5")
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: True,
    )

    assert worker.compile_or_warm_up_model() == "community-warmup"


@pytest.mark.parametrize("model_type", ["qwen2", "llama"])
def test_memory_probe_skips_other_models(monkeypatch, model_type):
    released = Mock()

    class FakeModelRunner:
        max_num_tokens = 2048

        def _dummy_run(self, num_tokens, **kwargs):
            raise AssertionError("unexpected ixaite prefill warmup")

    worker = _make_worker(FakeModelRunner())
    _set_model_type(worker, model_type)
    _stub_runner_class_swap(worker, monkeypatch)
    enabled = Mock()
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        enabled,
    )
    monkeypatch.setattr(worker, "_release_ixaite_warmup_scratch", released)

    assert worker.determine_available_memory() == "community-profile"
    enabled.assert_not_called()
    released.assert_not_called()


def test_memory_probe_skips_ixformer_linear(monkeypatch):
    released = Mock()

    class FakeModelRunner:
        max_num_tokens = 2048

        def _dummy_run(self, num_tokens, **kwargs):
            raise AssertionError("unexpected ixaite prefill warmup")

    worker = _make_worker(FakeModelRunner())
    _set_model_type(worker, "qwen3_5")
    _stub_runner_class_swap(worker, monkeypatch)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.runtime.is_ixaite_op_enabled",
        lambda op_name: False,
    )
    monkeypatch.setattr(worker, "_release_ixaite_warmup_scratch", released)

    assert worker.determine_available_memory() == "community-profile"
    released.assert_not_called()


def test_compile_or_warm_up_model_has_no_ixaite_warmup_calls():
    worker_source = _GPU_WORKER_SRC.parents[2] / "backends" / "ixaite" / "worker.py"
    source = worker_source.read_text(encoding="utf-8")
    tree = ast.parse(source)
    warmup_fn = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "IxaiteWorker":
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "compile_or_warm_up_model"
                ):
                    warmup_fn = item
                    break
    assert warmup_fn is not None
    called = {
        child.func.attr
        for child in ast.walk(warmup_fn)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }
    called.update(
        child.func.id
        for child in ast.walk(warmup_fn)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    )
    leaked = sorted(called & _FORBIDDEN_WARMUP_CALLS)
    assert not leaked, (
        "IxaiteWorker.compile_or_warm_up_model must not run ixaite JIT dummy; "
        f"move {leaked} to _warm_up_ixaite_before_kv"
    )


def test_release_returns_allocator_blocks(monkeypatch):
    collect = Mock()
    sync = Mock()
    empty = Mock()
    fake_cuda = SimpleNamespace(synchronize=sync, empty_cache=empty)
    fake_torch = SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setattr("gc.collect", collect)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    worker = _make_worker(SimpleNamespace())
    worker._release_ixaite_warmup_scratch()

    collect.assert_called_once()
    sync.assert_called_once()
    empty.assert_called_once()


def test_release_clears_ixaite_workspace_caches(monkeypatch):
    collect = Mock()
    sync = Mock()
    empty = Mock()
    fake_cuda = SimpleNamespace(synchronize=sync, empty_cache=empty)
    fake_torch = SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setattr("gc.collect", collect)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    staging_a = {("dev",): object()}
    ws = {("k",): {"buf": object()}}
    monkeypatch.setitem(
        sys.modules,
        "ixaite.ops.linear",
        SimpleNamespace(_STAGING_A=staging_a, _STAGING_W_N={}, _STAGING_C_N={}),
    )
    monkeypatch.setitem(
        sys.modules,
        "ixaite.ops.gdn_chunk_wy",
        SimpleNamespace(_WS_CACHE=ws),
    )

    worker = _make_worker(SimpleNamespace())
    worker._release_ixaite_warmup_scratch()

    assert staging_a == {}
    assert ws == {}
    collect.assert_called_once()
    sync.assert_called_once()
    empty.assert_called_once()


@pytest.mark.parametrize("fail", [False, True])
def test_ixaite_warmup_restores_block_kernel_hooks(monkeypatch, fail):
    from vllm_iluvatar.v1.worker.gpu_worker import IluWorker

    warmup = sys.modules["vllm.model_executor.warmup"]
    modules = (warmup.kernel_warmup, warmup.v1_block_table_warmup)
    originals = [module.warm_v1_block_table_kernels for module in modules]

    def compile_model(self):
        assert all(
            module.warm_v1_block_table_kernels(self) is None for module in modules
        )
        if fail:
            raise RuntimeError("compile failed")
        return "compiled"

    monkeypatch.setattr(IluWorker, "compile_or_warm_up_model", compile_model)
    worker = _make_worker(SimpleNamespace())
    if fail:
        with pytest.raises(RuntimeError, match="compile failed"):
            worker.compile_or_warm_up_model()
    else:
        assert worker.compile_or_warm_up_model() == "compiled"
    assert [module.warm_v1_block_table_kernels for module in modules] == originals
