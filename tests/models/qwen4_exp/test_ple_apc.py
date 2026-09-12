# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PLE window declarations and GDN-page prefix-cache plumbing."""

from math import prod
from pathlib import Path
from types import SimpleNamespace

import torch
import vllm
from torch import nn

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm_iluvatar.models.qwen4_exp.ple_utils import (
    ple_conv_window_state_shape,
)
from vllm_iluvatar.models.qwen4_exp.model import (
    Qwen4ExpForCausalLM,
    _install_ple_window_states,
)
from vllm_iluvatar.models.qwen4_exp.model_state import (
    Qwen4ExpModelState,
)
from vllm_iluvatar.models.qwen4_exp.ple import Qwen4ExpPLELayer

BLOCK_SIZE = 1024
PAGE_SIZE = 1_048_576
GDN_BYTES = 801_792
PLE_BYTES = 184_320


def _vllm_config(*, ple_layer_ids=(2,)) -> SimpleNamespace:
    text_config = SimpleNamespace(
        ple_layer_ids=list(ple_layer_ids),
        hidden_size=2560,
        hc_count=4,
        ple_conv_kernel_size=4,
        ngram_size=3,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        mamba_ssm_dtype="float32",
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=text_config,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
        ),
        speculative_config=None,
    )


def _state_nbytes(shapes, dtypes) -> int:
    return sum(
        prod(shape) * torch.empty((), dtype=dtype).element_size()
        for shape, dtype in zip(shapes, dtypes)
    )


def test_ple_window_fits_the_gdn_page_padding() -> None:
    config = _vllm_config()
    shapes = Qwen4ExpForCausalLM.get_mamba_state_shape_from_config(config)
    dtypes = Qwen4ExpForCausalLM.get_mamba_state_dtype_from_config(config)

    assert ple_conv_window_state_shape(config) == (10240, 9)
    assert _state_nbytes(shapes[:-1], dtypes[:-1]) == GDN_BYTES
    assert _state_nbytes(shapes[-1:], dtypes[-1:]) == PLE_BYTES
    assert _state_nbytes(shapes, dtypes) == 986_112
    assert _state_nbytes(shapes, dtypes) <= PAGE_SIZE


def test_mamba_spec_is_byte_identical_without_ple() -> None:
    config = _vllm_config(ple_layer_ids=())
    shapes = Qwen4ExpForCausalLM.get_mamba_state_shape_from_config(config)
    dtypes = Qwen4ExpForCausalLM.get_mamba_state_dtype_from_config(config)

    assert len(shapes) == len(dtypes) == 2
    assert _state_nbytes(shapes, dtypes) == GDN_BYTES


def test_copy_funcs_track_the_state_shapes() -> None:
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    nn.Module.__init__(model)
    model.vllm_config = _vllm_config()

    funcs = model.get_mamba_state_copy_func()

    assert len(funcs) == 3
    assert funcs[0] is get_conv_copy_spec
    assert funcs[2] is get_temporal_copy_spec


def test_mamba_copy_funcs_are_only_called_on_model_instances() -> None:
    worker_root = Path(vllm.__file__).resolve().parent / "v1" / "worker"
    call_lines = []
    for path in worker_root.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "get_mamba_state_copy_func()" in line and not line.lstrip().startswith(
                "def "
            ):
                call_lines.append(line.strip())

    assert len(call_lines) == 5
    assert all("self.model.get_mamba_state_copy_func()" in line for line in call_lines)


class _FakeGDN(QwenGatedDeltaNetAttention):
    def get_state_shape(self):
        return ((2, 3), (4, 5, 6))

    def get_state_dtype(self):
        return (torch.bfloat16, torch.float32)

    def _warmup_prefill_kernels(self, *_args, **_kwargs):
        self.warmup_dtypes = self.get_state_dtype()


def test_gdn_class_swap_keeps_the_existing_implementation_and_weights() -> None:
    gdn = _FakeGDN.__new__(_FakeGDN)
    nn.Module.__init__(gdn)
    gdn.layer_idx = 0
    gdn.weight = nn.Parameter(torch.randn(3))
    weight_ptr = gdn.weight.data_ptr()

    ple = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(ple)
    ple.layer_idx = 1

    container = nn.Module()
    container.gdn = gdn
    container.ple = ple
    _install_ple_window_states(container, _vllm_config())

    assert isinstance(gdn, _FakeGDN)
    assert gdn.weight.data_ptr() == weight_ptr
    assert gdn.get_state_shape()[-1] == (10240, 9)
    assert gdn.get_state_dtype()[-1] is torch.bfloat16
    assert gdn.hosts_ple_window
    assert ple.state_host is gdn
    gdn._warmup_prefill_kernels(None, 0)
    assert gdn.warmup_dtypes == (torch.bfloat16, torch.float32)
    assert len(gdn.get_state_dtype()) == 3

    raw_pages = torch.empty((2, 1, 1, PAGE_SIZE), dtype=torch.uint8)
    gdn.bind_kv_cache(raw_pages)
    assert gdn.ple_conv_window.shape == (2, 10240, 9)
    assert gdn.ple_conv_window.stride() == (PAGE_SIZE // 2, 9, 1)


def test_ple_state_indices_match_each_gdn_group_column() -> None:
    state = Qwen4ExpModelState.__new__(Qwen4ExpModelState)
    state.cache_config = SimpleNamespace(mamba_cache_mode="align")
    ple_a = SimpleNamespace(
        state_group_row=None,
        state_host=SimpleNamespace(prefix="gdn.1"),
    )
    ple_b = SimpleNamespace(
        state_group_row=None,
        state_host=SimpleNamespace(prefix="gdn.0"),
    )
    state.ple_layers = (ple_a, ple_b)
    state._ple_group_ids = None
    state._ple_idx_buf = torch.full((2, 2), -1, dtype=torch.int32)

    mamba_spec = MambaSpec(
        block_size=64,
        shapes=((4, 3), (2, 8, 8), (16, 9)),
        dtypes=(torch.bfloat16, torch.float32, torch.bfloat16),
        mamba_cache_mode="align",
    )
    full_spec = FullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
    )
    groups = [
        KVCacheGroupSpec(["attn.0"], full_spec),
        KVCacheGroupSpec(["gdn.0", "gdn.3"], mamba_spec),
        KVCacheGroupSpec(["gdn.1", "gdn.4"], mamba_spec),
    ]
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )
    block_tables = (
        torch.zeros((2, 4), dtype=torch.int32),
        torch.tensor([[10, 11, 12, 13], [14, 15, 16, 17]], dtype=torch.int32),
        torch.tensor([[20, 21, 22, 23], [24, 25, 26, 27]], dtype=torch.int32),
    )
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=2,
        seq_lens=torch.tensor([65, 130], dtype=torch.int32),
    )

    indices = state._gather_ple_state_indices(
        input_batch, block_tables, kv_cache_config
    )

    assert ple_a.state_group_row == 0
    assert ple_b.state_group_row == 1
    assert torch.equal(indices, torch.tensor([[21, 26], [11, 16]], dtype=torch.int32))


def test_profile_inputs_are_stateless_before_kv_cache_is_ready(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        MambaHybridModelState,
        "prepare_inputs",
        lambda self, input_batch, req_states: {},
    )
    state = Qwen4ExpModelState.__new__(Qwen4ExpModelState)
    state.ple = object()
    state._ple_state_indices_ready = False
    state.ngram_ctx_len = 2
    state.ngram_eos_token_id = 42
    state.device = torch.device("cpu")
    state._state_indices_buf = torch.full((2,), -1, dtype=torch.int32)
    state._query_start_loc_buf = torch.zeros(3, dtype=torch.int32)
    state._ngram_ctx_buf = torch.full((2, 2), 42, dtype=torch.int32)
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=2,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 7], dtype=torch.int32),
    )

    model_inputs = state.prepare_inputs(input_batch, SimpleNamespace())

    assert model_inputs["state_indices"] is None
    assert model_inputs["ple_state_indices"] is None
    assert model_inputs["ple_conv_state"] is None
    assert torch.equal(
        model_inputs["ngram_context"],
        torch.tensor([[42, 42], [42, 42]], dtype=torch.int32),
    )


def test_dummy_graph_inputs_use_persistent_non_none_buffers(monkeypatch) -> None:
    monkeypatch.setattr(
        MambaHybridModelState,
        "prepare_dummy_inputs",
        lambda self, num_reqs, num_tokens: {},
    )
    state = Qwen4ExpModelState.__new__(Qwen4ExpModelState)
    state.ple = object()
    state.ngram_ctx_len = 2
    state.ngram_eos_token_id = 42
    state.device = torch.device("cpu")
    state._state_indices_buf = torch.empty(4, dtype=torch.int32)
    state._query_start_loc_buf = torch.empty(5, dtype=torch.int32)
    state._ngram_ctx_buf = torch.empty(4, 2, dtype=torch.int32)
    state._ple_idx_buf = torch.empty(2, 4, dtype=torch.int32)

    first = state.prepare_dummy_inputs(4, 4)
    pointers = {
        name: tensor.data_ptr()
        for name, tensor in first.items()
        if isinstance(tensor, torch.Tensor)
    }
    second = state.prepare_dummy_inputs(3, 3)

    assert second["state_indices"] is not None
    assert second["ple_state_indices"] is not None
    assert second["query_start_loc"].tolist() == [0, 1, 2, 3]
    assert second["state_indices"].tolist() == [-1, -1, -1]
    assert second["ple_state_indices"].tolist() == [[0, 0, 0], [0, 0, 0]]
    for name in ("state_indices", "query_start_loc", "ngram_context"):
        assert second[name].data_ptr() == pointers[name]


def test_live_graph_inputs_refresh_padded_tails_in_place(monkeypatch) -> None:
    monkeypatch.setattr(
        MambaHybridModelState,
        "prepare_inputs",
        lambda self, input_batch, req_states: {},
    )
    state = Qwen4ExpModelState.__new__(Qwen4ExpModelState)
    state.ple = object()
    state._ple_state_indices_ready = True
    state.ngram_ctx_len = 2
    state.ngram_eos_token_id = 42
    state._state_indices_buf = torch.full((4,), 99, dtype=torch.int32)
    state._query_start_loc_buf = torch.full((5,), 99, dtype=torch.int32)
    state._ngram_ctx_buf = torch.full((4, 2), 99, dtype=torch.int32)
    state._ple_idx_buf = torch.tensor([[10, 11, 12, 0]], dtype=torch.int32)
    state._build_ngram_context = lambda state_indices, req_states, out: out.fill_(7)
    input_batch = SimpleNamespace(
        num_reqs=3,
        num_reqs_after_padding=4,
        idx_mapping=torch.tensor([5, 6, 7], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32),
    )

    model_inputs = state.prepare_inputs(input_batch, SimpleNamespace())

    assert model_inputs["state_indices"].tolist() == [5, 6, 7, -1]
    assert model_inputs["query_start_loc"].tolist() == [0, 1, 2, 3, 3]
    assert model_inputs["ngram_context"].tolist() == [
        [7, 7],
        [7, 7],
        [7, 7],
        [42, 42],
    ]
    assert model_inputs["ple_state_indices"].tolist() == [[10, 11, 12, 0]]
    assert not state._ple_state_indices_ready
