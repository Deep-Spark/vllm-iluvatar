# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from conftest import requires_cuda
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm_iluvatar.models.qwen4_exp import ple as ple_layer_module
from vllm_iluvatar.models.qwen4_exp.ple import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpNGramTable,
    Qwen4ExpPLELayer,
    _ple_embedding_quant_kind,
)
from vllm_iluvatar.models.qwen4_exp.ple_utils import (
    PLEShardOverlap,
    compute_ple_shard_overlap,
    copy_ple_embedding_shard_,
)


def _single_rank(monkeypatch: pytest.MonkeyPatch, rank: int = 0, size: int = 1) -> None:
    monkeypatch.setattr(
        ple_layer_module, "get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(
        ple_layer_module, "get_tensor_model_parallel_world_size", lambda: size
    )


def _make_ngram_embedding_for_load_test() -> Qwen4ExpNGramEmbedding:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.split_ngram_parts = 2
    module.register_buffer("layer_multipliers", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_offsets", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_vocab_sizes", torch.zeros(1, dtype=torch.long))
    module.ngram_embedding = SimpleNamespace(
        org_vocab_size=8,
        embedding_dim=2,
        weight=nn.Parameter(torch.full((4, 2), -1.0)),
        weight_scale=None,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=2,
            org_vocab_end_index=6,
        ),
    )
    return module


def test_ple_shard_overlap_and_copy() -> None:
    overlap = compute_ple_shard_overlap(
        checkpoint_start=2, checkpoint_rows=5, tp_start=4, tp_end=8
    )
    assert overlap == PLEShardOverlap(source_start=2, destination_start=0, row_count=3)

    destination = torch.full((4, 2), -1.0)
    loaded = torch.arange(10, dtype=torch.float64).reshape(5, 2)
    copied = copy_ple_embedding_shard_(
        destination,
        loaded,
        checkpoint_start=2,
        tp_start=4,
        tp_end=8,
    )

    assert copied == 3
    torch.testing.assert_close(destination[:3], loaded[2:5].float())
    torch.testing.assert_close(destination[3], torch.tensor([-1.0, -1.0]))


def test_ple_shard_copy_is_a_noop_without_overlap() -> None:
    destination = torch.ones(4, 2)
    copied = copy_ple_embedding_shard_(
        destination,
        torch.zeros(2, 2),
        checkpoint_start=10,
        tp_start=4,
        tp_end=8,
    )

    assert copied == 0
    assert torch.equal(destination, torch.ones_like(destination))


def test_ple_embedding_tp_shard_copy_covers_full_range() -> None:
    destination = torch.zeros(8, 2)
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    copied_0 = copy_ple_embedding_shard_(
        destination, shard_0, checkpoint_start=0, tp_start=0, tp_end=8
    )
    copied_1 = copy_ple_embedding_shard_(
        destination, shard_1, checkpoint_start=4, tp_start=0, tp_end=8
    )
    assert copied_0 == 4
    assert copied_1 == 4
    torch.testing.assert_close(destination, torch.cat((shard_0, shard_1)))


def test_ngram_embedding_loads_shards_and_ignores_legacy_token_lookup() -> None:
    module = _make_ngram_embedding_for_load_test()
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)

    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
            ("token_lookup", torch.tensor([2, 1, 0])),
        ]
    )

    assert loaded == {"ngram_embedding.weight"}
    torch.testing.assert_close(
        module.ngram_embedding.weight,
        torch.cat((shard_0[2:4], shard_1[0:2])),
    )


def test_ngram_embedding_rejects_mismatched_checkpoint_shard() -> None:
    module = _make_ngram_embedding_for_load_test()

    with pytest.raises(
        ValueError,
        match=r"Shape mismatch for PLE embedding shard 0",
    ):
        module.load_weights([("ngram_embedding.shard_0.weight", torch.zeros(3, 2))])


def test_ngram_table_rows_are_split_evenly_across_tp_ranks(monkeypatch) -> None:
    shards = []
    for rank in range(4):
        _single_rank(monkeypatch, rank=rank, size=4)
        table = Qwen4ExpNGramTable(
            1024, 2, quant_kind=None, params_dtype=torch.float32, divisible_by=128
        )
        shards.append(table.shard_indices)
        assert table.weight.shape == (256, 2)

    assert shards[0].org_vocab_start_index == 0
    assert [s.org_vocab_end_index for s in shards] == [256, 512, 768, 1024]
    assert [s.org_vocab_start_index for s in shards[1:]] == [256, 512, 768]


def test_ngram_table_dequantizes_int8_rows(monkeypatch) -> None:
    _single_rank(monkeypatch)
    table = Qwen4ExpNGramTable(
        3, 2, quant_kind="int8", params_dtype=torch.bfloat16, divisible_by=1
    )
    table.weight.data.copy_(torch.tensor([[1, 2], [4, 8], [16, 32]], dtype=torch.int8))
    table.weight_scale.data.copy_(
        torch.tensor([[0.5], [0.25], [0.125]], dtype=torch.bfloat16)
    )

    output = table(torch.tensor([2, 0]))

    assert output.dtype == torch.bfloat16
    expected = torch.tensor([[16.0, 32.0], [1.0, 2.0]]) * torch.tensor([[0.125], [0.5]])
    torch.testing.assert_close(output.float(), expected)


def test_ngram_table_masks_foreign_ids_and_sums_across_ranks(monkeypatch) -> None:
    """Each id lives on exactly one rank, so the all-reduce rebuilds the table."""
    _single_rank(monkeypatch, rank=1, size=2)
    reduced = []
    monkeypatch.setattr(
        ple_layer_module,
        "tensor_model_parallel_all_reduce",
        lambda t: reduced.append(t) or t,
    )
    table = Qwen4ExpNGramTable(
        4, 2, quant_kind=None, params_dtype=torch.float32, divisible_by=2
    )
    table.weight.data.copy_(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))  # global rows 2, 3

    output = table(torch.tensor([0, 2, 3]))

    # Rank 1 owns rows 2 and 3; row 0 belongs to rank 0 and must read as zero.
    torch.testing.assert_close(
        output, torch.tensor([[0.0, 0.0], [5.0, 6.0], [7.0, 8.0]])
    )
    assert reduced, "TP lookups must be summed across ranks"
    assert reduced[0].dtype == torch.float32


def test_ngram_table_quant_kind_follows_the_checkpoint() -> None:
    prefix = "model.layers.1.ple.ple_embedding.ngram_embedding"
    fp8 = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        ignored_layers=[],
        weight_block_size=[128, 128],
    )
    assert _ple_embedding_quant_kind(fp8, prefix) == "fp8"
    fp8.ignored_layers = [f"{prefix}.shard_0"]
    assert _ple_embedding_quant_kind(fp8, prefix) is None

    compressed = SimpleNamespace(
        get_name=lambda: "compressed-tensors", ignore=[], packed_modules_mapping={}
    )
    assert _ple_embedding_quant_kind(compressed, prefix) == "int8"
    compressed.ignore = [f"{prefix}.shard_0"]
    assert _ple_embedding_quant_kind(compressed, prefix) is None

    assert _ple_embedding_quant_kind(None, prefix) is None


def _make_conv_layer(channels: int, kernel: int, dilation: int) -> Qwen4ExpPLELayer:
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.hc_hidden_size = channels
    module.conv_kernel_size = kernel
    module.short_conv_dilation = dilation
    module.conv_state_len = (kernel - 1) * dilation
    module.conv1d = nn.Conv1d(
        channels, channels, kernel, groups=channels, dilation=dilation, bias=False
    )
    with torch.no_grad():
        module.conv1d.weight.copy_(
            torch.linspace(-1.0, 1.0, channels * kernel).reshape(channels, 1, kernel)
        )
    return module


def _dilated_short_conv_reference(
    module: Qwen4ExpPLELayer,
    inputs: torch.Tensor,
    conv_state: torch.Tensor | None,
    state_indices: torch.Tensor | None,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Test-only torch oracle for the production ixformer CUDA operator."""
    weight = module.conv1d.weight.to(dtype=inputs.dtype)
    channels = inputs.shape[-1]
    state_len = module.conv_state_len
    starts = query_start_loc.to(torch.int64)
    carries = conv_state
    if carries is not None and state_indices is not None:
        state_indices = state_indices.to(torch.int64)
    else:
        carries, state_indices = None, None

    outputs = []
    new_states = []
    for req in range(starts.numel() - 1):
        lo = int(starts[req])
        hi = int(starts[req + 1])
        if hi <= lo:
            if carries is not None:
                new_states.append(carries[state_indices[req]])
            continue
        tokens = inputs[lo:hi].transpose(0, 1).unsqueeze(0)
        if state_len:
            carry = (
                carries[state_indices[req]].unsqueeze(0)
                if carries is not None
                else tokens.new_zeros((1, channels, state_len))
            )
            history = torch.cat([carry.to(tokens.dtype), tokens], dim=-1)
        else:
            history = tokens
        conv = F.conv1d(
            history,
            weight,
            groups=channels,
            dilation=module.short_conv_dilation,
        )
        outputs.append(F.silu(conv).squeeze(0).transpose(0, 1))
        if state_len:
            new_states.append(history[0, :, -state_len:])
    result = torch.cat(outputs, dim=0) if outputs else inputs.new_zeros((0, channels))
    if carries is not None and new_states:
        carries.index_copy_(
            0,
            state_indices,
            torch.stack(new_states, dim=0).to(carries.dtype),
        )
    return result


def test_dilated_conv_output_is_independent_of_prefill_chunking() -> None:
    """The carry must make N chunks equal one shot -- silent drift otherwise.

    Writing the roll forward wrong does not raise; it only makes long prompts
    diverge slowly, so this equivalence is the real acceptance check for the
    ModelState-owned conv state.
    """
    channels, tokens = 3, 14
    module = _make_conv_layer(channels, kernel=4, dilation=3)
    inputs = (
        torch.arange(tokens * channels, dtype=torch.float32).reshape(tokens, channels)
        / tokens
    )

    one_shot_state = torch.zeros(2, channels, module.conv_state_len)
    one_shot = _dilated_short_conv_reference(
        module,
        inputs,
        one_shot_state,
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0, tokens]),
    )

    chunked_state = torch.zeros(2, channels, module.conv_state_len)
    pieces = []
    for lo, hi in ((0, 5), (5, 10), (10, tokens)):
        pieces.append(
            _dilated_short_conv_reference(
                module,
                inputs[lo:hi],
                chunked_state,
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([0, hi - lo]),
            )
        )

    torch.testing.assert_close(torch.cat(pieces), one_shot)
    torch.testing.assert_close(chunked_state, one_shot_state)
    # The carry is exactly the trailing window the next chunk needs.
    torch.testing.assert_close(
        one_shot_state[1], inputs[-module.conv_state_len :].transpose(0, 1)
    )
    assert torch.count_nonzero(one_shot_state[0]) == 0


def test_dilated_conv_matches_a_zero_padded_full_sequence_convolution() -> None:
    channels, tokens = 3, 11
    module = _make_conv_layer(channels, kernel=4, dilation=3)
    inputs = torch.randn(tokens, channels)

    output = _dilated_short_conv_reference(
        module,
        inputs,
        torch.zeros(1, channels, module.conv_state_len),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0, tokens]),
    )

    padded = F.pad(inputs.transpose(0, 1).unsqueeze(0), (module.conv_state_len, 0))
    expected = F.silu(
        F.conv1d(
            padded,
            module.conv1d.weight,
            groups=channels,
            dilation=module.short_conv_dilation,
        )
    )
    torch.testing.assert_close(output, expected.squeeze(0).transpose(0, 1))


def test_dilated_conv_writes_each_request_to_its_own_state_slot() -> None:
    channels = 2
    module = _make_conv_layer(channels, kernel=2, dilation=1)
    inputs = torch.arange(10, dtype=torch.float32).reshape(5, channels)
    conv_state = torch.zeros(4, channels, module.conv_state_len)

    # Batch rows 0 and 1 map to state slots 3 and 0; slots 1 and 2 stay untouched.
    _dilated_short_conv_reference(
        module,
        inputs,
        conv_state,
        torch.tensor([3, 0], dtype=torch.int32),
        torch.tensor([0, 2, 5]),
    )

    torch.testing.assert_close(conv_state[3], inputs[1:2].transpose(0, 1))
    torch.testing.assert_close(conv_state[0], inputs[4:5].transpose(0, 1))
    assert torch.count_nonzero(conv_state[1]) == 0
    assert torch.count_nonzero(conv_state[2]) == 0


def test_dilated_conv_resumes_from_a_block_snapshot() -> None:
    """A copied block snapshot must be indistinguishable from one-shot state."""

    channels, boundary, tokens = 4, 12, 29
    module = _make_conv_layer(channels, kernel=4, dilation=3)
    inputs = torch.randn(tokens, channels)

    one_shot_state = torch.zeros(16, channels, module.conv_state_len)
    one_shot = _dilated_short_conv_reference(
        module,
        inputs,
        one_shot_state,
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([0, tokens]),
    )

    paged_state = torch.zeros_like(one_shot_state)
    first = _dilated_short_conv_reference(
        module,
        inputs[:boundary],
        paged_state,
        torch.tensor([7], dtype=torch.int32),
        torch.tensor([0, boundary]),
    )
    paged_state[13].copy_(paged_state[7])
    second = _dilated_short_conv_reference(
        module,
        inputs[boundary:],
        paged_state,
        torch.tensor([13], dtype=torch.int32),
        torch.tensor([0, tokens - boundary]),
    )

    torch.testing.assert_close(torch.cat((first, second)), one_shot)
    torch.testing.assert_close(paged_state[13], one_shot_state[3])


def test_resolve_conv_state_indexes_the_assigned_gdn_group() -> None:
    module = _make_conv_layer(channels=4, kernel=4, dilation=3)
    host_window = torch.randn(17, 4, module.conv_state_len)
    module.state_host = SimpleNamespace(ple_conv_window=host_window)
    module.state_group_row = 1
    group_indices = torch.tensor([[2, 4], [9, 11]], dtype=torch.int32)

    state, indices = module._resolve_conv_state(None, group_indices)

    assert state is host_window
    assert torch.equal(indices, torch.tensor([9, 11], dtype=torch.int32))


@requires_cuda
def test_cuda_dilated_conv_accepts_a_padded_kv_page_stride() -> None:
    channels, state_len, num_blocks, padding = 256, 9, 5, 128
    module = _make_conv_layer(channels, kernel=4, dilation=3).cuda().bfloat16()
    storage = torch.randn(
        num_blocks,
        channels * state_len + padding,
        device="cuda",
        dtype=torch.bfloat16,
    )
    paged_state = torch.as_strided(
        storage,
        (num_blocks, channels, state_len),
        (channels * state_len + padding, state_len, 1),
    )
    dense_state = paged_state.clone(memory_format=torch.contiguous_format)
    conv_in = torch.randn(11, channels, device="cuda", dtype=torch.bfloat16)
    gated = torch.randn_like(conv_in)
    indices = torch.tensor([3], device="cuda", dtype=torch.int32)
    query_start_loc = torch.tensor([0, 11], device="cuda", dtype=torch.int32)

    expected = gated + _dilated_short_conv_reference(
        module, conv_in, dense_state, indices, query_start_loc
    )
    actual = module._cuda_dilated_conv(
        conv_in, gated, paged_state, indices, query_start_loc
    )

    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(paged_state, dense_state, atol=0, rtol=0)


def test_dilated_conv_runs_on_a_zero_carry_when_no_state_is_supplied() -> None:
    """Profile and dummy runs pass no state; they must not touch live slots."""
    channels = 2
    module = _make_conv_layer(channels, kernel=2, dilation=1)
    inputs = torch.randn(4, channels)

    stateless = _dilated_short_conv_reference(
        module, inputs, None, None, torch.tensor([0, 4])
    )
    with_zero_state = _dilated_short_conv_reference(
        module,
        inputs,
        torch.zeros(1, channels, module.conv_state_len),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0, 4]),
    )

    torch.testing.assert_close(stateless, with_zero_state)


def test_splitmix64_and_ngram_vocab_primes_are_deterministic() -> None:
    from vllm_iluvatar.models.qwen4_exp.ple import _nth_prime_after, _splitmix64

    assert _splitmix64(0) == _splitmix64(0)
    assert _splitmix64(1) != _splitmix64(2)
    first = _nth_prime_after(20_000 - 1, 1)
    second = _nth_prime_after(20_000 - 1, 2)
    assert first > 19_999
    assert second > first
