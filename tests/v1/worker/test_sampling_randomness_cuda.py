"""Native sampling correctness against a CPU integer RNG and categorical oracle.

Do not use the old CoreX Triton output as the oracle: sharing a random key
between 64 vocabulary items reproduces its defect instead of sampling p(x).
"""

import numpy as np
import pytest
import torch
import vllm_iluvatar._C  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _rotl(x, n):
    return (x << np.uint32(n)) | (x >> np.uint32(32 - n))


def _hash(seeds, positions, vocab):
    seeds = np.asarray(seeds, dtype=np.int64).view(np.uint64)[:, None]
    positions = np.asarray(positions, dtype=np.int64).view(np.uint64)[:, None]
    with np.errstate(over="ignore"):
        h = np.zeros((len(seeds), 1), dtype=np.uint32)
        for key in [
            seeds.astype(np.uint32),
            (seeds >> np.uint64(32)).astype(np.uint32),
            positions.astype(np.uint32),
            np.arange(vocab, dtype=np.uint32)[None, :],
        ]:
            k = _rotl(key * np.uint32(0xCC9E2D51), 15) * np.uint32(0x1B873593)
            h = _rotl(h ^ k, 13) * np.uint32(5) + np.uint32(0xE6546B64)
        h ^= np.uint32(16)
        h = (h ^ (h >> np.uint32(16))) * np.uint32(0x85EBCA6B)
        h = (h ^ (h >> np.uint32(13))) * np.uint32(0xC2B2AE35)
        return h ^ (h >> np.uint32(16))


def _reference(scores, seeds, positions, temperatures=None):
    """FP64 CPU arithmetic, preserving the FP32 sampler's 1-u convention."""
    scores = np.asarray(scores, dtype=np.float64)
    u = (_hash(seeds, positions, scores.shape[1]).astype(np.float64) + 0.5) * 2**-32
    noise = -np.log(-np.log1p(-u))
    if temperatures is not None:
        noise = np.where(np.asarray(temperatures)[:, None] != 0, noise, 0.0)
    return np.where(np.isnan(scores + noise), -np.inf, scores + noise).argmax(axis=1)


def _buffers(rows, vocab):
    blocks = (vocab + 1023) // 1024
    ids = torch.full((rows, blocks + 2), -177, dtype=torch.int64, device="cuda")
    maxima = torch.full((rows, blocks + 2), 12345.0, dtype=torch.float32, device="cuda")
    return ids, maxima, ids[:, 1:-1], maxima[:, 1:-1]


def _assert_guards(ids, maxima):
    assert torch.all(ids[:, [0, -1]] == -177)
    assert torch.all(maxima[:, [0, -1]] == 12345.0)


def _sampled(ids, maxima):
    return ids.gather(1, maxima.argmax(1, keepdim=True)).view(-1).cpu().numpy()


def _gumbel(
    ids,
    maxima,
    logits,
    mapping,
    seeds,
    positions,
    temperatures,
    *,
    drafting=False,
    cache=None,
    columns=None,
    fp64=False,
):
    torch.ops._C.triton_gumbel_sample(
        ids,
        ids.stride(0),
        maxima,
        maxima.stride(0),
        cache,
        cache.stride(0) if cache is not None else 0,
        cache.stride(1) if cache is not None else 0,
        columns,
        logits,
        logits.stride(0),
        mapping,
        seeds,
        positions,
        temperatures,
        logits.shape[1],
        1024,
        drafting,
        True,
        fp64,
        columns is not None and columns.ndim > 0,
    )


@pytest.mark.parametrize("vocab", [32, 64, 128, 129, 1025])
def test_uniform_gumbel_has_per_token_noise(vocab):
    rows = 8192 if vocab <= 129 else 512
    logits = torch.zeros((rows, vocab), device="cuda")
    seeds = torch.arange(rows, dtype=torch.int64, device="cuda")
    positions = torch.zeros_like(seeds)
    mapping = torch.arange(rows, dtype=torch.int32, device="cuda")
    temperatures = torch.ones(rows, device="cuda")
    all_ids, all_max, ids, maxima = _buffers(rows, vocab)
    _gumbel(ids, maxima, logits, mapping, seeds, positions, temperatures)
    got = _sampled(ids, maxima)
    expected = _hash(np.arange(rows), np.zeros(rows, dtype=np.int64), vocab).argmin(1)
    np.testing.assert_array_equal(got, expected)
    if vocab <= 129:
        assert np.unique(got).size == vocab
    _gumbel(ids, maxima, logits, mapping, seeds, positions, temperatures)
    np.testing.assert_array_equal(_sampled(ids, maxima), got)
    _assert_guards(all_ids, all_max)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("drafting", [False, True])
@pytest.mark.parametrize("per_token_col", [False, True])
def test_gumbel_strides_mapping_cache_and_temperature(dtype, drafting, per_token_col):
    rows, vocab = 32, 1025
    rng = np.random.default_rng(7261)
    source = torch.from_numpy(rng.normal(size=(rows, vocab)).astype(np.float32)).to(
        dtype
    )
    storage = torch.full((rows, vocab + 11), float("nan"), device="cuda", dtype=dtype)
    logits = storage[:, :vocab]
    logits.copy_(source)
    mapping_cpu = rng.permutation(rows).astype(np.int64)
    seed_cpu = np.arange(rows, dtype=np.int64) * 91873282931 - 123415876789
    pos_cpu = np.arange(rows, dtype=np.int64) * 1267 + 9
    temp_cpu = np.asarray([0.0, 0.5, 1.0, 1.75] * (rows // 4), dtype=np.float32)
    mapping = torch.tensor(mapping_cpu, device="cuda", dtype=torch.int64)
    seeds = torch.tensor(seed_cpu, device="cuda")
    positions = torch.tensor(pos_cpu, device="cuda")
    temperatures = torch.tensor(temp_cpu, device="cuda")
    columns = (
        torch.arange(rows, device="cuda", dtype=torch.int32) % 3
        if per_token_col
        else torch.tensor(2, dtype=torch.int32, device="cuda")
    )
    cache = torch.full((rows, 3, vocab + 7), float("nan"), device="cuda", dtype=dtype)
    all_ids, all_max, ids, maxima = _buffers(rows, vocab)
    _gumbel(
        ids,
        maxima,
        logits,
        mapping,
        seeds,
        positions,
        temperatures,
        drafting=drafting,
        cache=cache,
        columns=columns,
    )
    actual_temp = temp_cpu[mapping_cpu]
    scores = (
        source.float().numpy().astype(np.float64)
        / np.where(actual_temp == 0.0, 1.0, actual_temp)[:, None]
    )
    expected = _reference(
        scores,
        seed_cpu[mapping_cpu],
        pos_cpu + (1 << 30 if drafting else 0),
        actual_temp,
    )
    np.testing.assert_array_equal(_sampled(ids, maxima), expected)
    saved = cache.cpu()
    for row in range(rows):
        col = row % 3 if per_token_col else 2
        assert torch.equal(saved[mapping_cpu[row], col, :vocab], source[row])
        assert torch.isnan(
            saved[mapping_cpu[row], [c for c in range(3) if c != col], :]
        ).all()
    assert torch.isnan(cache[:, :, vocab:]).all()
    assert torch.isnan(storage[:, vocab:]).all()
    _assert_guards(all_ids, all_max)


def _resample_inputs(rows, vocab, mode):
    target = torch.zeros((rows * 2, vocab), device="cuda")
    draft = torch.zeros((rows, 1, vocab), device="cuda")
    draft[:, :, vocab // 2 :] = float(np.log(7.0))
    return dict(
        target=target,
        draft=draft,
        target_lse=torch.full((rows,), float(np.log(vocab)), device="cuda"),
        draft_lse=torch.full(
            (rows,), float(np.log(vocab // 2 + 7 * (vocab - vocab // 2))), device="cuda"
        ),
        rejected=torch.full(
            (rows,), int(mode == "bonus"), dtype=torch.int32, device="cuda"
        ),
        cu=torch.arange(rows + 1, dtype=torch.int32, device="cuda") * 2,
        mapping=torch.arange(rows, dtype=torch.int32, device="cuda").repeat_interleave(
            2
        ),
        draft_sampled=torch.zeros(rows * 2, dtype=torch.int64, device="cuda"),
        temperatures=torch.ones(rows, device="cuda"),
        seeds=torch.arange(rows, dtype=torch.int64, device="cuda") + (1 << 40),
        positions=torch.arange(rows * 2, dtype=torch.int64, device="cuda") * 17 + 23,
    )


def _resample(ids, maxima, x, mode, *, fp64=False):
    draft = x["draft"] if mode == "residual" else None
    torch.ops._C.triton_resample(
        ids,
        ids.stride(0),
        maxima,
        maxima.stride(0),
        x["target"],
        x["target"].stride(0),
        x["target_lse"],
        draft,
        draft.stride(0) if draft is not None else 0,
        draft.stride(1) if draft is not None else 0,
        x["draft_lse"],
        x["rejected"],
        x["cu"],
        x["mapping"],
        x["draft_sampled"],
        x["temperatures"],
        x["seeds"],
        x["positions"],
        None,
        x["target"].shape[1],
        1024,
        mode == "residual",
        fp64,
        False,
    )


@pytest.mark.parametrize("mode", ["bonus", "exclude_rejected", "residual"])
@pytest.mark.parametrize("vocab", [128, 1025])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_resample_distribution_and_insert(mode, vocab, dtype):
    rows = 8192 if vocab == 128 else 64
    x = _resample_inputs(rows, vocab, mode)
    x["target"] = x["target"].to(dtype)
    x["draft"] = x["draft"].to(dtype)
    all_ids, all_max, ids, maxima = _buffers(rows, vocab)
    _resample(ids, maxima, x, mode)
    scores = np.zeros((rows, vocab), dtype=np.float64)
    if mode == "exclude_rejected":
        scores[:, 0] = -np.inf
    elif mode == "residual":
        scores[:, vocab // 2 :] = -np.inf
    row = np.arange(rows) * 2 + int(mode == "bonus")
    expected = _reference(
        scores, x["seeds"].cpu().numpy(), x["positions"].cpu().numpy()[row]
    )
    np.testing.assert_array_equal(_sampled(ids, maxima), expected)
    expected_support = (
        vocab
        if mode == "bonus"
        else vocab - 1
        if mode == "exclude_rejected"
        else vocab // 2
    )
    if vocab == 128:
        assert np.unique(expected).size == expected_support
    sampled = torch.full((rows, 4), -77, dtype=torch.int64, device="cuda")
    counts = x["rejected"].clone()
    torch.ops._C.triton_insert_resampled(
        sampled,
        sampled.stride(0),
        counts,
        ids,
        ids.stride(0),
        maxima,
        maxima.stride(0),
        ids.shape[1],
        x["cu"],
        x["mapping"],
        x["temperatures"],
        False,
    )
    col = int(mode == "bonus")
    np.testing.assert_array_equal(sampled[:, col].cpu().numpy(), expected)
    assert torch.all(counts == col + 1)
    assert torch.all(sampled[:, [c for c in range(4) if c != col]] == -77)
    _assert_guards(all_ids, all_max)


@pytest.mark.parametrize("kind", ["gumbel", "resample"])
def test_changed_input_cuda_graph_replay(kind):
    rows, vocab = 8, 129
    x = _resample_inputs(rows, vocab, "bonus")
    all_ids, all_max, ids, maxima = _buffers(rows, vocab)
    if kind == "gumbel":
        mapping = torch.arange(rows, dtype=torch.int32, device="cuda")
        logits = x["target"][::2]
        positions = x["positions"][::2].contiguous()

        def launch():
            _gumbel(
                ids, maxima, logits, mapping, x["seeds"], positions, x["temperatures"]
            )
    else:

        def launch():
            _resample(ids, maxima, x, "bonus")

    for _ in range(3):
        launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for replay in range(24):
        seeds = np.arange(rows, dtype=np.int64) + replay * 1321567 - (1 << 41)
        pos = np.arange(rows * 2, dtype=np.int64) + replay * 521
        x["seeds"].copy_(torch.tensor(seeds, device="cuda"))
        x["positions"].copy_(torch.tensor(pos, device="cuda"))
        if kind == "gumbel":
            positions.copy_(torch.tensor(pos[::2], device="cuda"))
        graph.replay()
        expected = _reference(
            np.zeros((rows, vocab)), seeds, pos[::2] if kind == "gumbel" else pos[1::2]
        )
        np.testing.assert_array_equal(_sampled(ids, maxima), expected)
    _assert_guards(all_ids, all_max)


@pytest.mark.parametrize("mapping_dtype", [torch.int32, torch.int64])
def test_greedy_invalid_mapping_and_nonfinite_fallback(mapping_dtype):
    logits = torch.zeros((4, 129), device="cuda")
    logits[0].fill_(float("nan"))
    logits[1].fill_(float("-inf"))
    logits[2, 17] = 3.0
    logits[3, 23] = 4.0
    mapping = torch.tensor([0, 1, 2, -1], device="cuda", dtype=mapping_dtype)
    seeds = torch.arange(4, dtype=torch.int64, device="cuda")
    positions = torch.arange(4, dtype=torch.int64, device="cuda")
    temperatures = torch.tensor([1.0, 1.0, 0.0, 1.0], device="cuda")
    all_ids, all_max, ids, maxima = _buffers(4, 129)
    _gumbel(ids, maxima, logits, mapping, seeds, positions, temperatures)
    np.testing.assert_array_equal(_sampled(ids, maxima), [0, 0, 17, 23])
    _assert_guards(all_ids, all_max)


def test_greedy_resample_preserves_nonbonus_rows():
    x = _resample_inputs(4, 129, "exclude_rejected")
    x["temperatures"].zero_()
    x["target"][:, 17] = 3.0
    all_ids, all_max, ids, maxima = _buffers(4, 129)
    _resample(ids, maxima, x, "exclude_rejected")
    assert torch.all(all_ids == -177) and torch.all(all_max == 12345.0)
    x["rejected"].fill_(1)
    _resample(ids, maxima, x, "bonus")
    np.testing.assert_array_equal(_sampled(ids, maxima), [17] * 4)
    _assert_guards(all_ids, all_max)


@pytest.mark.parametrize(
    "kind",
    ["gumbel_mode", "gumbel_logits", "resample_mode", "resample_logits", "insert"],
)
def test_mr_rejects_fp64_sampling(kind):
    if torch.ops._C.compiled_cuda_arch() != "ivcore11":
        pytest.skip("MR-specific unsupported precision boundary")
    x = _resample_inputs(2, 64, "bonus")
    _, _, ids, maxima = _buffers(2, 64)
    with pytest.raises(RuntimeError, match="FP64 sampling is not supported"):
        if kind.startswith("gumbel"):
            logits = x["target"][::2]
            if kind == "gumbel_logits":
                logits = logits.double()
            _gumbel(
                ids,
                maxima.double(),
                logits,
                torch.arange(2, dtype=torch.int32, device="cuda"),
                x["seeds"],
                x["positions"][::2].contiguous(),
                x["temperatures"],
                fp64=kind == "gumbel_mode",
            )
        elif kind.startswith("resample"):
            if kind == "resample_logits":
                x["target"] = x["target"].double()
            _resample(ids, maxima.double(), x, "bonus", fp64=kind == "resample_mode")
        else:
            sampled = torch.zeros((2, 2), dtype=torch.int64, device="cuda")
            torch.ops._C.triton_insert_resampled(
                sampled,
                2,
                x["rejected"],
                ids,
                ids.stride(0),
                maxima.double(),
                maxima.stride(0),
                1,
                x["cu"],
                x["mapping"],
                x["temperatures"],
                True,
            )


@pytest.mark.parametrize(
    "name", ["triton_compute_local_logits_stats", "triton_rejection"]
)
def test_mr_rejects_double_logits_before_launch(name):
    if torch.ops._C.compiled_cuda_arch() != "ivcore11":
        pytest.skip("MR-specific unsupported precision boundary")
    op = getattr(torch.ops._C, name).default
    args = {}
    # Empty tensors make this an early API-boundary check. A requested
    # unsupported precision must be rejected before any pointer access/launch.
    for arg in op._schema.arguments:
        kind = str(arg.type)
        if kind == "Tensor":
            args[arg.name] = torch.empty((0, 1), device="cuda")
        elif "Tensor" in kind:
            args[arg.name] = None
        elif kind == "int":
            args[arg.name] = 1
        elif kind == "bool":
            args[arg.name] = False
        else:
            raise AssertionError((arg.name, kind))
    args["target_logits"] = torch.empty((0, 1), device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="FP64 sampling is not supported"):
        op(**args)
