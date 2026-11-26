# SPDX-License-Identifier: Apache-2.0
"""GLM record/planner/commit lifecycle and cache-budget regression tests."""

from types import SimpleNamespace as NS

import pytest
import torch
from ixformer.inference.functions.ops.linear_attn.float.kda import fused_recurrent_kda
from ixformer.inference.functions.ops.linear_attn.float.kda_recoverssm import (
    kda_recoverssm_verify,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.config.mamba import MambaBackendEnum
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_iluvatar.config.recoverssm import (
    make_validate_mamba_cached_kernel,
    validate_glm_recoverssm,
)
from vllm_iluvatar.models.glm5next.recoverssm import GLMRecoverSSMResources
from vllm_iluvatar.v1.attention.backends.ilu_glm5next_kda import (
    IluGlm5NextKDAMetadataBuilder,
)
from vllm_iluvatar.v1.worker.gpu.model_states.glm5next import IluGLMRecoverSSMState


def config(mode="align", dtype=torch.float32):
    return NS(
        model_config=NS(
            architecture="Glm5NextForCausalLM",
            dtype=torch.bfloat16,
            supports_replayssm=True,
            hf_text_config=NS(
                linear_head_dim=128,
                linear_conv_kernel_dim=4,
                linear_lower_bound=-5.0,
                linear_key_head_dim=128,
            ),
        ),
        speculative_config=NS(num_speculative_tokens=5, parallel_drafting=False),
        num_speculative_tokens=5,
        use_v2_model_runner=True,
        additional_config={},
        scheduler_config=NS(max_num_seqs=6, max_num_batched_tokens=256),
        parallel_config=NS(
            pipeline_parallel_size=1, enable_dbo=False, decode_context_parallel_size=1
        ),
        mamba_config=NS(
            enable_stochastic_rounding=False, backend=MambaBackendEnum.TRITON
        ),
        kv_transfer_config=None,
        cache_config=NS(
            use_replayssm=True,
            use_kda_recoverssm=True,
            mamba_cache_mode=mode,
            mamba_block_size=8,
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="bfloat16" if dtype == torch.bfloat16 else "auto",
        ),
        compilation_config=NS(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            max_cudagraph_capture_size=36,
            static_forward_context={},
        ),
    )


def layer(index, dtype):
    h, p, r, t = 2, 16, 6, 6
    torch.manual_seed(93 + index)
    state = torch.randn(p, h, 128, 128, device="cuda").to(dtype) * 0.1
    conv = torch.randn(p, t + 2, 3 * h * 128, device="cuda").bfloat16() * 0.1
    a = torch.randn(h, device="cuda") * 0.1
    bias = torch.randn(h, 128, device="cuda") * 0.1
    return NS(
        prefix=f"model.layers.{index}.self_attn",
        local_num_heads=h,
        kda_lower_bound=-5.0,
        get_state_dtype=lambda: (torch.bfloat16, dtype),
        A_log=a,
        A_log_view=a,
        dt_bias_view=bias,
        kv_cache=(conv, state),
        _conv_state_dim_first=False,
        _recoverssm_correction=torch.full((r, h, t, 128), 7.25, device="cuda"),
        _recoverssm_key_gate=torch.full(
            (r, h, t, 256), 7.25, dtype=torch.bfloat16, device="cuda"
        ),
        _merged_conv_weight=torch.randn(3 * h * 128, 4, device="cuda") * 0.1,
        _recoverssm_weights_ready=True,
    )


def setup(mode, dtype=torch.float32):
    cfg = config(mode, dtype)
    layers = [layer(i, dtype) for i in range(2)]
    cfg.compilation_config.static_forward_context = {x.prefix: x for x in layers}
    resources = GLMRecoverSSMResources(layers, cfg)
    resources.begin_cache_binding(False)
    spec = MambaSpec(
        shapes=((8, 3 * 2 * 128), (2, 128, 128)),
        dtypes=(torch.bfloat16, dtype),
        block_size=8,
        mamba_cache_mode=mode,
        num_speculative_blocks=0,
    )
    builder = IluGlm5NextKDAMetadataBuilder(
        spec, [x.prefix for x in layers], cfg, torch.device("cuda:0")
    )
    resources.finish_cache_binding()
    return cfg, layers, resources, builder


def common(lengths, previous, prefills=None):
    cu_cpu = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32
    )
    seq_cpu = torch.tensor(previous, dtype=torch.int32) + torch.tensor(
        lengths, dtype=torch.int32
    )
    b = len(lengths)
    table = torch.arange(1, b * 3 + 1, dtype=torch.int32, device="cuda").view(b, 3)
    return CommonAttentionMetadata(
        query_start_loc=cu_cpu.cuda(),
        query_start_loc_cpu=cu_cpu,
        seq_lens=seq_cpu.cuda(),
        seq_lens_cpu_upper_bound=seq_cpu,
        num_reqs=b,
        num_actual_tokens=int(cu_cpu[-1]),
        max_query_len=max(lengths),
        max_seq_len=int(seq_cpu.max()),
        block_table_tensor=table,
        slot_mapping=torch.zeros(int(cu_cpu[-1]), dtype=torch.int64, device="cuda"),
        is_prefilling=torch.tensor(prefills or [False] * b, dtype=torch.bool),
    )


def build(builder, m):
    if builder.kv_cache_spec.mamba_cache_mode == "align":
        cols = ((m.seq_lens - 1).clamp_min(0) // 8).long()
        builder.mamba_aligned_state_indices = m.block_table_tensor.gather(
            1, cols[:, None]
        )
    return builder.build(
        0, m, num_decode_draft_tokens_cpu=torch.zeros(m.num_reqs, dtype=torch.int32)
    )


def verify_with_snapshot(layers, metadata):
    n, h, t = metadata.num_actual_tokens, 2, 6
    proj = torch.randn(n, 3 * h * 128, device="cuda").bfloat16() * 0.2
    q, k, v = [x.view(n, h, 128) for x in proj.chunk(3, dim=-1)]
    g = torch.randn(n, h, 128, device="cuda").bfloat16() * 0.1
    beta = torch.randn(n, h, device="cuda").bfloat16() * 0.1
    b = metadata.num_spec_decodes
    cu = metadata.spec_query_start_loc[: b + 1]
    source = metadata.spec_state_indices_tensor[:b, 0]
    metadata._test_inputs = (q, k, v, g, beta)
    refs = []
    for x in layers:
        conv, state = x.kv_cache
        original = state.clone()
        old_state = torch.zeros(
            b * t + 1, h, 128, 128, device="cuda", dtype=state.dtype
        )
        old_conv = torch.zeros(
            b * t + 1, 3 * h * 128, t + 2, device="cuda", dtype=torch.bfloat16
        )
        bt = torch.arange(1, b * t + 1, dtype=torch.int32, device="cuda").view(b, t)
        for row, src in enumerate(source.cpu().tolist()):
            old_state[1 + row * t].copy_(state[src])
            old_conv[1 + row * t].copy_(conv[src].T)
        old_out = torch.full((n, h, 128), -8.0, dtype=torch.bfloat16, device="cuda")
        fused_recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            x.A_log_view,
            x.dt_bias_view,
            old_out,
            old_state,
            cu,
            bt,
            128**-0.5,
            -5.0,
            num_accepted_tokens=torch.ones(b, dtype=torch.int32, device="cuda"),
            token_index=metadata.spec_token_indx,
            conv_weight=x._merged_conv_weight,
            conv_state_pool=old_conv,
        )
        out = torch.full_like(old_out, -8.0)
        kda_recoverssm_verify(
            q,
            k,
            v,
            g,
            beta,
            x.A_log_view,
            x.dt_bias_view,
            state,
            x._recoverssm_correction,
            x._recoverssm_key_gate,
            cu,
            source,
            spec_query_len=t,
            lower_bound=-5.0,
            out=out,
            record_rows=metadata.recoverssm_record_rows,
            token_index=metadata.spec_token_indx,
            conv_weight=x._merged_conv_weight,
            conv_state_pool=conv.transpose(1, 2),
        )
        torch.testing.assert_close(out, old_out, rtol=0, atol=0)
        torch.testing.assert_close(state, original, rtol=0, atol=0)
        refs.append((original, old_state, bt))
    return refs


def assert_state(actual, expected, dtype):
    rms = (
        (
            (actual.double() - expected.double()).square().mean()
            / expected.double().square().mean().clamp_min(1e-24)
        )
        .sqrt()
        .item()
    )
    assert rms <= (2.5e-3 if dtype == torch.bfloat16 else 5e-4), rms


@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("counts", [[0, 0, 0], [1, 1, 1], [2, 1, 2], [6, 1, 3]])
def test_immediate_commit_boundaries_and_single_use(mode, dtype, counts):
    cfg, layers, resources, builder = setup(mode, dtype)
    m = common([6, 1, 3], [6, 7, 8])
    metadata = build(builder, m)
    assert metadata.num_spec_decodes == 3  # includes the zero-draft L=1 row
    refs = verify_with_snapshot(layers, metadata)
    coordinator = IluGLMRecoverSSMState()
    group = NS(layer_names=[x.prefix for x in layers])
    coordinator.record_step(
        {x.prefix: metadata for x in layers}, [[group], [group]], for_capture=False
    )
    idx = torch.tensor([5, 2, 4], dtype=torch.int32, device="cuda")
    sampled = torch.tensor(counts, dtype=torch.int32, device="cuda")
    accepted = torch.full((6,), 9, dtype=torch.int32, device="cuda")
    running = torch.full_like(accepted, -1) if mode == "align" else None
    coordinator.commit_step(
        sampled, idx, state_indices=running, num_accepted_tokens=accepted
    )
    plan = builder.group
    for x, (initial, snapshots, bt) in zip(layers, refs):
        expected = initial.clone()
        for row, count in enumerate(counts):
            if count:
                target = int(plan.final[row])
                expected[target] = snapshots[bt[row, count - 1]]
                bound = int(plan.boundary[row])
                if bound:
                    expected[bound] = snapshots[
                        bt[row, int(plan.boundary_lens[row]) - 1]
                    ]
        for slot in range(16):
            assert_state(x.kv_cache[1][slot], expected[slot], dtype)
    for row, count in enumerate(counts):
        req = int(idx[row])
        assert int(accepted[req]) == (1 if count else 9)
        if running is not None:
            expected_col = max([6, 7, 8][row] + count - 1, 0) // 8 if count else -1
            assert int(running[req]) == expected_col
    before = [x.kv_cache[1].clone() for x in layers]
    coordinator.commit_step(
        sampled, idx, state_indices=running, num_accepted_tokens=accepted
    )
    for x, old in zip(layers, before):
        torch.testing.assert_close(x.kv_cache[1], old, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["none", "align"])
def test_mixed_prefill_request_mapping_and_filtered_commit(mode):
    _, layers, resources, builder = setup(mode)
    m = common([6, 2, 1], [6, 0, 7], [False, True, False])
    metadata = build(builder, m)
    assert metadata.num_prefills == 1 and metadata.num_spec_decodes == 2
    assert builder.group.request_rows[:2].cpu().tolist() == [0, 2]
    assert metadata.spec_token_indx.cpu().tolist() == [0, 1, 2, 3, 4, 5, 8]
    verify_with_snapshot(layers, metadata)
    before = [x.kv_cache[1].clone() for x in layers]
    metadata.commit_recoverssm_state(
        torch.tensor([2, 0, 1], dtype=torch.int32, device="cuda"),
        idx_mapping=torch.tensor([3, 2, -1], dtype=torch.int32, device="cuda"),
    )
    assert builder.group.commit_lens[:2].cpu().tolist() == [2, 0]
    source = int(builder.group.source[1])
    for x, old in zip(layers, before):
        torch.testing.assert_close(x.kv_cache[1][source], old[source], rtol=0, atol=0)
    if mode == "align":
        assert builder.group.final[:2].cpu().tolist() == [1, 0]
        assert int(builder.group.computed_by_request[2]) == 7


@pytest.mark.parametrize("mode", ["none", "align"])
def test_overlong_decode_uses_prefill_without_entering_recoverssm(mode):
    _, _, resources, builder = setup(mode)
    assert resources.spec_query_len == 6
    assert not resources.is_profiling

    metadata = build(builder, common([6, 7, 1], [2, 1, 7]))

    assert metadata.num_spec_decodes == 2
    assert metadata.num_spec_decode_tokens == 7
    assert metadata.num_prefills == 1
    assert metadata.num_prefill_tokens == 7
    assert metadata.non_spec_query_start_loc_cpu.tolist() == [0, 7]
    assert metadata.non_spec_token_indx.cpu().tolist() == list(range(6, 13))
    assert metadata.spec_token_indx.cpu().tolist() == [0, 1, 2, 3, 4, 5, 13]
    assert builder.group.request_rows[:2].cpu().tolist() == [0, 2]

    all_overlong = build(builder, common([7, 8], [1, 2]))
    assert all_overlong.num_prefills == 2
    assert all_overlong.num_prefill_tokens == 15
    assert all_overlong.num_spec_decodes == 0
    assert all_overlong.recoverssm_group is None


def test_capture_and_profiling_do_not_enqueue_commit():
    _, layers, resources, builder = setup("none")
    metadata = build(builder, common([1, 0, 0], [8, 0, 0]))
    coordinator = IluGLMRecoverSSMState()
    groups = [[NS(layer_names=[x.prefix for x in layers])]]
    mapping = {x.prefix: metadata for x in layers}
    coordinator.record_step(mapping, groups, for_capture=True)
    assert coordinator._step is None
    assert resources.captured_generation == resources.generation
    resources.is_profiling = True
    coordinator.record_step(mapping, groups, for_capture=False)
    assert coordinator._step is None
    with pytest.raises(RuntimeError, match="recapture"):
        resources.before_layer_binding()


def test_cache_rebind_reuses_reservation_and_rejects_overlapping_groups():
    cfg, layers, resources, builder = setup("align")
    addresses = resources.context.state_addresses.data_ptr()
    plans = resources.plans.data_ptr()
    resources.begin_cache_binding(False)
    for x in layers:
        x.kv_cache = tuple(t.clone() for t in x.kv_cache)
    group = resources.get_group([layers[0].prefix])
    with pytest.raises(RuntimeError, match="share a layer"):
        resources.get_group([layers[0].prefix, layers[1].prefix])
    resources.get_group([layers[1].prefix])
    resources.finish_cache_binding()
    assert resources.context.state_addresses.data_ptr() == addresses
    assert resources.plans.data_ptr() == plans
    assert group.context.state_pools[0] is layers[0].kv_cache[1]
    assert group.bound_generation == resources.generation


def test_cache_budget_drops_speculative_pages_only():
    for mode, base_pages in (("none", 1), ("align", 2)):
        cfg = config(mode)
        kwargs = dict(
            shapes=((8, 768), (2, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            block_size=8,
            mamba_cache_mode=mode,
        )
        before = MambaSpec(**kwargs, num_speculative_blocks=5)
        after = MambaSpec(**kwargs, num_speculative_blocks=0)
        assert (
            before.max_memory_usage_bytes(cfg)
            == (base_pages + 5) * before.page_size_bytes
        )
        assert after.max_memory_usage_bytes(cfg) == base_pages * after.page_size_bytes
        assert after.shapes == before.shapes and len(after.shapes) == 2


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("use_v2_model_runner", False, "V2"),
        ("num_speculative_tokens", 0, "speculative"),
        ("num_speculative_tokens", 64, "<= 64"),
        ("parallel_config.pipeline_parallel_size", 2, "pipeline"),
        ("parallel_config.enable_dbo", True, "DBO"),
        ("cache_config.mamba_cache_mode", "all", "none/align"),
        ("cache_config.mamba_ssm_cache_dtype", "float16", "SSM state"),
        ("cache_config.mamba_cache_dtype", "float32", "Conv history"),
        ("model_config.dtype", torch.float16, "activations"),
    ],
)
def test_config_rejects_unsupported_modes(monkeypatch, field, value, match):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_ILU_GDN_ATTN_BACKEND", True, raising=False)
    cfg = config()
    target = cfg
    parts = field.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    with pytest.raises(ValueError, match=match):
        validate_glm_recoverssm(cfg)


def test_config_enables_glm_only_and_preserves_other_validation(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_ILU_GDN_ATTN_BACKEND", True, raising=False)
    cfg = config()
    assert validate_glm_recoverssm(cfg) is cfg
    assert cfg.cache_config.use_kda_recoverssm
    cfg.model_config.architecture = "KimiLinearForCausalLM"
    sentinel = object()
    wrapped = make_validate_mamba_cached_kernel(lambda _: sentinel)
    assert wrapped(cfg) is sentinel


@pytest.mark.parametrize("mode", ["none", "align"])
def test_scheduler_really_releases_speculative_block_capacity(mode):
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    free_counts = []
    for spec_blocks in (5, 0):
        spec = MambaSpec(
            shapes=((8, 768), (2, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            block_size=8,
            mamba_cache_mode=mode,
            num_speculative_blocks=spec_blocks,
        )
        pool = BlockPool(100, enable_caching=False, hash_block_size=8)
        manager = MambaManager(
            spec,
            pool,
            enable_caching=False,
            kv_cache_group_id=0,
            scheduler_block_size=8,
        )
        blocks = manager.allocate_new_blocks("request", 1, 1)
        assert len(blocks) == 1 + spec_blocks
        if mode == "align":
            manager.allocate_new_blocks("request", 9, 9)
        free_counts.append(pool.get_num_free_blocks())
    assert free_counts[1] - free_counts[0] == 5


@pytest.mark.parametrize("mode", ["none", "align"])
def test_graph_replay_changes_commit_and_record_maps(mode):
    _, layers, _, builder = setup(mode)
    m = common([6, 1, 3], [6, 7, 8])
    metadata = build(builder, m)
    initial_conv = [x.kv_cache[0].clone() for x in layers]
    refs = verify_with_snapshot(layers, metadata)
    q, k, v, g, beta = metadata._test_inputs
    outputs = [torch.empty_like(q) for _ in layers]
    sampled = torch.tensor([2, 1, 2], dtype=torch.int32, device="cuda")
    idx = torch.tensor([5, 2, 4], dtype=torch.int32, device="cuda")
    accepted = torch.ones(6, dtype=torch.int32, device="cuda")
    running = torch.zeros_like(accepted) if mode == "align" else None

    def execute():
        for x, out in zip(layers, outputs):
            conv, state = x.kv_cache
            kda_recoverssm_verify(
                q,
                k,
                v,
                g,
                beta,
                x.A_log_view,
                x.dt_bias_view,
                state,
                x._recoverssm_correction,
                x._recoverssm_key_gate,
                metadata.spec_query_start_loc,
                metadata.spec_state_indices_tensor[:, 0],
                spec_query_len=6,
                lower_bound=-5.0,
                out=out,
                record_rows=metadata.recoverssm_record_rows,
                conv_weight=x._merged_conv_weight,
                conv_state_pool=conv.transpose(1, 2),
            )
        metadata.commit_recoverssm_state(sampled, idx_mapping=idx)
        builder.group.finish(3, running, accepted)

    def reset():
        for x, (initial, _, _), conv in zip(layers, refs, initial_conv):
            x.kv_cache[1].copy_(initial)
            x.kv_cache[0].copy_(conv)

    reset()
    execute()
    reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    for counts, rows in [([1, 1, 1], [3, 0, 4]), ([6, 1, 3], [2, 5, 1])]:
        reset()
        sampled.copy_(torch.tensor(counts, dtype=torch.int32))
        builder.group.record_rows[:3].copy_(torch.tensor(rows, dtype=torch.int32))
        graph.replay()
        for x, (_, snapshots, bt) in zip(layers, refs):
            for b, count in enumerate(counts):
                dest = int(builder.group.final[b])
                assert_state(
                    x.kv_cache[1][dest], snapshots[bt[b, count - 1]], torch.float32
                )
        assert accepted[idx.long()].cpu().tolist() == [1, 1, 1]


def test_upstream_runner_binding_and_profiling_teardown_release_cache_views():
    import gc
    import weakref

    from vllm_iluvatar.v1.worker.gpu.model_runner import (
        make_recoverssm_initialize_kv_cache,
        make_recoverssm_profiling_teardown,
    )

    cfg, layers, resources, _ = setup("align")
    runner = NS(
        model=NS(_recoverssm_resources=resources),
        model_state=NS(recoverssm=IluGLMRecoverSSMState(), _align_mode=True),
        kv_caches=[],
    )
    names = [x.prefix for x in layers]

    def initialize(self, cache, is_profiling=False, kv_cache_allocation_context=None):
        assert resources.binding and not resources.ready
        self.kv_caches = [x.kv_cache for x in layers]
        resources.get_group(names)
        assert is_profiling
        return "initialized"

    # Nested wrappers must not reset a reservation twice (a subclass may delegate).
    wrapped = make_recoverssm_initialize_kv_cache(
        make_recoverssm_initialize_kv_cache(initialize)
    )
    assert wrapped(runner, object(), is_profiling=True) == "initialized"
    assert resources.ready and not resources.binding and resources.is_profiling
    group = resources.groups[tuple(names)]
    assert group.context.bound
    state_ref = weakref.ref(layers[0].kv_cache[1])
    record_ref = weakref.ref(layers[0]._recoverssm_correction)

    def teardown(self):
        assert not resources.ready and not resources.groups
        assert not group.context.state_pools and not group.context.bound
        self.kv_caches.clear()
        for x in layers:
            x.kv_cache = ()

    make_recoverssm_profiling_teardown(teardown)(runner)
    gc.collect()
    assert state_ref() is None
    assert record_ref() is layers[0]._recoverssm_correction


def test_empty_metadata_has_no_pending_commit():
    _, layers, _, builder = setup("none")
    m = CommonAttentionMetadata(
        query_start_loc=torch.zeros(1, dtype=torch.int32, device="cuda"),
        query_start_loc_cpu=torch.zeros(1, dtype=torch.int32),
        seq_lens=torch.empty(0, dtype=torch.int32, device="cuda"),
        seq_lens_cpu_upper_bound=torch.empty(0, dtype=torch.int32),
        num_reqs=0,
        num_actual_tokens=0,
        max_query_len=0,
        max_seq_len=0,
        block_table_tensor=torch.empty(0, 3, dtype=torch.int32, device="cuda"),
        slot_mapping=torch.empty(0, dtype=torch.int64, device="cuda"),
        is_prefilling=torch.empty(0, dtype=torch.bool),
    )
    metadata = builder.build(0, m)
    assert metadata.num_spec_decodes == 0
    coordinator = IluGLMRecoverSSMState()
    coordinator.record_step(
        {x.prefix: metadata for x in layers},
        [[NS(layer_names=[x.prefix for x in layers])]],
        for_capture=False,
    )
    assert coordinator._step is None


@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("sample_rows,index_rows", [(1, None), (1, 3), (3, 1)])
def test_padded_metadata_never_reads_sampler_or_index_tail(
    mode, sample_rows, index_rows
):
    _, layers, _, builder = setup(mode)
    metadata = build(builder, common([1, 1, 1], [1, 1, 1]))
    verify_with_snapshot(layers, metadata)
    before = [x.kv_cache[1].clone() for x in layers]
    # Nonzero allocation tails make an accidental read observable without
    # relying on a device fault. Only row zero exists in both logical inputs.
    sampled = torch.tensor([1, 1, 1], dtype=torch.int32, device="cuda")[:sample_rows]
    idx = (
        torch.tensor([2, 3, 4], dtype=torch.int32, device="cuda")[:index_rows]
        if index_rows is not None
        else None
    )
    metadata.commit_recoverssm_state(sampled, idx_mapping=idx)
    assert builder.group.commit_lens[:3].cpu().tolist() == [1, 0, 0]
    assert builder.group.request_slots[1:3].cpu().tolist() == [-1, -1]
    for x, old in zip(layers, before):
        for row in (1, 2):
            source = int(builder.group.source[row])
            torch.testing.assert_close(
                x.kv_cache[1][source], old[source], rtol=0, atol=0
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cache_binding_enforces_configured_state_dtype(dtype):
    _, layers, resources, builder = setup("none", dtype)
    conv, state = layers[0].kv_cache
    other = torch.bfloat16 if dtype == torch.float32 else torch.float32
    layers[0].kv_cache = (conv, state.to(other))
    group = next(iter(resources.groups.values()))
    with pytest.raises(ValueError, match="cache dtypes"):
        group.bind()
