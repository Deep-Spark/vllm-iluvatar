"""Sampler/dummy phase contracts without GPU or process-group initialization."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_iluvatar.distributed import dp_vocab_projection as impl


@pytest.fixture
def harness(monkeypatch):
    def original(processor, head, hidden, bias):
        return F.linear(hidden, head.weight, bias)

    monkeypatch.setattr(LogitsProcessor, "_apply_head", original)
    weight = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 16
    head = SimpleNamespace(weight=weight)
    projector = SimpleNamespace(
        dummy=torch.zeros(1, 4), calls=[], capacity=24, capacities=[]
    )

    def project(hidden, current_weight, capacity=None):
        capacity = projector.capacity if capacity is None else capacity
        if hidden.shape[0] > capacity:
            raise ValueError(
                "Hidden states exceed the configured vocabulary projection extent"
            )
        projector.calls.append((hidden.clone(), current_weight))
        projector.capacities.append(capacity)
        return F.linear(hidden, current_weight)

    projector.project = project
    controller = impl.DPVocabController(
        projector,
        head,
        head,
        per_request_bound=6,
        dynamic_capacity_supported=True,
    )
    impl._enable_scoped_head_dispatch()
    processor = SimpleNamespace(head_dtype=None)

    def head_call(hidden, bias=None):
        return LogitsProcessor._apply_head(processor, head, hidden, bias)

    sequential = controller.wrap_sequential(lambda hidden: head_call(hidden))
    propose = controller.wrap_propose(lambda hidden, **kwargs: sequential(hidden))
    sample = controller.wrap_sample(head_call)
    return SimpleNamespace(
        controller=controller,
        projector=projector,
        head=head,
        head_call=head_call,
        sample=sample,
        propose=propose,
    )


@pytest.mark.parametrize("rows", [0, 1, 2, 4, 12, 24])
def test_target_and_draft_have_one_projection_each(harness, rows):
    x = torch.ones(rows, 4)
    expected = F.linear(x, harness.head.weight)
    torch.testing.assert_close(harness.sample(x), expected)
    torch.testing.assert_close(harness.propose(x), expected)
    assert len(harness.projector.calls) == 2
    assert not harness.controller.pending_target
    assert impl._active_projection.get() is None


def test_idle_dummy_replaces_the_missing_target_projection(harness):
    for idle in (False, True, True, False):
        x = torch.ones(2, 4)
        if not idle:
            harness.sample(x)
        harness.propose(x, dummy_run=idle)
    assert len(harness.projector.calls) == 8
    assert harness.projector.calls[2][0].shape[0] == 1
    assert not harness.controller.pending_target


def test_unscoped_prompt_logits_keep_the_original_path(harness):
    x = torch.ones(2, 4)
    harness.head_call(x)
    assert not harness.projector.calls


def test_parameter_replacement_is_resolved_at_the_call(harness):
    weight = torch.full_like(harness.head.weight, 3)
    harness.head.weight = weight
    expected = F.linear(torch.ones(1, 4), weight)
    torch.testing.assert_close(harness.sample(torch.ones(1, 4)), expected)
    harness.propose(torch.ones(1, 4))
    assert all(call[1] is weight for call in harness.projector.calls)


def test_real_draft_requires_target_phase(harness):
    with pytest.raises(RuntimeError, match="preceding target"):
        harness.propose(torch.ones(1, 4))


def test_duplicate_target_or_dummy_is_rejected(harness):
    harness.sample(torch.ones(1, 4))
    with pytest.raises(RuntimeError, match="Previous target"):
        harness.sample(torch.ones(1, 4))
    with pytest.raises(RuntimeError, match="duplicate"):
        harness.propose(torch.ones(1, 4), dummy_run=True)


def test_scope_is_cleared_after_failure(harness):
    with pytest.raises(ValueError), harness.controller.scope():
        raise ValueError("test")
    assert impl._active_projection.get() is None


def test_dispatch_install_is_idempotent(harness):
    original = LogitsProcessor._apply_head
    impl._enable_scoped_head_dispatch()
    assert LogitsProcessor._apply_head is original


def test_projection_scope_rejects_bias_without_changing_default(harness):
    x = torch.ones(1, 4)
    with pytest.raises(ValueError, match="precision or bias"):
        harness.sample(x, torch.ones(8))
    expected = F.linear(x, harness.head.weight, torch.ones(8))
    torch.testing.assert_close(harness.head_call(x, torch.ones(8)), expected)


@pytest.mark.parametrize("maxseq", [1, 2, 3, 5, 9])
@pytest.mark.parametrize("steps", [1, 2, 5, 9])
def test_capacity_uses_only_shared_request_bound_after_warmup(maxseq, steps):
    full = maxseq * (steps + 1)
    controller = impl.DPVocabController(
        SimpleNamespace(capacity=full),
        None,
        None,
        per_request_bound=steps + 1,
        dynamic_capacity_supported=True,
    )
    for requests in range(1, maxseq + 1):
        assert controller.capacity_for(SimpleNamespace(num_reqs=requests)) == full
    controller.enable_dynamic_capacity()
    for requests in range(1, maxseq + 1):
        assert controller.capacity_for(
            SimpleNamespace(num_reqs=requests)
        ) == requests * (steps + 1)
    for unknown in [
        None,
        SimpleNamespace(),
        *[SimpleNamespace(num_reqs=n) for n in (0, -1, maxseq + 1, True, 1.5)],
    ]:
        assert controller.capacity_for(unknown) == full


def test_unsupported_executor_policy_stays_full_capacity():
    controller = impl.DPVocabController(
        SimpleNamespace(capacity=18),
        None,
        None,
        per_request_bound=6,
        dynamic_capacity_supported=False,
    )
    controller.enable_dynamic_capacity()
    assert not controller.dynamic_capacity
    assert controller.capacity_for(SimpleNamespace(num_reqs=2)) == 18


@pytest.mark.parametrize("requests", [1, 2, 4, None])
@pytest.mark.parametrize("dummy", [False, True])
def test_one_common_capacity_covers_real_and_dummy_pair(harness, requests, dummy):
    controller = harness.controller
    controller.enable_dynamic_capacity()
    sync = SimpleNamespace(num_reqs=requests) if requests is not None else None
    expected = requests * 6 if requests is not None else harness.projector.capacity
    target, draft = torch.ones(3, 4), torch.ones(2, 4)
    if dummy:
        harness.propose(draft, dp_sync=sync, dummy_run=True)
    else:
        with controller.pair(sync):
            harness.sample(target)
            harness.propose(draft, dp_sync=sync)
    assert harness.projector.capacities == [expected, expected]
    assert not controller.pending_target and controller._pair_capacity is None


def test_positional_propose_arguments_keep_dummy_and_real_agreement(harness):
    controller = harness.controller
    controller.enable_dynamic_capacity()
    sync = SimpleNamespace(num_reqs=2)
    sequential = controller.wrap_sequential(harness.head_call)

    def original(hidden, dp_sync=None, dummy_run=False):
        return sequential(hidden)

    propose = controller.wrap_propose(original)
    x = torch.ones(2, 4)
    propose(x, sync, True)
    with controller.pair(sync):
        harness.sample(x)
        propose(x, sync, False)
    assert harness.projector.capacities == [12, 12, 12, 12]


def test_changed_draft_agreement_raises_without_local_capacity_fallback(harness):
    controller = harness.controller
    controller.enable_dynamic_capacity()
    with (
        pytest.raises(RuntimeError, match="same DP capacity"),
        controller.pair(SimpleNamespace(num_reqs=2)),
    ):
        harness.sample(torch.ones(2, 4))
        harness.propose(torch.ones(2, 4), dp_sync=SimpleNamespace(num_reqs=1))
    assert harness.projector.capacities == [12]
    assert not controller.pending_target and controller._pair_capacity is None


def test_local_overflow_raises_without_changing_collective_extent(harness):
    controller = harness.controller
    controller.enable_dynamic_capacity()
    with (
        pytest.raises(ValueError, match="extent"),
        controller.pair(SimpleNamespace(num_reqs=1)),
    ):
        harness.sample(torch.ones(7, 4))
    assert not harness.projector.capacities
    assert not controller.pending_target and controller._pair_capacity is None


def test_pair_aborted_after_target_clears_state_for_next_rpc(harness):
    controller = harness.controller
    controller.enable_dynamic_capacity()
    sync = SimpleNamespace(num_reqs=2)
    with pytest.raises(RuntimeError, match="abort"), controller.pair(sync):
        harness.sample(torch.ones(2, 4))
        raise RuntimeError("abort before propose")
    assert not controller.pending_target and controller._pair_capacity is None
    with controller.pair(sync):
        harness.sample(torch.ones(2, 4))
        harness.propose(torch.ones(2, 4), dp_sync=sync)
    assert harness.projector.capacities == [12, 12, 12]
    assert impl._active_projection.get() is None


def test_pair_rejects_missing_extra_and_nested_phases(harness):
    controller = harness.controller
    controller.enable_dynamic_capacity()
    sync = SimpleNamespace(num_reqs=2)
    with (
        pytest.raises(RuntimeError, match="requires target and draft"),
        controller.pair(sync),
    ):
        pass
    with (
        pytest.raises(RuntimeError, match="must not overlap"),
        controller.pair(sync),
        controller.pair(sync),
    ):
        pass
    with pytest.raises(RuntimeError, match="extra vocabulary"), controller.pair(sync):
        harness.sample(torch.ones(2, 4))
        harness.propose(torch.ones(2, 4), dp_sync=sync)
        controller.project(torch.ones(2, 4), harness.head.weight)
    assert not controller.pending_target and controller._pair_capacity is None


@pytest.mark.parametrize(
    "group,full,capacity,rank",
    [
        (g, full, c, rank)
        for g, full, c in [(2, 4, 2), (4, 18, 6), (4, 18, 12), (4, 18, 18)]
        for rank in range(g)
    ],
)
def test_packed_peer_routing_preserves_owner_and_unused_storage(
    monkeypatch, group, full, capacity, rank
):
    """Simulate unequal peers against the full-weight oracle, not each other."""
    hidden_size, shard_rows = 4, 3
    weight = (
        torch.arange(group * shard_rows * hidden_size, dtype=torch.float32).reshape(
            group * shard_rows, hidden_size
        )
        / 8
    )
    peers = [
        torch.arange(
            (capacity - rank % (capacity + 1)) * hidden_size, dtype=torch.float32
        ).reshape(-1, hidden_size)
        + rank * 100
        for rank in range(group)
    ]
    peers[0] = torch.empty(0, hidden_size)
    p = impl.DPVocabProjection.__new__(impl.DPVocabProjection)
    p.capacity, p.group_size, p.group_rank = full, group, rank
    p.hidden_size, p.shard_rows, p.vocab_size = (
        hidden_size,
        shard_rows,
        group * shard_rows,
    )
    p.dtype, p.device, p.group = weight.dtype, weight.device, object()
    p.input = torch.full((full, hidden_size), -123.0)
    p.gathered = torch.full((group * full, hidden_size), -123.0)
    p.partial = torch.full((group * full, shard_rows), -123.0)
    p.received = torch.full_like(p.partial, -123.0)
    padded = []
    for peer in peers:
        x = torch.zeros(capacity, hidden_size)
        x[: len(peer)] = peer
        padded.append(x)

    def gather(out, inp, *, group):
        assert group is p.group and inp.shape[0] == capacity
        torch.testing.assert_close(inp, padded[rank], rtol=0, atol=0)
        out.copy_(torch.cat(padded))

    def exchange(out, partial, *, group):
        assert group is p.group and partial.shape[0] == p.group_size * capacity
        torch.testing.assert_close(
            partial,
            F.linear(
                torch.cat(padded), weight[rank * shard_rows : (rank + 1) * shard_rows]
            ),
            rtol=0,
            atol=0,
        )
        out.copy_(
            torch.cat(
                [
                    F.linear(
                        padded[rank], weight[j * shard_rows : (j + 1) * shard_rows]
                    )
                    for j in range(p.group_size)
                ]
            )
        )

    monkeypatch.setattr(impl.dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(impl.dist, "all_to_all_single", exchange)
    result = p.project(peers[rank], weight, capacity=capacity)
    expected = F.linear(peers[rank], weight)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for tensor, extent in [
        (p.input, capacity),
        (p.gathered, group * capacity),
        (p.partial, group * capacity),
        (p.received, group * capacity),
    ]:
        assert torch.all(tensor[extent:] == -123)
    # Later draft traffic must not overwrite returned target logits.
    p.received.fill_(987)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "tp,dp,external,group", [(1, 16, 1, 4), (2, 8, 1, 2), (2, 8, 1, 4), (2, 8, 2, 4)]
)
def test_every_global_rank_creates_all_vocab_groups_in_the_same_order(
    monkeypatch, tp, dp, external, group
):
    # Use the actual vLLM ExternalDP x DP x TP transpose to form the oracle.
    ranks = torch.arange(external * dp * tp).view(external, dp, tp)
    lanes = ranks.transpose(1, 2).reshape(-1, dp).tolist()
    expected = [lane[lo : lo + group] for lane in lanes for lo in range(0, dp, group)]
    monkeypatch.setattr(impl.dist, "get_world_size", lambda: ranks.numel())
    for external_rank in range(external):
        for dp_rank in range(dp):
            for tp_rank in range(tp):
                rank = int(ranks[external_rank, dp_rank, tp_rank])
                dp_ranks = ranks[external_rank, :, tp_rank].tolist()
                tp_ranks = ranks[external_rank, dp_rank].tolist()
                monkeypatch.setattr(impl.dist, "get_rank", lambda rank=rank: rank)
                monkeypatch.setattr(
                    impl,
                    "get_tp_group",
                    lambda tp_ranks=tp_ranks: SimpleNamespace(ranks=tp_ranks),
                )
                created = []

                def new_group(members, *, backend, timeout, created=created):
                    assert backend == "nccl" and timeout.total_seconds() == 90
                    created.append(members.copy())
                    return tuple(members)

                monkeypatch.setattr(impl.dist, "new_group", new_group)
                owner, owner_rank = impl._create_projection_group(dp_ranks, group)
                assert created == expected
                assert owner[owner_rank] == rank
                assert set(owner).issubset(dp_ranks)
                assert all(member % tp == tp_rank for member in owner)
    assert sorted(member for members in expected for member in members) == list(
        range(ranks.numel())
    )


@pytest.mark.parametrize(
    "dp_ranks,tp_ranks,rank,world,group",
    [
        ([0, 2, 4, 6], [0, 1], 0, 8, 3),  # group does not divide DP
        ([0, 2, 4, 6], [0, 1], 0, 8, 1),
        ([0, 2, 4, 6], [0, 1], 0, 8, True),
        ([0, 2, 4, 6], [0, 1], 0, 9, 2),  # incomplete replica
        ([0, 2, 4, 6], [0, 1], 1, 8, 2),  # other TP lane
        ([0, 4, 2, 6], [0, 1], 0, 8, 2),  # rank-order mismatch
        ([0, 2, 4, 6], [1, 0], 0, 8, 2),
        ([0, 4, 8, 12], [0, 1], 0, 16, 2),  # PCP>1 or noncartesian DP
        ([0, 2, 4, 4], [0, 1], 0, 8, 2),
    ],
)
def test_invalid_vocab_topology_fails_before_any_group_creation(
    monkeypatch, dp_ranks, tp_ranks, rank, world, group
):
    monkeypatch.setattr(impl.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(impl.dist, "get_world_size", lambda: world)
    monkeypatch.setattr(impl, "get_tp_group", lambda: SimpleNamespace(ranks=tp_ranks))
    created = []
    monkeypatch.setattr(impl.dist, "new_group", lambda *a, **k: created.append(a))
    with pytest.raises(ValueError, match="static"):
        impl._create_projection_group(dp_ranks, group)
    assert not created


@pytest.mark.parametrize("group", [2, 4])
@pytest.mark.parametrize("capacity", [6, 12, 18])
def test_tp2_projection_returns_local_vocab_before_the_callers_tp_gather(
    monkeypatch, group, capacity
):
    hidden_size, local_vocab = 4, 16
    weight = torch.arange(2 * local_vocab * hidden_size).float().view(-1, hidden_size)
    # Include an idle owner and unequal target/draft row counts under one bound.
    peers = [
        torch.full((rows, hidden_size), 1.0 + owner)
        for owner, rows in enumerate([0, 1, capacity - 1, capacity][:group])
    ]
    for owner, hidden in enumerate(peers):
        local_results = []
        for lane in range(2):
            local_weight = weight.chunk(2)[lane]
            p = impl.DPVocabProjection.__new__(impl.DPVocabProjection)
            p.capacity, p.group_size, p.group_rank = 18, group, owner
            p.hidden_size, p.vocab_size = hidden_size, local_vocab
            p.shard_rows, p.dtype, p.device = (
                local_vocab // group,
                weight.dtype,
                weight.device,
            )
            p.group = (lane, group)
            p.input = torch.empty(18, hidden_size)
            p.gathered = torch.empty(group * 18, hidden_size)
            p.partial = torch.empty(group * 18, p.shard_rows)
            p.received = torch.empty_like(p.partial)
            padded = [F.pad(x, (0, 0, 0, capacity - len(x))) for x in peers]

            def gather(out, inp, *, group, p=p, padded=padded, owner=owner):
                assert group == p.group
                torch.testing.assert_close(inp, padded[owner])
                out.copy_(torch.cat(padded))

            def exchange(
                out,
                partial,
                *,
                group,
                p=p,
                local_weight=local_weight,
                padded=padded,
                owner=owner,
            ):
                assert group == p.group
                shard = local_weight.chunk(p.group_size)[owner]
                torch.testing.assert_close(partial, F.linear(torch.cat(padded), shard))
                out.copy_(
                    torch.cat(
                        [
                            F.linear(padded[owner], w)
                            for w in local_weight.chunk(p.group_size)
                        ]
                    )
                )

            monkeypatch.setattr(impl.dist, "all_gather_into_tensor", gather)
            monkeypatch.setattr(impl.dist, "all_to_all_single", exchange)
            result = p.project(hidden, local_weight, capacity)
            assert result.shape == (len(hidden), local_vocab)
            torch.testing.assert_close(result, F.linear(hidden, local_weight))
            local_results.append(result)
        # Existing target and draft callers concatenate TP lanes in vocab order.
        full_logits = torch.cat(local_results, dim=-1)
        torch.testing.assert_close(full_logits, F.linear(hidden, weight))


def _install_harness(monkeypatch, tp=2):
    weight = torch.arange(32).view(8, 4).to(torch.bfloat16)
    head = SimpleNamespace(
        weight=weight,
        quant_method=impl.UnquantizedLinearMethod(),
        tp_size=tp,
        num_embeddings_per_partition=len(weight),
        shard_indices=(0, len(weight)),
    )
    language = SimpleNamespace(
        lm_head=head, logits_processor=SimpleNamespace(head_dtype=None)
    )
    draft = type("DSparkSpeculator", (), {})()
    draft.model = SimpleNamespace(lm_head=head)
    draft.speculative_config = SimpleNamespace(enforce_eager=True)
    draft._draft_topk, draft.num_speculative_steps = None, 5
    draft._sample_sequential = lambda *args: None
    draft.propose = lambda hidden, dp_sync=None, dummy_run=False: None
    runner = SimpleNamespace(
        model=language,
        speculator=draft,
        scheduler_config=SimpleNamespace(max_num_seqs=3),
        parallel_config=SimpleNamespace(distributed_executor_backend="mp"),
        model_state=SimpleNamespace(num_new_sampled_tokens_per_step=1),
        sample=lambda *args: None,
    )
    monkeypatch.setattr(impl, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(impl, "get_tp_group", lambda: SimpleNamespace(world_size=tp))
    monkeypatch.setattr(
        impl, "get_dp_group", lambda: SimpleNamespace(ranks=[0, 2, 4, 6])
    )
    calls = []

    def projector(current_weight, capacity, ranks, group):
        calls.append((current_weight, capacity, ranks, group))
        return SimpleNamespace(
            capacity=capacity,
            shard_rows=len(weight) // group,
            vocab_size=len(weight),
            dtype=weight.dtype,
            dummy=torch.zeros(1, 4),
            project=lambda *args: None,
        )

    monkeypatch.setattr(impl, "DPVocabProjection", projector)
    monkeypatch.setattr(impl, "_enable_scoped_head_dispatch", lambda: None)
    monkeypatch.setattr(impl.torch.cuda, "synchronize", lambda: None)
    return runner, calls


@pytest.mark.parametrize("tp", [1, 2])
def test_installer_keeps_local_head_metadata_and_dynamic_pair_contract(monkeypatch, tp):
    runner, calls = _install_harness(monkeypatch, tp)
    head = runner.model.lm_head
    controller = impl.install_dp_vocab_projection(runner, 4)
    assert calls == [(head.weight, 18, [0, 2, 4, 6], 4)]
    assert runner.model.lm_head is head and runner.speculator.model.lm_head is head
    assert head.tp_size == tp and head.num_embeddings_per_partition == 8
    assert controller.heads == (head, head)
    assert controller.dynamic_capacity_supported and not controller.dynamic_capacity
    controller.enable_dynamic_capacity()
    assert controller.capacity_for(SimpleNamespace(num_reqs=1)) == 6


@pytest.mark.parametrize("mismatch", ["tp", "local_rows", "draft_indices"])
def test_installer_rejects_inconsistent_local_vocab_before_collectives(
    monkeypatch, mismatch
):
    runner, calls = _install_harness(monkeypatch)
    if mismatch == "tp":
        runner.model.lm_head.tp_size = 1
    elif mismatch == "local_rows":
        runner.model.lm_head.num_embeddings_per_partition = 16
    else:
        runner.speculator.model.lm_head = SimpleNamespace(**vars(runner.model.lm_head))
        runner.speculator.model.lm_head.shard_indices = (8, 16)
    with pytest.raises(ValueError, match="same TP vocabulary shard"):
        impl.install_dp_vocab_projection(runner, 4)
    assert not calls
