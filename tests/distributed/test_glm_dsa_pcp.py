# SPDX-License-Identifier: Apache-2.0
"""Two-GPU PCP sparse attention, using real collectives and Iluvatar kernels.

Run with two free/available devices exposed via CUDA_VISIBLE_DEVICES. No model
weights are required: projections are identities; indexer, cache insertion,
Q packing, global-slot conversion and attention execute their native kernels.
"""

import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _check_sparse_indexer(rank, device, group):
    """Compare actual sparse selection and decode against an unpartitioned run."""
    from vllm.v1.attention.ops import pcp

    from vllm_iluvatar.model_executor.layers import sparse_attn_indexer as idx
    from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
        IluDeepSeekV32IndexerDecodeMetadata,
        IluDeepseekV32IndexerMetadata,
        IluDeepseekV32IndexerMetadataBuilder,
    )

    gen = torch.Generator().manual_seed(149)
    # Exact integer scores avoid near-ties at the Top-K boundary. A permutation
    # makes the selected slots sparse rather than a contiguous prefix/suffix.
    scores = torch.randperm(257, generator=gen)
    keys = torch.zeros(257, 128, dtype=torch.bfloat16, device=device)
    keys[:, 0] = (scores % 128 + 1).to(device)
    keys[:, 1] = (scores // 128).to(device)
    queries = torch.zeros(257, 32, 128, dtype=torch.bfloat16, device=device)
    queries[:, :, 0] = 1
    queries[:, :, 1] = 128
    weights = torch.ones(257, 32, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(5, 64, 128, dtype=torch.bfloat16, device=device)
    idx.ops.indexer_k_cache(keys[:249], cache, torch.arange(249, device=device))
    reference_cache = cache.clone()
    builder = object.__new__(IluDeepseekV32IndexerMetadataBuilder)
    builder.compress_ratio = 1
    builder.max_prefill_buffer_size = 1024
    builder.device = device

    def metadata(ids, slots, world_size, decode):
        # Split non-contiguous DualChunkSwap positions into local requests.
        boundaries = (
            [0]
            + [i for i in range(1, len(ids)) if ids[i] != ids[i - 1] + 1]
            + [len(ids)]
        )
        starts = torch.tensor(boundaries, dtype=torch.int32)
        ends = torch.tensor([ids[i - 1] + 1 for i in boundaries[1:]], dtype=torch.int32)
        block_table = torch.arange(5, device=device, dtype=torch.int32).repeat(
            len(ends), 1
        )
        chunk = builder.build_one_prefill_chunk(
            slice(0, len(ends)), slice(0, len(ids)), starts, ends, block_table
        )
        decode_meta = None
        if decode:
            decode_meta = IluDeepSeekV32IndexerDecodeMetadata(
                block_table=block_table,
                seq_lens=ends.to(device),
                decode_lens=(starts[1:] - starts[:-1]).to(device),
                requires_padding=False,
                cu_seqlen_ks=chunk.cu_seqlen_ks,
                cu_seqlen_ke=chunk.cu_seqlen_ke,
                valid_lens=chunk.valid_lens,
                cu_seqlens_kv=chunk.cu_seq_lens,
                cu_seqlens_q=chunk.cu_seqlens_q,
                max_context_len=chunk.max_context_len,
                max_q_len=chunk.max_q_len,
                max_kv_len=chunk.max_kv_len,
            )
        return IluDeepseekV32IndexerMetadata(
            seq_lens=ends.to(device),
            num_reqs=len(ends),
            max_query_len=chunk.max_q_len,
            max_seq_len=int(ends.max()),
            num_actual_tokens=len(ids),
            query_start_loc=starts.to(device),
            slot_mapping=torch.tensor(slots, device=device),
            head_dim=128,
            num_decodes=len(ids) if decode else 0,
            num_decode_tokens=len(ids) if decode else 0,
            num_prefills=0 if decode else len(ends),
            num_prefill_tokens=0 if decode else len(ids),
            pcp_world_size=world_size,
            prefill=None if decode else SimpleNamespace(chunks=[chunk]),
            decode=decode_meta,
        )

    def run(ids, meta, target_cache, padded):
        def local(tensor):
            out = tensor.new_zeros((padded, *tensor.shape[1:]))
            out[: len(ids)] = tensor[ids]
            return out

        result = torch.full((padded, 128), -1, dtype=torch.int32, device=device)
        with patch.object(
            idx,
            "get_forward_context",
            return_value=SimpleNamespace(attn_metadata={"idx": meta}),
        ):
            idx.ilu_sparse_attn_indexer(
                local(keys),
                "idx",
                target_cache,
                local(queries),
                local(keys),
                local(weights),
                128,
                128,
                320,
                1024,
                result,
            )
        return result[: len(ids)]

    for rank_ids, slots, decode in (
        (
            [[249, 250, 255], [251, 252, 253, 254]],
            [249, 250, 255, -1, 251, 252, 253, 254],
            False,
        ),
        ([[256], [256]], [256, -1], True),
    ):
        ids = rank_ids[rank]
        all_ids = sorted(set(sum(rank_ids, [])))
        with patch.object(pcp, "get_pcp_group", return_value=group):
            selected = run(
                ids,
                metadata(ids, slots, 2, decode),
                cache,
                max(map(len, rank_ids)),
            )
        reference = run(
            all_ids,
            metadata(all_ids, all_ids, 1, decode),
            reference_cache,
            len(all_ids),
        )
        assert torch.equal(cache, reference_cache)
        for row, pos in enumerate(ids):
            actual = selected[row].sort().values
            expected = reference[all_ids.index(pos)].sort().values
            oracle = scores[: pos + 1].topk(128).indices.sort().values.to(device)
            assert actual.unique().numel() == 128
            assert actual.min() >= 0 and actual.max() <= pos
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual.long(), oracle, rtol=0, atol=0)
        dist.barrier()


def _worker(rank, rendezvous, native_comm):
    import ixformer.distributed as ixfd
    import ixformer.inference.functions as ixf
    from ixformer.inference.backend_registry import registry
    from vllm.v1.attention.ops import pcp

    from vllm_iluvatar.model_executor.layers import sparse_attn_indexer as idx
    from vllm_iluvatar.models.glm_moe_dsa import attention as attn
    from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
        IluDeepseekV32IndexerMetadata,
        IluDeepseekV32IndexerMetadataBuilder,
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    # Initialize the barrier communicator before starting ixformer collectives.
    dist.barrier(device_ids=[rank])
    ixfd_group = ixfd.init_comm_with_store(dist.group.WORLD) if native_comm else None

    class Group:
        world_size = 2

        def all_gather(self, x, dim=0):
            assert dim == 0
            output = x.new_empty((2 * x.shape[0], *x.shape[1:]))
            if native_comm:
                ixfd.all_gather_into_tensor(
                    output, x.contiguous(), group=ixfd_group, async_op=True
                )
            else:
                dist.all_gather_into_tensor(output, x.contiguous())
            return output

    gen = torch.Generator().manual_seed(521)
    latent = (torch.randn(17, 512, generator=gen) / 4).bfloat16()
    rope = (torch.randn(17, 1, 64, generator=gen) / 4).bfloat16()
    q_full = (torch.randn(17, 16, 576, generator=gen) / 4).bfloat16()
    index_k = torch.randn(17, 128, generator=gen).bfloat16()
    index_q = torch.randn(17, 32, 128, generator=gen).bfloat16()
    scale = torch.tensor([0.01], device=device)
    pack_ref = registry.get_reference_impl("glm_dsa_fp8_q_pack")
    cache_ref = registry.get_reference_impl("glm_dsa_fp8_kv_insert")

    try:
        _check_sparse_indexer(rank, device, Group())
        # Exercise packed communication with mixed decode/prefill, non-contiguous
        # inputs, saturated FP8 values, negative slots and an entirely empty rank.
        for rank_ids, decodes in (
            ([[0, 1, 6], [0, 2, 3, 4]], 1),
            ([[0], []], 0),
        ):
            padded = max(map(len, rank_ids))
            source = torch.randn(7, 640, generator=gen).bfloat16()
            source[:, :2] = torch.tensor([1000, -1000]).bfloat16()
            all_kv, all_pe = source[:, :512], source[:, 512:576, None].transpose(1, 2)
            slots = torch.tensor(
                [
                    -1 if i >= len(ids) or (r > 0 and i < decodes) else ids[i]
                    for r, ids in enumerate(rank_ids)
                    for i in range(padded)
                ],
                device=device,
            )
            cache = torch.full((2, 4, 640), 177, dtype=torch.uint8, device=device)
            expected = cache.cpu()
            ids = sorted(set(sum(rank_ids, [])))
            cache_ref(
                all_kv[ids], all_pe[ids], expected, torch.tensor(ids), scale.cpu(), 4
            )
            local = source[rank_ids[rank]].to(device)
            with patch.object(pcp, "get_pcp_group", return_value=Group()):
                attn._gather_and_cache_packed_fp8_kv(
                    local[:, :512],
                    local[:, None, 512:576],
                    cache,
                    slots,
                    scale,
                    2,
                    decodes,
                )
            torch.testing.assert_close(cache.cpu(), expected, atol=0, rtol=0)

        for empty_rank in (False, True):
            layer = attn.IluGlmMoeDsaSparseMLA.__new__(attn.IluGlmMoeDsaSparseMLA)
            torch.nn.Module.__init__(layer)
            layer.prefix = "attn"
            layer.num_heads = layer._fp8_q_heads = 16
            layer.qk_nope_head_dim = layer.kv_lora_rank = layer.v_head_dim = 512
            layer.qk_rope_head_dim = 64
            layer.softmax_scale = 1 / 16
            layer.use_fp8_attn = True
            layer._decode_prepare_enabled = False
            layer.use_rms_rope_fp8_kv_fusion = False
            layer._fp8_kv_scale_frozen = True
            layer._k_scale = scale
            layer.kv_cache = torch.zeros(1, 64, 640, dtype=torch.uint8, device=device)
            layer.topk_indices_buffer = torch.full(
                (4, 128), -1, dtype=torch.int32, device=device
            )
            layer.topk_convert_cache = None
            layer.rotary_emb = lambda positions, q, k: (q, k)
            layer._k_up_proj = lambda q: q
            layer._v_up_proj = lambda x, outputs=None: (
                x if outputs is None else outputs.copy_(x)
            )
            ixf.glm_dsa_fp8_kv_insert(
                latent[:9].to(device),
                rope[:9].to(device),
                layer.kv_cache,
                torch.arange(9, device=device),
                scale,
                64,
            )
            index_cache = torch.zeros(1, 64, 128, dtype=torch.bfloat16, device=device)
            idx.ops.indexer_k_cache(
                index_k[:9].to(device), index_cache, torch.arange(9, device=device)
            )
            # Prefill after a cached prefix, then one replicated decode step.
            batches = (
                [([[9], []], [9, -1], 0)]
                if empty_rank
                else [
                    (
                        [[9, 10, 15], [11, 12, 13, 14]],
                        [9, 10, 15, -1, 11, 12, 13, 14],
                        0,
                    ),
                    ([[16], [16]], [16, -1], 1),
                ]
            )
            for rank_ids, slot_ids, decode_tokens in batches:
                ids = rank_ids[rank]
                n = len(ids)
                padded = max(map(len, rank_ids))
                # Each contiguous DualChunkSwap segment becomes one local request.
                lengths = [2, n - 2] if n > 2 else [n]
                starts = torch.tensor(
                    [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
                )
                ends = torch.tensor(
                    [ids[int(end) - 1] + 1 if end else 9 for end in starts[1:]],
                    dtype=torch.int32,
                )
                block_table = torch.zeros(
                    len(lengths), 1, dtype=torch.int32, device=device
                )
                builder = object.__new__(IluDeepseekV32IndexerMetadataBuilder)
                builder.compress_ratio = 1
                builder.max_prefill_buffer_size = 128
                builder.device = device
                chunk = builder.build_one_prefill_chunk(
                    slice(0, len(lengths)), slice(0, n), starts, ends, block_table
                )
                slots = torch.tensor(slot_ids, device=device)
                # Use prefill logits for the single decode row too; cache gather
                # still receives num_decode_tokens=1 and must bypass collectives.
                imeta = IluDeepseekV32IndexerMetadata(
                    seq_lens=ends.to(device),
                    num_reqs=len(lengths),
                    max_query_len=max(lengths),
                    max_seq_len=int(ends.max()),
                    num_actual_tokens=n,
                    query_start_loc=starts.to(device),
                    slot_mapping=slots,
                    head_dim=128,
                    num_decodes=0,
                    num_decode_tokens=decode_tokens,
                    num_prefills=1,
                    num_prefill_tokens=n - decode_tokens,
                    pcp_world_size=2,
                    prefill=SimpleNamespace(chunks=[chunk]),
                )
                meta = SimpleNamespace(
                    num_actual_tokens=n,
                    pcp_world_size=2,
                    num_decode_tokens=decode_tokens,
                    slot_mapping=slots,
                    block_size=64,
                    req_id_per_token=torch.repeat_interleave(
                        torch.arange(len(lengths), dtype=torch.int32),
                        torch.tensor(lengths),
                    ).to(device),
                    block_table=block_table,
                )

                def local(x, padded=padded, n=n, ids=ids):
                    y = torch.zeros(
                        (padded, *x.shape[1:]), dtype=x.dtype, device=device
                    )
                    y[:n] = x[ids].to(device)
                    return y

                with (
                    patch.object(pcp, "get_pcp_group", return_value=Group()),
                    patch.object(
                        idx,
                        "get_forward_context",
                        return_value=SimpleNamespace(attn_metadata={"idx": imeta}),
                    ),
                    patch.object(
                        attn,
                        "get_forward_context",
                        return_value=SimpleNamespace(attn_metadata={"attn": meta}),
                    ),
                ):
                    idx.ilu_sparse_attn_indexer(
                        local(latent),
                        "idx",
                        index_cache,
                        local(index_q),
                        local(index_k),
                        torch.ones(padded, 32, device=device, dtype=torch.bfloat16),
                        128,
                        128,
                        64,
                        128,
                        layer.topk_indices_buffer,
                    )
                    for row, pos in enumerate(ids):
                        selected = layer.topk_indices_buffer[row].cpu()
                        assert sorted(selected[selected >= 0].tolist()) == list(
                            range(pos + 1)
                        )
                    output = layer.forward(
                        local(q_full),
                        local(latent),
                        local(rope)[:, 0],
                        torch.tensor(ids + [0] * (padded - n), device=device),
                    )
                torch.cuda.synchronize()
                if n:
                    packed_q = torch.empty(n, 16, 640, dtype=torch.uint8)
                    q_scale = torch.empty(n, 16)
                    pack_ref(
                        q_full[ids, :, :512], q_full[ids, :, 512:], packed_q, q_scale
                    )
                    packed_kv = torch.zeros(1, 64, 640, dtype=torch.uint8)
                    end = max(max(items, default=-1) for items in rank_ids) + 1
                    cache_ref(
                        latent[:end],
                        rope[:end],
                        packed_kv,
                        torch.arange(end),
                        scale.cpu(),
                        64,
                    )
                    assert torch.equal(layer.kv_cache.cpu(), packed_kv)
                    torch.testing.assert_close(
                        index_cache[0, :end].cpu(), index_k[:end], rtol=0, atol=0
                    )
                    expected_idx = torch.arange(128, dtype=torch.int32).repeat(n, 1)
                    lengths_ref = torch.tensor(ids, dtype=torch.int32) + 1
                    expected_idx[expected_idx >= lengths_ref[:, None]] = -1
                    reference = ixf.dsa_flash_mla_sparse_with_fp8_kvcache_diff_kv(
                        packed_q[:, None],
                        packed_kv.unsqueeze(2),
                        expected_idx[:, None],
                        lengths_ref,
                        q_scale,
                        scale.cpu(),
                        1 / 16,
                        backend="torch",
                    )
                    # PCP=1 uses the same device packer. CPU/GPU division
                    # differs at FP8 rounding boundaries, so retain the CPU
                    # pack above for the numerical reference, not bit equality.
                    native_q = torch.empty_like(packed_q, device=device)
                    native_scale = torch.empty_like(q_scale, device=device)
                    full_q_device = q_full.to(device)
                    all_q = torch.empty(17, 16, 640, dtype=torch.uint8, device=device)
                    all_scale = torch.empty(17, 16, device=device)
                    ixf.glm_dsa_fp8_q_pack(
                        full_q_device[:, :, :512],
                        full_q_device[:, :, 512:],
                        all_q,
                        all_scale,
                    )
                    native_q.copy_(all_q[ids])
                    native_scale.copy_(all_scale[ids])
                    unpartitioned = ixf.dsa_flash_mla_sparse_with_fp8_kvcache_diff_kv(
                        native_q[:, None],
                        packed_kv.unsqueeze(2).to(device),
                        layer.topk_indices_buffer[:n, None],
                        lengths_ref.to(device),
                        native_scale,
                        scale,
                        1 / 16,
                    ).squeeze(1)
                    torch.testing.assert_close(
                        output[:n].view_as(unpartitioned), unpartitioned, atol=0, rtol=0
                    )
                    # FP8 quantized softmax/PV tolerance against the CPU reference.
                    torch.testing.assert_close(
                        output[:n].cpu().view_as(reference),
                        reference,
                        atol=0.15,
                        rtol=0.15,
                    )
                assert torch.count_nonzero(output[n:]) == 0
                dist.barrier()
    finally:
        # On failure, let mp.spawn report it and terminate the peer. Destroying
        # a group here can wait forever while the peer is inside a collective.
        if sys.exc_info()[0] is None:
            if ixfd_group is not None:
                ixfd.destroy_process_group(ixfd_group)
            dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two Iluvatar GPUs")
@pytest.mark.parametrize("native_comm", [False, True])
def test_glm_dsa_pcp_two_gpus(tmp_path, native_comm):
    mp.spawn(
        _worker,
        args=(f"file://{tmp_path / 'rendezvous'}", native_comm),
        nprocs=2,
        join=True,
    )
